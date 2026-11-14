# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
from collections.abc import Callable, Mapping
from typing import NamedTuple

import torch


logger = logging.getLogger(__name__)

# Dequantise a large host tensor on the device (see `_should_dequantize_on_gpu`). The
# threshold is measured: from ~16K elements up the arithmetic is memory-bound and the host
# path is two orders of magnitude slower, while below it the round trip dominates.
_GPU_DEQUANT_MIN_NUMEL = 1 << 14
import torch.nn.functional as F


FP8_BLOCK_SIZE = 128
# Keep the dequantized tensor on the device instead of returning it to the
# host. Measured on the GLM-5.3-Flash import: the pageable D2H of the 33 MB
# BF16 result dominated the round trip (20.9 ms/tensor -> ~13 min total),
# while the device multiply itself is 0.1 ms. Callers on this import path
# (the conversion mappings) are device-agnostic reshape/slice/cat chains and
# the destination param lives on the device, so no copy back is needed.
# Set False to restore the CPU-returning contract for a caller that needs it.
# On the DSV4.1 load path the peft init broadcasts the dequantized
# shared-expert weights with the CPU-tensor contract
# (_broadcast_shared_expert_weights); a device-returning dequant wedges that
# broadcast at load time (measured: ranks spin in c10d broadcast with 0%
# GPU and no progress). Keep False here.
_DEQUANT_RETURN_ON_DEVICE = False
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
MXFP4_BLOCK_SIZE = 32

_FP4_E2M1_TABLE_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def is_fp8_tensor(tensor: torch.Tensor) -> bool:
    """Return whether *tensor* uses one of PyTorch's FP8 dtypes."""
    return tensor.dtype in FP8_DTYPES


def is_float8_e8m0_dtype(dtype: torch.dtype) -> bool:
    """Return whether *dtype* is PyTorch's E8M0 scale dtype."""
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    return e8m0_dtype is not None and dtype == e8m0_dtype


def scale_from_amax(amax: torch.Tensor, max_quantized_value: float, scale_dtype: torch.dtype) -> torch.Tensor:
    """Build positive quantization scales in the same scale family as ``scale_dtype``."""
    scale = torch.where(amax > 0, amax / max_quantized_value, torch.ones_like(amax))
    if is_float8_e8m0_dtype(scale_dtype):
        scale = scale.clamp(min=2.0**-127, max=2.0**127)
        return torch.exp2(torch.ceil(torch.log2(scale)))
    return scale


def _dequantize_blockwise_exact(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Exact-block dequantization as ONE broadcast multiply (any device).

    ``weight`` is viewed as ``(sM, block, sN, block)`` so block ``(bi, bj)``
    indexes ``scale_inv[bi, bj]``; the caller guarantees both dims are exact
    multiples of ``block_size``.
    """
    M, N = weight.shape
    sM, sN = scale_inv.shape
    w = weight.view(sM, block_size, sN, block_size).float()
    out = w * scale_inv[:sM, :sN].float()[:, None, :, None]
    return out.reshape(M, N).to(dtype)


def dequantize_fp8_blockwise(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    block_size: int = FP8_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 weights with one scale per 2D block.

    DeepSeek-V3 and MiniMax-M2 store linear weights as FP8 tensors with a
    separate ``*_scale_inv`` tensor. Each scale applies to one 128x128 weight
    block by default.

    The block scales are applied as ONE broadcast multiply when the weight is
    an exact multiple of ``block_size`` — the geometry every checkpoint in
    this tree uses (``scale_inv`` is ``(M // 128, N // 128)`` and both weight
    dims are multiples of 128). The loop form this replaced cost ~36 ms per
    4096x2048 expert tensor on CPU (512 slice-multiplies); over the
    GLM-5.3-Flash checkpoint's 37,338 FP8 tensors that was ~22 minutes of
    pure Python-loop overhead — essentially the entire worker boot time
    (measured 2026-10-11). The ragged-edge loop is kept for shapes that are
    not exact multiples. Both paths compute the same float32 product before
    casting, so the result is numerically identical (pinned by
    test_dequantize_fp8_blockwise_matches_the_ragged_loop).
    """
    M, N = weight.shape
    sM, sN = scale_inv.shape
    if M % block_size == 0 and N % block_size == 0:
        if weight.device.type == "cpu" and weight.numel() >= _GPU_DEQUANT_MIN_NUMEL and torch.cuda.is_available():
            # GPU offload (measured 2026-10-11 on H20-3e): the exact-block
            # broadcast is memory-bound and ~340x faster on the device than on
            # the host (0.1 ms vs 34 ms for a 4096x2048 expert tensor), which
            # over this checkpoint's 37,338 FP8 tensors is the difference
            # between ~21 minutes and ~1 minute of import wall clock. The
            # round trip returns a CPU tensor so the caller's contract is
            # unchanged, and the arithmetic is the same float32 product the
            # host path computes (bit-identical: same multiply, same cast).
            out = _dequantize_blockwise_exact(weight.cuda(), scale_inv.cuda(), block_size, dtype)
            return out if _DEQUANT_RETURN_ON_DEVICE else out.cpu()
        return _dequantize_blockwise_exact(weight, scale_inv, block_size, dtype)
    w = weight.float()
    out = torch.empty_like(w)
    for bi in range(sM):
        for bj in range(sN):
            r0, r1 = bi * block_size, min((bi + 1) * block_size, M)
            c0, c1 = bj * block_size, min((bj + 1) * block_size, N)
            out[r0:r1, c0:c1] = w[r0:r1, c0:c1] * scale_inv[bi, bj]
    return out.to(dtype)


def maybe_dequantize_fp8_blockwise(
    weight: torch.Tensor,
    scale_inv: torch.Tensor | None = None,
    *,
    block_size: int = FP8_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 block-scaled weights, falling back to a plain cast."""
    if not is_fp8_tensor(weight):
        return weight
    if weight.ndim == 2 and scale_inv is not None:
        return dequantize_fp8_blockwise(weight, scale_inv, block_size=block_size, dtype=dtype)
    return weight.float().to(dtype)


def maybe_dequantize_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 weights with a scalar or broadcastable scale tensor."""
    if not is_fp8_tensor(weight):
        return weight
    if scale_inv is None:
        return weight.to(dtype)
    return weight.to(dtype) * scale_inv.to(dtype)


class Fp8BlockwisePair(NamedTuple):
    """A checkpoint blockwise-FP8 linear weight: FP8 payload plus its F32 block scales.

    The two tensors travel together through the HF->Megatron mapping so the scale
    undergoes exactly the layout transform the payload does. ``data`` is held in its
    raw byte view, which makes every layout op (chunk/cat/transpose/scatter) a byte
    move that cannot round a value the checkpoint already rounded.
    """

    data: torch.Tensor
    scale_inv: torch.Tensor


def as_fp8_bytes(weight: torch.Tensor) -> torch.Tensor:
    """Return the raw bytes of an 8-bit payload without reinterpreting its values."""
    if weight.dtype == torch.uint8:
        return weight
    if is_fp8_tensor(weight) or is_float8_e8m0_dtype(weight.dtype):
        return weight.view(torch.uint8)
    raise ValueError(f"expected an 8-bit FP8 payload, got dtype {weight.dtype}")


def is_blockwise_fp8_param(param: object) -> bool:
    """Whether *param* is a TransformerEngine blockwise-FP8 quantized tensor.

    Duck-typed on the rowwise storage instead of ``isinstance`` against TE's class:
    importing TE would make the conversion path importable only on a host with a
    working CUDA TE build, while the only thing a direct write needs to know is
    where the parameter keeps its bytes.
    """
    return getattr(param, "_rowwise_data", None) is not None and getattr(param, "_rowwise_scale_inv", None) is not None


def blockwise_fp8_scale_block_len(param: object, default: int = FP8_BLOCK_SIZE) -> int:
    """Block length of a blockwise-FP8 parameter's scale grid (128 for this checkpoint family)."""
    metadata = {}
    get_metadata = getattr(param, "get_metadata", None)
    if callable(get_metadata):
        try:
            metadata = get_metadata() or {}
        except (AttributeError, RuntimeError, TypeError):
            metadata = {}
    block_len = getattr(metadata.get("quantizer"), "block_len", None)
    return int(block_len) if block_len else default


def write_blockwise_fp8_param(
    param: torch.Tensor,
    rowwise_data: torch.Tensor,
    rowwise_scale_inv: torch.Tensor,
    *,
    param_name: str,
) -> None:
    """Install checkpoint FP8 bytes and F32 block scales into a TE blockwise-FP8 param.

    ``QuantizedTensor.copy_`` is not usable here: it dequantizes the source and
    requantizes with the destination's quantizer, which re-rounds every value. The
    four TE storages are therefore written directly, and the columnwise orientation
    (which TE needs for the backward GEMM) is derived as the byte transpose of the
    rowwise one: with whole 128x128 blocks the transpose maps every block onto a
    block, so payload bytes and scales are carried over unchanged.

    Fails loudly rather than degrading to a round trip: a parameter whose storage
    geometry does not match the transported payload is a mapping bug, and silently
    dequantizing it would reintroduce exactly the error this path exists to remove.
    """
    if not is_blockwise_fp8_param(param):
        raise ValueError(f"{param_name}: destination is not a blockwise FP8 parameter")

    if rowwise_data.ndim != 2:
        raise ValueError(f"{param_name}: expected a 2D blockwise FP8 payload, got shape {tuple(rowwise_data.shape)}")
    data = as_fp8_bytes(rowwise_data)
    block_size = blockwise_fp8_scale_block_len(param)
    rows, cols = data.shape
    if rows % block_size or cols % block_size:
        raise ValueError(
            f"{param_name}: blockwise FP8 direct write requires whole {block_size}x{block_size} blocks, "
            f"got payload shape {tuple(data.shape)}"
        )
    if tuple(rowwise_scale_inv.shape) != (rows // block_size, cols // block_size):
        raise ValueError(
            f"{param_name}: scale grid {tuple(rowwise_scale_inv.shape)} does not match the "
            f"{block_size}x{block_size} grid of payload shape {tuple(data.shape)}"
        )
    if os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES") != "1":
        # The requirement is the ENV VAR, not the dtype of what we are holding: with the
        # variable unset TransformerEngine's blockwise quantiser defaults to
        # `force_pow_2_scales=True` and reads every F32 scale as an exponent of two, so
        # installing `amax / 448` scales here would dequantise every block wrongly — the
        # bytes would differ from TE's own `copy_` on all of them (measured on the GPU,
        # 2026-10-30: 65536 of 65536 bytes differ under the default, identical with the
        # variable set).
        raise ValueError(
            f"{param_name}: the blockwise FP8 direct write requires "
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1 to be exported before TransformerEngine "
            "is imported. Without it TE reads blockwise scales as powers of two, so the "
            "F32 scales installed here would dequantise into different weights."
        )
    if rowwise_scale_inv.dtype != torch.float32:
        raise ValueError(
            f"{param_name}: expected F32 block scales — TE only reproduces the checkpoint's "
            f"dequantization in that scale family (NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1) — "
            f"got {rowwise_scale_inv.dtype}"
        )

    target_data = param._rowwise_data
    target_scale = param._rowwise_scale_inv
    if target_data.element_size() != data.element_size():
        raise ValueError(
            f"{param_name}: destination FP8 storage has element size {target_data.element_size()}, "
            f"cannot carry {data.dtype} bytes"
        )
    if target_scale.dtype != rowwise_scale_inv.dtype:
        raise ValueError(
            f"{param_name}: destination FP8 storage carries {target_scale.dtype} scales while the "
            f"checkpoint stores {rowwise_scale_inv.dtype}. These only agree when TE is configured "
            "for the checkpoint's scale family (NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1): writing F32 "
            "scales into an E8M0 grid would silently mis-scale every block."
        )

    device = target_data.device
    data = data.to(device=device)
    scale = rowwise_scale_inv.to(device=device)
    # TE reads both storages as dense buffers, so a strided view (e.g. a column shard
    # taken without materializing it) would dequantize to garbage while every shape
    # check above still passed. Materializing is byte-preserving, which is why it is the
    # cheap half of this guard; the geometry was already validated against the destination.
    if not data.is_contiguous():
        data = data.contiguous()
    if not scale.is_contiguous():
        scale = scale.contiguous()
    # TE stores the payload as uint8 bytes; a TE build that keeps the FP8 dtype on the
    # storage is served by the same bytes.
    if target_data.dtype != torch.uint8 and target_data.dtype != data.dtype:
        data = data.view(target_data.dtype)

    # The payload IS the parameter's logical weight: TransformerEngine pads the SCALE
    # grids (to a whole number of quantiser blocks) but keeps the data storages at exactly
    # the logical shape and its transpose (measured on this model: a [32768, 1280] weight
    # arrives with `_rowwise_scale_inv` (256, 12) beside an exact `_columnwise_data`).
    # So data may never be padded (that would install bytes over nothing) and may never be
    # trimmed (that would drop real weights) — only the scale grids are zero-filled.
    if tuple(data.shape) != tuple(target_data.shape):
        raise ValueError(
            f"{param_name}: transported payload {tuple(data.shape)} is not the destination "
            f"FP8 storage {tuple(target_data.shape)}. TransformerEngine does not pad the "
            "data storages, so a mismatch means the payload is not this weight"
        )
    # The LOGICAL grid stays separate from the padded rowwise storage: TransformerEngine
    # pads each 2D scale storage so its inner dimension is a multiple of four, and it does
    # that to the two storages INDEPENDENTLY — a [32768, 1280] weight's logical 128x128 grid
    # is (256, 10), TE stores it rowwise as (256, 12) and columnwise as (10, 256). Padding
    # first and then transposing the padded grid yields (12, 256), which cannot fit the
    # columnwise storage and is not what TE's own transpose produces (it copies only the
    # rows the destination can hold). Measured on the real checkpoint: the load failed with
    # "destination columnwise scale grid (10, 256) is smaller than the grid the checkpoint
    # carries (12, 256)" — the "checkpoint" there was this intermediate padded grid.
    logical_scale = scale
    rowwise_scale = _pad_scale_grid_to(
        target_scale, logical_scale, param_name=param_name, label="rowwise scale grid"
    )
    param._rowwise_data = data
    param._rowwise_scale_inv = rowwise_scale
    if getattr(param, "_columnwise_data", None) is not None:
        columnwise = data.t().contiguous()
        if tuple(columnwise.shape) != tuple(param._columnwise_data.shape):
            raise ValueError(
                f"{param_name}: the transpose of the payload {tuple(columnwise.shape)} is not "
                f"the destination columnwise storage {tuple(param._columnwise_data.shape)} — "
                "TE keeps that storage at exactly the logical transpose, so this is not the "
                "same weight"
            )
        param._columnwise_data = columnwise
        # Transpose the LOGICAL grid, then pad — the destination's own padding is applied
        # to the transposed layout, exactly as TE allocates it.
        logical_columnwise = logical_scale.t().contiguous()
        if logical_columnwise.shape[0] > param._columnwise_scale_inv.shape[0] or (
            logical_columnwise.shape[1] > param._columnwise_scale_inv.shape[1]
        ):
            raise ValueError(
                f"{param_name}: the logical columnwise scale grid "
                f"{tuple(logical_columnwise.shape)} exceeds the destination storage "
                f"{tuple(param._columnwise_scale_inv.shape)} — the destination is narrower "
                "than the checkpoint's weight, which is not a padding difference"
            )
        param._columnwise_scale_inv = _pad_scale_grid_to(
            param._columnwise_scale_inv,
            logical_columnwise,
            param_name=param_name,
            label="columnwise scale grid",
        )


def _pad_scale_grid_to(destination, source, *, param_name: str, label: str):
    """Zero-fill a scale grid up to the destination's shape (never trim, never shrink)."""
    rows_pad = destination.shape[0] - source.shape[0]
    cols_pad = destination.shape[1] - source.shape[1]
    if rows_pad < 0 or cols_pad < 0:
        raise ValueError(
            f"{param_name}: destination {label} {tuple(destination.shape)} is smaller than the "
            f"grid the checkpoint carries {tuple(source.shape)}"
        )
    if not rows_pad and not cols_pad:
        return source
    padded = torch.zeros(
        (source.shape[0] + rows_pad, source.shape[1] + cols_pad),
        dtype=source.dtype,
        device=source.device,
    )
    padded[: source.shape[0], : source.shape[1]] = source
    return padded


def _should_dequantize_on_gpu(tensor: torch.Tensor) -> bool:
    """Return whether *tensor* should be dequantized on the current CUDA device.

    Only a CPU tensor at least ``_GPU_DEQUANT_MIN_NUMEL`` elements large is eligible: a tensor
    that already lives on an accelerator needs no transfer at all, and below the cutoff the
    transfers and kernel launches cost the same order as the whole host op, so the saving would
    be a fraction of a millisecond. The choice is automatic -- there is no configuration knob:
    callers see one path, and the host implementation stays as the fallback when no device is
    available or the device attempt runs out of memory.
    """
    if tensor.device.type != "cpu" or tensor.numel() < _GPU_DEQUANT_MIN_NUMEL:
        return False
    return torch.cuda.is_available()


_dequant_gpu_oom_logged = False


def _log_gpu_dequant_oom_once(kind: str, name: str, error: torch.cuda.OutOfMemoryError) -> None:
    """Warn once about a device OOM that pushed a dequantization back onto the host."""
    global _dequant_gpu_oom_logged
    if _dequant_gpu_oom_logged:
        return
    _dequant_gpu_oom_logged = True
    label = f" for {name!r}" if name else ""
    logger.warning(
        "GPU %s dequantization%s ran out of device memory and fell back to the host path"
        " (this warning is logged once): %s",
        kind,
        label,
        error,
    )


def dequantize_fp8_e4m3fn_with_scale(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    name: str = "",
    block_size: int = FP8_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 E4M3 weights with a companion scale tensor.

    Supports three common HF checkpoint scale layouts:
    1-D per-row or row-block scales, 2-D per-row K tiles, and 2-D block scales.

    A host ``weight`` of at least :data:`_GPU_DEQUANT_MIN_NUMEL` elements is
    dequantized on the current CUDA device and the result is returned **on that
    device**: the DSV4.1 import hands it to a mapping chain that ends in a
    ``copy_`` into a device parameter, so a trip back to the host would throw
    the gain away. Every other call — small host tensors, device tensors, or a
    host without CUDA — behaves exactly as before. Both paths run the same
    float32 math and return the same dtype and values; a device out-of-memory
    error falls back to the host path instead of failing the load. There is no
    switch: the choice is automatic, so callers see one path.
    """
    if _should_dequantize_on_gpu(weight):
        device = torch.device("cuda", torch.cuda.current_device())
        try:
            return _dequantize_fp8_e4m3fn_with_scale_math(
                weight.to(device),
                scale.to(device),
                name=name,
                block_size=block_size,
                dtype=dtype,
            )
        except torch.cuda.OutOfMemoryError as error:
            # Never let the offload turn a loadable checkpoint into a failed
            # load. Drop the cache the failed attempt may have fragmented
            # before the host path allocates the same result again.
            _log_gpu_dequant_oom_once("FP8", name, error)
            torch.cuda.empty_cache()
    return _dequantize_fp8_e4m3fn_with_scale_math(weight, scale, name=name, block_size=block_size, dtype=dtype)


def _dequantize_fp8_e4m3fn_with_scale_math(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    name: str = "",
    block_size: int = FP8_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Device-agnostic math behind :func:`dequantize_fp8_e4m3fn_with_scale`."""
    weight_f32 = weight.to(torch.float32)
    scale_f32 = scale.to(torch.float32)
    if scale_f32.dim() == 1:
        if scale_f32.shape[0] == weight_f32.shape[0]:
            scale_exp = scale_f32.unsqueeze(1)
        else:
            scale_exp = scale_f32.repeat_interleave(block_size)[: weight_f32.shape[0]].unsqueeze(1)
        scale_exp = scale_exp.expand_as(weight_f32)
    elif scale_f32.dim() != 2:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(f"Unsupported FP8 scale rank{label}: scale={tuple(scale.shape)}")
    elif scale_f32.shape[0] == weight_f32.shape[0]:
        tile_k = weight_f32.shape[1] // scale_f32.shape[1]
        scale_exp = scale_f32.repeat_interleave(tile_k, dim=1)[:, : weight_f32.shape[1]]
    else:
        scale_exp = scale_f32.repeat_interleave(block_size, dim=0)[: weight_f32.shape[0]]
        scale_exp = scale_exp.repeat_interleave(block_size, dim=1)[:, : weight_f32.shape[1]]
    if scale_exp.shape != weight_f32.shape:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"FP8 dequant shape mismatch{label}: "
            f"weight={tuple(weight_f32.shape)} scale={tuple(scale.shape)} "
            f"scale_exp={tuple(scale_exp.shape)}"
        )
    return (weight_f32 * scale_exp).to(dtype)


def _dequantize_e4m3_scale_impl(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    name: str = "",
    block_size: int = FP8_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """The dequantization itself, on whatever device ``weight`` lives on."""
    weight_f32 = weight.to(torch.float32)
    scale_f32 = scale.to(torch.float32)
    if scale_f32.dim() == 1:
        if scale_f32.shape[0] == weight_f32.shape[0]:
            scale_exp = scale_f32.unsqueeze(1)
        else:
            scale_exp = scale_f32.repeat_interleave(block_size)[: weight_f32.shape[0]].unsqueeze(1)
        scale_exp = scale_exp.expand_as(weight_f32)
    elif scale_f32.dim() != 2:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(f"Unsupported FP8 scale rank{label}: scale={tuple(scale.shape)}")
    elif scale_f32.shape[0] == weight_f32.shape[0]:
        tile_k = weight_f32.shape[1] // scale_f32.shape[1]
        scale_exp = scale_f32.repeat_interleave(tile_k, dim=1)[:, : weight_f32.shape[1]]
    else:
        scale_exp = scale_f32.repeat_interleave(block_size, dim=0)[: weight_f32.shape[0]]
        scale_exp = scale_exp.repeat_interleave(block_size, dim=1)[:, : weight_f32.shape[1]]
    if scale_exp.shape != weight_f32.shape:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"FP8 dequant shape mismatch{label}: "
            f"weight={tuple(weight_f32.shape)} scale={tuple(scale.shape)} "
            f"scale_exp={tuple(scale_exp.shape)}"
        )
    return (weight_f32 * scale_exp).to(dtype)


def _quantize_fp8_2d_blocks(
    weight: torch.Tensor,
    source_scale: torch.Tensor,
    *,
    name: str = "",
    block_size: int = FP8_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    scale_rows, scale_cols = source_scale.shape
    expected_shape = (
        (rows + block_size - 1) // block_size,
        (cols + block_size - 1) // block_size,
    )
    if (scale_rows, scale_cols) != expected_shape:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"Unsupported FP8 scale geometry{label}: "
            f"weight={tuple(weight.shape)} scale={tuple(source_scale.shape)} expected={expected_shape}"
        )

    weight_f32 = weight.to(torch.float32)
    pad_rows = scale_rows * block_size - rows
    pad_cols = scale_cols * block_size - cols
    if pad_rows or pad_cols:
        weight_f32 = F.pad(weight_f32, (0, pad_cols, 0, pad_rows))

    blocks = weight_f32.view(scale_rows, block_size, scale_cols, block_size).transpose(1, 2)
    scale_f32 = scale_from_amax(blocks.abs().amax(dim=(-1, -2)), FP8_E4M3_MAX, source_scale.dtype)
    q_blocks = (blocks / scale_f32[:, :, None, None]).to(torch.float8_e4m3fn)
    q_weight = q_blocks.transpose(1, 2).reshape(scale_rows * block_size, scale_cols * block_size)[:rows, :cols]
    return q_weight.contiguous(), scale_f32.to(dtype=source_scale.dtype)


def _quantize_fp8_per_row_tiles(
    weight: torch.Tensor,
    source_scale: torch.Tensor,
    *,
    name: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    scale_rows, scale_cols = source_scale.shape
    if scale_rows != rows or scale_cols <= 0 or cols % scale_cols != 0:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"Unsupported per-row FP8 scale geometry{label}: "
            f"weight={tuple(weight.shape)} scale={tuple(source_scale.shape)}"
        )

    tile_cols = cols // scale_cols
    grouped = weight.to(torch.float32).reshape(rows, scale_cols, tile_cols)
    scale_f32 = scale_from_amax(grouped.abs().amax(dim=-1), FP8_E4M3_MAX, source_scale.dtype)
    q_weight = (grouped / scale_f32[:, :, None]).to(torch.float8_e4m3fn).reshape(rows, cols)
    return q_weight.contiguous(), scale_f32.to(dtype=source_scale.dtype)


def _quantize_fp8_1d_scale(
    weight: torch.Tensor,
    source_scale: torch.Tensor,
    *,
    name: str = "",
    block_size: int = FP8_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, _ = weight.shape
    scale_len = source_scale.numel()
    weight_f32 = weight.to(torch.float32)

    if scale_len == rows:
        scale_f32 = scale_from_amax(weight_f32.abs().amax(dim=1), FP8_E4M3_MAX, source_scale.dtype)
        q_weight = (weight_f32 / scale_f32[:, None]).to(torch.float8_e4m3fn)
        return q_weight.contiguous(), scale_f32.to(dtype=source_scale.dtype)

    expected_len = (rows + block_size - 1) // block_size
    if scale_len != expected_len:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"Unsupported 1-D FP8 scale geometry{label}: "
            f"weight={tuple(weight.shape)} scale={tuple(source_scale.shape)} expected_len={expected_len}"
        )

    q_weight = torch.empty_like(weight_f32, dtype=torch.float8_e4m3fn)
    scale_f32 = torch.empty(scale_len, dtype=torch.float32, device=weight.device)
    for block_idx in range(scale_len):
        row_start = block_idx * block_size
        row_end = min(row_start + block_size, rows)
        block = weight_f32[row_start:row_end]
        block_scale = scale_from_amax(block.abs().amax().reshape(()), FP8_E4M3_MAX, source_scale.dtype)
        q_weight[row_start:row_end] = (block / block_scale).to(torch.float8_e4m3fn)
        scale_f32[block_idx] = block_scale
    return q_weight.contiguous(), scale_f32.to(dtype=source_scale.dtype)


def quantize_fp8_e4m3fn_like_scale(
    weight: torch.Tensor,
    source_scale: torch.Tensor,
    *,
    name: str = "",
    block_size: int = FP8_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight to FP8 E4M3 using ``source_scale`` geometry and dtype."""
    if weight.ndim != 2:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(f"FP8 quantized export expects a 2-D weight{label}, got {weight.ndim}D")

    if source_scale.ndim == 1:
        return _quantize_fp8_1d_scale(weight, source_scale, name=name, block_size=block_size)

    if source_scale.ndim != 2:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(f"Unsupported FP8 scale rank{label}: scale={tuple(source_scale.shape)}")

    if source_scale.shape[0] == weight.shape[0]:
        return _quantize_fp8_per_row_tiles(weight, source_scale, name=name)

    return _quantize_fp8_2d_blocks(weight, source_scale, name=name, block_size=block_size)


def dequantize_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    """Dequantize GPT-OSS MXFP4 block/scales tensors."""
    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"
    scales = scales.to(torch.int32) - 127
    lut = torch.tensor(_FP4_E2M1_TABLE_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)


def dequantize_mxfp4_e2m1_packed(
    weight_packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize MXFP4 E2M1 weights packed two values per byte.

    ``scale`` is expected to be one scale per row and per K tile. ``uint8``
    E8M0 tensors use exponent bias 127 and are decoded to powers of two.
    """
    w_u8 = weight_packed.view(torch.uint8)
    lo = (w_u8 & 0xF).to(torch.int64)
    hi = (w_u8 >> 4).to(torch.int64)

    table = torch.tensor(_FP4_E2M1_TABLE_VALUES, dtype=torch.float32, device=weight_packed.device)
    logical = torch.stack([table[lo], table[hi]], dim=-1).reshape(weight_packed.shape[0], -1)

    if scale.dtype == torch.uint8:
        scale_f32 = torch.ldexp(
            torch.ones_like(scale, dtype=torch.float32),
            scale.to(torch.int32) - 127,
        )
    else:
        scale_f32 = scale.to(torch.float32)
    if scale_f32.dim() != 2 or scale_f32.shape[0] != logical.shape[0] or logical.shape[1] % scale_f32.shape[1] != 0:
        raise RuntimeError(
            f"Unsupported MXFP4 scale geometry: "
            f"weight={tuple(weight_packed.shape)} logical={tuple(logical.shape)} scale={tuple(scale.shape)}"
        )
    block_size = logical.shape[1] // scale_f32.shape[1]
    scale_exp = scale_f32.repeat_interleave(block_size, dim=1)

    return (logical * scale_exp).to(dtype)


def e8m0_to_float32(scale: torch.Tensor) -> torch.Tensor:
    """Decode an E8M0 scale grid to float32 powers of two.

    E8M0 stores only an exponent (bias 127), so the value is ``2 ** (x - 127)`` and
    ``0xFF`` is NaN by definition. Checkpoints in this family carry their scales that
    way — one byte per 32x32 block (or per row-K tile for the packed 4-bit experts) —
    while a TransformerEngine blockwise parameter stores F32 scales over 128x128
    blocks. Anything that wants to write those bytes into such a parameter has to
    decode the exponent grid first, which is this function.
    """
    if scale.dtype == torch.uint8 or is_float8_e8m0_dtype(scale.dtype):
        raw = scale.view(torch.uint8).to(torch.int32)
        if bool((raw == 0xFF).any()):
            raise RuntimeError("E8M0 scale grid contains the NaN code 0xFF")
        return torch.ldexp(torch.ones_like(raw, dtype=torch.float32), raw - 127)
    return scale.to(torch.float32)


def dequantize_block_scaled_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    name: str = "",
    block_size: int = 32,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize an FP8 payload whose scale grid is fine-grained and possibly E8M0.

    Two grids occur in this checkpoint family: a 2-D block grid (``rows/32 x cols/32``
    for the dense and shared-expert linears) and a per-row K-tile grid (one scale per
    32-element group for the packed 4-bit experts' FP8 siblings). Both are finer than
    the 128x128 F32 grid a TransformerEngine blockwise parameter wants, so this is the
    first half of "requantise onto the parameter's own grid".
    """
    weight_f32 = weight.to(torch.float32)
    scale_f32 = e8m0_to_float32(scale)
    label = f" for {name!r}" if name else ""
    if scale_f32.dim() == 1:
        if scale_f32.shape[0] != weight_f32.shape[0]:
            raise RuntimeError(
                f"Unsupported 1-D FP8 scale geometry{label}: weight={tuple(weight.shape)} "
                f"scale={tuple(scale.shape)}"
            )
        scale_exp = scale_f32.unsqueeze(1).expand_as(weight_f32)
    elif scale_f32.dim() == 2 and scale_f32.shape[0] == weight_f32.shape[0]:
        tile = weight_f32.shape[1] // scale_f32.shape[1]
        scale_exp = scale_f32.repeat_interleave(tile, dim=1)[:, : weight_f32.shape[1]]
    elif scale_f32.dim() == 2:
        scale_exp = scale_f32.repeat_interleave(block_size, dim=0)[: weight_f32.shape[0]]
        scale_exp = scale_exp.repeat_interleave(block_size, dim=1)[:, : weight_f32.shape[1]]
    else:
        raise RuntimeError(
            f"Unsupported FP8 scale rank{label}: scale={tuple(scale.shape)}"
        )
    return (weight_f32 * scale_exp).to(dtype)


def requantize_to_blockwise_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    name: str = "",
    block_size: int = 32,
    target_block: int = FP8_BLOCK_SIZE,
    mxfp4: bool = False,
    device: torch.device | None = None,
) -> Fp8BlockwisePair:
    """Re-quantise a checkpoint's quantised weight onto a 128x128 F32 blockwise grid.

    This is the generic half of "write the checkpoint straight into an FP8 parameter":
    a checkpoint that is already quantised CANNOT be copied byte-for-byte unless its
    scale grid is the parameter's own (the DSV4.1 checkpoint is 32x32 E8M0, and its
    routed experts are packed MXFP4 — neither is a TransformerEngine 128x128 F32
    blockwise layout). What can be skipped is the whole BF16 materialisation and the
    host round trip: decode to F32, requantise in one pass, and hand back the payload
    bytes plus the F32 scale grid the parameter stores.

    Returned ``data`` is a byte view so the caller's mapping ops move bytes; the
    caller writes it with :func:`write_blockwise_fp8_param`.
    """
    # Run the arithmetic where the destination lives: the checkpoint hunks arrive on the
    # host, and a 20 GB model's float32 dequantise + requantise on the CPU is minutes of
    # work (measured 2026-10-30: 250 s in the profiler's `other` bucket, versus 1.9 s
    # once the same tensors were dequantised on device).
    if device is not None:
        weight = weight.to(device=device)
        scale = scale.to(device=device)
    if mxfp4:
        dequantized = dequantize_mxfp4_e2m1_packed(weight, scale, dtype=torch.float32)
    else:
        dequantized = dequantize_block_scaled_fp8(
            weight, scale, name=name, block_size=block_size, dtype=torch.float32
        )
    rows, cols = dequantized.shape
    scale_rows = (rows + target_block - 1) // target_block
    scale_cols = (cols + target_block - 1) // target_block
    template = torch.empty((scale_rows, scale_cols), dtype=torch.float32, device=dequantized.device)
    q_weight, scale_f32 = _quantize_fp8_2d_blocks(
        dequantized, template, name=name, block_size=target_block
    )
    return Fp8BlockwisePair(
        data=as_fp8_bytes(q_weight).contiguous(),
        scale_inv=scale_f32.to(torch.float32).contiguous(),
    )


def is_mxfp4_e2m1_scale_geometry(
    weight: torch.Tensor,
    source_scale: torch.Tensor,
    *,
    block_size: int = MXFP4_BLOCK_SIZE,
) -> bool:
    """Return whether ``source_scale`` describes packed MXFP4 E2M1 K tiles."""
    return (
        weight.ndim == 2
        and source_scale.ndim == 2
        and source_scale.shape[0] == weight.shape[0]
        and source_scale.shape[1] * block_size == weight.shape[1]
    )


def quantize_mxfp4_e2m1_like_scale(
    weight: torch.Tensor,
    source_scale: torch.Tensor,
    *,
    name: str = "",
    block_size: int = MXFP4_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight to packed MXFP4 E2M1 using source scale geometry."""
    if weight.ndim != 2:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(f"MXFP4 quantized export expects a 2-D weight{label}, got {weight.ndim}D")

    rows, cols = weight.shape
    if cols % 2 != 0 or cols % block_size != 0 or source_scale.shape != (rows, cols // block_size):
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"Unsupported MXFP4 geometry{label}: weight={tuple(weight.shape)} scale={tuple(source_scale.shape)}"
        )

    weight_f32 = weight.to(torch.float32)
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device=weight.device)
    scale_f32 = torch.empty(tuple(source_scale.shape), dtype=torch.float32, device=weight.device)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32, device=weight.device)

    max_chunk_elements = 16_000_000
    rows_per_chunk = max(1, min(rows, max_chunk_elements // max(cols, 1)))
    scale_cols = source_scale.shape[1]
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        chunk = weight_f32[row_start:row_end].reshape(-1, scale_cols, block_size)
        chunk_amax = chunk.abs().amax(dim=-1)
        if source_scale.dtype == torch.uint8:
            unrounded_scale = torch.where(
                chunk_amax > 0,
                chunk_amax / FP4_E2M1_MAX,
                torch.ones_like(chunk_amax),
            )
            chunk_scale = torch.exp2(torch.ceil(torch.log2(unrounded_scale)).clamp(min=-127, max=127))
        else:
            chunk_scale = scale_from_amax(chunk_amax, FP4_E2M1_MAX, source_scale.dtype)
        scale_f32[row_start:row_end] = chunk_scale

        normalized = chunk / chunk_scale[:, :, None]
        codes = torch.bucketize(normalized.abs(), boundaries).to(torch.uint8)
        codes = (codes | ((normalized < 0).to(torch.uint8) * 8)).reshape(row_end - row_start, cols)

        lo = codes[:, 0::2].to(torch.int16)
        hi = codes[:, 1::2].to(torch.int16)
        packed[row_start:row_end] = (lo | (hi << 4)).to(torch.uint8)

    if source_scale.dtype == torch.uint8:
        output_scale = (torch.log2(scale_f32).round() + 127).clamp(min=0, max=254).to(torch.uint8)
    else:
        output_scale = scale_f32.to(dtype=source_scale.dtype)
    return packed.contiguous().view(torch.int8), output_scale


def write_blockwise_fp8_params_batch(entries) -> None:
    """Write a batch of (param, rowwise_data, rowwise_scale_inv, param_name)
    triples with ONE multi-tensor copy per tensor family.

    The per-parameter pageable H2D in write_blockwise_fp8_param was the
    measured dominant cost of the direct-write import (~31 ms/param over
    ~4700 params per rank, ~145 s): each payload.to(device) allocates a
    staging buffer and synchronizes. Grouping the parameters and copying
    with torch._foreach_copy_ amortizes the launches and the sync across the
    batch; the destinations keep TE's own storages (in-place, no rebinding).

    Validates every entry BEFORE copying anything, so a bad entry fails loud
    without half-writing the batch. Callers must batch under no_grad.
    """
    if not entries:
        return
    # Validate first: the same checks as the single writer, none of the writes.
    prepared = []
    for param, data, scale_inv, param_name in entries:
        if not is_blockwise_fp8_param(param):
            raise ValueError(f"{param_name}: destination is not a blockwise FP8 parameter")
        if data.ndim != 2:
            raise ValueError(
                f"{param_name}: blockwise FP8 direct import requires a 2-D payload, "
                f"got ndim={data.ndim} shape {tuple(data.shape)}"
            )
        rows, cols = data.shape
        block_size = blockwise_fp8_scale_block_len(param)
        if rows % block_size or cols % block_size:
            raise ValueError(
                f"{param_name}: blockwise FP8 direct import requires whole "
                f"{block_size}x{block_size} blocks, got payload shape {tuple(data.shape)}"
            )
        target_data = param._rowwise_data
        target_scale = param._rowwise_scale_inv
        if tuple(target_data.shape) != (rows, cols):
            raise ValueError(
                f"{param_name}: destination FP8 storage has shape {tuple(target_data.shape)}, "
                f"payload shape is {(rows, cols)}"
            )
        # TransformerEngine pads each 2D scale storage independently of the payload, so the
        # destination is allowed to be LARGER than the checkpoint's logical grid; the pad
        # carries no weights and is zero-filled. A destination SMALLER than the grid is not a
        # padding difference and still fails inside _pad_scale_grid_to.
        padded_scale_inv = _pad_scale_grid_to(
            target_scale, scale_inv, param_name=param_name, label="rowwise scale grid"
        )
        if scale_inv.dtype != torch.float32 or target_scale.dtype != torch.float32:
            raise ValueError(
                f"{param_name}: blockwise FP8 scales must be F32 on both sides (the "
                "checkpoint family and the destination family), got "
                f"{scale_inv.dtype}/{target_scale.dtype}"
            )
        if target_data.dtype != torch.uint8 or data.dtype != torch.uint8:
            raise ValueError(
                f"{param_name}: the batched writer copies raw bytes, expected uint8 on both "
                f"sides, got {data.dtype} -> {target_data.dtype}"
            )
        prepared.append((param, data, scale_inv, padded_scale_inv, param_name, rows, cols))

    # One multi-tensor H2D per family: N payload copies + N scale copies in
    # two launches, one synchronization each.
    with torch.no_grad():
        torch._foreach_copy_([entry[0]._rowwise_data for entry in prepared], [entry[1] for entry in prepared])
        torch._foreach_copy_([entry[0]._rowwise_scale_inv for entry in prepared], [entry[3] for entry in prepared])
        # The columnwise orientation is derived per parameter — GPU transposes,
        # cheap next to the batched H2D.
        for param, _data, logical_scale, _padded_scale, param_name, rows, cols in prepared:
            if getattr(param, "_columnwise_data", None) is not None:
                if tuple(param._columnwise_data.shape) != (cols, rows):
                    raise ValueError(
                        f"{param_name}: destination columnwise storage has shape "
                        f"{tuple(param._columnwise_data.shape)}, expected ({cols}, {rows})"
                    )
                param._columnwise_data = param._rowwise_data.t().contiguous()
                # Transpose the LOGICAL grid, then pad the destination: transposing the
                # already-padded grid yields a shape the columnwise storage cannot hold, and
                # it is not what TransformerEngine's own transpose produces.
                logical_columnwise = logical_scale.t().contiguous()
                param._columnwise_scale_inv = _pad_scale_grid_to(
                    param._columnwise_scale_inv,
                    logical_columnwise,
                    param_name=param_name,
                    label="columnwise scale grid",
                ).to(param._rowwise_data.device)


def maybe_dequantize_hf_quantized_weight(
    hf_param: str | dict[str, str],
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = FP8_BLOCK_SIZE,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Load and dequantize HF ``*.weight`` tensors that carry sibling ``*.scale`` tensors.

    ``block_size`` is the edge of the checkpoint's FP8 scale tile — the value its
    ``quantization_config.weight_block_size`` declares — and is forwarded to
    :func:`dequantize_fp8_e4m3fn_with_scale`. The default is the 128x128 grid used
    by DeepSeek-V3 / MiniMax-M2; a checkpoint quantized on a finer grid must pass
    its own edge, because the wrong edge still satisfies the dequant shape assert
    (the scale grid is repeated and then sliced) and silently returns mis-scaled
    values. The int8/MXFP4 branch ignores ``block_size``: those scales are
    per-row over 32-element K tiles.
    """
    if isinstance(hf_param, dict):
        return {
            k: maybe_dequantize_hf_quantized_weight(v, hf_state_dict, dtype=dtype, block_size=block_size)
            for k, v in hf_param.items()
        }

    weight = hf_state_dict[hf_param]

    if weight.dtype == torch.int8:
        scale_key = hf_param[: -len(".weight")] + ".scale" if hf_param.endswith(".weight") else None
        if scale_key is None or scale_key not in hf_state_dict:
            return weight.to(dtype)
        return dequantize_mxfp4_e2m1_packed(weight, hf_state_dict[scale_key], dtype=dtype)

    if weight.dtype != torch.float8_e4m3fn:
        return weight

    if not hf_param.endswith(".weight"):
        return weight.to(dtype)

    scale_key = hf_param[: -len(".weight")] + ".scale"
    if scale_key not in hf_state_dict:
        return weight.to(dtype)

    return dequantize_fp8_e4m3fn_with_scale(
        weight, hf_state_dict[scale_key], name=hf_param, block_size=block_size, dtype=dtype
    )


def requantize_hf_weight_scale_pairs(
    converted_weights_dict: Mapping[str, torch.Tensor],
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    use_mxfp4: Callable[[str, torch.Tensor, torch.Tensor], bool] | None = None,
) -> dict[str, torch.Tensor]:
    """Recreate quantized HF ``*.weight``/``*.scale`` pairs using source scale layout.

    ``use_mxfp4`` lets model bridges opt specific parameters into packed MXFP4
    output. Other scaled weights are emitted as FP8 E4M3.
    """
    result: dict[str, torch.Tensor] = {}
    for hf_param, weight in converted_weights_dict.items():
        if not hf_param.endswith(".weight"):
            if hf_param.endswith(".scale"):
                weight_key = hf_param[: -len(".scale")] + ".weight"
                if weight_key in converted_weights_dict:
                    continue
            result[hf_param] = weight
            continue

        scale_key = hf_param[: -len(".weight")] + ".scale"
        if scale_key not in hf_state_dict:
            result[hf_param] = weight
            continue

        source_scale = hf_state_dict[scale_key]
        if use_mxfp4 is not None and use_mxfp4(hf_param, weight, source_scale):
            q_weight, q_scale = quantize_mxfp4_e2m1_like_scale(weight, source_scale, name=hf_param)
        else:
            q_weight, q_scale = quantize_fp8_e4m3fn_like_scale(weight, source_scale, name=hf_param)

        result[hf_param] = q_weight
        result[scale_key] = q_scale
    return result


def dequantize_int4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    group_size: int = 32,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Dequantize Kimi INT4 packed weights to bfloat16.

    The checkpoint stores eight offset-binary INT4 values in each int32 slot and
    carries per-group scales beside the packed tensor.
    """
    del weight_shape, group_size

    local_out, local_packed_in = weight_packed.shape
    local_in = local_packed_in * 8

    target_device = weight_packed.device if device is None else torch.device(device)
    use_cuda = target_device.type == "cuda" and torch.cuda.is_available()

    if use_cuda:
        weight_packed = weight_packed.to(target_device)
        weight_scale = weight_scale.to(target_device)

    shifts = torch.arange(8, device=weight_packed.device) * 4

    packed_unsqueezed = weight_packed.unsqueeze(-1)
    unpacked = ((packed_unsqueezed >> shifts) & 0xF).float()
    unpacked = unpacked.reshape(local_out, local_in)

    unpacked = unpacked - 8

    scale = weight_scale.float()
    if scale.ndim == 1:
        local_num_groups = scale.numel() // local_out
        scale = scale.view(local_out, local_num_groups)
    else:
        scale = scale.view(local_out, -1)

    local_num_groups = scale.shape[1]
    elements_per_group = local_in // local_num_groups

    scale_expanded = scale.repeat_interleave(elements_per_group, dim=1)

    if scale_expanded.shape[1] < local_in:
        scale_expanded = torch.nn.functional.pad(
            scale_expanded, (0, local_in - scale_expanded.shape[1]), value=scale_expanded[:, -1:].mean()
        )
    scale_expanded = scale_expanded[:, :local_in]
    result = unpacked * scale_expanded

    return result.to(torch.bfloat16)


def quantize_to_int4(
    weight: torch.Tensor,
    group_size: int = 32,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize bfloat16/float16 weights to Kimi INT4 packed format."""
    out_features, in_features = weight.shape
    weight_shape = torch.tensor([out_features, in_features], dtype=torch.int32, device=weight.device)

    w = weight.float()

    num_groups = (in_features + group_size - 1) // group_size
    w_grouped = w.view(out_features, num_groups, -1)

    group_max = w_grouped.abs().amax(dim=-1)
    scale = group_max / 7.0
    scale = scale.clamp(min=1e-10)

    scale_expanded = scale.unsqueeze(-1).expand_as(w_grouped)
    w_q = (w_grouped / scale_expanded).round().clamp(-8, 7)

    w_q = w_q.view(out_features, -1)[:, :in_features]
    w_q = (w_q + 8).to(torch.uint8)

    assert in_features % 8 == 0, f"in_features must be divisible by 8, got {in_features}"

    w_q_grouped = w_q.view(out_features, in_features // 8, 8).to(torch.int32)

    packed = torch.zeros(out_features, in_features // 8, dtype=torch.int32, device=weight.device)
    for i in range(8):
        packed |= (w_q_grouped[:, :, i] & 0xF) << (i * 4)

    weight_packed = packed
    weight_scale = scale.to(scale_dtype)

    return weight_packed, weight_scale, weight_shape
