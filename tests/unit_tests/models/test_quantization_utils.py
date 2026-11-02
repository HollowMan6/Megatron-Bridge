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

import pytest
import torch

from megatron.bridge.models.conversion.quantization_utils import (
    dequantize_fp8_blockwise,
    dequantize_fp8_e4m3fn_with_scale,
    dequantize_int4,
    dequantize_mxfp4,
    dequantize_mxfp4_e2m1_packed,
    is_mxfp4_e2m1_scale_geometry,
    maybe_dequantize_fp8,
    maybe_dequantize_fp8_blockwise,
    maybe_dequantize_hf_quantized_weight,
    quantize_fp8_e4m3fn_like_scale,
    quantize_mxfp4_e2m1_like_scale,
    quantize_to_int4,
    requantize_hf_weight_scale_pairs,
)


def test_dequantize_fp8_blockwise_applies_distinct_scales():
    weight = torch.ones(256, 256, dtype=torch.float8_e4m3fn)
    scale_inv = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = dequantize_fp8_blockwise(weight, scale_inv).float()

    assert result.dtype == torch.float32
    assert torch.all(result[:128, :128] == 1.0)
    assert torch.all(result[:128, 128:] == 2.0)
    assert torch.all(result[128:, :128] == 3.0)
    assert torch.all(result[128:, 128:] == 4.0)


def test_maybe_dequantize_fp8_blockwise_passthrough_and_fallback_cast():
    bf16_weight = torch.ones(4, 4, dtype=torch.bfloat16)
    assert maybe_dequantize_fp8_blockwise(bf16_weight) is bf16_weight

    fp8_weight = torch.ones(4, 4, dtype=torch.float8_e4m3fn)
    result = maybe_dequantize_fp8_blockwise(fp8_weight)

    assert result.dtype == torch.bfloat16
    assert torch.all(result == 1.0)


def test_maybe_dequantize_fp8_applies_broadcastable_scale():
    fp8_weight = torch.ones(2, 2, dtype=torch.float8_e4m3fn)
    scale_inv = torch.tensor([2.0])

    result = maybe_dequantize_fp8(fp8_weight, scale_inv)

    assert result.dtype == torch.bfloat16
    assert torch.all(result == 2.0)


def test_dequantize_mxfp4_uses_low_then_high_nibbles():
    blocks = torch.tensor([[[0x21]]], dtype=torch.uint8)
    scales = torch.tensor([[127]], dtype=torch.uint8)

    result = dequantize_mxfp4(blocks, scales, dtype=torch.float32)

    assert result.shape == (1, 2)
    assert torch.equal(result, torch.tensor([[0.5, 1.0]]))


def test_quantize_fp8_e4m3fn_like_scale_roundtrips_scaled_weight():
    weight = torch.full((4, 4), 2.0, dtype=torch.bfloat16)
    source_scale = torch.ones((1, 1), dtype=torch.float32)

    q_weight, q_scale = quantize_fp8_e4m3fn_like_scale(weight, source_scale)
    result = dequantize_fp8_e4m3fn_with_scale(q_weight, q_scale)

    assert q_weight.dtype == torch.float8_e4m3fn
    assert q_scale.shape == source_scale.shape
    assert q_scale.dtype == source_scale.dtype
    assert torch.allclose(result.float(), weight.float())


@pytest.mark.parametrize(
    ("weight", "source_scale", "block_size"),
    [
        (
            torch.tensor([[1.0, -1.0], [2.0, -2.0]], dtype=torch.bfloat16),
            torch.ones(2),
            128,
        ),
        (
            torch.tensor(
                [
                    [2.0, -2.0, 2.0, -2.0],
                    [2.0, -2.0, 2.0, -2.0],
                    [4.0, -4.0, 4.0, -4.0],
                ],
                dtype=torch.bfloat16,
            ),
            torch.ones(2),
            2,
        ),
        (
            torch.tensor([[1.0, -1.0, 2.0, -2.0], [3.0, -3.0, 4.0, -4.0]], dtype=torch.bfloat16),
            torch.ones(2, 2),
            128,
        ),
    ],
)
def test_quantize_fp8_e4m3fn_like_scale_roundtrips_supported_scale_layouts(weight, source_scale, block_size):
    q_weight, q_scale = quantize_fp8_e4m3fn_like_scale(weight, source_scale, block_size=block_size)
    result = dequantize_fp8_e4m3fn_with_scale(q_weight, q_scale, block_size=block_size)

    assert q_weight.dtype == torch.float8_e4m3fn
    assert q_scale.shape == source_scale.shape
    assert q_scale.dtype == source_scale.dtype
    assert torch.allclose(result.float(), weight.float())


@pytest.mark.parametrize(
    ("weight", "source_scale", "message"),
    [
        (torch.ones(2, 2, 2), torch.ones(2), "expects a 2-D weight"),
        (torch.ones(2, 4), torch.ones(1, 1, 1), "Unsupported FP8 scale rank"),
        (torch.ones(3, 4), torch.ones(4), "Unsupported 1-D FP8 scale geometry"),
        (torch.ones(2, 5), torch.ones(2, 2), "Unsupported per-row FP8 scale geometry"),
        (torch.ones(4, 4), torch.ones(3, 1), "Unsupported FP8 scale geometry"),
    ],
)
def test_quantize_fp8_e4m3fn_like_scale_rejects_unsupported_geometry(weight, source_scale, message):
    with pytest.raises(RuntimeError, match=message):
        quantize_fp8_e4m3fn_like_scale(weight, source_scale, block_size=2)


def test_dequantize_fp8_e4m3fn_with_scale_rejects_unsupported_geometry():
    fp8_weight = torch.ones(2, 3, dtype=torch.float8_e4m3fn)

    with pytest.raises(RuntimeError, match="Unsupported FP8 scale rank"):
        dequantize_fp8_e4m3fn_with_scale(fp8_weight, torch.ones(1, 1, 1))

    with pytest.raises(RuntimeError, match="FP8 dequant shape mismatch"):
        dequantize_fp8_e4m3fn_with_scale(fp8_weight, torch.ones(2, 2))


def test_quantize_dequantize_mxfp4_e2m1_packed_roundtrips_representable_values():
    values = torch.tensor(
        [
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
        ],
        dtype=torch.float32,
    ).repeat(2)
    weight = values.reshape(1, 32).to(torch.bfloat16)
    source_scale = torch.ones((1, 1), dtype=torch.float32)

    packed, scale = quantize_mxfp4_e2m1_like_scale(weight, source_scale)
    result = dequantize_mxfp4_e2m1_packed(packed, scale)

    assert packed.dtype == torch.int8
    assert packed.shape == (1, 16)
    assert scale.shape == source_scale.shape
    assert torch.equal(result.float(), weight.float())


def test_quantize_dequantize_mxfp4_e2m1_packed_supports_uint8_e8m0_scales():
    values = torch.tensor(
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -1.0,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
            0.0,
            1.0,
            2.0,
            3.0,
        ],
        dtype=torch.float32,
    ).repeat(2)
    weight = values.reshape(1, 32).to(torch.bfloat16)
    source_scale = torch.full((1, 1), 127, dtype=torch.uint8)

    packed, scale = quantize_mxfp4_e2m1_like_scale(weight, source_scale)
    result = dequantize_mxfp4_e2m1_packed(packed, scale)

    assert scale.dtype == torch.uint8
    assert torch.equal(scale, source_scale)
    assert torch.equal(result.float(), weight.float())

    doubled = dequantize_mxfp4_e2m1_packed(packed, torch.full((1, 1), 128, dtype=torch.uint8))
    assert torch.equal(doubled.float(), weight.float() * 2)


def test_mxfp4_scale_geometry_checks_logical_unpacked_shape():
    weight = torch.ones(2, 64)

    assert is_mxfp4_e2m1_scale_geometry(weight, torch.ones(2, 2))
    assert not is_mxfp4_e2m1_scale_geometry(weight, torch.ones(2, 1))
    assert not is_mxfp4_e2m1_scale_geometry(torch.ones(2, 2, 32), torch.ones(2, 2))


def test_mxfp4_helpers_reject_unsupported_geometry():
    with pytest.raises(RuntimeError, match="Unsupported MXFP4 scale geometry"):
        dequantize_mxfp4_e2m1_packed(torch.zeros(1, 16, dtype=torch.int8), torch.ones(2, 1))

    with pytest.raises(RuntimeError, match="expects a 2-D weight"):
        quantize_mxfp4_e2m1_like_scale(torch.ones(1, 1, 32), torch.ones(1, 1))

    with pytest.raises(RuntimeError, match="Unsupported MXFP4 geometry"):
        quantize_mxfp4_e2m1_like_scale(torch.ones(1, 30), torch.ones(1, 1))


def test_maybe_dequantize_hf_quantized_weight_dispatches_by_dtype_and_sibling_scale():
    fp8_weight = torch.ones(2, 2, dtype=torch.float8_e4m3fn)
    mxfp4_weight = torch.zeros(1, 16, dtype=torch.int8)
    hf_state_dict = {
        "fp8.weight": fp8_weight,
        "fp8.scale": torch.ones(2),
        "fp8.activation": fp8_weight,
        "mxfp4.weight": mxfp4_weight,
        "mxfp4.scale": torch.ones(1, 1),
        "int8_without_scale.weight": mxfp4_weight,
        "bf16.weight": torch.ones(1, dtype=torch.bfloat16),
    }

    result = maybe_dequantize_hf_quantized_weight(
        {"fp8": "fp8.weight", "mxfp4": "mxfp4.weight"},
        hf_state_dict,
        dtype=torch.float32,
    )

    assert set(result) == {"fp8", "mxfp4"}
    assert result["fp8"].dtype == torch.float32
    assert result["mxfp4"].dtype == torch.float32
    assert maybe_dequantize_hf_quantized_weight("fp8.activation", hf_state_dict).dtype == torch.bfloat16
    assert maybe_dequantize_hf_quantized_weight("int8_without_scale.weight", hf_state_dict).dtype == torch.bfloat16
    assert maybe_dequantize_hf_quantized_weight("bf16.weight", hf_state_dict) is hf_state_dict["bf16.weight"]


def test_requantize_hf_weight_scale_pairs_emits_scale_siblings():
    weight_key = "layers.0.ffn.experts.0.w1.weight"
    scale_key = "layers.0.ffn.experts.0.w1.scale"
    weight = torch.ones((1, 32), dtype=torch.bfloat16)
    source_scale = torch.ones((1, 1), dtype=torch.float32)
    stale_scale = torch.full((1, 1), 9.0, dtype=torch.float32)

    result = requantize_hf_weight_scale_pairs(
        {weight_key: weight, scale_key: stale_scale},
        {scale_key: source_scale},
        use_mxfp4=lambda *_: True,
    )

    assert set(result) == {weight_key, scale_key}
    assert result[weight_key].dtype == torch.int8
    assert result[scale_key].shape == source_scale.shape
    assert not torch.equal(result[scale_key], stale_scale)


def test_requantize_hf_weight_scale_pairs_preserves_unscaled_weights_and_stale_scales():
    weight_key = "layers.0.self_attn.q_proj.weight"
    stale_scale_key = "layers.0.self_attn.k_proj.scale"
    scaled_weight_key = "layers.0.mlp.down_proj.weight"
    scaled_scale_key = "layers.0.mlp.down_proj.scale"
    unscaled_weight = torch.ones(1, 2, dtype=torch.bfloat16)
    stale_scale = torch.full((1, 1), 9.0)
    scaled_weight = torch.full((2, 4), 2.0, dtype=torch.bfloat16)
    source_scale = torch.ones(2)

    result = requantize_hf_weight_scale_pairs(
        {
            weight_key: unscaled_weight,
            stale_scale_key: stale_scale,
            scaled_weight_key: scaled_weight,
            scaled_scale_key: stale_scale,
            "metadata": torch.tensor(1.0),
        },
        {scaled_scale_key: source_scale},
    )

    assert result[weight_key] is unscaled_weight
    assert result["metadata"].item() == 1.0
    assert stale_scale_key in result
    assert scaled_scale_key in result
    assert result[scaled_weight_key].dtype == torch.float8_e4m3fn
    assert result[scaled_scale_key].shape == source_scale.shape


def test_quantize_dequantize_int4_preserves_shape_and_dtype():
    weight = torch.linspace(-1.0, 1.0, steps=32).view(1, 32).to(torch.bfloat16)

    packed, scale, shape = quantize_to_int4(weight)
    result = dequantize_int4(packed, scale, shape)

    assert packed.shape == (1, 4)
    assert shape.tolist() == [1, 32]
    assert result.shape == weight.shape
    assert result.dtype == torch.bfloat16


# A 64x64 float8_e4m3fn weight whose scales are a genuine 2x2 grid of 32x32
# blocks -- the DeepSeek-V4.1 geometry (``weight_block_size: [32, 32]``), which a
# 128x128 default cannot describe. Both grids are asymmetric on purpose, so a
# transposed expansion changes the quadrant values too. Every product of a
# payload entry and a scale entry is exact in bfloat16.
_BLOCK32_PAYLOAD_QUADRANTS = [[1.0, 2.0], [3.0, 4.0]]
_BLOCK32_SCALE_QUADRANTS = [[1.0, 2.0], [4.0, 8.0]]
_BLOCK32_DEQUANTIZED_QUADRANTS = [[1.0, 4.0], [12.0, 32.0]]


def _block32_fp8_weight_and_scale():
    """Build a 64x64 FP8 weight plus the 2x2 scale grid of its 32x32 blocks."""
    payload = torch.ones((2, 32, 2, 32), dtype=torch.float8_e4m3fn)
    for row_block in range(2):
        for col_block in range(2):
            payload[row_block, :, col_block, :] = _BLOCK32_PAYLOAD_QUADRANTS[row_block][col_block]
    return payload.reshape(64, 64), torch.tensor(_BLOCK32_SCALE_QUADRANTS, dtype=torch.float32)


def _quadrant_values(tensor, block_size=32):
    """Return the single value held by each ``block_size``-square tile."""
    quadrants = []
    for row_block in range(tensor.shape[0] // block_size):
        row = []
        for col_block in range(tensor.shape[1] // block_size):
            tile = tensor[
                row_block * block_size : (row_block + 1) * block_size,
                col_block * block_size : (col_block + 1) * block_size,
            ]
            uniques = tile.float().unique()
            assert uniques.numel() == 1, f"tile ({row_block}, {col_block}) is not constant: {uniques.tolist()}"
            row.append(float(uniques.item()))
        quadrants.append(row)
    return quadrants


def test_dequantize_fp8_e4m3fn_with_scale_honours_the_checkpoint_block_grid():
    """A 32-block checkpoint must be dequantized on its own grid, not on the 128 default."""
    weight, scale = _block32_fp8_weight_and_scale()

    exact = dequantize_fp8_e4m3fn_with_scale(weight, scale, block_size=32)

    assert _quadrant_values(exact) == _BLOCK32_DEQUANTIZED_QUADRANTS

    # The FP8_BLOCK_SIZE default still satisfies the dequant shape assert: the
    # 2x2 grid is repeat_interleaved by 128 and sliced back to 64, which keeps
    # only scale[0, 0], so the whole weight comes back as payload * 1.0.
    default = dequantize_fp8_e4m3fn_with_scale(weight, scale)

    assert _quadrant_values(default) == _BLOCK32_PAYLOAD_QUADRANTS
    assert not torch.equal(default, exact)


def test_maybe_dequantize_hf_quantized_weight_threads_block_size_to_fp8_dequant():
    weight, scale = _block32_fp8_weight_and_scale()
    hf_state_dict = {
        "layers.0.attn.wkv.weight": weight,
        "layers.0.attn.wkv.scale": scale,
    }

    exact = maybe_dequantize_hf_quantized_weight("layers.0.attn.wkv.weight", hf_state_dict, block_size=32)

    assert _quadrant_values(exact) == _BLOCK32_DEQUANTIZED_QUADRANTS

    # The keyword defaults to the 128 grid, so every pre-existing caller keeps its
    # current behaviour -- and that behaviour is visibly wrong for this checkpoint.
    default = maybe_dequantize_hf_quantized_weight("layers.0.attn.wkv.weight", hf_state_dict)
    assert _quadrant_values(default) == _BLOCK32_PAYLOAD_QUADRANTS

    # dict hf_param (GatedMLPMapping) threads the same edge through the recursion.
    as_dict = maybe_dequantize_hf_quantized_weight({"gate": "layers.0.attn.wkv.weight"}, hf_state_dict, block_size=32)
    assert _quadrant_values(as_dict["gate"]) == _BLOCK32_DEQUANTIZED_QUADRANTS


def test_block_size_leaves_the_mxfp4_branch_alone():
    """MXFP4 expert scales are per-row over 32-element K tiles, not block tiles."""
    hf_state_dict = {
        "layers.0.ffn.experts.0.w1.weight": torch.zeros(1, 16, dtype=torch.int8),
        "layers.0.ffn.experts.0.w1.scale": torch.ones(1, 1),
    }

    default = maybe_dequantize_hf_quantized_weight("layers.0.ffn.experts.0.w1.weight", hf_state_dict)
    with_block_size = maybe_dequantize_hf_quantized_weight(
        "layers.0.ffn.experts.0.w1.weight", hf_state_dict, block_size=32
    )

    assert torch.equal(default, with_block_size)

# ---------------------------------------------------------------------------
# Tests for the DSV4.1 re-quantisation units, ported with them. Kept alongside
# main's own tests above; this block only adds cases main did not have.
# ---------------------------------------------------------------------------

def test_e4m3_with_scale_matches_the_host_path(monkeypatch):
    """The device offload must be a pure speed-up, not a different number.

    The DSV4.1 bridge dequantizes every FP8 leaf through this helper; moving the
    broadcast multiply to the GPU is what turns the import's dominant phase from
    minutes into seconds, so the two paths computing different values would silently
    change every loaded weight. Host and device are compared bitwise on the three
    scale layouts the helper accepts.
    """
    import torch

    from megatron.bridge.models.conversion import quantization_utils as qu

    torch.manual_seed(0)
    block = 32

    def _pair(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
        weight = (torch.rand(rows, cols) * 4 - 2).to(torch.float8_e4m3fn)
        return weight, torch.rand(rows // block, cols // block, dtype=torch.float32) + 0.5

    cases = {
        "block2d": _pair(64, 64),
        "row1d": (
            (torch.rand(32, 64) * 4 - 2).to(torch.float8_e4m3fn),
            torch.rand(32, dtype=torch.float32) + 0.5,
        ),
        "rowtile2d": (
            (torch.rand(32, 64) * 4 - 2).to(torch.float8_e4m3fn),
            torch.rand(32, 4, dtype=torch.float32) + 0.5,
        ),
    }

    for label, (weight, scale) in cases.items():
        host = qu._dequantize_e4m3_scale_impl(weight, scale, block_size=block)
        public = qu.dequantize_fp8_e4m3fn_with_scale(weight, scale, block_size=block)
        assert torch.equal(host, public), label

        if torch.cuda.is_available():
            device = qu._dequantize_e4m3_scale_impl(
                weight.cuda(), scale.cuda(), block_size=block
            ).cpu()
            assert torch.equal(host, device), label


def _te_reads_f32_scales(monkeypatch):
    """Every test here assumes TransformerEngine reads blockwise scales as F32.

    The direct write refuses to install scales without it (TE would reinterpret each one as
    an exponent of two), so the variable is part of the environment these tests describe.
    `test_the_writer_refuses_without_the_scale_family_variable` removes it on purpose.
    """
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")


def test_the_writer_refuses_without_the_scale_family_variable(monkeypatch):
    import torch

    qu = _load_quant_utils()
    monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    param = _FakeBlockwiseParam(256, 512)
    data = torch.zeros((256, 512), dtype=torch.uint8)
    scale = torch.ones((2, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="NVTE_FP8_BLOCK_SCALING_FP32_SCALES"):
        qu.write_blockwise_fp8_param(param, data, scale, param_name="proj")


def _load_quant_utils():
    """The module under test (imported the way every other test in this file does)."""
    from megatron.bridge.models.conversion import quantization_utils

    return quantization_utils


def test_e8m0_to_float32_decodes_powers_of_two():
    """E8M0 is an exponent with bias 127, so the value is 2 ** (x - 127)."""
    import torch

    qu = _load_quant_utils()
    raw = torch.tensor([[127, 128], [126, 130]], dtype=torch.uint8)
    got = qu.e8m0_to_float32(raw)
    assert got.tolist() == [[1.0, 2.0], [0.5, 8.0]]
    assert got.dtype == torch.float32


def test_e8m0_to_float32_rejects_the_nan_code():
    import torch

    qu = _load_quant_utils()
    raw = torch.tensor([[255]], dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="NaN code"):
        qu.e8m0_to_float32(raw)


def test_dequantize_block_scaled_fp8_matches_the_ground_truth():
    """A 32x32 E8M0 grid expands exactly, in float32, before any re-quantisation."""
    import torch

    qu = _load_quant_utils()
    torch.manual_seed(0)
    weight = (torch.rand(64, 64) * 4 - 2).to(torch.float8_e4m3fn)
    exps = torch.tensor([[127, 126], [129, 127]], dtype=torch.uint8)

    got = qu.dequantize_block_scaled_fp8(weight, exps, block_size=32)
    scale = qu.e8m0_to_float32(exps)
    expanded = scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
    expected = weight.to(torch.float32) * expanded
    assert torch.equal(got, expected)


def test_requantize_to_blockwise_fp8_produces_the_te_grid_and_bounded_error():
    """The pair lands on the parameter's 128-block grid, within one half-step.

    The checkpoint grid is 32x32 E8M0; the TransformerEngine parameter wants 128x128
    F32. The bytes therefore cannot be copied — they must be re-quantised — and the
    only correctness claim available is that the re-quantised weight dequantises back
    to the float32 ground truth within the 128-block quantiser's half-step.
    """
    import torch

    qu = _load_quant_utils()
    torch.manual_seed(0)
    weight = (torch.rand(256, 256) * 4 - 2).to(torch.float8_e4m3fn)
    exps = torch.full((8, 8), 127, dtype=torch.uint8)  # scale 1.0 everywhere

    pair = qu.requantize_to_blockwise_fp8(weight, exps, block_size=32)
    assert pair.data.dtype == torch.uint8 and pair.data.shape == (256, 256)
    assert pair.scale_inv.dtype == torch.float32 and pair.scale_inv.shape == (2, 2)

    ground_truth = qu.dequantize_block_scaled_fp8(weight, exps, block_size=32)
    back = qu.dequantize_fp8_blockwise(
        pair.data.view(torch.float8_e4m3fn), pair.scale_inv
    )
    # The destination grid has THREE mantissa bits, so the error budget is one step at
    # the block's peak magnitude — 1/16 of the block amax, not 0.5 * scale_inv (which
    # would only hold for a value sitting at the top of the block).
    block_amax = ground_truth.abs().amax().item()
    max_err = (back.float() - ground_truth).abs().max().item()
    assert max_err <= block_amax / 16.0, (max_err, block_amax)


def test_requantize_to_blockwise_fp8_handles_packed_mxfp4():
    """The routed experts are packed 4-bit; they still land on the 128-block grid."""
    import torch

    qu = _load_quant_utils()
    torch.manual_seed(0)
    packed = torch.randint(0, 256, (32, 32), dtype=torch.uint8)  # 32 x 64 logical
    exps = torch.full((32, 2), 127, dtype=torch.uint8)  # per-row, 32-element tiles

    pair = qu.requantize_to_blockwise_fp8(packed, exps, mxfp4=True)
    assert pair.data.shape == (32, 64) and pair.data.dtype == torch.uint8
    assert pair.scale_inv.shape == (1, 1) and pair.scale_inv.dtype == torch.float32

    ground_truth = qu.dequantize_mxfp4_e2m1_packed(packed, exps, dtype=torch.float32)
    back = qu.dequantize_fp8_blockwise(pair.data.view(torch.float8_e4m3fn), pair.scale_inv)
    block_amax = ground_truth.abs().amax().item()
    max_err = (back.float() - ground_truth).abs().max().item()
    assert max_err <= block_amax / 16.0, (max_err, block_amax)


class _FakeBlockwiseParam:
    """The four storages a TransformerEngine blockwise-FP8 parameter exposes.

    Duck-typed on purpose — the writer is duck-typed too, so this needs no CUDA TE
    build. ``pad_cols`` reproduces what this model's projections actually see: TE
    stores the columnwise orientation wider than the logical weight (measured
    2026-10-30: a [1280, 5120] weight arrives with columnwise storages [5120, 1536]).
    """

    def __init__(self, rows: int, cols: int, *, pad_cols: int = 0, block: int = 128):
        import torch

        self.block = block
        self._rowwise_data = torch.zeros((rows, cols), dtype=torch.uint8)
        self._rowwise_scale_inv = torch.zeros(
            (rows // block, cols // block), dtype=torch.float32
        )
        self._columnwise_data = torch.zeros((cols, rows + pad_cols), dtype=torch.uint8)
        self._columnwise_scale_inv = torch.zeros(
            (cols // block, (rows + pad_cols) // block), dtype=torch.float32
        )

    def get_metadata(self):
        from types import SimpleNamespace

        return {"quantizer": SimpleNamespace(block_len=self.block)}


def test_the_writer_zero_fills_a_padded_columnwise_scale_grid():
    """TE pads the columnwise SCALE grid independently of the payload.

    Measured on this model: a [1280, 5120] projection arrives with columnwise data
    [5120, 1280] (exact transpose) beside a columnwise scale grid of (40, 12) — two
    whole blocks wider. The pad carries no weights, so it is zero-filled; refusing it
    (the previous behaviour) made every such projection abort the load.
    """
    import torch

    qu = _load_quant_utils()
    torch.manual_seed(0)
    rows, cols = 256, 512
    param = _FakeBlockwiseParam(rows, cols)
    # the destination's columnwise grid is two whole blocks wider than the transpose
    param._columnwise_scale_inv = torch.zeros((cols // 128, rows // 128 + 2), dtype=torch.float32)
    data = torch.randint(0, 200, (rows, cols), dtype=torch.uint8)
    scale = torch.rand((rows // 128, cols // 128), dtype=torch.float32) + 0.5

    qu.write_blockwise_fp8_param(param, data, scale, param_name="proj")

    assert torch.equal(param._rowwise_data, data)
    assert torch.equal(param._rowwise_scale_inv, scale)
    assert tuple(param._columnwise_data.shape) == (cols, rows)
    assert torch.equal(param._columnwise_data, data.t())
    assert torch.equal(param._columnwise_scale_inv[:, : rows // 128], scale.t())
    assert bool((param._columnwise_scale_inv[:, rows // 128 :] == 0).all())


def test_the_writer_refuses_a_columnwise_storage_narrower_than_the_transpose():
    """Trimming is not allowed: a smaller destination means a different tensor.

    TransformerEngine pads the SCALE grids (to a multiple of the quantiser's block count)
    but keeps the data storages at exactly the logical shape and its transpose, so a
    narrower columnwise storage is not "padding on the other side" — dropping the excess
    would silently discard real weights.
    """
    import torch

    qu = _load_quant_utils()
    rows, cols = 256, 512
    param = _FakeBlockwiseParam(rows, cols)
    param._columnwise_data = torch.zeros((cols, 128), dtype=torch.uint8)  # one block wide
    data = torch.randint(0, 200, (rows, cols), dtype=torch.uint8)
    scale = torch.ones((2, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="columnwise storage"):
        qu.write_blockwise_fp8_param(param, data, scale, param_name="proj")


def test_the_writer_refuses_a_mixed_columnwise_pad_and_trim():
    import torch

    qu = _load_quant_utils()
    param = _FakeBlockwiseParam(256, 512)
    param._columnwise_data = torch.zeros((512 + 128, 128), dtype=torch.uint8)  # wrong both ways
    data = torch.zeros((256, 512), dtype=torch.uint8)
    scale = torch.ones((2, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="columnwise storage"):
        qu.write_blockwise_fp8_param(param, data, scale, param_name="proj")


def test_the_writer_refuses_a_payload_smaller_than_the_destination():
    """A shard that did not arrive must not become zeros over the destination's rows."""
    import torch

    qu = _load_quant_utils()
    param = _FakeBlockwiseParam(512, 256)
    data = torch.full((256, 256), 7, dtype=torch.uint8)  # half the rows
    scale = torch.ones((2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="not the destination FP8 storage"):
        qu.write_blockwise_fp8_param(param, data, scale, param_name="proj")


