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

"""Unit tests for the direct blockwise-FP8 import path.

An FP8 checkpoint imported into an ``fp8_param=True`` model must land byte-for-byte
in the TransformerEngine parameter: dequantizing to BF16 and letting ``copy_``
requantize adds one rounding the checkpoint never had. These tests pin the gate, the
scale-carrying transport through the mappings, and the loud failures that keep a
mis-shaped checkpoint from being silently requantized instead.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    DirectMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.glm5next.glm53flash_bridge import GLM53FlashBridge


BLOCK = 128


class FakeBlockwiseFp8Param:
    """Duck-typed stand-in for TE's ``Float8BlockwiseQTensor``.

    Mirrors the storage contract the direct write relies on: both orientations exist,
    ``dtype`` reports the *fake* (logical) dtype rather than the byte dtype, and the
    quantizer carries the block length.
    """

    def __init__(self, rows: int, cols: int, *, block: int = BLOCK, dtype: torch.dtype = torch.bfloat16):
        self._rowwise_data = torch.zeros((rows, cols), dtype=torch.uint8)
        self._rowwise_scale_inv = torch.zeros((rows // block, cols // block), dtype=torch.float32)
        self._columnwise_data = torch.zeros((cols, rows), dtype=torch.uint8)
        self._columnwise_scale_inv = torch.zeros((cols // block, rows // block), dtype=torch.float32)
        self.dtype = dtype
        self.shape = torch.Size((rows, cols))
        self.device = torch.device("cpu")
        self._block = block

    def get_metadata(self):
        return {"quantizer": SimpleNamespace(block_len=self._block), "is_2D_scaled": True}

    def copy_(self, source):
        """TE's ``copy_`` requantizes; the fake only records that it was called.

        The legacy-path tests assert the direct writer did NOT run, which is exactly
        "the checkpoint's bytes are not in the parameter's FP8 storage".
        """
        self.last_copy_source = source
        return self


def _module_with(param, name: str = "weight"):
    return SimpleNamespace(**{name: param})


# A class name AutoMapping recognises, so the auto-detected delegate is a row mapping.
_FakeRowParallelModule = type("TERowParallelLinear", (SimpleNamespace,), {})


def _fp8_payload(rows: int, cols: int, *, seed: int = 0) -> torch.Tensor:
    """E4M3 bytes with varied bit patterns (so a byte swap cannot pass unnoticed)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-120, 120, (rows, cols), generator=generator).to(torch.float32).to(torch.float8_e4m3fn)


def _scale_grid(rows: int, cols: int, *, seed: int = 1, block: int = BLOCK) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(rows // block, cols // block, generator=generator) + 0.25).to(torch.float32)


def _pair(rows: int, cols: int, *, seed: int = 0):
    return quantization_utils.Fp8BlockwisePair(
        data=quantization_utils.as_fp8_bytes(_fp8_payload(rows, cols, seed=seed)),
        scale_inv=_scale_grid(rows, cols, seed=seed + 100),
    )


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Byte equality, including the F32 scales' bit patterns."""
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.uint8), expected.contiguous().view(torch.uint8))


def _write(mapping, pair, param, module=None):
    module = module if module is not None else _module_with(param)
    data, scale = mapping.hf_to_megatron_fp8(pair, module)
    quantization_utils.write_blockwise_fp8_param(
        param,
        data,
        scale,
        param_name=mapping.megatron_param,
    )
    return param


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config, expected",
    [
        (SimpleNamespace(fp8_param=True, fp8_recipe="blockwise"), True),
        (SimpleNamespace(fp8_param=False, fp8_recipe="blockwise"), False),
        (SimpleNamespace(fp8_param=True, fp8_recipe="delayed"), False),
        (SimpleNamespace(fp8_param=True, fp8_recipe=None), False),
        (SimpleNamespace(fp8_param=True), False),
        (None, False),
    ],
)
def test_glm53_gate_needs_fp8_param_and_blockwise_recipe(config, expected):
    assert GLM53FlashBridge().fp8_import_direct_write_enabled(config) is expected


def test_default_gate_is_off_for_every_other_bridge():
    """Other models read the same dequantize helper; their path must not change."""
    config = SimpleNamespace(fp8_param=True, fp8_recipe="blockwise")
    assert MegatronModelBridge.fp8_import_direct_write_enabled(Mock(), config) is False


def test_maybe_load_fp8_direct_weight_default_returns_none():
    assert MegatronModelBridge.maybe_load_fp8_direct_weight(Mock(), Mock(), Mock(), {}) is None


# ---------------------------------------------------------------------------
# What the checkpoint loader hands to the mapping
# ---------------------------------------------------------------------------


def test_direct_pair_keeps_the_checkpoint_bytes_and_scale():
    weight = _fp8_payload(2 * BLOCK, 3 * BLOCK)
    scale = _scale_grid(2 * BLOCK, 3 * BLOCK)
    pair = GLM53FlashBridge._load_fp8_direct_pair("hf.w", {"hf.w": weight, "hf.w_scale_inv": scale})

    assert pair.data.dtype == torch.uint8
    assert pair.data.shape == weight.shape
    _assert_bitwise_equal(pair.data, weight.view(torch.uint8))
    _assert_bitwise_equal(pair.scale_inv, scale)


def test_bf16_source_stays_on_the_legacy_path():
    """lm_head / vision tower / kv_b_proj are BF16 in the checkpoint but land in FP8 params."""
    weight = torch.randn(2 * BLOCK, 3 * BLOCK, dtype=torch.bfloat16)
    assert GLM53FlashBridge._load_fp8_direct_pair("hf.w", {"hf.w": weight}) is None

    bridge = GLM53FlashBridge()
    param = FakeBlockwiseFp8Param(2 * BLOCK, 3 * BLOCK)
    # A fused mapping with a BF16 half cannot be written directly either.
    assert (
        bridge.maybe_load_fp8_direct_weight(
            SimpleNamespace(hf_param={"gate": "hf.gate", "up": "hf.up"}),
            param,
            {
                "hf.gate": _fp8_payload(2 * BLOCK, 3 * BLOCK),
                "hf.gate_scale_inv": _scale_grid(2 * BLOCK, 3 * BLOCK),
                "hf.up": weight,
            },
        )
        is None
    )


def test_missing_scale_raises_instead_of_dequantizing(monkeypatch):
    bridge = GLM53FlashBridge()
    weight = _fp8_payload(BLOCK, BLOCK)
    mapping = SimpleNamespace(hf_param="hf.w")
    param = FakeBlockwiseFp8Param(BLOCK, BLOCK)

    with pytest.raises(ValueError, match=r"hf\.w_scale_inv"):
        bridge.maybe_load_fp8_direct_weight(mapping, param, {"hf.w": weight})


def test_plain_destination_and_synthetic_keys_stay_on_the_legacy_path():
    bridge = GLM53FlashBridge()
    state = {"hf.w": _fp8_payload(BLOCK, BLOCK), "hf.w_scale_inv": _scale_grid(BLOCK, BLOCK)}

    # A parameter that is not blockwise FP8 (fp8_param off, or a norm/bias) is untouched.
    assert bridge.maybe_load_fp8_direct_weight(SimpleNamespace(hf_param="hf.w"), torch.zeros(2, 2), state) is None
    # InitOnlyMapping's synthetic key has no checkpoint entry at all.
    assert (
        bridge.maybe_load_fp8_direct_weight(
            SimpleNamespace(hf_param="mg.weight.__init_only__"), FakeBlockwiseFp8Param(BLOCK, BLOCK), state
        )
        is None
    )


def test_legacy_dequantize_path_is_unchanged(monkeypatch):
    """With the gate off the bridge still returns the dequantized BF16 weight."""
    monkeypatch.setattr(quantization_utils, "_DEQUANT_RETURN_ON_DEVICE", False)
    bridge = GLM53FlashBridge()
    weight = _fp8_payload(BLOCK, BLOCK)
    scale = _scale_grid(BLOCK, BLOCK)
    state = {"hf.w": weight, "hf.w_scale_inv": scale}

    loaded = bridge.maybe_modify_loaded_hf_weight("hf.w", state)

    expected = quantization_utils.maybe_dequantize_fp8_blockwise(weight, scale)
    assert loaded.dtype == torch.bfloat16
    _assert_bitwise_equal(loaded, expected)


# ---------------------------------------------------------------------------
# Transport: the scale rides the same layout transform as the payload
# ---------------------------------------------------------------------------


def test_row_parallel_direct_write_is_bit_exact():
    rows, cols = 2 * BLOCK, 3 * BLOCK
    pair = _pair(rows, cols)
    param = FakeBlockwiseFp8Param(rows, cols)
    mapping = RowParallelMapping(megatron_param="linear_fc2.weight", hf_param="hf.down_proj.weight")

    _write(mapping, pair, param)

    _assert_bitwise_equal(param._rowwise_data, pair.data)
    _assert_bitwise_equal(param._rowwise_scale_inv, pair.scale_inv)
    # The columnwise orientation TE needs for the backward GEMM is the byte transpose.
    _assert_bitwise_equal(param._columnwise_data, pair.data.t().contiguous())
    _assert_bitwise_equal(param._columnwise_scale_inv, pair.scale_inv.t().contiguous())


def test_row_parallel_transpose_turns_the_scale_with_the_payload():
    """A checkpoint stored as [in, out] transposes payload and scale grid alike."""
    in_features, out_features = 3 * BLOCK, 2 * BLOCK
    stored = _pair(in_features, out_features, seed=7)
    param = FakeBlockwiseFp8Param(out_features, in_features)
    mapping = RowParallelMapping(megatron_param="mlp.linear_fc2.weight", hf_param="hf.experts.0.down_proj.weight")

    _write(mapping, stored, param)

    _assert_bitwise_equal(param._rowwise_data, stored.data.t().contiguous())
    _assert_bitwise_equal(param._rowwise_scale_inv, stored.scale_inv.t().contiguous())


def test_replicated_direct_write_is_bit_exact():
    pair = _pair(2 * BLOCK, 2 * BLOCK, seed=3)
    param = FakeBlockwiseFp8Param(2 * BLOCK, 2 * BLOCK)
    mapping = ReplicatedMapping(megatron_param="linear_q_down_proj.weight", hf_param="hf.q_a_proj.weight")

    _write(mapping, pair, param)

    _assert_bitwise_equal(param._rowwise_data, pair.data)
    _assert_bitwise_equal(param._rowwise_scale_inv, pair.scale_inv)


def test_auto_mapping_permutes_payload_and_scale_together():
    rows, cols = 2 * BLOCK, 2 * BLOCK
    pair = _pair(rows, cols, seed=5)
    param = FakeBlockwiseFp8Param(cols, rows)
    mapping = AutoMapping(
        megatron_param="linear_fc2.weight",
        hf_param="hf.down_proj.weight",
        permute_dims=(1, 0),
    )

    _write(mapping, pair, param, module=_FakeRowParallelModule(weight=param))

    _assert_bitwise_equal(param._rowwise_data, pair.data.permute(1, 0).contiguous())
    _assert_bitwise_equal(param._rowwise_scale_inv, pair.scale_inv.permute(1, 0).contiguous())


def test_gated_mlp_fuses_payload_and_scale_along_dim0():
    rows, cols = 2 * BLOCK, 2 * BLOCK
    gate, up = _pair(rows, cols, seed=11), _pair(rows, cols, seed=12)
    param = FakeBlockwiseFp8Param(2 * rows, cols)
    mapping = GatedMLPMapping(
        megatron_param="mlp.linear_fc1.weight",
        gate="hf.gate_proj.weight",
        up="hf.up_proj.weight",
    )

    _write(mapping, {"gate": gate, "up": up}, param)

    _assert_bitwise_equal(param._rowwise_data, torch.cat([gate.data, up.data], dim=0))
    _assert_bitwise_equal(param._rowwise_scale_inv, torch.cat([gate.scale_inv, up.scale_inv], dim=0))


@pytest.mark.parametrize(
    "mapping_cls, dim, extra",
    [
        (ColumnParallelMapping, 0, {}),
        (RowParallelMapping, 1, {}),
    ],
)
def test_tp_shards_slice_the_scale_with_the_payload(monkeypatch, mapping_cls, dim, extra):
    """Rank 0 splits payload and scale grid on the same axis, so they stay in step."""
    tp_size = 2
    rows, cols = 2 * BLOCK, 4 * BLOCK
    pair = _pair(rows, cols, seed=13)
    mapping = mapping_cls(megatron_param="linear.weight", hf_param="hf.weight", **extra)
    monkeypatch.setattr(type(mapping), "tp_size", property(lambda _self: tp_size))
    monkeypatch.setattr(type(mapping), "tp_rank", property(lambda _self: 0))

    captured = {}

    def fake_scatter(splits, output_shape, dtype, device, src_rank=0):
        captured.setdefault("splits", []).append(splits)
        return splits[0]

    monkeypatch.setattr(mapping, "scatter_to_tp_ranks", fake_scatter)

    param = FakeBlockwiseFp8Param(rows // tp_size if dim == 0 else rows, cols if dim == 0 else cols // tp_size)
    # The isolation stub accepts the optional resolved-target kwarg the
    # transports now pass through (the production signature grew it to halve
    # the per-parameter name resolution).
    mapping._fp8_validate_pair = lambda data, scale, module, target=None: (data, scale)
    data, scale = mapping.hf_to_megatron_fp8(pair, _module_with(param))

    data_splits, scale_splits = captured["splits"]
    assert len(data_splits) == len(scale_splits) == tp_size
    _assert_bitwise_equal(data, data_splits[0])
    _assert_bitwise_equal(scale, scale_splits[0])
    # The same block range reaches the same rank in both domains.
    assert data_splits[0].shape[dim] // BLOCK == scale_splits[0].shape[dim]


def test_gated_mlp_tp_shards_pair_each_rank_gate_with_its_own_up(monkeypatch):
    """Rank 0 must fuse gate shard 0 with up shard 0 in BOTH the payload and the grid."""
    tp_size = 2
    rows, cols = 2 * BLOCK, 2 * BLOCK
    gate, up = _pair(rows, cols, seed=41), _pair(rows, cols, seed=42)
    mapping = GatedMLPMapping(megatron_param="linear_fc1.weight", gate="hf.gate", up="hf.up")
    monkeypatch.setattr(type(mapping), "tp_size", property(lambda _self: tp_size))
    monkeypatch.setattr(type(mapping), "tp_rank", property(lambda _self: 0))

    def fake_scatter(splits, output_shape, dtype, device, src_rank=0):
        return splits[0]

    monkeypatch.setattr(mapping, "scatter_to_tp_ranks", fake_scatter)
    param = FakeBlockwiseFp8Param(rows, cols)
    # The isolation stub accepts the optional resolved-target kwarg the
    # transports now pass through (the production signature grew it to halve
    # the per-parameter name resolution).
    mapping._fp8_validate_pair = lambda data, scale, module, target=None: (data, scale)

    data, scale = mapping.hf_to_megatron_fp8({"gate": gate, "up": up}, _module_with(param))

    rank0 = lambda t: torch.chunk(t, tp_size, dim=0)[0]  # noqa: E731
    _assert_bitwise_equal(data, torch.cat([rank0(gate.data), rank0(up.data)], dim=0))
    _assert_bitwise_equal(scale, torch.cat([rank0(gate.scale_inv), rank0(up.scale_inv)], dim=0))


# ---------------------------------------------------------------------------
# Loud failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows, cols", [(100, BLOCK), (BLOCK, 64), (BLOCK + 1, 2 * BLOCK)])
def test_non_block_multiple_dims_fail_loud(rows, cols):
    param = FakeBlockwiseFp8Param(rows, cols)
    payload = torch.zeros((rows, cols), dtype=torch.uint8)
    scale = torch.zeros((max(rows // BLOCK, 1), max(cols // BLOCK, 1)), dtype=torch.float32)

    with pytest.raises(ValueError, match="linear_fc2.weight"):
        quantization_utils.write_blockwise_fp8_param(param, payload, scale, param_name="linear_fc2.weight")


def test_non_block_multiple_dim_fails_loud_in_the_transport():
    """A payload whose dim is not a whole 128-block: the BLOCK guard must be
    the one that fires (the payload was (256, 256) with a mismatched grid
    before, which exercised the GRID guard and pinned nothing about the
    block guard)."""
    mapping = RowParallelMapping(megatron_param="linear_fc2.weight", hf_param="hf.down_proj.weight")
    param = FakeBlockwiseFp8Param(2 * BLOCK, 2 * BLOCK)
    bad = quantization_utils.Fp8BlockwisePair(
        data=torch.zeros((2 * BLOCK, 2 * BLOCK + 64), dtype=torch.uint8),
        scale_inv=torch.zeros((2 * BLOCK // 128, 3), dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="whole 128x128 blocks"):
        # match the DISTINGUISHING text: "linear_fc2.weight" is also
        # satisfied by the next guard (scale-grid mismatch), so a mutation
        # deleting the whole-block guard would pass this test silently.
        mapping.hf_to_megatron_fp8(bad, _module_with(param))


def test_write_rejects_scale_grid_mismatch():
    """Storage matches the bad grid, so the grid check is the only one that can fire."""
    param = FakeBlockwiseFp8Param(2 * BLOCK, 2 * BLOCK)
    param._rowwise_scale_inv = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="scale grid"):
        quantization_utils.write_blockwise_fp8_param(
            param,
            torch.zeros((2 * BLOCK, 2 * BLOCK), dtype=torch.uint8),
            torch.zeros((2, 3), dtype=torch.float32),
            param_name="linear_fc2.weight",
        )


def test_transport_rejects_scale_grid_mismatch():
    """Same isolation for the mapping-level check: 2 payload blocks need a 2x2 grid."""
    mapping = RowParallelMapping(megatron_param="linear_fc2.weight", hf_param="hf.down_proj.weight")
    param = FakeBlockwiseFp8Param(2 * BLOCK, 2 * BLOCK)
    param._rowwise_scale_inv = torch.zeros((2, 3), dtype=torch.float32)
    bad = quantization_utils.Fp8BlockwisePair(
        data=torch.zeros((2 * BLOCK, 2 * BLOCK), dtype=torch.uint8),
        scale_inv=torch.zeros((2, 3), dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="scale grid"):
        mapping.hf_to_megatron_fp8(bad, _module_with(param))


def test_non_f32_scale_family_fails_loud():
    """E8M0 scales would dequantize differently; the write refuses them."""
    param = FakeBlockwiseFp8Param(BLOCK, BLOCK)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is None:
        pytest.skip("torch build without float8_e8m0fnu")
    with pytest.raises(ValueError, match="F32"):
        quantization_utils.write_blockwise_fp8_param(
            param,
            torch.zeros((BLOCK, BLOCK), dtype=torch.uint8),
            torch.zeros((1, 1), dtype=torch.float32).to(e8m0),
            param_name="linear_fc2.weight",
        )


def test_scale_family_must_match_the_destination_storage():
    """An E8M0 destination grid (NVTE_FP8_BLOCK_SCALING_FP32_SCALES unset) is refused."""
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is None:
        pytest.skip("torch build without float8_e8m0fnu")
    param = FakeBlockwiseFp8Param(2 * BLOCK, 2 * BLOCK)
    param._rowwise_scale_inv = torch.zeros((2, 2), dtype=torch.float32).to(e8m0)
    with pytest.raises(ValueError, match="scale family"):
        quantization_utils.write_blockwise_fp8_param(
            param,
            torch.zeros((2 * BLOCK, 2 * BLOCK), dtype=torch.uint8),
            torch.zeros((2, 2), dtype=torch.float32),
            param_name="linear_fc2.weight",
        )


def test_transported_payload_must_match_the_destination_storage():
    mapping = RowParallelMapping(megatron_param="linear_fc2.weight", hf_param="hf.down_proj.weight")
    param = FakeBlockwiseFp8Param(4 * BLOCK, 2 * BLOCK)

    with pytest.raises(ValueError, match="linear_fc2.weight"):
        mapping.hf_to_megatron_fp8(_pair(2 * BLOCK, 2 * BLOCK), _module_with(param))


def test_mapping_without_fp8_support_fails_loud():
    mapping = QKVMapping(
        megatron_param="self_attention.linear_qkv.weight",
        q="hf.q_proj.weight",
        k="hf.k_proj.weight",
        v="hf.v_proj.weight",
    )
    with pytest.raises(NotImplementedError) as excinfo:
        mapping.hf_to_megatron_fp8(_pair(BLOCK, BLOCK), _module_with(FakeBlockwiseFp8Param(BLOCK, BLOCK)))

    assert "QKVMapping" in str(excinfo.value)
    assert "linear_qkv.weight" in str(excinfo.value)


def test_ragged_tp_split_fails_loud():
    mapping = ColumnParallelMapping(megatron_param="linear.weight", hf_param="hf.weight")
    param = FakeBlockwiseFp8Param(2 * BLOCK, BLOCK)
    object.__setattr__(mapping, "_tp_group", None)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(type(mapping), "tp_size", property(lambda _self: 3))
        mp.setattr(type(mapping), "tp_rank", property(lambda _self: 0))
        with pytest.raises(ValueError, match="evenly"):
            mapping.hf_to_megatron_fp8(_pair(4 * BLOCK, BLOCK), _module_with(param))


# ---------------------------------------------------------------------------
# End to end: the import loop wires the gate to the direct write
# ---------------------------------------------------------------------------


class _DirectImportBridge(MegatronModelBridge):
    """Bridge that opts in, exposing the same hooks the GLM-5.3-Flash bridge does."""

    direct_write = True
    gate_calls = 0
    loader_calls = 0

    def provider_bridge(self, hf_pretrained):  # pragma: no cover - unused
        return None

    def mapping_registry(self):  # pragma: no cover - unused
        return None

    def fp8_import_direct_write_enabled(self, megatron_config) -> bool:
        type(self).gate_calls += 1
        return self.direct_write

    def maybe_load_fp8_direct_weight(self, mapping, param_weight, hf_state_dict):
        type(self).loader_calls += 1
        if not quantization_utils.is_blockwise_fp8_param(param_weight):
            return None
        name = mapping.hf_param
        weight = hf_state_dict[name]
        if not quantization_utils.is_fp8_tensor(weight):
            return None
        return quantization_utils.Fp8BlockwisePair(
            data=quantization_utils.as_fp8_bytes(weight),
            scale_inv=hf_state_dict[name + "_scale_inv"],
        )


def _run_import(monkeypatch, *, direct_write: bool, capture_unquantized: bool = False):
    rows, cols = 2 * BLOCK, 2 * BLOCK
    payload = _fp8_payload(rows, cols, seed=21)
    scale = _scale_grid(rows, cols, seed=22)
    param = FakeBlockwiseFp8Param(rows, cols)
    mapping = RowParallelMapping(megatron_param="linear.weight", hf_param="hf.w")
    task = WeightConversionTask(
        param_name="decoder.layers.0.linear.weight",
        global_param_name="decoder.layers.0.linear.weight",
        mapping=mapping,
        pp_rank=0,
        vp_stage=0,
        megatron_module=_module_with(param),
        param_weight=param,
    )
    bridge = _DirectImportBridge()
    bridge.direct_write = direct_write
    bridge.export_weight_dtype = "fp8" if capture_unquantized else "bf16"
    monkeypatch.setattr(_DirectImportBridge, "build_conversion_tasks", lambda self, *_a, **_k: [task])
    monkeypatch.setattr(_DirectImportBridge, "_with_progress_tracking", lambda self, tasks, *_a, **_k: tasks)
    monkeypatch.setattr(_DirectImportBridge, "_broadcast_shared_embeddings", lambda self, *_a, **_k: None)
    # Mirrors the bridge hook the loop uses when it is NOT on the direct path.
    monkeypatch.setattr(
        _DirectImportBridge,
        "maybe_modify_loaded_hf_weight",
        lambda self, hf_param, state: quantization_utils.maybe_dequantize_fp8_blockwise(
            state[hf_param], state[hf_param + "_scale_inv"]
        ),
    )

    hf_pretrained = SimpleNamespace(state={"hf.w": payload, "hf.w_scale_inv": scale}, model_name_or_path="dummy")
    model = SimpleNamespace(config=SimpleNamespace(fp8_param=True, fp8_recipe="blockwise"))
    bridge.load_weights_hf_to_megatron(hf_pretrained, [model])
    return bridge, param, payload, scale


def test_import_loop_writes_fp8_bytes_directly_when_the_gate_is_on(monkeypatch):
    _DirectImportBridge.gate_calls = 0
    _DirectImportBridge.loader_calls = 0
    _bridge, param, payload, scale = _run_import(monkeypatch, direct_write=True)

    assert _DirectImportBridge.gate_calls == 1
    assert _DirectImportBridge.loader_calls == 1
    _assert_bitwise_equal(param._rowwise_data, payload.view(torch.uint8))
    _assert_bitwise_equal(param._rowwise_scale_inv, scale)
    _assert_bitwise_equal(param._columnwise_data, payload.view(torch.uint8).t().contiguous())
    _assert_bitwise_equal(param._columnwise_scale_inv, scale.t().contiguous())
    # The `continue` after the direct write keeps the legacy copy_ out: a
    # trailing copy_ would requantize OVER the bit-exact storages (real TE
    # would corrupt them silently; the fake records the call without
    # writing), so the absence of the recorded call IS the pin.
    assert not hasattr(param, "last_copy_source")


def test_import_loop_never_touches_the_direct_path_when_the_gate_is_off(monkeypatch):
    _DirectImportBridge.gate_calls = 0
    _DirectImportBridge.loader_calls = 0
    _bridge, param, payload, scale = _run_import(monkeypatch, direct_write=False)

    assert _DirectImportBridge.loader_calls == 0
    # Legacy path: the FP8 payload was dequantized to BF16 and handed to copy_ (which
    # requantizes inside TE), so the parameter's FP8 storage is not the checkpoint's.
    assert param.last_copy_source.dtype == torch.bfloat16
    _assert_bitwise_equal(param.last_copy_source, quantization_utils.maybe_dequantize_fp8_blockwise(payload, scale))


def test_import_loop_requantizes_a_bf16_source_into_the_fp8_param(monkeypatch):
    """A BF16 source has no scale grid to carry: the loop must fall back to copy_."""
    rows, cols = 2 * BLOCK, 2 * BLOCK
    payload = torch.randn(rows, cols, dtype=torch.bfloat16)
    param = FakeBlockwiseFp8Param(rows, cols)
    mapping = RowParallelMapping(megatron_param="linear.weight", hf_param="hf.w")
    task = WeightConversionTask(
        param_name="decoder.layers.0.linear.weight",
        global_param_name="decoder.layers.0.linear.weight",
        mapping=mapping,
        pp_rank=0,
        vp_stage=0,
        megatron_module=_module_with(param),
        param_weight=param,
    )
    bridge = _DirectImportBridge()
    bridge.direct_write = True
    monkeypatch.setattr(_DirectImportBridge, "build_conversion_tasks", lambda self, *_a, **_k: [task])
    monkeypatch.setattr(_DirectImportBridge, "_with_progress_tracking", lambda self, tasks, *_a, **_k: tasks)
    monkeypatch.setattr(_DirectImportBridge, "_broadcast_shared_embeddings", lambda self, *_a, **_k: None)
    monkeypatch.setattr(
        _DirectImportBridge, "maybe_modify_loaded_hf_weight", lambda self, hf_param, state: state[hf_param]
    )

    hf_pretrained = SimpleNamespace(state={"hf.w": payload}, model_name_or_path="dummy")
    model = SimpleNamespace(config=SimpleNamespace(fp8_param=True, fp8_recipe="blockwise"))
    bridge.load_weights_hf_to_megatron(hf_pretrained, [model])

    assert param.last_copy_source is payload


def test_unquantized_capture_keeps_the_legacy_path(monkeypatch):
    """Export-side capture needs the logical BF16 values of every parameter."""
    _DirectImportBridge.loader_calls = 0
    bridge, _param, _payload, _scale = _run_import(monkeypatch, direct_write=True, capture_unquantized=True)

    assert _DirectImportBridge.loader_calls == 0
    assert "decoder.layers.0.linear.weight" in bridge.unquantized_state_dict["model"]


def test_strided_payload_is_materialized_before_it_reaches_te():
    """TE reads the storages as dense buffers; a strided view would dequantize to garbage."""
    rows, cols = 4 * BLOCK, 4 * BLOCK
    backing = _fp8_payload(rows, cols, seed=31)
    pair = quantization_utils.Fp8BlockwisePair(
        data=quantization_utils.as_fp8_bytes(backing)[:, : 2 * BLOCK],
        scale_inv=_scale_grid(rows, 4 * BLOCK, seed=32)[:, :2],
    )
    param = FakeBlockwiseFp8Param(rows, 2 * BLOCK)
    mapping = RowParallelMapping(megatron_param="linear_fc2.weight", hf_param="hf.down_proj.weight")

    _write(mapping, pair, param)

    assert param._rowwise_data.is_contiguous()
    assert param._rowwise_scale_inv.is_contiguous()
    _assert_bitwise_equal(param._rowwise_data, quantization_utils.as_fp8_bytes(backing)[:, : 2 * BLOCK])
    _assert_bitwise_equal(param._rowwise_scale_inv, _scale_grid(rows, 4 * BLOCK, seed=32)[:, :2])


def test_direct_mapping_transport_fails_loud_on_a_non_block_multiple():
    """DirectMapping rides the same choke point as the other transports."""
    mapping = DirectMapping(megatron_param="linear_fc1.weight", hf_param="hf.up_proj.weight")
    param = FakeBlockwiseFp8Param(2 * BLOCK, 2 * BLOCK)
    bad = quantization_utils.Fp8BlockwisePair(
        data=torch.zeros((2 * BLOCK, 2 * BLOCK + 64), dtype=torch.uint8),
        scale_inv=torch.zeros((2 * BLOCK // 128, 2 * BLOCK // 128), dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="whole 128x128 blocks"):
        mapping.hf_to_megatron_fp8(bad, _module_with(param))


def test_column_parallel_mapping_transport_fails_loud_on_a_scale_grid_mismatch():
    """ColumnParallelMapping rides the same choke point."""
    mapping = ColumnParallelMapping(megatron_param="linear_fc1.weight", hf_param="hf.gate_proj.weight")
    param = FakeBlockwiseFp8Param(2 * BLOCK, 2 * BLOCK)
    param._rowwise_scale_inv = torch.zeros((2, 3), dtype=torch.float32)
    bad = quantization_utils.Fp8BlockwisePair(
        data=torch.zeros((2 * BLOCK, 2 * BLOCK), dtype=torch.uint8),
        scale_inv=torch.zeros((2, 3), dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="scale grid"):
        mapping.hf_to_megatron_fp8(bad, _module_with(param))


def test_replicated_mapping_transport_fails_loud_on_a_non_block_multiple():
    """ReplicatedMapping rides the same choke point (mutating its
    _fp8_validate_pair call away must turn this red)."""
    mapping = ReplicatedMapping(megatron_param="linear_fc1.weight", hf_param="hf.gate_proj.weight")
    param = FakeBlockwiseFp8Param(2 * BLOCK, 2 * BLOCK)
    bad = quantization_utils.Fp8BlockwisePair(
        data=torch.zeros((2 * BLOCK, 2 * BLOCK + 64), dtype=torch.uint8),
        scale_inv=torch.zeros((2 * BLOCK // 128, 2 * BLOCK // 128), dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="whole 128x128 blocks"):
        mapping.hf_to_megatron_fp8(bad, _module_with(param))


def test_gate_accepts_the_enum_form_of_the_recipe():
    """Fp8Recipe is a (str, Enum): `Fp8Recipe.blockwise == "blockwise"` is
    True but str() renders 'Fp8Recipe.blockwise'. The gate must compare
    DIRECTLY — an enum-valued config (the form MCore's own determinism tests
    build) must enable the direct path, not silently fall back to the round
    trip."""
    from megatron.core.enums import Fp8Recipe

    bridge = GLM53FlashBridge()
    assert (
        bridge.fp8_import_direct_write_enabled(SimpleNamespace(fp8_param=True, fp8_recipe=Fp8Recipe.blockwise)) is True
    )
    assert bridge.fp8_import_direct_write_enabled(SimpleNamespace(fp8_param=True, fp8_recipe="blockwise")) is True
    assert (
        bridge.fp8_import_direct_write_enabled(SimpleNamespace(fp8_param=True, fp8_recipe=Fp8Recipe.delayed)) is False
    )
    assert (
        bridge.fp8_import_direct_write_enabled(SimpleNamespace(fp8_param=False, fp8_recipe=Fp8Recipe.blockwise))
        is False
    )
