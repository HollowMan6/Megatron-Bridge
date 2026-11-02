"""The DSV4.1 line's own coverage of the direct FP8 write.

`test_fp8_direct_import.py` skips entirely where the GLM-5.3-Flash bridge cannot import,
which is this line — so the guards that protect this line's load had no running test here.
These do not need a megatron runtime: a fake blockwise parameter plus the real validator and
the real bridge gate.
"""

import pytest


torch = pytest.importorskip("torch")

from megatron.bridge.models.conversion import quantization_utils as qu  # noqa: E402
from megatron.bridge.models.conversion.param_mapping import AutoMapping  # noqa: E402


class _FakeParam(torch.nn.Parameter if False else object):
    """A blockwise-FP8 parameter: logical shape plus the two storage pairs."""

    def __init__(self, rows, cols, *, block=128, scale_pad_blocks=0):
        self.shape = (rows, cols)
        self._rowwise_data = torch.zeros((rows, cols), dtype=torch.uint8)
        self._rowwise_scale_inv = torch.zeros(
            (rows // block, (cols // block) + scale_pad_blocks), dtype=torch.float32
        )
        self._columnwise_data = torch.zeros((cols, rows), dtype=torch.uint8)
        self._columnwise_scale_inv = torch.zeros(
            ((cols // block), (rows // block)), dtype=torch.float32
        )
        self.block = block

    def get_metadata(self):
        from types import SimpleNamespace

        return {"quantizer": SimpleNamespace(block_len=self.block)}

    def __getattr__(self, item):  # pragma: no cover - defensive
        raise AttributeError(item)


class _Module(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(param.shape))


def _mapping(param):
    mapping = AutoMapping(megatron_param="w", hf_param="hf.w")
    mapping._fp8_target_param = lambda module: param  # type: ignore[assignment]
    return mapping


def test_a_payload_that_misses_rows_is_refused():
    """A shard that never arrived must not become zeros over real rows."""
    param = _FakeParam(512, 256)
    mapping = _mapping(param)
    data = torch.full((256, 256), 7, dtype=torch.uint8)
    scale = torch.ones((2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="logical weight"):
        mapping._fp8_validate_pair(data, scale, _Module(param))


def test_a_logical_payload_with_a_padded_scale_grid_is_accepted():
    param = _FakeParam(512, 256, scale_pad_blocks=2)
    mapping = _mapping(param)
    data = torch.zeros((512, 256), dtype=torch.uint8)
    scale = torch.ones((4, 2), dtype=torch.float32)
    out_data, out_scale = mapping._fp8_validate_pair(data, scale, _Module(param))
    assert tuple(out_data.shape) == (512, 256)
    assert tuple(out_scale.shape) == (4, 2)


def test_the_bridge_refuses_to_opt_in_above_one_tensor_parallel_rank(monkeypatch):
    """TP>1 transposes into padded storages the transport cannot scatter into."""
    import megatron.bridge.models.deepseek.deepseek_v41_bridge as bridge_module

    class _Config:
        fp8_param = True
        fp8_recipe = "blockwise"

    class _PS:
        @staticmethod
        def get_tensor_model_parallel_world_size():
            return 2

    monkeypatch.setattr("megatron.core.parallel_state", _PS, raising=False)
    bridge = bridge_module.DeepSeekV41Bridge.__new__(bridge_module.DeepSeekV41Bridge)
    assert bridge.fp8_import_direct_write_enabled(_Config()) is False

    class _PS1(_PS):
        @staticmethod
        def get_tensor_model_parallel_world_size():
            return 1

    monkeypatch.setattr("megatron.core.parallel_state", _PS1, raising=False)
    assert bridge.fp8_import_direct_write_enabled(_Config()) is True


def test_the_writer_refuses_without_the_f32_scale_family(monkeypatch):
    monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    param = _FakeParam(256, 256)
    with pytest.raises(ValueError, match="NVTE_FP8_BLOCK_SCALING_FP32_SCALES"):
        qu.write_blockwise_fp8_param(
            param, torch.zeros((256, 256), dtype=torch.uint8),
            torch.ones((2, 2), dtype=torch.float32), param_name="w",
        )


def test_a_padded_rowwise_storage_does_not_widen_the_columnwise_grid(monkeypatch):
    """TE pads each scale storage independently, so the transpose must use the logical grid.

    TransformerEngine pads a 2D scale storage so its inner dimension is a multiple of four,
    and it does that to the rowwise and columnwise storages separately: a [512, 128] weight
    with 128x128 blocks has the logical grid (4, 1), stored rowwise as (4, 4) and
    columnwise as (1, 4). Transposing the *padded* rowwise grid gives (4, 4), which does not
    fit the columnwise storage — measured on the real checkpoint as
    `destination columnwise scale grid (10, 256) is smaller than the grid the checkpoint
    carries (12, 256)`, where the "checkpoint" grid was this intermediate.
    """
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")

    param = _FakeParam(512, 128, scale_pad_blocks=3)  # rowwise (4, 4), columnwise (1, 4)
    data = torch.zeros((512, 128), dtype=torch.uint8)
    scale = torch.arange(4, dtype=torch.float32).reshape(4, 1)

    qu.write_blockwise_fp8_param(param, data, scale, param_name="w")

    # The rowwise storage keeps TE's padding; the columnwise storage carries only the
    # logical transposed values, in the destination's own layout.
    assert tuple(param._rowwise_scale_inv.shape) == (4, 4)
    assert torch.equal(param._rowwise_scale_inv[:, :1], scale)
    assert tuple(param._columnwise_scale_inv.shape) == (1, 4)
    assert torch.equal(
        param._columnwise_scale_inv[: 1, : 4], scale.t()
    ), "the columnwise grid must be the logical transpose, not the padded one"
