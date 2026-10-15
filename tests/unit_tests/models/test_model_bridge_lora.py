# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.bridge.models.conversion.model_bridge as model_bridge_module
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import (
    AdapterWeightConversionTask,
    MegatronModelBridge,
    MegatronWeightTuple,
    WeightConversionTask,
)
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    RowParallelMapping,
    _fuse_gdn_separate_to_grouped,
    merge_gdn_linear_weights,
    merge_qkv_weights,
)
from megatron.bridge.models.conversion.peft_bridge import AdapterWeight


class DummyBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):  # pragma: no cover - not used in tests
        return None

    def mapping_registry(self):  # pragma: no cover - not used in tests
        return MegatronMappingRegistry()


@pytest.fixture(autouse=True)
def _patch_parallel_state(monkeypatch):
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_tensor_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "megatron.bridge.peft.lora.parallel_state.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "megatron.bridge.peft.lora.parallel_state.get_tensor_model_parallel_group",
        lambda: None,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_tensor_model_parallel_group",
        lambda: None,
    )


def _patch_pipeline_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    is_first_stage: bool,
    is_last_stage: bool,
    rank: int,
) -> tuple[object, Mock]:
    pp_group = object()
    embedding_group = object()
    broadcast = Mock()

    monkeypatch.setattr(model_bridge_module, "_get_pp_group", lambda _model: pp_group)
    monkeypatch.setattr(model_bridge_module, "_get_embedding_group", lambda _model: embedding_group)
    monkeypatch.setattr(model_bridge_module, "is_pp_first_stage", lambda _group: is_first_stage)
    monkeypatch.setattr(model_bridge_module, "is_pp_last_stage", lambda _group: is_last_stage)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _group: [0, 1])
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)

    return embedding_group, broadcast


def test_broadcast_shared_embeddings_uses_first_vpp_chunk_on_first_pp_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(share_embeddings_and_output_weights=True, pipeline_model_parallel_size=2)
    word_embeddings = torch.nn.Embedding(2, 3)
    output_layer = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        output_layer.weight.fill_(-1)

    model_chunks = [
        SimpleNamespace(config=config, embedding=SimpleNamespace(word_embeddings=word_embeddings)),
        SimpleNamespace(config=config, output_layer=output_layer),
    ]
    embedding_group, broadcast = _patch_pipeline_state(
        monkeypatch,
        is_first_stage=True,
        is_last_stage=False,
        rank=0,
    )

    DummyBridge()._broadcast_shared_embeddings(model_chunks)

    broadcast.assert_called_once()
    broadcasted_weight = broadcast.call_args.args[0]
    assert broadcasted_weight.data_ptr() == word_embeddings.weight.data_ptr()
    assert broadcast.call_args.kwargs == {"src": 0, "group": embedding_group}
    torch.testing.assert_close(output_layer.weight, torch.full_like(output_layer.weight, -1))


def test_broadcast_shared_embeddings_uses_last_vpp_chunk_on_last_pp_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(share_embeddings_and_output_weights=True, pipeline_model_parallel_size=2)
    output_layer = torch.nn.Linear(3, 2, bias=False)
    received_weight = torch.full_like(output_layer.weight, 4)
    model_chunks = [
        SimpleNamespace(config=config),
        SimpleNamespace(config=config, output_layer=output_layer),
    ]
    embedding_group, broadcast = _patch_pipeline_state(
        monkeypatch,
        is_first_stage=False,
        is_last_stage=True,
        rank=1,
    )

    def receive_weight(tensor: torch.Tensor, *, src: int, group: object) -> None:
        assert src == 0
        assert group is embedding_group
        tensor.copy_(received_weight)

    broadcast.side_effect = receive_weight

    DummyBridge()._broadcast_shared_embeddings(model_chunks)

    broadcast.assert_called_once()
    torch.testing.assert_close(output_layer.weight, received_weight)


def test_merge_lora_adapter_weights_merges(monkeypatch):
    bridge = DummyBridge()
    base_weight = torch.zeros(4, 4)
    converted = {"hf.weight": base_weight.clone()}
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.linear_fc1",
        adapter_key=None,
        alpha=4,
        dim=4,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(4), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", torch.eye(4), vp_stage=0),
    )

    updated = bridge._merge_lora_adapter_weights(
        [Mock(config=SimpleNamespace(num_moe_experts=0))],
        converted,
        [adapter_weight],
    )
    expected = base_weight + torch.eye(4)
    torch.testing.assert_close(updated["hf.weight"], expected)


def test_merge_lora_adapter_weights_empty_returns_empty():
    bridge = DummyBridge()
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.linear_fc1",
        adapter_key=None,
        alpha=4,
        dim=4,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(4), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", torch.eye(4), vp_stage=0),
    )

    updated = bridge._merge_lora_adapter_weights(
        [SimpleNamespace(config=SimpleNamespace(num_moe_experts=0))],
        {},
        [adapter_weight],
    )
    assert updated == {}


def test_merge_single_adapter_weight_matches_loramerge():
    bridge = DummyBridge()
    base = torch.zeros(2, 2)
    linear_in = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    linear_out = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    merged = bridge._merge_single_adapter_weight(
        base, alpha=2, dim=2, linear_in_weight=linear_in, linear_out_weight=linear_out
    )
    expected = base + 2 / 2 * (linear_out @ linear_in)
    torch.testing.assert_close(merged, expected)


def test_merge_lora_adapter_weights_fused_fc1(monkeypatch):
    bridge = DummyBridge()
    base = torch.zeros(4, 4)
    converted = {
        "decoder.layers.0.mlp.gate_proj.weight": base.clone(),
        "decoder.layers.0.mlp.up_proj.weight": base.clone(),
    }

    linear_out = torch.cat([torch.eye(4), 2 * torch.eye(4)], dim=0)
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.linear_fc1",
        adapter_key=None,
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(4), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", linear_out, vp_stage=0),
    )

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    updated = bridge._merge_lora_adapter_weights(
        [Mock(config=SimpleNamespace(num_moe_experts=0))],
        converted,
        [adapter_weight],
    )
    torch.testing.assert_close(updated["decoder.layers.0.mlp.gate_proj.weight"], torch.eye(4))
    torch.testing.assert_close(updated["decoder.layers.0.mlp.up_proj.weight"], 2 * torch.eye(4))


def test_merge_lora_adapter_weights_fused_fc1_tp_aware(monkeypatch):
    bridge = DummyBridge()
    base = torch.zeros(4, 4)
    converted = {
        "decoder.layers.0.mlp.gate_proj.weight": base.clone(),
        "decoder.layers.0.mlp.up_proj.weight": base.clone(),
    }

    gate0 = torch.arange(0, 8, dtype=base.dtype).reshape(2, 4)
    up0 = torch.arange(100, 108, dtype=base.dtype).reshape(2, 4)
    gate1 = torch.arange(200, 208, dtype=base.dtype).reshape(2, 4)
    up1 = torch.arange(300, 308, dtype=base.dtype).reshape(2, 4)
    linear_out = torch.cat([gate0, up0, gate1, up1], dim=0)
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.linear_fc1",
        adapter_key=None,
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(4), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", linear_out, vp_stage=0),
    )

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_tensor_model_parallel_world_size",
        lambda: 2,
    )

    updated = bridge._merge_lora_adapter_weights(
        [Mock(config=SimpleNamespace(num_moe_experts=0))],
        converted,
        [adapter_weight],
    )
    expected_gate = torch.cat([gate0, gate1], dim=0)
    expected_up = torch.cat([up0, up1], dim=0)
    torch.testing.assert_close(updated["decoder.layers.0.mlp.gate_proj.weight"], expected_gate)
    torch.testing.assert_close(updated["decoder.layers.0.mlp.up_proj.weight"], expected_up)


def test_merge_lora_adapter_weights_non_expert_uses_regular_tp(monkeypatch):
    bridge = DummyBridge()
    converted = {"decoder.layers.0.self_attn.q_proj.weight": torch.zeros(288, 2048)}

    linear_in_full = torch.arange(24 * 2048, dtype=torch.float32).reshape(24, 2048)
    linear_out_full = torch.arange(288 * 24, dtype=torch.float32).reshape(288, 24)
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.self_attention.linear_q_proj",
        adapter_key=None,
        alpha=24,
        dim=24,
        linear_in_weight=MegatronWeightTuple("in", linear_in_full, vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", linear_out_full, vp_stage=0),
    )

    updated = bridge._merge_lora_adapter_weights(
        [SimpleNamespace(config=SimpleNamespace(num_moe_experts=0))],
        converted,
        [adapter_weight],
    )

    expected = linear_out_full @ linear_in_full
    torch.testing.assert_close(updated["decoder.layers.0.self_attn.q_proj.weight"], expected)


def test_merge_lora_adapter_weights_fused_fc1_minimax_w13(monkeypatch):
    bridge = DummyBridge()
    base = torch.zeros(4, 4)
    converted = {
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": base.clone(),
        "model.layers.0.block_sparse_moe.experts.0.w3.weight": base.clone(),
    }

    linear_out = torch.cat([torch.eye(4), 2 * torch.eye(4)], dim=0)
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc1",
        adapter_key=None,
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(4), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", linear_out, vp_stage=0),
    )

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_tensor_parallel_world_size",
        lambda: 1,
    )

    updated = bridge._merge_lora_adapter_weights(
        [Mock(config=SimpleNamespace(num_moe_experts=1))],
        converted,
        [adapter_weight],
    )
    torch.testing.assert_close(updated["model.layers.0.block_sparse_moe.experts.0.w1.weight"], torch.eye(4))
    torch.testing.assert_close(updated["model.layers.0.block_sparse_moe.experts.0.w3.weight"], 2 * torch.eye(4))


def test_merge_lora_adapter_weights_qkv_split(monkeypatch):
    bridge = DummyBridge()
    config = SimpleNamespace(
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=None,
        hidden_size=4,
        attention_output_gate=False,
        num_moe_experts=0,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        kda_two_stage_gates=False,
    )
    megatron_model = [SimpleNamespace(config=config)]
    converted = {
        "q_proj.weight": torch.zeros(4, 4),
        "k_proj.weight": torch.zeros(2, 4),
        "v_proj.weight": torch.zeros(2, 4),
    }

    q_weight = torch.eye(4)
    k_weight = torch.ones(2, 4)
    v_weight = torch.full((2, 4), 2.0)
    linear_out = merge_qkv_weights(config, q_weight, k_weight, v_weight)

    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.self_attention.linear_qkv",
        adapter_key=None,
        alpha=4,
        dim=4,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(4), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", linear_out, vp_stage=0),
    )

    updated = bridge._merge_lora_adapter_weights(megatron_model, converted, [adapter_weight])
    torch.testing.assert_close(updated["q_proj.weight"], q_weight)
    torch.testing.assert_close(updated["k_proj.weight"], k_weight)
    torch.testing.assert_close(updated["v_proj.weight"], v_weight)


@pytest.mark.parametrize("trailing_shape", [(), (4,)])
def test_split_kda_qkv_linear_out_weight(trailing_shape):
    config = SimpleNamespace(
        kda_two_stage_gates=True,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
    )
    sections = [torch.full((4, *trailing_shape), value) for value in (1.0, 2.0, 3.0)]
    actual = DummyBridge()._split_qkv_linear_out_weight([SimpleNamespace(config=config)], torch.cat(sections))
    for name, expected in zip(("q_proj", "k_proj", "v_proj"), sections, strict=True):
        torch.testing.assert_close(actual[name], expected)


def test_merge_lora_adapter_weights_grouped_expert_missing_expert_idx(monkeypatch):
    bridge = DummyBridge()
    base = torch.zeros(2, 2)
    converted = {"model.layers.0.mlp.experts.down_proj.weight": base.clone()}

    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
        adapter_key=None,
        alpha=2,
        dim=2,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", 2 * torch.eye(2), vp_stage=0),
    )

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_model_parallel_world_size",
        lambda: 1,
    )

    updated = bridge._merge_lora_adapter_weights(
        [SimpleNamespace(config=SimpleNamespace(num_moe_experts=2))],
        converted,
        [adapter_weight],
    )

    torch.testing.assert_close(updated["model.layers.0.mlp.experts.down_proj.weight"], 2 * torch.eye(2))


def test_merge_lora_adapter_weights_grouped_expert_gate_up_proj_unfused(monkeypatch):
    bridge = DummyBridge()
    base = torch.zeros(2, 2)
    converted = {"model.language_model.layers.0.mlp.experts.gate_up_proj": base.clone()}

    adapter_weight = AdapterWeight(
        global_base_prefix="language_model.decoder.layers.0.mlp.experts.linear_fc1",
        adapter_key=None,
        alpha=2,
        dim=2,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", 2 * torch.eye(2), vp_stage=0),
    )

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_model_parallel_world_size",
        lambda: 1,
    )

    updated = bridge._merge_lora_adapter_weights(
        [SimpleNamespace(config=SimpleNamespace(num_moe_experts=2))],
        converted,
        [adapter_weight],
    )

    torch.testing.assert_close(updated["model.language_model.layers.0.mlp.experts.gate_up_proj"], 2 * torch.eye(2))


def test_merge_lora_adapter_weights_grouped_expert_per_expert_tensors(monkeypatch):
    bridge = DummyBridge()
    converted = {
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.zeros(2, 2),
        "model.layers.0.mlp.experts.1.down_proj.weight": torch.zeros(2, 2),
    }

    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
        adapter_key=None,
        alpha=2,
        dim=2,
        linear_in_weight=MegatronWeightTuple(
            "in",
            torch.stack([torch.eye(2), 2 * torch.eye(2)], dim=0),
            vp_stage=0,
        ),
        linear_out_weight=MegatronWeightTuple(
            "out",
            torch.stack([torch.eye(2), 3 * torch.eye(2)], dim=0),
            vp_stage=0,
        ),
    )

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_model_parallel_world_size",
        lambda: 1,
    )

    updated = bridge._merge_lora_adapter_weights(
        [SimpleNamespace(config=SimpleNamespace(num_moe_experts=2))],
        converted,
        [adapter_weight],
    )

    torch.testing.assert_close(updated["model.layers.0.mlp.experts.0.down_proj.weight"], torch.eye(2))
    torch.testing.assert_close(updated["model.layers.0.mlp.experts.1.down_proj.weight"], 6 * torch.eye(2))


def test_merge_lora_adapter_weights_grouped_expert_fused_fc1_uses_expert_tp(monkeypatch):
    bridge = DummyBridge()
    converted = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(2, 4),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.zeros(2, 4),
    }

    linear_in_full = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    gate_local = torch.arange(16, 32, dtype=torch.float32).reshape(2, 8)
    up_local = torch.arange(32, 48, dtype=torch.float32).reshape(2, 8)
    linear_out_local = torch.cat([gate_local, up_local], dim=0)

    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc1",
        adapter_key=None,
        alpha=8,
        dim=8,
        linear_in_weight=MegatronWeightTuple(
            "in",
            torch.stack([linear_in_full, linear_in_full + 100], dim=0),
            vp_stage=0,
        ),
        linear_out_weight=MegatronWeightTuple(
            "out",
            torch.stack([linear_out_local, linear_out_local + 100], dim=0),
            vp_stage=0,
        ),
    )

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_model_parallel_world_size",
        lambda: 1,
    )

    updated = bridge._merge_lora_adapter_weights(
        [SimpleNamespace(config=SimpleNamespace(num_moe_experts=2))],
        converted,
        [adapter_weight],
    )

    expected_gate = gate_local @ linear_in_full
    expected_up = up_local @ linear_in_full

    torch.testing.assert_close(updated["model.layers.0.mlp.experts.0.gate_proj.weight"], expected_gate)
    torch.testing.assert_close(updated["model.layers.0.mlp.experts.0.up_proj.weight"], expected_up)


def test_select_expert_adapter_weight_per_expert_gathered_tensor(monkeypatch):
    bridge = DummyBridge()
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_model_parallel_world_size",
        lambda: 2,
    )

    gathered = [
        torch.stack([torch.full((2, 2), 1.0), torch.full((2, 2), 2.0)], dim=0),
        torch.stack([torch.full((2, 2), 3.0), torch.full((2, 2), 4.0)], dim=0),
    ]

    selected = bridge._select_expert_adapter_weight(
        gathered[0],
        gathered,
        expert_idx=3,
        num_experts=4,
    )
    torch.testing.assert_close(selected, torch.full((2, 2), 4.0))


def test_merge_canonical_adapter_from_weights(monkeypatch):
    bridge = DummyBridge()
    converted = {
        "decoder.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        "decoder.layers.0.self_attn.k_proj.weight": torch.zeros(1, 2),
        "decoder.layers.0.self_attn.v_proj.weight": torch.zeros(1, 2),
    }

    adapter_q = AdapterWeight(
        global_base_prefix="decoder.layers.0.self_attn.linear_qkv",
        adapter_key="adapter_q",
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in_q", torch.eye(2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out_q", torch.ones(2, 2), vp_stage=0),
    )
    adapter_k = AdapterWeight(
        global_base_prefix="decoder.layers.0.self_attn.linear_qkv",
        adapter_key="adapter_k",
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in_k", torch.eye(2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out_k", 2 * torch.ones(1, 2), vp_stage=0),
    )
    adapter_v = AdapterWeight(
        global_base_prefix="decoder.layers.0.self_attn.linear_qkv",
        adapter_key="adapter_v",
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in_v", torch.eye(2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out_v", 3 * torch.ones(1, 2), vp_stage=0),
    )

    megatron_model = [SimpleNamespace(config=SimpleNamespace(num_moe_experts=0))]
    updated = bridge._merge_canonical_adapter_from_weights(
        megatron_model,
        converted,
        [adapter_q, adapter_k, adapter_v],
    )
    torch.testing.assert_close(updated["decoder.layers.0.self_attn.q_proj.weight"], torch.ones(2, 2))
    torch.testing.assert_close(updated["decoder.layers.0.self_attn.k_proj.weight"], 2 * torch.ones(1, 2))
    torch.testing.assert_close(updated["decoder.layers.0.self_attn.v_proj.weight"], 3 * torch.ones(1, 2))


def test_column_parallel_mapping_gathers_3d_expert_adapter_along_tp(monkeypatch):
    mapping = ColumnParallelMapping(
        "decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_out.weight",
        "hf_param",
    )

    monkeypatch.setattr(ColumnParallelMapping, "broadcast_from_pp_rank", lambda self, tensor, cache_key=None: tensor)
    monkeypatch.setattr(ColumnParallelMapping, "tp_size", property(lambda self: 2))
    monkeypatch.setattr(
        ColumnParallelMapping,
        "gather_from_tp_ranks",
        lambda self, tensor: [tensor, 2 * tensor],
    )

    local_weight = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    result = mapping.megatron_to_hf(local_weight, None)

    expected = torch.cat([local_weight, 2 * local_weight], dim=0)
    torch.testing.assert_close(result["hf_param"], expected)


def test_row_parallel_mapping_gathers_3d_expert_adapter_along_tp(monkeypatch):
    mapping = RowParallelMapping(
        "decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_in.weight",
        "hf_param",
    )

    monkeypatch.setattr(RowParallelMapping, "broadcast_from_pp_rank", lambda self, tensor, cache_key=None: tensor)
    monkeypatch.setattr(RowParallelMapping, "tp_size", property(lambda self: 2))
    monkeypatch.setattr(
        RowParallelMapping,
        "gather_from_tp_ranks",
        lambda self, tensor: [tensor, 2 * tensor],
    )

    local_weight = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    result = mapping.megatron_to_hf(local_weight, None)

    expected = torch.cat([local_weight, 2 * local_weight], dim=1)
    torch.testing.assert_close(result["hf_param"], expected)


def test_global_param_names_skip_adapter(monkeypatch):
    bridge = DummyBridge()

    class DummyGroup:
        def size(self):
            return 1

        def rank(self):
            return 0

    fake_param = torch.nn.Parameter(torch.zeros(1, 1))

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace()

        def named_parameters(self):
            return [
                ("decoder.layers.0.mlp.adapter.linear_in.weight", fake_param),
                ("decoder.layers.0.mlp.linear_fc1.to_wrap.weight", fake_param),
            ]

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_group",
        lambda: DummyGroup(),
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.persistent_buffers",
        lambda *_: [],
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge._megatron_local_name_to_global",
        lambda *_args, **_kwargs: _args[2],
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.unwrap_model",
        lambda models: models if isinstance(models, list) else [models],
    )
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda output, obj, group=None: output.__setitem__(0, obj),
    )

    names = bridge._megatron_global_param_names_all_pp_ranks([FakeModel()])
    assert names == ["decoder.layers.0.mlp.linear_fc1.to_wrap.weight"]


def test_megatron_global_adapters_info_all_pp_ranks(monkeypatch):
    bridge = DummyBridge()

    class DummyGroup:
        def size(self):
            return 1

        def rank(self):
            return 0

    class FakeAdapter:
        def __init__(self):
            self.linear_in = SimpleNamespace(weight=torch.ones(2, 2))
            self.linear_out = SimpleNamespace(weight=torch.ones(2, 2))
            self.input_is_parallel = True
            self.base_linear_is_parallel = False
            self.alpha = 8
            self.dim = 2

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace()
            param = torch.nn.Parameter(torch.zeros(2, 2))
            self._params = [
                ("decoder.layers.0.mlp.linear_fc1.adapter.linear_in.weight", param),
                ("decoder.layers.0.mlp.linear_fc1.adapter.linear_out.weight", param),
            ]

        def named_parameters(self):
            return self._params

        def named_modules(self):
            # No submodules required for this test
            return []

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_group",
        lambda: DummyGroup(),
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.persistent_buffers",
        lambda *_: [],
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.persistent_buffers",
        lambda *_: [],
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge._megatron_local_name_to_global",
        lambda *_args, **_kwargs: _args[2],
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.unwrap_model",
        lambda models: models if isinstance(models, list) else [models],
    )
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda output, obj, group=None: output.__setitem__(0, obj),
    )

    adapter = FakeAdapter()
    monkeypatch.setattr(bridge, "_get_adapter_wrap_module", lambda *_: (adapter, None))

    info = bridge._megatron_global_adapters_info_all_pp_ranks([FakeModel()])
    assert len(info) == 1
    (
        global_base_name,
        local_base_prefix,
        input_is_parallel,
        base_linear_is_parallel,
        requires_expert_splits,
        alpha,
        dim,
        pp_rank,
        vp_stage,
        is_split_gate_up,
        is_split_qkv,
    ) = info[0]
    assert global_base_name == "decoder.layers.0.mlp.linear_fc1.adapter"
    assert local_base_prefix == "decoder.layers.0.mlp.linear_fc1"
    assert input_is_parallel is True and base_linear_is_parallel is True
    assert requires_expert_splits is False
    assert alpha == 8 and dim == 2 and pp_rank == 0 and vp_stage == 0
    # Split gate/up adapters (SplitGateUpAdapter) are flagged so the
    # conversion emits one task per pair; a plain ParallelLinearAdapter is not.
    assert is_split_gate_up is False
    # Same flag for split q/k/v adapters (SplitQKVAdapter).
    assert is_split_qkv is False


def test_construct_adapters_names():
    bridge = DummyBridge()
    linear_in, linear_out = bridge._construct_adapters_names("decoder.layers.0.mlp.linear_fc1", None)
    assert linear_in == "decoder.layers.0.mlp.linear_fc1.adapter.linear_in.weight"
    assert linear_out == "decoder.layers.0.mlp.linear_fc1.adapter.linear_out.weight"

    linear_in_k, linear_out_k = bridge._construct_adapters_names("decoder.layers.0.attn.q_proj", "adapter_q")
    assert linear_in_k.endswith("adapter_q.linear_in.weight")
    assert linear_out_k.endswith("adapter_q.linear_out.weight")


def test_make_lora_param_name_without_weight_suffix():
    bridge = DummyBridge()
    base_name = "model.layers.0.mlp.experts.down_proj"
    assert bridge._make_lora_param_name(base_name, ".linear_in.weight") == base_name + ".lora_A.weight"
    assert bridge._make_lora_param_name(base_name, ".linear_out.weight") == base_name + ".lora_B.weight"


def test_resolve_hf_adapter_param_name_without_weight_suffix():
    bridge = DummyBridge()
    registry = MegatronMappingRegistry(
        AutoMapping(
            megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
            hf_param="model.layers.*.mlp.experts.gate_up_proj",
        )
    )

    name = bridge._resolve_hf_adapter_param_name(
        registry,
        "decoder.layers.0.mlp.linear_fc1",
        ".linear_in.weight",
        ".weight",
        None,
    )
    assert name == "model.layers.0.mlp.experts.gate_up_proj.lora_A.weight"


def test_build_adapter_conversion_tasks(monkeypatch):
    bridge = DummyBridge()
    bridge.hf_pretrained = SimpleNamespace()
    bridge.hf_config = bridge.hf_pretrained

    adapters_info = [
        (
            "decoder.layers.0.mlp.linear_fc1.adapter",
            "decoder.layers.0.mlp.linear_fc1",
            False,
            False,
            False,
            4,
            8,
            0,
            0,
            False,
            False,
        )
    ]

    adapter = SimpleNamespace(
        linear_in=SimpleNamespace(weight=torch.ones(2, 2)),
        linear_out=SimpleNamespace(weight=torch.ones(2, 2)),
        alpha=4,
        dim=8,
    )

    monkeypatch.setattr(bridge, "_megatron_global_adapters_info_all_pp_ranks", lambda *_: adapters_info)
    monkeypatch.setattr(bridge, "_get_adapter_wrap_module", lambda *_: (adapter, None))
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_rank",
        lambda: 0,
    )

    # Provide a minimal mapping so adapter base lookups succeed during task construction.
    registry = MegatronMappingRegistry(
        AutoMapping(
            megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
            hf_param="decoder.layers.*.mlp.linear_fc1.weight",
        )
    )
    monkeypatch.setattr(bridge, "mapping_registry", lambda: registry)

    tasks_by_base = bridge.build_adapter_conversion_tasks([Mock()])
    assert "decoder.layers.0.mlp.linear_fc1" in tasks_by_base
    tasks = tasks_by_base["decoder.layers.0.mlp.linear_fc1"]
    assert len(tasks) == 1
    task = tasks[0]
    assert task.adapter_key is None
    assert task.linear_in_task.param_weight.shape == torch.Size([2, 2])
    assert task.linear_out_task.param_weight.shape == torch.Size([2, 2])


def test_build_adapter_conversion_tasks_excludes_base_prefix_before_mapping(monkeypatch):
    bridge = DummyBridge()
    bridge.hf_pretrained = SimpleNamespace()
    bridge.hf_config = bridge.hf_pretrained

    adapters_info = [
        (
            "mtp.layers.0.mtp_model_layer.layers.0.self_attention.linear_proj.adapter",
            "mtp.layers.0.mtp_model_layer.layers.0.self_attention.linear_proj",
            False,
            False,
            False,
            4,
            8,
            0,
            0,
            False,
            False,
        )
    ]

    monkeypatch.setattr(bridge, "_megatron_global_adapters_info_all_pp_ranks", lambda *_: adapters_info)
    monkeypatch.setattr(bridge, "mapping_registry", lambda: MegatronMappingRegistry())

    tasks_by_base = bridge.build_adapter_conversion_tasks(
        [Mock()],
        exclude_adapter_base_prefixes=("mtp.layers",),
    )

    assert tasks_by_base == {}


def test_materialize_adapter_weights(monkeypatch):
    bridge = DummyBridge()

    class DummyMapping:
        def __init__(self, payload):
            self.payload = payload

        def megatron_to_hf(self, weight, module):
            return {"hf": self.payload}

    adapter_tasks = [
        AdapterWeightConversionTask(
            global_base_prefix="decoder.layers.0.mlp.linear_fc1",
            adapter_key=None,
            alpha=2,
            dim=4,
            linear_in_task=WeightConversionTask(
                param_name="in_name",
                global_param_name="in_name",
                mapping=DummyMapping(torch.ones(2, 2)),
                megatron_module=None,
                param_weight=None,
            ),
            linear_out_task=WeightConversionTask(
                param_name="out_name",
                global_param_name="out_name",
                mapping=DummyMapping(2 * torch.ones(2, 2)),
                megatron_module=None,
                param_weight=None,
            ),
        )
    ]

    materials = bridge.materialize_adapter_weights(adapter_tasks)
    assert len(materials) == 1
    assert torch.all(materials[0].linear_in_weight.weight == torch.ones(2, 2))
    assert torch.all(materials[0].linear_out_weight.weight == 2 * torch.ones(2, 2))


def test_materialize_adapter_weights_grouped_expert_fc1_uses_expert_tp_axis(monkeypatch):
    bridge = DummyBridge()

    linear_in_mapping = ColumnParallelMapping(
        "decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_in.weight",
        "hf.in",
    )
    linear_out_mapping = ColumnParallelMapping(
        "decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_out.weight",
        "hf.out",
    )

    monkeypatch.setattr(
        ColumnParallelMapping,
        "broadcast_from_pp_rank",
        lambda self, tensor, cache_key=None: tensor,
    )
    monkeypatch.setattr(ColumnParallelMapping, "maybe_dequantize", lambda self, tensor: tensor)
    monkeypatch.setattr(ColumnParallelMapping, "tp_size", property(lambda self: 2))
    monkeypatch.setattr(
        ColumnParallelMapping,
        "gather_from_tp_ranks",
        lambda self, tensor: [tensor, tensor + 100],
    )

    linear_in_local = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    linear_out_local = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    adapter_tasks = [
        AdapterWeightConversionTask(
            global_base_prefix="decoder.layers.0.mlp.experts.linear_fc1",
            adapter_key=None,
            alpha=2,
            dim=4,
            linear_in_task=WeightConversionTask(
                param_name="in_name",
                global_param_name="decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_in.weight",
                mapping=linear_in_mapping,
                megatron_module=None,
                param_weight=linear_in_local,
            ),
            linear_out_task=WeightConversionTask(
                param_name="out_name",
                global_param_name="decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_out.weight",
                mapping=linear_out_mapping,
                megatron_module=None,
                param_weight=linear_out_local,
            ),
            requires_expert_splits=True,
        )
    ]

    materials = bridge.materialize_adapter_weights(adapter_tasks)

    torch.testing.assert_close(
        materials[0].linear_in_weight.weight,
        torch.cat([linear_in_local, linear_in_local + 100], dim=1),
    )
    torch.testing.assert_close(
        materials[0].linear_out_weight.weight,
        torch.cat([linear_out_local, linear_out_local + 100], dim=1),
    )


def test_materialize_adapter_weights_grouped_expert_fc2_uses_expert_tp_axis(monkeypatch):
    bridge = DummyBridge()

    linear_in_mapping = RowParallelMapping(
        "decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_in.weight",
        "hf.in",
    )
    linear_out_mapping = ColumnParallelMapping(
        "decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_out.weight",
        "hf.out",
    )

    monkeypatch.setattr(
        RowParallelMapping,
        "broadcast_from_pp_rank",
        lambda self, tensor, cache_key=None: tensor,
    )
    monkeypatch.setattr(RowParallelMapping, "maybe_dequantize", lambda self, tensor: tensor)
    monkeypatch.setattr(RowParallelMapping, "tp_size", property(lambda self: 2))
    monkeypatch.setattr(
        RowParallelMapping,
        "gather_from_tp_ranks",
        lambda self, tensor: [tensor, tensor + 100],
    )
    monkeypatch.setattr(
        ColumnParallelMapping,
        "broadcast_from_pp_rank",
        lambda self, tensor, cache_key=None: tensor,
    )
    monkeypatch.setattr(ColumnParallelMapping, "maybe_dequantize", lambda self, tensor: tensor)
    monkeypatch.setattr(ColumnParallelMapping, "tp_size", property(lambda self: 2))
    monkeypatch.setattr(
        ColumnParallelMapping,
        "gather_from_tp_ranks",
        lambda self, tensor: [tensor, tensor + 100],
    )

    linear_in_local = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    linear_out_local = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    adapter_tasks = [
        AdapterWeightConversionTask(
            global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
            adapter_key=None,
            alpha=2,
            dim=4,
            linear_in_task=WeightConversionTask(
                param_name="in_name",
                global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_in.weight",
                mapping=linear_in_mapping,
                megatron_module=None,
                param_weight=linear_in_local,
            ),
            linear_out_task=WeightConversionTask(
                param_name="out_name",
                global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_out.weight",
                mapping=linear_out_mapping,
                megatron_module=None,
                param_weight=linear_out_local,
            ),
            requires_expert_splits=True,
        )
    ]

    materials = bridge.materialize_adapter_weights(adapter_tasks)

    torch.testing.assert_close(
        materials[0].linear_in_weight.weight,
        torch.cat([linear_in_local, linear_in_local + 100], dim=2),
    )
    torch.testing.assert_close(
        materials[0].linear_out_weight.weight,
        torch.cat([linear_out_local, linear_out_local + 100], dim=1),
    )


def test_materialize_adapter_weights_shared_expert_adapter_uses_regular_mapping(monkeypatch):
    bridge = DummyBridge()

    linear_in_mapping = Mock()
    linear_in_mapping.megatron_to_hf.return_value = {"hf.in": torch.ones(2, 3)}
    linear_out_mapping = Mock()
    linear_out_mapping.megatron_to_hf.return_value = {"hf.out": 2 * torch.ones(4, 2)}

    adapter_tasks = [
        AdapterWeightConversionTask(
            global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
            adapter_key=None,
            alpha=2,
            dim=4,
            linear_in_task=WeightConversionTask(
                param_name="in_name",
                global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_in.weight",
                mapping=linear_in_mapping,
                megatron_module=None,
                param_weight=torch.zeros(2, 3),
            ),
            linear_out_task=WeightConversionTask(
                param_name="out_name",
                global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_out.weight",
                mapping=linear_out_mapping,
                megatron_module=None,
                param_weight=torch.zeros(4, 2),
            ),
        )
    ]

    materialize_grouped = Mock(
        side_effect=AssertionError("shared expert adapters should not use grouped materialization")
    )
    monkeypatch.setattr(bridge, "_materialize_grouped_expert_adapter_tensor", materialize_grouped)

    materials = bridge.materialize_adapter_weights(adapter_tasks)

    linear_in_mapping.megatron_to_hf.assert_called_once()
    linear_out_mapping.megatron_to_hf.assert_called_once()
    materialize_grouped.assert_not_called()
    torch.testing.assert_close(materials[0].linear_in_weight.weight, torch.ones(2, 3))
    torch.testing.assert_close(materials[0].linear_out_weight.weight, 2 * torch.ones(4, 2))


def test_stream_adapter_weights_megatron_to_hf(monkeypatch):
    bridge = DummyBridge()
    bridge.hf_pretrained = SimpleNamespace()
    bridge.hf_config = bridge.hf_pretrained

    adapter_task = AdapterWeightConversionTask(
        global_base_prefix="decoder.layers.0.mlp.linear_fc1",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_task=WeightConversionTask(
            param_name="local_in",
            global_param_name="decoder.layers.0.mlp.linear_fc1.adapter.linear_in.weight",
            mapping=Mock(),
        ),
        linear_out_task=WeightConversionTask(
            param_name="local_out",
            global_param_name="decoder.layers.0.mlp.linear_fc1.adapter.linear_out.weight",
            mapping=Mock(),
        ),
    )

    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.linear_fc1",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_weight=MegatronWeightTuple("local_in", torch.ones(2, 2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("local_out", 2 * torch.ones(2, 2), vp_stage=0),
    )

    monkeypatch.setattr(
        bridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {"decoder.layers.0.mlp.linear_fc1": [adapter_task]},
    )
    monkeypatch.setattr(
        bridge,
        "materialize_adapter_weights",
        lambda *_: [adapter_weight],
    )

    # Provide a base HF weight name so stream_adapter_weights_megatron_to_hf can
    # translate it into lora_A/lora_B names.
    monkeypatch.setattr(
        bridge,
        "_get_base_hf_param_names_for_adapter",
        lambda *_args, **_kwargs: ["model.layers.0.mlp.linear_fc1.weight"],
    )

    megatron_model = [SimpleNamespace(config=SimpleNamespace(num_moe_experts=0))]
    weights = list(
        bridge.stream_adapter_weights_megatron_to_hf(
            megatron_model,
            cpu=False,
            show_progress=False,
        )
    )
    assert len(weights) == 2
    assert weights[0].param_name.endswith("lora_A.weight")
    assert weights[1].param_name.endswith("lora_B.weight")
    torch.testing.assert_close(weights[0].weight, torch.ones(2, 2))
    torch.testing.assert_close(weights[1].weight, 2 * torch.ones(2, 2))


def test_stream_adapter_weights_megatron_to_hf_qkv(monkeypatch):
    bridge = DummyBridge()
    bridge.hf_pretrained = SimpleNamespace()
    bridge.hf_config = bridge.hf_pretrained

    adapter_task = AdapterWeightConversionTask(
        global_base_prefix="decoder.layers.0.self_attn.linear_qkv",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_task=WeightConversionTask(
            param_name="local_in",
            global_param_name="decoder.layers.0.self_attn.linear_qkv.adapter.linear_in.weight",
            mapping=Mock(),
        ),
        linear_out_task=WeightConversionTask(
            param_name="local_out",
            global_param_name="decoder.layers.0.self_attn.linear_qkv.adapter.linear_out.weight",
            mapping=Mock(),
        ),
    )

    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.self_attn.linear_qkv",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_weight=MegatronWeightTuple("local_in", torch.ones(2, 2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("local_out", torch.ones(6, 2), vp_stage=0),
    )

    monkeypatch.setattr(
        bridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {"decoder.layers.0.self_attn.linear_qkv": [adapter_task]},
    )
    monkeypatch.setattr(bridge, "materialize_adapter_weights", lambda *_: [adapter_weight])
    monkeypatch.setattr(
        bridge,
        "_get_base_hf_param_names_for_adapter",
        lambda *_: [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ],
    )

    qkv_slices = {
        "q_proj": torch.full((2, 2), 1.0),
        "k_proj": torch.full((2, 2), 2.0),
        "v_proj": torch.full((2, 2), 3.0),
    }
    monkeypatch.setattr(bridge, "_split_qkv_linear_out_weight", lambda *_: qkv_slices)

    weights = list(
        bridge.stream_adapter_weights_megatron_to_hf(
            [SimpleNamespace(config=SimpleNamespace(num_moe_experts=0))],
            cpu=False,
            show_progress=False,
        )
    )

    assert len(weights) == 6
    names = [w.param_name for w in weights]
    assert names[0].endswith("q_proj.lora_A.weight")
    assert names[1].endswith("q_proj.lora_B.weight")
    assert names[2].endswith("k_proj.lora_A.weight")
    assert names[3].endswith("k_proj.lora_B.weight")
    assert names[4].endswith("v_proj.lora_A.weight")
    assert names[5].endswith("v_proj.lora_B.weight")

    torch.testing.assert_close(weights[0].weight, torch.ones(2, 2))
    torch.testing.assert_close(weights[1].weight, qkv_slices["q_proj"])
    torch.testing.assert_close(weights[3].weight, qkv_slices["k_proj"])


def test_stream_adapter_weights_megatron_to_hf_fused_fc1(monkeypatch):
    bridge = DummyBridge()
    bridge.hf_pretrained = SimpleNamespace()
    bridge.hf_config = bridge.hf_pretrained

    adapter_task = AdapterWeightConversionTask(
        global_base_prefix="decoder.layers.0.mlp.linear_fc1",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_task=WeightConversionTask(
            param_name="local_in",
            global_param_name="decoder.layers.0.mlp.linear_fc1.adapter.linear_in.weight",
            mapping=Mock(),
        ),
        linear_out_task=WeightConversionTask(
            param_name="local_out",
            global_param_name="decoder.layers.0.mlp.linear_fc1.adapter.linear_out.weight",
            mapping=Mock(),
        ),
    )

    # Megatron stores fused FC1 as [gate; up] along dim=0. Streaming should split B
    # and duplicate A for gate_proj and up_proj.
    linear_out = torch.cat([torch.full((2, 2), 1.0), torch.full((2, 2), 2.0)], dim=0)
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.linear_fc1",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_weight=MegatronWeightTuple("local_in", torch.ones(2, 2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("local_out", linear_out, vp_stage=0),
    )

    monkeypatch.setattr(
        bridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {"decoder.layers.0.mlp.linear_fc1": [adapter_task]},
    )
    monkeypatch.setattr(bridge, "materialize_adapter_weights", lambda *_: [adapter_weight])
    monkeypatch.setattr(
        bridge,
        "_get_base_hf_param_names_for_adapter",
        lambda *_: [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ],
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    weights = list(
        bridge.stream_adapter_weights_megatron_to_hf(
            [SimpleNamespace(config=SimpleNamespace(num_moe_experts=0))],
            cpu=False,
            show_progress=False,
        )
    )

    assert len(weights) == 4
    names = [w.param_name for w in weights]
    assert names[0].endswith("gate_proj.lora_A.weight")
    assert names[1].endswith("gate_proj.lora_B.weight")
    assert names[2].endswith("up_proj.lora_A.weight")
    assert names[3].endswith("up_proj.lora_B.weight")

    torch.testing.assert_close(weights[0].weight, torch.ones(2, 2))
    torch.testing.assert_close(weights[1].weight, torch.full((2, 2), 1.0))
    torch.testing.assert_close(weights[2].weight, torch.ones(2, 2))
    torch.testing.assert_close(weights[3].weight, torch.full((2, 2), 2.0))


def test_stream_adapter_weights_megatron_to_hf_fused_fc1_minimax_w13(monkeypatch):
    bridge = DummyBridge()

    adapter_task = AdapterWeightConversionTask(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc1",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_task=WeightConversionTask(
            param_name="local_in",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_in.weight",
            mapping=Mock(),
        ),
        linear_out_task=WeightConversionTask(
            param_name="local_out",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_out.weight",
            mapping=Mock(),
        ),
    )

    linear_out = torch.cat([torch.full((2, 2), 1.0), torch.full((2, 2), 2.0)], dim=0)
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc1",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_weight=MegatronWeightTuple("local_in", torch.ones(2, 2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("local_out", linear_out, vp_stage=0),
    )

    monkeypatch.setattr(
        bridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {"decoder.layers.0.mlp.experts.linear_fc1": [adapter_task]},
    )
    monkeypatch.setattr(bridge, "materialize_adapter_weights", lambda *_: [adapter_weight])
    monkeypatch.setattr(
        bridge,
        "_get_base_hf_param_names_for_adapter",
        lambda *_: [
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.0.w3.weight",
        ],
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_tensor_parallel_world_size",
        lambda: 1,
    )

    weights = list(
        bridge.stream_adapter_weights_megatron_to_hf(
            [SimpleNamespace(config=SimpleNamespace(num_moe_experts=1))],
            cpu=False,
            show_progress=False,
        )
    )

    assert len(weights) == 4
    names = [w.param_name for w in weights]
    assert names[0].endswith("w1.lora_A.weight")
    assert names[1].endswith("w1.lora_B.weight")
    assert names[2].endswith("w3.lora_A.weight")
    assert names[3].endswith("w3.lora_B.weight")

    torch.testing.assert_close(weights[0].weight, torch.ones(2, 2))
    torch.testing.assert_close(weights[1].weight, torch.full((2, 2), 1.0))
    torch.testing.assert_close(weights[2].weight, torch.ones(2, 2))
    torch.testing.assert_close(weights[3].weight, torch.full((2, 2), 2.0))


def test_stream_adapter_weights_megatron_to_hf_packed_expert_stacks(monkeypatch):
    bridge = DummyBridge()

    adapter_task = AdapterWeightConversionTask(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_task=WeightConversionTask(
            param_name="local_in",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_in.weight",
            mapping=Mock(),
        ),
        linear_out_task=WeightConversionTask(
            param_name="local_out",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_out.weight",
            mapping=Mock(),
        ),
    )

    linear_in = torch.stack(
        [torch.full((2, 2), 1.0), torch.full((2, 2), 2.0)],
        dim=0,
    )
    linear_out = torch.stack(
        [torch.full((2, 2), 3.0), torch.full((2, 2), 4.0)],
        dim=0,
    )
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_weight=MegatronWeightTuple("local_in", linear_in, vp_stage=0),
        linear_out_weight=MegatronWeightTuple("local_out", linear_out, vp_stage=0),
    )

    monkeypatch.setattr(
        bridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {"decoder.layers.0.mlp.experts.linear_fc2": [adapter_task]},
    )
    monkeypatch.setattr(bridge, "materialize_adapter_weights", lambda *_: [adapter_weight])
    monkeypatch.setattr(
        bridge,
        "_get_base_hf_param_names_for_adapter",
        lambda *_args, **_kwargs: ["model.layers.0.mlp.experts.down_proj"],
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_model_parallel_world_size",
        lambda: 1,
    )

    weights = list(
        bridge.stream_adapter_weights_megatron_to_hf(
            [SimpleNamespace(config=SimpleNamespace(num_moe_experts=2))],
            cpu=False,
            show_progress=False,
        )
    )

    assert len(weights) == 2
    assert weights[0].param_name.endswith("down_proj.lora_A.weight")
    assert weights[1].param_name.endswith("down_proj.lora_B.weight")
    torch.testing.assert_close(weights[0].weight, linear_in)
    torch.testing.assert_close(weights[1].weight, linear_out)


def test_stream_adapter_weights_megatron_to_hf_grouped_expert_exports_per_expert_names(monkeypatch):
    bridge = DummyBridge()

    adapter_task = AdapterWeightConversionTask(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_task=WeightConversionTask(
            param_name="local_in",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_in.weight",
            mapping=Mock(),
        ),
        linear_out_task=WeightConversionTask(
            param_name="local_out",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_out.weight",
            mapping=Mock(),
        ),
    )

    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_weight=MegatronWeightTuple("local_in", torch.ones(2, 2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("local_out", 3 * torch.ones(2, 2), vp_stage=0),
    )

    monkeypatch.setattr(
        bridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {"decoder.layers.0.mlp.experts.linear_fc2": [adapter_task]},
    )
    monkeypatch.setattr(bridge, "materialize_adapter_weights", lambda *_: [adapter_weight])
    monkeypatch.setattr(
        bridge,
        "_get_base_hf_param_names_for_adapter",
        lambda *_args, **_kwargs: ["model.layers.0.mlp.experts.0.down_proj.weight"],
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_model_parallel_world_size",
        lambda: 1,
    )

    weights = list(
        bridge.stream_adapter_weights_megatron_to_hf(
            [SimpleNamespace(config=SimpleNamespace(num_moe_experts=1))],
            cpu=False,
            show_progress=False,
        )
    )

    assert len(weights) == 2
    assert weights[0].param_name.endswith("down_proj.lora_A.weight")
    assert weights[1].param_name.endswith("down_proj.lora_B.weight")
    assert weights[0].param_name == "model.layers.0.mlp.experts.0.down_proj.lora_A.weight"
    assert weights[1].param_name == "model.layers.0.mlp.experts.0.down_proj.lora_B.weight"


def test_stream_adapter_weights_megatron_to_hf_shared_outer_fc1_gate_up(monkeypatch):
    # Shared-outer FC1 (SGLang PR #21466): linear_in (lora_A) is a single 2D matrix
    # shared across experts, while linear_out (lora_B) is a per-expert 3D pack of the
    # fused gate/up projection. The shared side is emitted once under an
    # expert-agnostic name; the per-expert side is split into gate/up per expert.
    bridge = DummyBridge()

    adapter_task = AdapterWeightConversionTask(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc1",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_task=WeightConversionTask(
            param_name="local_in",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_in.weight",
            mapping=Mock(),
        ),
        linear_out_task=WeightConversionTask(
            param_name="local_out",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_out.weight",
            mapping=Mock(),
        ),
    )

    # Shared lora_A: [rank=2, hidden=3]. Per-expert lora_B: [num_experts=2, 2*inter=4, rank=2],
    # gate = first 2 rows, up = last 2 rows, with distinct values per expert/projection.
    shared_lora_a = torch.ones(2, 3)
    expert0_lora_b = torch.cat([torch.full((2, 2), 10.0), torch.full((2, 2), 20.0)], dim=0)
    expert1_lora_b = torch.cat([torch.full((2, 2), 30.0), torch.full((2, 2), 40.0)], dim=0)
    per_expert_lora_b = torch.stack([expert0_lora_b, expert1_lora_b], dim=0)

    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc1",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_weight=MegatronWeightTuple("local_in", shared_lora_a, vp_stage=0),
        linear_out_weight=MegatronWeightTuple("local_out", per_expert_lora_b, vp_stage=0),
    )

    def fake_base_names(_registry, _prefix, _adapter_key, base_suffix):
        # base_suffix is ".weight0"/".weight1"/...; reflect the expert index into the
        # HF names so the per-expert side keeps experts.<idx> and the shared side strips it.
        idx = base_suffix[len(".weight") :]
        return [
            f"model.layers.0.mlp.experts.{idx}.gate_proj.weight",
            f"model.layers.0.mlp.experts.{idx}.up_proj.weight",
        ]

    monkeypatch.setattr(
        bridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {"decoder.layers.0.mlp.experts.linear_fc1": [adapter_task]},
    )
    monkeypatch.setattr(bridge, "materialize_adapter_weights", lambda *_: [adapter_weight])
    monkeypatch.setattr(bridge, "_get_base_hf_param_names_for_adapter", fake_base_names)
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_model_parallel_world_size",
        lambda: 1,
    )

    weights = list(
        bridge.stream_adapter_weights_megatron_to_hf(
            [SimpleNamespace(config=SimpleNamespace(num_moe_experts=2))],
            cpu=False,
            show_progress=False,
        )
    )

    # Shared lora_A emitted once per fused projection under the expert-agnostic name,
    # then per-expert lora_B split into gate/up for each expert.
    assert [w.param_name for w in weights] == [
        "model.layers.0.mlp.experts.gate_proj.lora_A.weight",
        "model.layers.0.mlp.experts.up_proj.lora_A.weight",
        "model.layers.0.mlp.experts.0.gate_proj.lora_B.weight",
        "model.layers.0.mlp.experts.0.up_proj.lora_B.weight",
        "model.layers.0.mlp.experts.1.gate_proj.lora_B.weight",
        "model.layers.0.mlp.experts.1.up_proj.lora_B.weight",
    ]

    # Shared side is unsqueezed to [1, ...] and identical for gate and up.
    assert weights[0].weight.shape == (1, 2, 3)
    torch.testing.assert_close(weights[0].weight, shared_lora_a.unsqueeze(0))
    torch.testing.assert_close(weights[1].weight, shared_lora_a.unsqueeze(0))
    # Per-expert side carries each expert's gate/up halves.
    torch.testing.assert_close(weights[2].weight, torch.full((2, 2), 10.0))
    torch.testing.assert_close(weights[3].weight, torch.full((2, 2), 20.0))
    torch.testing.assert_close(weights[4].weight, torch.full((2, 2), 30.0))
    torch.testing.assert_close(weights[5].weight, torch.full((2, 2), 40.0))


def test_stream_adapter_weights_megatron_to_hf_shared_outer_fc2_down(monkeypatch):
    # Shared-outer FC2 is the mirror of FC1: linear_in (lora_A) is the per-expert 3D
    # pack and linear_out (lora_B) is the shared 2D matrix. down_proj is not fused, so
    # the per-expert side emits one weight per expert and the shared side emits once.
    bridge = DummyBridge()

    adapter_task = AdapterWeightConversionTask(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_task=WeightConversionTask(
            param_name="local_in",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_in.weight",
            mapping=Mock(),
        ),
        linear_out_task=WeightConversionTask(
            param_name="local_out",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_out.weight",
            mapping=Mock(),
        ),
    )

    # Per-expert lora_A: [num_experts=2, rank=2, inter=3]. Shared lora_B: [hidden=3, rank=2].
    per_expert_lora_a = torch.stack([torch.full((2, 3), 5.0), torch.full((2, 3), 6.0)], dim=0)
    shared_lora_b = torch.full((3, 2), 7.0)

    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_weight=MegatronWeightTuple("local_in", per_expert_lora_a, vp_stage=0),
        linear_out_weight=MegatronWeightTuple("local_out", shared_lora_b, vp_stage=0),
    )

    def fake_base_names(_registry, _prefix, _adapter_key, base_suffix):
        idx = base_suffix[len(".weight") :]
        return [f"model.layers.0.mlp.experts.{idx}.down_proj.weight"]

    monkeypatch.setattr(
        bridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {"decoder.layers.0.mlp.experts.linear_fc2": [adapter_task]},
    )
    monkeypatch.setattr(bridge, "materialize_adapter_weights", lambda *_: [adapter_weight])
    monkeypatch.setattr(bridge, "_get_base_hf_param_names_for_adapter", fake_base_names)
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_model_parallel_world_size",
        lambda: 1,
    )

    weights = list(
        bridge.stream_adapter_weights_megatron_to_hf(
            [SimpleNamespace(config=SimpleNamespace(num_moe_experts=2))],
            cpu=False,
            show_progress=False,
        )
    )

    # Per-expert lora_A emitted once per expert (experts.<idx>), then the shared lora_B
    # emitted once under the expert-agnostic name.
    assert [w.param_name for w in weights] == [
        "model.layers.0.mlp.experts.0.down_proj.lora_A.weight",
        "model.layers.0.mlp.experts.1.down_proj.lora_A.weight",
        "model.layers.0.mlp.experts.down_proj.lora_B.weight",
    ]

    torch.testing.assert_close(weights[0].weight, torch.full((2, 3), 5.0))
    torch.testing.assert_close(weights[1].weight, torch.full((2, 3), 6.0))
    # Shared side is unsqueezed to [1, hidden, rank].
    assert weights[2].weight.shape == (1, 3, 2)
    torch.testing.assert_close(weights[2].weight, shared_lora_b.unsqueeze(0))


def test_split_gdn_in_proj_linear_out_weight_roundtrip(monkeypatch):
    bridge = DummyBridge()
    config = SimpleNamespace(
        hidden_size=4,
        linear_key_head_dim=1,
        linear_num_key_heads=2,
        linear_value_head_dim=1,
        linear_num_value_heads=2,
    )

    qkv = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    z = torch.full((2, 4), 100.0)
    b = torch.full((2, 4), 200.0)
    a = torch.full((2, 4), 300.0)

    qkvz, ba = _fuse_gdn_separate_to_grouped(config, qkv, z, b, a)
    linear_out = merge_gdn_linear_weights(config, qkvz, ba, tp_size=1)

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    split = bridge._split_gdn_in_proj_linear_out_weight(
        [SimpleNamespace(config=config)],
        linear_out,
    )

    torch.testing.assert_close(split["in_proj_qkv"], qkv)
    torch.testing.assert_close(split["in_proj_z"], z)
    torch.testing.assert_close(split["in_proj_b"], b)
    torch.testing.assert_close(split["in_proj_a"], a)


def test_get_fused_adapter_linear_out_slices_gdn_mapping(monkeypatch):
    bridge = DummyBridge()
    base_hf_weight_names = [
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
        "model.layers.0.linear_attn.in_proj_b.weight",
        "model.layers.0.linear_attn.in_proj_a.weight",
    ]
    gdn_slices = {
        "in_proj_qkv": torch.full((6, 2), 1.0),
        "in_proj_z": torch.full((2, 2), 2.0),
        "in_proj_b": torch.full((2, 2), 3.0),
        "in_proj_a": torch.full((2, 2), 4.0),
    }

    monkeypatch.setattr(
        bridge,
        "_split_gdn_in_proj_linear_out_weight",
        lambda *_args, **_kwargs: gdn_slices,
    )

    per_base = bridge._get_fused_adapter_linear_out_slices(
        [SimpleNamespace(config=SimpleNamespace())],
        base_hf_weight_names,
        torch.zeros(1, 1),
        is_expert=False,
    )

    assert per_base is not None
    torch.testing.assert_close(per_base["model.layers.0.linear_attn.in_proj_qkv.weight"], gdn_slices["in_proj_qkv"])
    torch.testing.assert_close(per_base["model.layers.0.linear_attn.in_proj_z.weight"], gdn_slices["in_proj_z"])
    torch.testing.assert_close(per_base["model.layers.0.linear_attn.in_proj_b.weight"], gdn_slices["in_proj_b"])
    torch.testing.assert_close(per_base["model.layers.0.linear_attn.in_proj_a.weight"], gdn_slices["in_proj_a"])
    assert bridge._infer_gdn_in_proj_projection_from_name("foo.bar.in_proj_z.weight") == "in_proj_z"
    assert bridge._infer_gdn_in_proj_projection_from_name("foo.bar.unknown.weight") is None


def test_merge_lora_adapter_weights_gdn_in_proj_split(monkeypatch):
    bridge = DummyBridge()
    config = SimpleNamespace(
        hidden_size=4,
        linear_key_head_dim=1,
        linear_num_key_heads=2,
        linear_value_head_dim=1,
        linear_num_value_heads=2,
        num_moe_experts=0,
    )
    megatron_model = [SimpleNamespace(config=config)]

    qkv_delta = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    z_delta = torch.full((2, 4), 10.0)
    b_delta = torch.full((2, 4), 20.0)
    a_delta = torch.full((2, 4), 30.0)

    base_qkv = torch.full((6, 4), -1.0)
    base_z = torch.full((2, 4), -2.0)
    base_b = torch.full((2, 4), -3.0)
    base_a = torch.full((2, 4), -4.0)
    converted = {
        "model.layers.0.linear_attn.in_proj_qkv.weight": base_qkv.clone(),
        "model.layers.0.linear_attn.in_proj_z.weight": base_z.clone(),
        "model.layers.0.linear_attn.in_proj_b.weight": base_b.clone(),
        "model.layers.0.linear_attn.in_proj_a.weight": base_a.clone(),
    }

    qkvz, ba = _fuse_gdn_separate_to_grouped(config, qkv_delta, z_delta, b_delta, a_delta)
    linear_out = merge_gdn_linear_weights(config, qkvz, ba, tp_size=1)
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.self_attention.linear_qkv",
        adapter_key=None,
        alpha=4,
        dim=4,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(4), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", linear_out, vp_stage=0),
    )

    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    updated = bridge._merge_lora_adapter_weights(megatron_model, converted, [adapter_weight])

    torch.testing.assert_close(
        updated["model.layers.0.linear_attn.in_proj_qkv.weight"],
        base_qkv + qkv_delta,
    )
    torch.testing.assert_close(
        updated["model.layers.0.linear_attn.in_proj_z.weight"],
        base_z + z_delta,
    )
    torch.testing.assert_close(
        updated["model.layers.0.linear_attn.in_proj_b.weight"],
        base_b + b_delta,
    )
    torch.testing.assert_close(
        updated["model.layers.0.linear_attn.in_proj_a.weight"],
        base_a + a_delta,
    )


def _stream_weights_with_merge_disabled(monkeypatch, converted_name: str):
    bridge = DummyBridge()

    class DummyMapping:
        def megatron_to_hf(self, weight, module):
            return {converted_name: torch.ones(1)}

    task = WeightConversionTask(
        param_name="decoder.layers.0.mlp.linear_fc1.to_wrap.weight",
        global_param_name="decoder.layers.0.mlp.linear_fc1.to_wrap.weight",
        mapping=DummyMapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=None,
        param_weight=torch.ones(1),
    )

    monkeypatch.setattr(
        DummyBridge,
        "_with_progress_tracking",
        lambda self, tasks, *_args, **_kwargs: tasks,
    )
    monkeypatch.setattr(
        DummyBridge,
        "maybe_modify_converted_hf_weight",
        lambda self, *_args, **_kwargs: _args[1],
    )
    monkeypatch.setattr(
        DummyBridge,
        "_share_embeddings_and_output_weights",
        lambda self, *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.unwrap_model",
        lambda *_args, **_kwargs: [SimpleNamespace(config=SimpleNamespace(pipeline_model_parallel_size=1))],
    )

    def _raise_on_build(*_args, **_kwargs):
        raise AssertionError("Adapter tasks should not be built when merge_adapter_weights=False")

    monkeypatch.setattr(DummyBridge, "build_adapter_conversion_tasks", _raise_on_build)

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=False,
            show_progress=False,
            conversion_tasks=[task],
            merge_adapter_weights=False,
        )
    )

    return weights


def _patch_stream_weights_megatron_to_hf_basics(monkeypatch, *, num_moe_experts: int = 0):
    monkeypatch.setattr(
        DummyBridge,
        "_with_progress_tracking",
        lambda self, tasks, *_args, **_kwargs: tasks,
    )
    monkeypatch.setattr(
        DummyBridge,
        "_share_embeddings_and_output_weights",
        lambda self, *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.unwrap_model",
        lambda *_args, **_kwargs: [
            SimpleNamespace(config=SimpleNamespace(num_moe_experts=num_moe_experts, pipeline_model_parallel_size=1))
        ],
    )


@pytest.mark.parametrize("cpu", [False, True])
def test_stream_weights_megatron_to_hf_detaches_standard_export_tensors(monkeypatch, cpu):
    bridge = DummyBridge()
    source_tensor = torch.ones(1, requires_grad=True)

    class DummyMapping:
        def megatron_to_hf(self, weight, module):
            return {"hf.weight": weight}

    task = WeightConversionTask(
        param_name="decoder.layers.0.mlp.linear_fc1.weight",
        global_param_name="decoder.layers.0.mlp.linear_fc1.weight",
        mapping=DummyMapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=None,
        param_weight=source_tensor,
    )

    _patch_stream_weights_megatron_to_hf_basics(monkeypatch)
    monkeypatch.setattr(
        DummyBridge,
        "maybe_modify_converted_hf_weight",
        lambda self, *_args, **_kwargs: _args[1],
    )

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=cpu,
            show_progress=False,
            conversion_tasks=[task],
            merge_adapter_weights=False,
        )
    )

    assert len(weights) == 1
    assert weights[0].param_name == "hf.weight"
    assert weights[0].weight.requires_grad is False
    assert weights[0].weight.grad_fn is None


@pytest.mark.parametrize("cpu", [False, True])
def test_stream_weights_megatron_to_hf_detaches_grouped_export_tensors(monkeypatch, cpu):
    bridge = DummyBridge()
    grouped_tensor = torch.ones(1, 1, requires_grad=True)

    class GroupedMapping:
        is_grouped_export = True
        group_key = "hf.grouped"

        def megatron_to_hf(self, weight, module):
            return {self.group_key: weight}

    task = WeightConversionTask(
        param_name="decoder.layers.0.mlp.experts.linear_fc2.weight0",
        global_param_name="decoder.layers.0.mlp.experts.linear_fc2.weight0",
        mapping=GroupedMapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=None,
        param_weight=grouped_tensor,
    )

    _patch_stream_weights_megatron_to_hf_basics(monkeypatch, num_moe_experts=1)
    monkeypatch.setattr(
        DummyBridge,
        "_accumulate_grouped_export",
        lambda *_args, **_kwargs: {"hf.grouped": grouped_tensor},
    )

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=cpu,
            show_progress=False,
            conversion_tasks=[task],
            merge_adapter_weights=False,
        )
    )

    assert len(weights) == 1
    assert weights[0].param_name == "hf.grouped"
    assert weights[0].weight.requires_grad is False
    assert weights[0].weight.grad_fn is None


@pytest.mark.parametrize(
    ("converted_name", "expected_name"),
    [
        ("hf.weight", "hf.base_layer.weight"),
        ("hf.tensor", "hf.base_layer.tensor"),
        ("hf", "hf.base_layer"),
    ],
)
def test_stream_weights_megatron_to_hf_skips_merge_when_disabled(monkeypatch, converted_name, expected_name):
    weights = _stream_weights_with_merge_disabled(monkeypatch, converted_name)

    assert len(weights) == 1
    assert weights[0].param_name == expected_name


def test_stream_weights_megatron_to_hf_merges_grouped_expert_adapters(monkeypatch):
    bridge = DummyBridge()

    class GroupedMapping:
        is_grouped_export = True
        group_key = "hf.experts.down_proj"
        ep_size = 1

        def megatron_to_hf(self, weight, module):
            return {self.group_key: torch.zeros(2, 2)}

    task = WeightConversionTask(
        param_name="decoder.layers.0.mlp.experts.linear_fc2.to_wrap.weight0",
        global_param_name="decoder.layers.0.mlp.experts.linear_fc2.to_wrap.weight0",
        mapping=GroupedMapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=None,
        param_weight=torch.ones(1),
    )

    adapter_task = AdapterWeightConversionTask(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
        adapter_key=None,
        alpha=2,
        dim=2,
        linear_in_task=WeightConversionTask(
            param_name="local_in",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_in.weight",
            mapping=Mock(),
        ),
        linear_out_task=WeightConversionTask(
            param_name="local_out",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_out.weight",
            mapping=Mock(),
        ),
    )
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
        adapter_key=None,
        alpha=2,
        dim=2,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", torch.eye(2), vp_stage=0),
    )

    monkeypatch.setattr(
        DummyBridge,
        "_with_progress_tracking",
        lambda self, tasks, *_args, **_kwargs: tasks,
    )
    monkeypatch.setattr(
        DummyBridge,
        "_share_embeddings_and_output_weights",
        lambda self, *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        DummyBridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {"decoder.layers.0.mlp.experts.linear_fc2": [adapter_task]},
    )
    monkeypatch.setattr(DummyBridge, "materialize_adapter_weights", lambda *_args, **_kwargs: [adapter_weight])
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.unwrap_model",
        lambda *_args, **_kwargs: [SimpleNamespace(config=SimpleNamespace(num_moe_experts=1))],
    )

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=False,
            show_progress=False,
            conversion_tasks=[task],
            merge_adapter_weights=True,
        )
    )

    assert len(weights) == 1
    assert weights[0].param_name == "hf.experts.down_proj"
    torch.testing.assert_close(weights[0].weight, torch.eye(2).unsqueeze(0))


def test_stream_weights_megatron_to_hf_merges_grouped_expert_adapters_before_transpose(monkeypatch):
    bridge = DummyBridge()

    class GroupedMapping:
        is_grouped_export = True
        group_key = "hf.experts.down_proj"
        ep_size = 1
        transpose_on_export = True

        def megatron_to_hf(self, weight, module):
            return {self.group_key: torch.zeros(2, 3)}

    task = WeightConversionTask(
        param_name="decoder.layers.0.mlp.experts.linear_fc2.to_wrap.weight0",
        global_param_name="decoder.layers.0.mlp.experts.linear_fc2.to_wrap.weight0",
        mapping=GroupedMapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=None,
        param_weight=torch.ones(1),
    )

    adapter_task = AdapterWeightConversionTask(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
        adapter_key=None,
        alpha=1,
        dim=1,
        linear_in_task=WeightConversionTask(
            param_name="local_in",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_in.weight",
            mapping=Mock(),
        ),
        linear_out_task=WeightConversionTask(
            param_name="local_out",
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_out.weight",
            mapping=Mock(),
        ),
    )
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
        adapter_key=None,
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in", torch.tensor([[3.0, 4.0, 5.0]]), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", torch.tensor([[1.0], [2.0]]), vp_stage=0),
    )

    monkeypatch.setattr(
        DummyBridge,
        "_with_progress_tracking",
        lambda self, tasks, *_args, **_kwargs: tasks,
    )
    monkeypatch.setattr(
        DummyBridge,
        "_share_embeddings_and_output_weights",
        lambda self, *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        DummyBridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {"decoder.layers.0.mlp.experts.linear_fc2": [adapter_task]},
    )
    monkeypatch.setattr(DummyBridge, "materialize_adapter_weights", lambda *_args, **_kwargs: [adapter_weight])
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.unwrap_model",
        lambda *_args, **_kwargs: [SimpleNamespace(config=SimpleNamespace(num_moe_experts=1))],
    )

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=False,
            show_progress=False,
            conversion_tasks=[task],
            merge_adapter_weights=True,
        )
    )

    expected = torch.tensor([[[3.0, 6.0], [4.0, 8.0], [5.0, 10.0]]])

    assert len(weights) == 1
    assert weights[0].param_name == "hf.experts.down_proj"
    torch.testing.assert_close(weights[0].weight, expected)


def test_merge_grouped_export_adapter_weights_uses_global_expert_idx(monkeypatch):
    bridge = DummyBridge()

    task = WeightConversionTask(
        param_name="decoder.layers.0.mlp.experts.linear_fc2.to_wrap.weight0",
        global_param_name="decoder.layers.0.mlp.experts.linear_fc2.to_wrap.weight1",
        mapping=Mock(),
    )

    converted = {"hf.experts.down_proj": torch.zeros(1, 1)}
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc2",
        adapter_key=None,
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in", torch.ones(1, 1), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", torch.ones(1, 1), vp_stage=0),
    )

    monkeypatch.setattr(DummyBridge, "_gather_expert_adapter_weight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        DummyBridge,
        "_select_expert_adapter_weight",
        lambda self, weight, gathered, expert_idx, num_moe_experts: torch.full_like(weight, float(expert_idx)),
    )
    monkeypatch.setattr(
        DummyBridge,
        "_merge_single_adapter_weight",
        lambda self, base_weight, alpha, dim, linear_in_weight, linear_out_weight: linear_out_weight,
    )

    updated = bridge._merge_grouped_export_adapter_weights(
        task,
        converted,
        [adapter_weight],
        num_moe_experts=2,
    )

    torch.testing.assert_close(updated["hf.experts.down_proj"], torch.tensor([[1.0]]))


def test_merge_grouped_export_adapter_weights_raises_for_canonical_adapters():
    bridge = DummyBridge()

    task = WeightConversionTask(
        param_name="decoder.layers.0.mlp.experts.linear_fc1.to_wrap.weight0",
        global_param_name="decoder.layers.0.mlp.experts.linear_fc1.to_wrap.weight0",
        mapping=Mock(),
    )
    converted = {"hf.experts.gate_up_proj": torch.zeros(2, 2)}
    adapter_weight = AdapterWeight(
        global_base_prefix="decoder.layers.0.mlp.experts.linear_fc1",
        adapter_key="adapter_gate",
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", torch.eye(2), vp_stage=0),
    )

    with pytest.raises(ValueError, match="Unsupported adapter configuration for grouped export weight merging"):
        bridge._merge_grouped_export_adapter_weights(
            task,
            converted,
            [adapter_weight],
            num_moe_experts=1,
        )


def test_stream_weights_megatron_to_hf_merges_shared_expert_fc1_adapters(monkeypatch):
    bridge = DummyBridge()

    class SharedExpertFc1Mapping:
        def megatron_to_hf(self, weight, module):
            return {
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": torch.zeros(2, 2),
                "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": torch.zeros(2, 2),
            }

    task = WeightConversionTask(
        param_name="language_model.decoder.layers.0.mlp.shared_experts.linear_fc1.to_wrap.weight",
        global_param_name="language_model.decoder.layers.0.mlp.shared_experts.linear_fc1.to_wrap.weight",
        mapping=SharedExpertFc1Mapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=None,
        param_weight=torch.ones(1),
    )

    adapter_task = AdapterWeightConversionTask(
        global_base_prefix="language_model.decoder.layers.0.mlp.shared_experts.linear_fc1",
        adapter_key=None,
        alpha=1,
        dim=1,
        linear_in_task=WeightConversionTask(
            param_name="local_in",
            global_param_name="language_model.decoder.layers.0.mlp.shared_experts.linear_fc1.adapter.linear_in.weight",
            mapping=Mock(),
        ),
        linear_out_task=WeightConversionTask(
            param_name="local_out",
            global_param_name="language_model.decoder.layers.0.mlp.shared_experts.linear_fc1.adapter.linear_out.weight",
            mapping=Mock(),
        ),
    )
    adapter_weight = AdapterWeight(
        global_base_prefix="language_model.decoder.layers.0.mlp.shared_experts.linear_fc1",
        adapter_key=None,
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", torch.cat([torch.eye(2), 2 * torch.eye(2)], dim=0), vp_stage=0),
    )

    monkeypatch.setattr(
        DummyBridge,
        "_with_progress_tracking",
        lambda self, tasks, *_args, **_kwargs: tasks,
    )
    monkeypatch.setattr(
        DummyBridge,
        "_share_embeddings_and_output_weights",
        lambda self, *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        DummyBridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {"language_model.decoder.layers.0.mlp.shared_experts.linear_fc1": [adapter_task]},
    )
    monkeypatch.setattr(DummyBridge, "materialize_adapter_weights", lambda *_args, **_kwargs: [adapter_weight])
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.unwrap_model",
        lambda *_args, **_kwargs: [SimpleNamespace(config=SimpleNamespace(num_moe_experts=1))],
    )

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=False,
            show_progress=False,
            conversion_tasks=[task],
            merge_adapter_weights=True,
        )
    )

    assert len(weights) == 2
    assert weights[0].param_name == "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"
    torch.testing.assert_close(weights[0].weight, torch.eye(2))
    assert weights[1].param_name == "model.language_model.layers.0.mlp.shared_expert.up_proj.weight"
    torch.testing.assert_close(weights[1].weight, 2 * torch.eye(2))


def test_stream_weights_megatron_to_hf_merges_shared_expert_fc2_adapters(monkeypatch):
    bridge = DummyBridge()

    class SharedExpertFc2Mapping:
        def megatron_to_hf(self, weight, module):
            return {"model.language_model.layers.0.mlp.shared_expert.down_proj.weight": torch.zeros(2, 2)}

    task = WeightConversionTask(
        param_name="language_model.decoder.layers.0.mlp.shared_experts.linear_fc2.to_wrap.weight",
        global_param_name="language_model.decoder.layers.0.mlp.shared_experts.linear_fc2.to_wrap.weight",
        mapping=SharedExpertFc2Mapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=None,
        param_weight=torch.ones(1),
    )

    adapter_task = AdapterWeightConversionTask(
        global_base_prefix="language_model.decoder.layers.0.mlp.shared_experts.linear_fc2",
        adapter_key=None,
        alpha=1,
        dim=1,
        linear_in_task=WeightConversionTask(
            param_name="local_in",
            global_param_name="language_model.decoder.layers.0.mlp.shared_experts.linear_fc2.adapter.linear_in.weight",
            mapping=Mock(),
        ),
        linear_out_task=WeightConversionTask(
            param_name="local_out",
            global_param_name="language_model.decoder.layers.0.mlp.shared_experts.linear_fc2.adapter.linear_out.weight",
            mapping=Mock(),
        ),
    )
    adapter_weight = AdapterWeight(
        global_base_prefix="language_model.decoder.layers.0.mlp.shared_experts.linear_fc2",
        adapter_key=None,
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in", torch.eye(2), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", 3 * torch.eye(2), vp_stage=0),
    )

    monkeypatch.setattr(
        DummyBridge,
        "_with_progress_tracking",
        lambda self, tasks, *_args, **_kwargs: tasks,
    )
    monkeypatch.setattr(
        DummyBridge,
        "_share_embeddings_and_output_weights",
        lambda self, *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        DummyBridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {"language_model.decoder.layers.0.mlp.shared_experts.linear_fc2": [adapter_task]},
    )
    monkeypatch.setattr(DummyBridge, "materialize_adapter_weights", lambda *_args, **_kwargs: [adapter_weight])
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.unwrap_model",
        lambda *_args, **_kwargs: [SimpleNamespace(config=SimpleNamespace(num_moe_experts=1))],
    )

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=False,
            show_progress=False,
            conversion_tasks=[task],
            merge_adapter_weights=True,
        )
    )

    assert len(weights) == 1
    assert weights[0].param_name == "model.language_model.layers.0.mlp.shared_expert.down_proj.weight"
    torch.testing.assert_close(weights[0].weight, 3 * torch.eye(2))


def test_stream_weights_megatron_to_hf_merges_router_adapters(monkeypatch):
    bridge = DummyBridge()

    class RouterMapping:
        def megatron_to_hf(self, weight, module):
            return {"model.language_model.layers.0.mlp.gate.weight": torch.zeros(2, 3)}

    task = WeightConversionTask(
        param_name="language_model.decoder.layers.0.mlp.router.to_wrap.weight",
        global_param_name="language_model.decoder.layers.0.mlp.router.to_wrap.weight",
        mapping=RouterMapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=None,
        param_weight=torch.ones(1),
    )

    adapter_task = AdapterWeightConversionTask(
        global_base_prefix="language_model.decoder.layers.0.mlp.router",
        adapter_key=None,
        alpha=1,
        dim=1,
        linear_in_task=WeightConversionTask(
            param_name="local_in",
            global_param_name="language_model.decoder.layers.0.mlp.router.adapter.linear_in.weight",
            mapping=Mock(),
        ),
        linear_out_task=WeightConversionTask(
            param_name="local_out",
            global_param_name="language_model.decoder.layers.0.mlp.router.adapter.linear_out.weight",
            mapping=Mock(),
        ),
    )
    adapter_weight = AdapterWeight(
        global_base_prefix="language_model.decoder.layers.0.mlp.router",
        adapter_key=None,
        alpha=1,
        dim=1,
        linear_in_weight=MegatronWeightTuple("in", torch.tensor([[1.0, 2.0, 3.0]]), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", torch.tensor([[1.0], [2.0]]), vp_stage=0),
    )

    monkeypatch.setattr(
        DummyBridge,
        "_with_progress_tracking",
        lambda self, tasks, *_args, **_kwargs: tasks,
    )
    monkeypatch.setattr(
        DummyBridge,
        "_share_embeddings_and_output_weights",
        lambda self, *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        DummyBridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {"language_model.decoder.layers.0.mlp.router": [adapter_task]},
    )
    monkeypatch.setattr(DummyBridge, "materialize_adapter_weights", lambda *_args, **_kwargs: [adapter_weight])
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.unwrap_model",
        lambda *_args, **_kwargs: [SimpleNamespace(config=SimpleNamespace(num_moe_experts=1))],
    )

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=False,
            show_progress=False,
            conversion_tasks=[task],
            merge_adapter_weights=True,
        )
    )

    expected = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])

    assert len(weights) == 1
    assert weights[0].param_name == "model.language_model.layers.0.mlp.gate.weight"
    torch.testing.assert_close(weights[0].weight, expected)


def test_stream_weights_megatron_to_hf_merges_lora_before_quantization(monkeypatch):
    """LoRA delta must be merged into the bf16 base *before* ``maybe_modify`` quantizes.

    If quantization runs first, the packed weight's shape no longer matches the LoRA
    delta and the merge is silently skipped, so vLLM receives un-merged weights.
    This test installs a ``maybe_modify`` that "quantizes" by recording the shape it
    received and padding a sentinel column, then asserts the recorded shape is the
    merged (base + delta) shape, not the raw base shape.
    """
    bridge = DummyBridge()

    base_weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    class QuantizingMapping:
        def megatron_to_hf(self, weight, module):
            return {"model.layers.0.mlp.fc1.weight": base_weight.clone()}

    task = WeightConversionTask(
        param_name="language_model.decoder.layers.0.mlp.fc1.to_wrap.weight",
        global_param_name="language_model.decoder.layers.0.mlp.fc1.to_wrap.weight",
        mapping=QuantizingMapping(),
        pp_rank=0,
        vp_stage=0,
        megatron_module=None,
        param_weight=torch.ones(1),
    )

    adapter_task = AdapterWeightConversionTask(
        global_base_prefix="language_model.decoder.layers.0.mlp.fc1",
        adapter_key=None,
        alpha=2,
        dim=2,
        linear_in_task=WeightConversionTask(
            param_name="local_in",
            global_param_name="language_model.decoder.layers.0.mlp.fc1.adapter.linear_in.weight",
            mapping=Mock(),
        ),
        linear_out_task=WeightConversionTask(
            param_name="local_out",
            global_param_name="language_model.decoder.layers.0.mlp.fc1.adapter.linear_out.weight",
            mapping=Mock(),
        ),
    )
    # delta = (alpha/dim=1) * linear_out @ linear_in = eye(2) @ in = in
    # merged = base + in = [[2,4,6],[8,10,12]]
    adapter_weight = AdapterWeight(
        global_base_prefix="language_model.decoder.layers.0.mlp.fc1",
        adapter_key=None,
        alpha=2,
        dim=2,
        linear_in_weight=MegatronWeightTuple("in", torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", torch.eye(2), vp_stage=0),
    )

    # Record what maybe_modify receives: if merge ran first, the weight is the
    # merged tensor (base + delta); if quantize ran first, it's the raw base.
    seen_by_maybe_modify: dict[str, torch.Tensor] = {}

    def fake_maybe_modify(self, task, converted_weights_dict, hf_state_dict):
        for name, tensor in converted_weights_dict.items():
            seen_by_maybe_modify[name] = tensor.clone()
        # Simulate "quantization" by tagging the tensor so we can tell it passed through.
        return {name: t.clone() for name, t in converted_weights_dict.items()}

    monkeypatch.setattr(DummyBridge, "maybe_modify_converted_hf_weight", fake_maybe_modify)
    monkeypatch.setattr(
        DummyBridge,
        "_with_progress_tracking",
        lambda self, tasks, *_args, **_kwargs: tasks,
    )
    monkeypatch.setattr(
        DummyBridge,
        "_share_embeddings_and_output_weights",
        lambda self, *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        DummyBridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {"language_model.decoder.layers.0.mlp.fc1": [adapter_task]},
    )
    monkeypatch.setattr(DummyBridge, "materialize_adapter_weights", lambda *_args, **_kwargs: [adapter_weight])
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.unwrap_model",
        lambda *_args, **_kwargs: [SimpleNamespace(config=SimpleNamespace(num_moe_experts=1))],
    )

    weights = list(
        bridge.stream_weights_megatron_to_hf(
            [Mock()],
            SimpleNamespace(),
            cpu=False,
            show_progress=False,
            conversion_tasks=[task],
            merge_adapter_weights=True,
        )
    )

    # maybe_modify must have received the MERGED weight (base + delta), not the raw base.
    merged_expected = base_weight + torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    received = seen_by_maybe_modify["model.layers.0.mlp.fc1.weight"]
    torch.testing.assert_close(received, merged_expected)
    # And the final exported weight carries the merged values through the fake quantize.
    torch.testing.assert_close(weights[0].weight, merged_expected)


def test_column_parallel_mapping_skips_ep_gather_for_adapters(monkeypatch):
    mapping = ColumnParallelMapping(
        "decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_in.weight",
        "hf_param",
    )

    # Avoid distributed calls
    monkeypatch.setattr(ColumnParallelMapping, "broadcast_from_pp_rank", lambda self, tensor, cache_key=None: tensor)
    monkeypatch.setattr(ColumnParallelMapping, "gather_from_tp_ranks", lambda self, tensor: [tensor])
    monkeypatch.setattr(ColumnParallelMapping, "tp_size", property(lambda self: 1))

    def _raise(*args, **kwargs):
        raise AssertionError("gather_from_ep_ranks should not be called for adapters")

    monkeypatch.setattr(ColumnParallelMapping, "gather_from_ep_ranks", _raise)

    result = mapping.megatron_to_hf(torch.ones(2, 2), None)
    torch.testing.assert_close(result["hf_param"], torch.ones(2, 2))


def test_split_gate_up_adapter_emits_one_task_per_half(monkeypatch):
    """Split gate/up adapters yield ONE conversion task per enumerated pair.

    A SplitGateUpAdapter carries two independent pairs, and
    _megatron_global_adapters_info_all_pp_ranks enumerates each pair
    separately (the dedup key is that pair's ``linear_in.weight``). Each
    entry must therefore emit exactly ONE task, tagged with its half —
    emitting both halves per entry doubles every split surface (observed as
    4 tasks per fused fc1 and 312 duplicate PEFT names in the mint shard
    document).
    """
    from megatron.bridge.models.conversion.peft_bridge import is_split_gate_up_adapter
    from megatron.bridge.peft.utils import SplitGateUpAdapter

    bridge = DummyBridge()
    bridge.hf_pretrained = SimpleNamespace()
    bridge.hf_config = bridge.hf_pretrained

    adapter = SplitGateUpAdapter.__new__(SplitGateUpAdapter)
    torch.nn.Module.__init__(adapter)

    def _holder():
        return SimpleNamespace(weight=torch.ones(2, 2))

    adapter.linear_in, adapter.linear_out = _holder(), _holder()
    adapter.up = SimpleNamespace(linear_in=_holder(), linear_out=_holder())
    adapter.alpha, adapter.dim = 32, 8
    assert is_split_gate_up_adapter(adapter), "predicate must see the real class"

    # Two entries, exactly as the enumeration emits them: the gate pair
    # (no suffix) and the up pair (``.up`` suffix), both on the same base.
    adapters_info = [
        (
            "decoder.layers.0.mlp.linear_fc1.adapter",
            "decoder.layers.0.mlp.linear_fc1",
            False,
            False,
            False,
            32,
            8,
            0,
            0,
            True,
            False,
        ),
        (
            "decoder.layers.0.mlp.linear_fc1.adapter.up",
            "decoder.layers.0.mlp.linear_fc1",
            False,
            False,
            False,
            32,
            8,
            0,
            0,
            True,
            False,
        ),
    ]

    monkeypatch.setattr(bridge, "_megatron_global_adapters_info_all_pp_ranks", lambda *_: adapters_info)
    monkeypatch.setattr(bridge, "_get_adapter_wrap_module", lambda *_: (adapter, None))
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_rank",
        lambda: 0,
    )
    registry = MegatronMappingRegistry(
        AutoMapping(
            megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
            hf_param="decoder.layers.*.mlp.linear_fc1.weight",
        )
    )
    monkeypatch.setattr(bridge, "mapping_registry", lambda: registry)

    tasks = bridge.build_adapter_conversion_tasks([Mock()])[
        "decoder.layers.0.mlp.linear_fc1"
    ]
    assert len(tasks) == 2, "one task per pair (not per pair per entry)"
    by_half = {t.gate_up_half: t for t in tasks}
    assert sorted(by_half) == ["gate", "up"]
    # Each task must carry ITS pair's weights, not the other half's.
    assert by_half["up"].linear_in_task.param_weight is adapter.up.linear_in.weight
    assert by_half["gate"].linear_in_task.param_weight is adapter.linear_in.weight




def test_dequantize_fp8_blockwise_matches_the_ragged_loop():
    """The vectorized exact-block path must agree bit-for-bit with the
    reference loop (the ragged path is the same code the fast path replaced).
    Guards the GLM-5.3-Flash import: 37,338 tensors ride this function, and a
    silent numeric drift here would move every weight in the model."""
    import torch

    from megatron.bridge.models.conversion.quantization_utils import (
        dequantize_fp8_blockwise,
    )

    torch.manual_seed(0)

    def ragged_reference(weight, scale_inv, block_size=128, dtype=torch.bfloat16):
        M, N = weight.shape
        w = weight.float()
        out = torch.empty_like(w)
        sM, sN = scale_inv.shape
        for bi in range(sM):
            for bj in range(sN):
                r0, r1 = bi * block_size, min((bi + 1) * block_size, M)
                c0, c1 = bj * block_size, min((bj + 1) * block_size, N)
                out[r0:r1, c0:c1] = w[r0:r1, c0:c1] * scale_inv[bi, bj]
        return out.to(dtype)

    # Exact-block geometry (the checkpoint's): both dims multiples of 128.
    for (M, N) in [(256, 128), (512, 256), (4096, 2048)]:
        w = torch.randn(M, N).to(torch.float8_e4m3fn)
        s = torch.rand(M // 128, N // 128, dtype=torch.float32) * 0.5 + 0.1
        fast = dequantize_fp8_blockwise(w, s)
        ref = ragged_reference(w, s)
        assert fast.shape == ref.shape
        assert torch.equal(fast, ref), f"fast path diverged at {(M, N)}"

    # Ragged geometry must still work (loop path).
    w = torch.randn(300, 200).to(torch.float8_e4m3fn)
    s = torch.rand(3, 2, dtype=torch.float32) * 0.5 + 0.1
    out = dequantize_fp8_blockwise(w, s)
    assert out.shape == (300, 200)
    assert torch.equal(out, ragged_reference(w, s))


def test_shard_readahead_is_best_effort(tmp_path):
    """The readahead helper must never break a load: a missing file, a
    directory, or a platform without posix_fadvise all fall through silently.
    The win it buys on the real mount is 4.5-7.6x on cold shards (measured),
    and the load must not trade that for a new failure mode."""
    from megatron.bridge.models.hf_pretrained import state as state_mod

    # A real file: call must succeed and leave the file readable.
    f = tmp_path / "shard.safetensors"
    f.write_bytes(b"\0" * 4096)
    state_mod._issue_shard_readahead(f)
    assert f.read_bytes() == b"\0" * 4096

    # Missing path and a directory: silent no-ops, no exception.
    state_mod._issue_shard_readahead(tmp_path / "nope.safetensors")
    state_mod._issue_shard_readahead(tmp_path)

    # Platform without fadvise: monkeypatch it away and confirm the helper
    # still swallows the AttributeError.
    import os

    had = hasattr(os, "posix_fadvise")
    if had:
        orig = os.posix_fadvise
        del os.posix_fadvise
        try:
            state_mod._issue_shard_readahead(f)
        finally:
            os.posix_fadvise = orig

def _cpu_mp_config(tp_size: int = 1):
    """Model-parallel config that keeps adapter construction off the GPU.

    The unit suite also runs on CPU-only hosts, where MCore's column-parallel
    linears would otherwise call ``torch.cuda.current_device()``. Asking for CPU
    initialization keeps the same code path on both host kinds.
    """
    from megatron.core import ModelParallelConfig

    return ModelParallelConfig(
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=False,
        params_dtype=torch.float32,
    )


def _build_split_qkv_adapter(
    heads: int,
    groups,
    head_size,
    *,
    fused_out: int,
    in_features: int = 64,
    dim: int = 8,
    tp_size: int = 1,
):
    """Build a real SplitQKVAdapter for a fused linear_qkv of ``fused_out`` rows.

    Adapter construction must not depend on an initialized Megatron MPU
    (``ProcessGroupCollection.use_mpu_process_groups`` reads distributed state a
    unit test does not set up), so a collection whose process groups are all
    None is injected; the adapters then behave as unsharded (``tp_size == 1``).
    """
    from unittest.mock import patch

    from megatron.core.process_groups_config import ProcessGroupCollection

    from megatron.bridge.peft.utils import SplitQKVAdapter

    # A real collection object with every field left at its None default: the
    # adapter then reaches for no process group at all.
    stub = ProcessGroupCollection()
    with patch.object(ProcessGroupCollection, "use_mpu_process_groups", classmethod(lambda cls, **kwargs: stub)):
        return SplitQKVAdapter(
            in_features,
            fused_out,
            dim,
            qkv_head_layout=(heads, groups, head_size),
            base_linear_name="decoder.layers.0.self_attn.linear_qkv",
            activation="identity",
            model_parallel_config=_cpu_mp_config(tp_size),
            input_is_parallel=False,
            base_linear_is_parallel=True,
        )


def test_split_qkv_adapter_builds_three_independent_pairs():
    """GQA case: q = heads * head_size, k = v = groups * head_size, three pairs."""
    adapter = _build_split_qkv_adapter(8, 4, 64, fused_out=(8 + 2 * 4) * 64)

    assert adapter._qkv_head_layout == (8, 4, 64)
    assert adapter.linear_out.weight.shape[0] == 512  # q face = heads * head_size
    assert adapter.k.linear_out.weight.shape[0] == 256  # k face = groups * head_size
    assert adapter.v.linear_out.weight.shape[0] == 256
    # One shared rank-dim projection per face, each seeing every input feature.
    assert adapter.linear_in.weight.shape == (8, 64)
    assert adapter.k.linear_in.weight.shape == (8, 64)
    assert adapter.v.linear_in.weight.shape == (8, 64)

    pairs = adapter.pairs()
    assert len(pairs) == 3
    (q_in, q_out), (k_in, k_out), (v_in, v_out) = pairs
    assert q_in is adapter.linear_in and q_out is adapter.linear_out
    assert k_in is adapter.k.linear_in and k_out is adapter.k.linear_out
    assert v_in is adapter.v.linear_in and v_out is adapter.v.linear_out
    # Three pairs must not alias: six distinct modules.
    assert len({id(module) for pair in pairs for module in pair}) == 6


def test_split_qkv_adapter_full_mha_and_inferred_head_size():
    """``num_query_groups=None`` means full MHA; ``kv_channels=None`` infers head_size."""
    mha = _build_split_qkv_adapter(8, None, 64, fused_out=(8 + 2 * 8) * 64)
    assert mha._qkv_head_layout == (8, 8, 64)
    assert mha.linear_out.weight.shape[0] == 512
    assert mha.k.linear_out.weight.shape[0] == 512

    # No kv_channels: head_size is recovered from the fused output dim.
    inferred = _build_split_qkv_adapter(8, 4, None, fused_out=(8 + 2 * 4) * 64)
    assert inferred._qkv_head_layout == (8, 4, 64)
    assert inferred.k.linear_out.weight.shape[0] == 256


def test_split_qkv_adapter_rejects_inconsistent_layout():
    """A layout whose face sizes do not reproduce the fused dim must fail loudly."""
    from megatron.bridge.peft.utils import SplitQKVAdapter

    # Full MHA needs (8 + 2 * 8) * 64 = 1536 rows, not 1024.
    with pytest.raises(ValueError, match="match neither"):
        _build_split_qkv_adapter(8, 8, 64, fused_out=1024)
    # num_attention_heads must be divisible by num_query_groups.
    with pytest.raises(ValueError, match="divisible by num_query_groups"):
        _build_split_qkv_adapter(8, 3, 64, fused_out=(8 + 2 * 3) * 64)
    # The layout is mandatory: the faces cannot be derived without it.
    with pytest.raises(ValueError, match="qkv_head_layout"):
        SplitQKVAdapter(
            64,
            1024,
            8,
            base_linear_name="decoder.layers.0.self_attn.linear_qkv",
            activation="identity",
            model_parallel_config=_cpu_mp_config(),
            input_is_parallel=False,
            base_linear_is_parallel=True,
        )


def test_split_qkv_adapter_interleave_matches_faces_and_row_indices_invert_it():
    """The packed output must be the per-group [q_heads; k; v] layout, invertible."""
    heads, groups, head_size = 8, 4, 64
    heads_per_group = heads // groups
    adapter = _build_split_qkv_adapter(heads, groups, head_size, fused_out=(heads + 2 * groups) * head_size)
    x = torch.randn(2, 3, 64)

    def face_delta(pair):
        hidden, _ = pair.linear_in(x)
        out, _ = pair.linear_out(pair.activation(hidden))
        return out * (pair.alpha / pair.dim)

    q_delta = face_delta(adapter)
    k_delta = face_delta(adapter.k)
    v_delta = face_delta(adapter.v)
    assert q_delta.shape[-1] == 512 and k_delta.shape[-1] == 256

    fused = adapter._interleave_qkv(q_delta, k_delta, v_delta)

    # Independent reference: slice each face by explicit per-group index math.
    chunks = []
    for group in range(groups):
        q_start = group * heads_per_group * head_size
        chunks.append(q_delta[..., q_start : q_start + heads_per_group * head_size])
        chunks.append(k_delta[..., group * head_size : (group + 1) * head_size])
        chunks.append(v_delta[..., group * head_size : (group + 1) * head_size])
    torch.testing.assert_close(fused, torch.cat(chunks, dim=-1))

    # The row indices must recover the faces from the packed tensor, and cover it.
    q_idx, k_idx, v_idx = adapter.face_row_indices(heads_per_group, head_size, groups)
    assert sorted(q_idx + k_idx + v_idx) == list(range(fused.shape[-1]))
    torch.testing.assert_close(fused[..., torch.as_tensor(q_idx)], q_delta)
    torch.testing.assert_close(fused[..., torch.as_tensor(k_idx)], k_delta)
    torch.testing.assert_close(fused[..., torch.as_tensor(v_idx)], v_delta)


def test_split_qkv_adapter_interleave_rejects_tensor_shapes_off_the_layout():
    """A per-rank tensor whose head/group ratio disagrees with the layout must fail."""
    adapter = _build_split_qkv_adapter(8, 4, 64, fused_out=1024)
    with pytest.raises(ValueError, match="not divisible by head_size"):
        adapter._interleave_qkv(torch.zeros(1, 1, 500), torch.zeros(1, 1, 256), torch.zeros(1, 1, 256))
    with pytest.raises(ValueError, match="disagree on the local query-group count"):
        adapter._interleave_qkv(torch.zeros(1, 1, 512), torch.zeros(1, 1, 256), torch.zeros(1, 1, 128))


def test_split_qkv_adapter_forward_packs_the_three_pairs(monkeypatch):
    """forward must call all three pairs and hand their deltas to the packer."""
    from megatron.bridge.peft.utils import ParallelLinearAdapter

    adapter = _build_split_qkv_adapter(8, 4, 64, fused_out=1024)
    q_delta = torch.full((1, 1, 512), 1.0)
    k_delta = torch.full((1, 1, 256), 2.0)
    v_delta = torch.full((1, 1, 256), 3.0)
    deltas = {id(adapter): q_delta, id(adapter.k): k_delta, id(adapter.v): v_delta}

    monkeypatch.setattr(ParallelLinearAdapter, "forward", lambda self, x, *args, **kwargs: deltas[id(self)])

    out = adapter.forward(torch.zeros(1, 1, 64))
    torch.testing.assert_close(out, adapter._interleave_qkv(q_delta, k_delta, v_delta))


def test_split_qkv_adapter_merged_weight_merges_each_face_into_its_own_rows():
    """base + interleaved per-face deltas, with every fused row written exactly once."""
    heads, groups, head_size = 8, 4, 64
    heads_per_group = heads // groups
    adapter = _build_split_qkv_adapter(heads, groups, head_size, fused_out=(heads + 2 * groups) * head_size)
    torch.manual_seed(0)
    with torch.no_grad():
        # row_init_method defaults to "zero", so randomize B to get a non-trivial delta.
        for pair in (adapter, adapter.k, adapter.v):
            pair.linear_out.weight.normal_()

    base = torch.randn((heads + 2 * groups) * head_size, 64)
    merged = adapter.merged_weight(base)
    assert merged.shape == base.shape
    assert not torch.allclose(merged, base), "merge must actually change the fused weight"

    q_idx, k_idx, v_idx = adapter.face_row_indices(heads_per_group, head_size, groups)
    expected = base.clone()
    for pair, indices in ((adapter, q_idx), (adapter.k, k_idx), (adapter.v, v_idx)):
        index = torch.as_tensor(indices)
        delta = (pair.alpha / pair.dim) * (pair.linear_out.weight @ pair.linear_in.weight)
        expected.index_copy_(0, index, torch.index_select(base, 0, index) + delta)
    torch.testing.assert_close(merged, expected)


def test_split_qkv_adapter_merged_weight_handles_a_tp_shard():
    """A per-rank fused shard carries fewer query groups; the merge must still line up."""
    heads, groups, head_size = 8, 4, 64
    heads_per_group = heads // groups
    adapter = _build_split_qkv_adapter(heads, groups, head_size, fused_out=(heads + 2 * groups) * head_size)
    torch.manual_seed(1)
    with torch.no_grad():
        for pair in (adapter, adapter.k, adapter.v):
            pair.linear_out.weight.normal_()

    # tp = 2: this rank holds the first two of the four query groups, so each
    # face carries only its rank slice of rows (q: heads/2 * head_size, k/v:
    # groups/2 * head_size) and the fused base only 2 * rows_per_group rows.
    local_groups = 2
    local_rows = (heads_per_group + 2) * head_size * local_groups
    base = torch.randn(local_rows, 64)
    with torch.no_grad():
        for pair, rank_rows in ((adapter, 256), (adapter.k, 128), (adapter.v, 128)):
            pair.linear_out.weight = torch.nn.Parameter(pair.linear_out.weight[:rank_rows].clone())

    merged = adapter.merged_weight(base)
    assert merged.shape == base.shape

    q_idx, k_idx, v_idx = adapter.face_row_indices(heads_per_group, head_size, local_groups)
    assert max(q_idx + k_idx + v_idx) == local_rows - 1
    expected = base.clone()
    for pair, indices in ((adapter, q_idx), (adapter.k, k_idx), (adapter.v, v_idx)):
        index = torch.as_tensor(indices)
        delta = (pair.alpha / pair.dim) * (pair.linear_out.weight @ pair.linear_in.weight)
        expected.index_copy_(0, index, torch.index_select(base, 0, index) + delta)
    torch.testing.assert_close(merged, expected)


def test_split_qkv_adapter_merged_weight_rejects_a_non_group_multiple():
    """A fused weight that is not a whole number of query groups cannot be split."""
    adapter = _build_split_qkv_adapter(8, 4, 64, fused_out=1024)
    with pytest.raises(ValueError, match="one query group's rows"):
        adapter.merged_weight(torch.randn(1000, 64))


def _split_qkv_adapters_info(base: str, include_face_suffix: bool = True):
    """Three enumeration entries, exactly as named_parameters yields them."""
    info = [(f"{base}.adapter", base, False, False, False, 32, 8, 0, 0, False, True)]
    if include_face_suffix:
        for face in ("k", "v"):
            info.append((f"{base}.adapter.{face}", base, False, False, False, 32, 8, 0, 0, False, True))
    return info


def _fake_split_qkv_adapter():
    from megatron.bridge.peft.utils import SplitQKVAdapter

    adapter = SplitQKVAdapter.__new__(SplitQKVAdapter)
    torch.nn.Module.__init__(adapter)

    def _holder():
        return SimpleNamespace(weight=torch.ones(2, 2))

    adapter.linear_in, adapter.linear_out = _holder(), _holder()
    adapter.k = SimpleNamespace(linear_in=_holder(), linear_out=_holder())
    adapter.v = SimpleNamespace(linear_in=_holder(), linear_out=_holder())
    adapter.alpha, adapter.dim = 32, 8
    return adapter


def _qkv_registry():
    from megatron.bridge.models.conversion.param_mapping import QKVMapping

    return MegatronMappingRegistry(
        QKVMapping(
            megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
            q="model.layers.*.self_attn.q_proj.weight",
            k="model.layers.*.self_attn.k_proj.weight",
            v="model.layers.*.self_attn.v_proj.weight",
        )
    )


def test_split_qkv_adapter_emits_one_task_per_face(monkeypatch):
    """Three enumerated pairs must yield three tasks, each owning one face.

    Emitting all three faces per entry would triple every split surface, the
    q/k/v analogue of the doubled gate/up surface the split-gate-up test pins.
    """
    from megatron.bridge.models.conversion.peft_bridge import is_split_qkv_adapter

    bridge = DummyBridge()
    bridge.hf_pretrained = SimpleNamespace()
    bridge.hf_config = bridge.hf_pretrained

    adapter = _fake_split_qkv_adapter()
    assert is_split_qkv_adapter(adapter), "predicate must see the real class"

    base = "decoder.layers.0.self_attention.linear_qkv"
    adapters_info = _split_qkv_adapters_info(base)
    monkeypatch.setattr(bridge, "_megatron_global_adapters_info_all_pp_ranks", lambda *_: adapters_info)
    monkeypatch.setattr(bridge, "_get_adapter_wrap_module", lambda *_: (adapter, None))
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(bridge, "mapping_registry", _qkv_registry)

    tasks_by_base = bridge.build_adapter_conversion_tasks([Mock()])
    assert list(tasks_by_base) == [base]
    tasks = tasks_by_base[base]
    assert len(tasks) == 3, "one task per face (not per face per entry)"
    by_face = {task.qkv_face: task for task in tasks}
    assert sorted(by_face) == ["k", "q", "v"]
    # Each task carries its OWN pair, not the q pair three times.
    assert by_face["q"].linear_in_task.param_weight is adapter.linear_in.weight
    assert by_face["k"].linear_in_task.param_weight is adapter.k.linear_in.weight
    assert by_face["v"].linear_out_task.param_weight is adapter.v.linear_out.weight
    # The q pair's names stay plain; k / v are tagged with their face.
    assert by_face["q"].linear_in_task.global_param_name.endswith(".adapter.linear_in.weight")
    assert by_face["k"].linear_in_task.global_param_name.endswith(".adapter.linear_in.weight.k")
    assert by_face["v"].linear_in_task.global_param_name.endswith(".adapter.linear_in.weight.v")


def test_stream_adapter_weights_megatron_to_hf_split_qkv_faces(monkeypatch):
    """Each face streams under its own HF name with its own tensor.

    The registry maps the fused ``linear_qkv`` base to all three q/k/v names at
    once, so the face is what selects the HF base. The fused contiguous slicing
    path must NOT run: a split adapter stores each face separately, so slicing
    the (already per-face) tensor again would emit a fraction of a face.
    """
    bridge = DummyBridge()
    bridge.hf_pretrained = SimpleNamespace()
    bridge.hf_config = bridge.hf_pretrained

    base = "decoder.layers.0.self_attention.linear_qkv"
    face_tensors = {
        "q": torch.full((2, 2), 1.0),
        "k": torch.full((2, 2), 2.0),
        "v": torch.full((2, 2), 3.0),
    }

    def _task(face: str) -> AdapterWeightConversionTask:
        tag = "" if face == "q" else f".{face}"
        return AdapterWeightConversionTask(
            global_base_prefix=base,
            adapter_key=None,
            alpha=2,
            dim=4,
            linear_in_task=WeightConversionTask(
                param_name="local_in",
                global_param_name=f"{base}.adapter.linear_in.weight{tag}",
                mapping=Mock(),
            ),
            linear_out_task=WeightConversionTask(
                param_name="local_out",
                global_param_name=f"{base}.adapter.linear_out.weight{tag}",
                mapping=Mock(),
            ),
            qkv_face=face,
        )

    monkeypatch.setattr(
        bridge,
        "build_adapter_conversion_tasks",
        lambda *_args, **_kwargs: {base: [_task("q"), _task("k"), _task("v")]},
    )
    monkeypatch.setattr(
        bridge,
        "materialize_adapter_weights",
        lambda tasks: [
            AdapterWeight(
                global_base_prefix=base,
                adapter_key=None,
                alpha=2,
                dim=4,
                linear_in_weight=MegatronWeightTuple("local_in", torch.ones(2, 2), vp_stage=0),
                linear_out_weight=MegatronWeightTuple(
                    "local_out", face_tensors[tasks[0].qkv_face], vp_stage=0
                ),
            )
        ],
    )
    monkeypatch.setattr(
        bridge,
        "_get_base_hf_param_names_for_adapter",
        lambda *_: [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ],
    )

    def _no_fused_slicing(*_args, **_kwargs):
        raise AssertionError("split q/k/v faces must not go through the fused slicing path")

    monkeypatch.setattr(bridge, "_get_fused_adapter_linear_out_slices", _no_fused_slicing)

    weights = list(
        bridge.stream_adapter_weights_megatron_to_hf(
            [SimpleNamespace(config=SimpleNamespace(num_moe_experts=0))],
            cpu=False,
            show_progress=False,
        )
    )

    assert len(weights) == 6, "two tensors (A and B) per face, no duplicates"
    names = [w.param_name for w in weights]
    assert names == [
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
        "model.layers.0.self_attn.k_proj.lora_A.weight",
        "model.layers.0.self_attn.k_proj.lora_B.weight",
        "model.layers.0.self_attn.v_proj.lora_A.weight",
        "model.layers.0.self_attn.v_proj.lora_B.weight",
    ]
    # Each face's B tensor is emitted as-is, unmixed with the other faces.
    torch.testing.assert_close(weights[1].weight, face_tensors["q"])
    torch.testing.assert_close(weights[3].weight, face_tensors["k"])
    torch.testing.assert_close(weights[5].weight, face_tensors["v"])


def test_stream_adapter_weights_megatron_to_hf_split_qkv_rejects_missing_face(monkeypatch):
    """A registry that exposes no per-face HF base for the fused qkv must fail loudly."""
    bridge = DummyBridge()
    bridge.hf_pretrained = SimpleNamespace()
    bridge.hf_config = bridge.hf_pretrained

    base = "decoder.layers.0.self_attention.linear_qkv"
    task = AdapterWeightConversionTask(
        global_base_prefix=base,
        adapter_key=None,
        alpha=2,
        dim=4,
        linear_in_task=WeightConversionTask(param_name="local_in", global_param_name="g_in", mapping=Mock()),
        linear_out_task=WeightConversionTask(param_name="local_out", global_param_name="g_out", mapping=Mock()),
        qkv_face="k",
    )
    monkeypatch.setattr(
        bridge, "build_adapter_conversion_tasks", lambda *_a, **_k: {base: [task]}
    )
    monkeypatch.setattr(
        bridge,
        "materialize_adapter_weights",
        lambda *_a, **_k: [
            AdapterWeight(
                global_base_prefix=base,
                adapter_key=None,
                alpha=2,
                dim=4,
                linear_in_weight=MegatronWeightTuple("local_in", torch.ones(2, 2), vp_stage=0),
                linear_out_weight=MegatronWeightTuple("local_out", torch.ones(2, 2), vp_stage=0),
            )
        ],
    )
    # Fused single-name mapping: no ``.k_proj.`` base to attach the face to.
    monkeypatch.setattr(
        bridge, "_get_base_hf_param_names_for_adapter", lambda *_: ["model.layers.0.self_attn.query.weight"]
    )

    with pytest.raises(ValueError, match="expected exactly one k_proj base"):
        list(
            bridge.stream_adapter_weights_megatron_to_hf(
                [SimpleNamespace(config=SimpleNamespace(num_moe_experts=0))],
                cpu=False,
                show_progress=False,
            )
        )


# ---------------------------------------------------------------------------
# Blocked (GDN / KDA ``in_proj``) split q/k/v mode
#
# The GDN / gated-delta-net linear-attention mixer fuses q, k and v into one
# ``in_proj`` whose output is the BLOCK concatenation ``[q (qk); k (qk); v (v);
# extra]`` — no per-query-group packing — and the reference platform carries the
# same three ``q_proj`` / ``k_proj`` / ``v_proj`` faces on it. The blocks that
# follow v (``z`` / ``f`` / ``b`` / ``w`` ... by variant) are not platform
# targets, so their delta must be exactly zero.
# ---------------------------------------------------------------------------


def _build_blocked_split_qkv_adapter(
    qk_dim: int,
    v_dim: int,
    extra: int,
    *,
    in_features: int = 64,
    dim: int = 8,
    tp_size: int = 1,
    fused_out: int | None = None,
):
    """Build a real SplitQKVAdapter in blocked mode (GDN ``in_proj`` layout)."""
    from unittest.mock import patch

    from megatron.core.process_groups_config import ProcessGroupCollection

    from megatron.bridge.peft.utils import SplitQKVAdapter

    # Same ProcessGroupCollection stub as _build_split_qkv_adapter: adapter
    # construction must not need an initialized Megatron MPU.
    stub = ProcessGroupCollection()
    with patch.object(ProcessGroupCollection, "use_mpu_process_groups", classmethod(lambda cls, **kwargs: stub)):
        return SplitQKVAdapter(
            in_features,
            fused_out if fused_out is not None else 2 * qk_dim + v_dim + extra,
            dim,
            qkv_block_layout=(qk_dim, v_dim, extra),
            base_linear_name="decoder.layers.0.inner_layer.self_attention.in_proj",
            activation="identity",
            model_parallel_config=_cpu_mp_config(tp_size),
            input_is_parallel=False,
            base_linear_is_parallel=True,
        )


def test_split_qkv_adapter_blocked_builds_contiguous_face_blocks():
    """q = k = qk_dim and v = v_dim on one contiguous block each; extra rows own nothing."""
    adapter = _build_blocked_split_qkv_adapter(6, 4, 2)

    assert adapter._qkv_layout_mode == "blocked"
    assert adapter._qkv_block_layout == (6, 4, 2)
    # The interleaved geometry is not derivable (and is not used) in this mode.
    assert adapter._qkv_head_layout is None

    assert adapter.linear_out.weight.shape[0] == 6  # q block
    assert adapter.k.linear_out.weight.shape[0] == 6  # k block
    assert adapter.v.linear_out.weight.shape[0] == 4  # v block
    assert adapter.linear_in.weight.shape == (8, 64)

    pairs = adapter.pairs()
    assert len(pairs) == 3
    (q_in, q_out), (k_in, k_out), (v_in, v_out) = pairs
    assert q_in is adapter.linear_in and q_out is adapter.linear_out
    assert k_in is adapter.k.linear_in and k_out is adapter.k.linear_out
    assert v_in is adapter.v.linear_in and v_out is adapter.v.linear_out
    assert len({id(module) for pair in pairs for module in pair}) == 6


def test_split_qkv_adapter_blocked_accepts_the_per_rank_fused_width():
    """The fused width is accepted GLOBAL (18) or per-rank (18 / tp = 9 for tp = 2)."""
    global_adapter = _build_blocked_split_qkv_adapter(6, 4, 2)
    assert global_adapter._qkv_block_layout == (6, 4, 2)

    per_rank = _build_blocked_split_qkv_adapter(6, 4, 2, tp_size=2, fused_out=9)
    assert per_rank._qkv_block_layout == (6, 4, 2)
    assert per_rank._qkv_tp_size == 2
    # The stored geometry stays GLOBAL; only the merge/index derivation shards it.
    assert per_rank._blocked_local_sizes(18) == (6, 4, 2)
    assert per_rank._blocked_local_sizes(9) == (3, 2, 1)


def test_split_qkv_adapter_rejects_conflicting_or_missing_layout_modes():
    """Exactly one layout kwarg is required, and neither may be inferred."""
    from unittest.mock import patch

    from megatron.core.process_groups_config import ProcessGroupCollection

    from megatron.bridge.peft.utils import SplitQKVAdapter

    stub = ProcessGroupCollection()
    common = dict(
        base_linear_name="decoder.layers.0.inner_layer.self_attention.in_proj",
        activation="identity",
        model_parallel_config=_cpu_mp_config(),
        input_is_parallel=False,
        base_linear_is_parallel=True,
    )
    with patch.object(ProcessGroupCollection, "use_mpu_process_groups", classmethod(lambda cls, **kwargs: stub)):
        with pytest.raises(ValueError, match="not both"):
            SplitQKVAdapter(64, 18, 8, qkv_head_layout=(8, 4, 64), qkv_block_layout=(6, 4, 2), **common)
        with pytest.raises(ValueError, match="requires either qkv_head_layout"):
            SplitQKVAdapter(64, 18, 8, **common)


def test_split_qkv_adapter_blocked_rejects_fused_widths_off_the_layout():
    """A fused width that does not reproduce 2 * qk + v + extra must fail loudly."""
    with pytest.raises(ValueError, match="matches neither out_features"):
        _build_blocked_split_qkv_adapter(6, 4, 2, fused_out=17)
    with pytest.raises(ValueError, match="must be positive"):
        _build_blocked_split_qkv_adapter(0, 4, 2)
    with pytest.raises(ValueError, match="extra_features_dim must be >= 0"):
        _build_blocked_split_qkv_adapter(6, 4, -1, fused_out=16)
    # Every face shards independently, so each block must divide by tp_size.
    with pytest.raises(ValueError, match="must be divisible by"):
        _build_blocked_split_qkv_adapter(6, 4, 3, tp_size=2, fused_out=19)


def test_split_qkv_adapter_blocked_forward_cats_a_zero_extra_block(monkeypatch):
    """forward = cat([q, k, v, zeros(extra)]); the interleaver must NOT run."""
    from megatron.bridge.peft.utils import ParallelLinearAdapter

    adapter = _build_blocked_split_qkv_adapter(6, 4, 2)
    q_delta = torch.full((2, 3, 6), 1.0)
    k_delta = torch.full((2, 3, 6), 2.0)
    v_delta = torch.full((2, 3, 4), 3.0)
    deltas = {id(adapter): q_delta, id(adapter.k): k_delta, id(adapter.v): v_delta}
    monkeypatch.setattr(ParallelLinearAdapter, "forward", lambda self, x, *args, **kwargs: deltas[id(self)])
    monkeypatch.setattr(
        adapter,
        "_interleave_qkv",
        lambda *_: pytest.fail("blocked mode must not interleave"),
    )

    out = adapter.forward(torch.zeros(2, 3, 64))

    assert out.shape == (2, 3, 18)
    torch.testing.assert_close(out[..., :6], q_delta)
    torch.testing.assert_close(out[..., 6:12], k_delta)
    torch.testing.assert_close(out[..., 12:16], v_delta)
    # The non-target rows must be present (the fused output is added wholesale)
    # and carry no adapter contribution.
    extra = out[..., 16:]
    assert extra.shape == (2, 3, 2)
    assert torch.count_nonzero(extra) == 0


def test_split_qkv_adapter_blocked_forward_without_extra_rows(monkeypatch):
    """extra == 0 emits the plain three-block concatenation."""
    from megatron.bridge.peft.utils import ParallelLinearAdapter

    adapter = _build_blocked_split_qkv_adapter(6, 4, 0)
    deltas = {
        id(adapter): torch.full((1, 6), 1.0),
        id(adapter.k): torch.full((1, 6), 2.0),
        id(adapter.v): torch.full((1, 4), 3.0),
    }
    monkeypatch.setattr(ParallelLinearAdapter, "forward", lambda self, x, *args, **kwargs: deltas[id(self)])

    out = adapter.forward(torch.zeros(1, 64))
    assert out.shape == (1, 16)
    torch.testing.assert_close(out, torch.cat([deltas[id(adapter)], deltas[id(adapter.k)], deltas[id(adapter.v)]], -1))


def test_split_qkv_adapter_blocked_face_row_indices_are_contiguous():
    """Each face's rows are one contiguous range; the extra rows belong to no face."""
    adapter = _build_blocked_split_qkv_adapter(6, 4, 2)

    q_idx, k_idx, v_idx = adapter.face_row_indices_for_layout(18)
    assert q_idx == list(range(0, 6))
    assert k_idx == list(range(6, 12))
    assert v_idx == list(range(12, 16))
    # The extra block is deliberately uncovered: nothing merges into it.
    assert sorted(q_idx + k_idx + v_idx) == list(range(16))

    # Per-rank (tp = 2): the shard keeps the first half of every block, so the
    # trailing local extra row is the one that passes through untouched.
    sharded = _build_blocked_split_qkv_adapter(6, 4, 2, tp_size=2, fused_out=9)
    q_idx, k_idx, v_idx = sharded.face_row_indices_for_layout(9)
    assert (q_idx, k_idx, v_idx) == ([0, 1, 2], [3, 4, 5], [6, 7])


def test_split_qkv_adapter_blocked_merged_weight_merges_contiguous_slices():
    """base + per-block deltas; the extra rows pass through unchanged."""
    adapter = _build_blocked_split_qkv_adapter(6, 4, 2)
    torch.manual_seed(0)
    with torch.no_grad():
        # row_init_method defaults to "zero", so randomize B for a non-trivial delta.
        for pair in (adapter, adapter.k, adapter.v):
            pair.linear_out.weight.normal_()

    base = torch.randn(18, 64)
    merged = adapter.merged_weight(base)
    assert merged.shape == base.shape
    assert not torch.allclose(merged, base), "merge must actually change the fused weight"

    expected = base.clone()
    for pair, (start, width) in ((adapter, (0, 6)), (adapter.k, (6, 6)), (adapter.v, (12, 4))):
        delta = (pair.alpha / pair.dim) * (pair.linear_out.weight @ pair.linear_in.weight)
        expected[start : start + width] = base[start : start + width] + delta
    torch.testing.assert_close(merged, expected)
    # Untouched extra rows carry no adapter contribution at all.
    torch.testing.assert_close(merged[16:], base[16:])

    # A fused width that is neither the global total nor its shard cannot be split.
    with pytest.raises(ValueError, match="match neither the blocked fused width"):
        adapter.merged_weight(torch.randn(15, 64))


def test_split_qkv_adapter_blocked_merged_weight_handles_a_tp_shard():
    """A per-rank shard carries half of every block; the merge must still line up."""
    adapter = _build_blocked_split_qkv_adapter(6, 4, 2, tp_size=2, fused_out=9)
    torch.manual_seed(1)
    with torch.no_grad():
        # This rank's slice of each block: q/k 3 rows, v 2 rows, and the extra
        # block's single local row.
        for pair, rank_rows in ((adapter, 3), (adapter.k, 3), (adapter.v, 2)):
            pair.linear_out.weight = torch.nn.Parameter(pair.linear_out.weight[:rank_rows].clone())
            pair.linear_out.weight.normal_()

    base = torch.randn(9, 64)
    merged = adapter.merged_weight(base)
    assert merged.shape == base.shape

    q_idx, k_idx, v_idx = adapter.face_row_indices_for_layout(9)
    assert max(q_idx + k_idx + v_idx) == 7
    expected = base.clone()
    for pair, indices in ((adapter, q_idx), (adapter.k, k_idx), (adapter.v, v_idx)):
        index = torch.as_tensor(indices)
        delta = (pair.alpha / pair.dim) * (pair.linear_out.weight @ pair.linear_in.weight)
        expected.index_copy_(0, index, torch.index_select(base, 0, index) + delta)
    torch.testing.assert_close(merged, expected)
    torch.testing.assert_close(merged[8:], base[8:])


# ---------------------------------------------------------------------------
# GDN geometry derivation and the LoRA walk gate
# ---------------------------------------------------------------------------


def _gdn_config(**overrides):
    """A real TransformerConfig carrying the GLM-5.3-Flash-style GDN head geometry."""
    from megatron.core.transformer.transformer_config import TransformerConfig

    kwargs = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        linear_num_key_heads=4,
        linear_key_head_dim=8,
        linear_num_value_heads=4,
        linear_value_head_dim=8,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_gdn_in_proj_block_layout_reads_the_config_geometry():
    """qk = key_heads * key_head_dim, v = value_heads * value_head_dim, extra = the rest."""
    from megatron.bridge.peft.lora import gdn_in_proj_block_layout

    # qk = v = 32, so the q/k/v blocks take 96 rows and the rest is extra.
    assert gdn_in_proj_block_layout(_gdn_config(), 96) == (32, 32, 0)
    assert gdn_in_proj_block_layout(_gdn_config(), 112) == (32, 32, 16)
    # The mixer's own shorter spellings are accepted too.
    short = SimpleNamespace(
        num_key_heads=4, key_head_dim=8, num_value_heads=4, value_head_dim=8
    )
    assert gdn_in_proj_block_layout(short, 112) == (32, 32, 16)
    # Dual reading: the same geometry fits the per-rank width of a tp = 2 shard.
    assert gdn_in_proj_block_layout(_gdn_config(tensor_model_parallel_size=2), 56, tp_size=2) == (
        32,
        32,
        16,
    )


def test_gdn_in_proj_block_layout_is_none_without_gdn_geometry():
    """A config with no GDN head config must not select the split adapter."""
    from megatron.bridge.peft.lora import gdn_in_proj_block_layout

    assert gdn_in_proj_block_layout(SimpleNamespace(), 112) is None
    assert (
        gdn_in_proj_block_layout(
            SimpleNamespace(linear_num_key_heads=4, linear_key_head_dim=8), 112
        )
        is None
    )


def test_gdn_in_proj_block_layout_rejects_a_width_below_the_qkv_blocks():
    """A GDN config whose fused width cannot hold q/k/v is a real mismatch, not a skip."""
    from megatron.bridge.peft.lora import gdn_in_proj_block_layout

    with pytest.raises(ValueError, match="does not fit the fused"):
        gdn_in_proj_block_layout(_gdn_config(), 40)


def _walk_with_gdn_config(
    module_name: str, module, config, prefix: str, *, dim: int = 4, split: bool = True
):
    """Run LoRA.transform on ``module`` with the TE-stub workarounds the CPU host needs.

    ``transformer_engine`` is stubbed on this host, so
    ``megatron.bridge.peft.utils.HAVE_TE`` is True while its ``TECL``/``TERL``
    tuples hold MagicMocks — ``isinstance()`` against them raises ``TypeError``.
    Forcing the flag off takes the real MCore ``ColumnParallelLinear`` /
    ``RowParallelLinear`` branch instead, which is the branch these modules use
    on a real host anyway (the TE wrappers subclass them).
    """
    from unittest.mock import patch

    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    import megatron.bridge.peft.utils as peft_utils
    from megatron.bridge.peft.lora import LoRA

    stub = ProcessGroupCollection()
    peft = LoRA(target_modules=[module_name], dim=dim, split_qkv_adapters=split)
    with (
        patch.object(peft_utils, "HAVE_TE", False),
        patch.object(ProcessGroupCollection, "use_mpu_process_groups", classmethod(lambda cls, **kwargs: stub)),
        patch.object(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1),
    ):
        return peft.transform(module, module_name, prefix)


def test_lora_walk_selects_the_blocked_split_adapter_for_gdn_in_proj():
    """The walk must build a SplitQKVAdapter in blocked mode from the GDN geometry."""
    from megatron.core.tensor_parallel import ColumnParallelLinear

    from megatron.bridge.peft.utils import ParallelLinearAdapter, SplitQKVAdapter, is_grouped_expert_linear

    config = _gdn_config()
    module = ColumnParallelLinear(
        input_size=64,
        output_size=112,  # 2 * 32 + 32 + 16
        config=config,
        init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
    )
    prefix = "decoder.layers.0.inner_layer.self_attention"
    assert not is_grouped_expert_linear(f"{prefix}.in_proj")

    wrapped = _walk_with_gdn_config("in_proj", module, config, prefix)

    adapter = wrapped.adapter
    assert isinstance(adapter, SplitQKVAdapter)
    assert adapter._qkv_layout_mode == "blocked"
    assert adapter._qkv_block_layout == (32, 32, 16)
    assert adapter._qkv_head_layout is None
    assert adapter.linear_out.weight.shape[0] == 32
    assert adapter.k.linear_out.weight.shape[0] == 32
    assert adapter.v.linear_out.weight.shape[0] == 32

    # Opting out must leave the fused single-pair topology in place.
    plain = _walk_with_gdn_config("in_proj", module, config, prefix, split=False)
    assert type(plain.adapter) is ParallelLinearAdapter


def test_lora_walk_keeps_a_plain_adapter_for_out_proj():
    """``out_proj`` is a plain single-pair target: name match only, no split adapter."""
    from megatron.core.tensor_parallel import RowParallelLinear

    from megatron.bridge.peft.lora import LoRA
    from megatron.bridge.peft.utils import ParallelLinearAdapter

    # The matcher accepts the bare name (mint adds "out_proj" to its target list).
    matcher = LoRA(target_modules=["out_proj"], dim=4, split_qkv_adapters=True)
    assert matcher.match(torch.nn.Linear(4, 4), "out_proj", "decoder.layers.0.inner_layer") is not None
    assert matcher.match(torch.nn.Linear(4, 4), "in_proj", "decoder.layers.0.inner_layer") is None

    config = _gdn_config()
    module = RowParallelLinear(
        input_size=112,
        output_size=64,
        config=config,
        init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
        bias=False,
        input_is_parallel=True,
        skip_bias_add=False,
        is_expert=False,
    )

    wrapped = _walk_with_gdn_config(
        "out_proj", module, config, "decoder.layers.0.inner_layer.self_attention"
    )

    assert type(wrapped.adapter) is ParallelLinearAdapter


# ---------------------------------------------------------------------------
# Conversion of the GDN ``in_proj`` / ``out_proj`` surfaces
#
# The real GLM-5.3-Flash KDA registry (``_kda_attn_mappings``) maps megatron's
# fused ``in_proj`` to the three HF ``self_attn.{q,k,v}_proj`` weights and
# ``out_proj`` to ``self_attn.o_proj``, so the split adapter's per-face export
# needs no in_proj-specific name handling.
# ---------------------------------------------------------------------------

_GDN_ATTN_LAYER = "decoder.layers.0"
_GDN_HF_LAYER = "model.language_model.layers.0"


def _gdn_kda_registry():
    """The GLM-5.3-Flash KDA attention mappings, without the language_model prefix."""
    from megatron.bridge.models.glm5next.glm53flash_bridge import GLM53FlashBridge

    return MegatronMappingRegistry(
        *GLM53FlashBridge._kda_attn_mappings(_GDN_ATTN_LAYER, _GDN_HF_LAYER)
    )


def _fake_blocked_split_qkv_adapter():
    """The real SplitQKVAdapter class standing in for a GDN in_proj split adapter.

    The conversion path is duck-typed on the class (``is_split_qkv_adapter``), so
    the interleaved fake doubles for the blocked mode — the geometry only matters
    to ``forward`` / ``merged_weight``, neither of which the export touches.
    """
    return _fake_split_qkv_adapter()


def _build_plain_adapter(in_features: int = 112, out_features: int = 64, dim: int = 8):
    """A real single-pair ParallelLinearAdapter, as a GDN ``out_proj`` target gets."""
    from unittest.mock import patch

    from megatron.core.process_groups_config import ProcessGroupCollection

    from megatron.bridge.peft.utils import ParallelLinearAdapter

    stub = ProcessGroupCollection()
    with patch.object(ProcessGroupCollection, "use_mpu_process_groups", classmethod(lambda cls, **kwargs: stub)):
        return ParallelLinearAdapter(
            in_features,
            out_features,
            dim,
            base_linear_name="decoder.layers.0.inner_layer.self_attention.out_proj",
            activation="identity",
            model_parallel_config=_cpu_mp_config(),
            input_is_parallel=True,
            base_linear_is_parallel=True,
        )


def plain_adapters_info(base: str):
    """One enumeration entry for a plain (single-pair) adapter on a row-parallel base."""
    return [(f"{base}.adapter", base, True, True, False, 8, 8, 0, 0, False, False)]


def test_gdn_in_proj_registry_exposes_the_three_faces():
    """The registry must hand the adapter path three per-face HF names, not the fused one."""
    registry = _gdn_kda_registry()
    bridge = DummyBridge()
    prefix = f"{_GDN_ATTN_LAYER}.inner_layer.self_attention.in_proj"

    base_names = bridge._get_base_hf_param_names_for_adapter(registry, prefix, None, ".weight")
    assert base_names == [
        f"{_GDN_HF_LAYER}.self_attn.q_proj.weight",
        f"{_GDN_HF_LAYER}.self_attn.k_proj.weight",
        f"{_GDN_HF_LAYER}.self_attn.v_proj.weight",
    ]
    for face in ("q", "k", "v"):
        assert [name for name in base_names if f".{face}_proj." in name] == [
            f"{_GDN_HF_LAYER}.self_attn.{face}_proj.weight"
        ]
    assert bridge._resolve_hf_adapter_param_name(
        registry, prefix, ".linear_in.weight", ".weight", None
    ) == f"{_GDN_HF_LAYER}.self_attn.q_proj.lora_A.weight"

    # out_proj is a plain single-name mapping.
    out_prefix = f"{_GDN_ATTN_LAYER}.inner_layer.self_attention.out_proj"
    assert bridge._get_base_hf_param_names_for_adapter(registry, out_prefix, None, ".weight") == [
        f"{_GDN_HF_LAYER}.self_attn.o_proj.weight"
    ]
    assert bridge._resolve_hf_adapter_param_name(
        registry, out_prefix, ".linear_out.weight", ".weight", None
    ) == f"{_GDN_HF_LAYER}.self_attn.o_proj.lora_B.weight"


def test_split_qkv_adapter_blocked_emits_one_task_per_face_for_in_proj(monkeypatch):
    """Three enumeration entries on ``in_proj`` must yield three face tasks."""
    from megatron.bridge.models.conversion.peft_bridge import is_split_qkv_adapter

    bridge = DummyBridge()
    bridge.hf_pretrained = SimpleNamespace()
    bridge.hf_config = bridge.hf_pretrained

    adapter = _fake_blocked_split_qkv_adapter()
    assert is_split_qkv_adapter(adapter), "predicate must see the real class"

    base = f"{_GDN_ATTN_LAYER}.inner_layer.self_attention.in_proj"
    monkeypatch.setattr(
        bridge, "_megatron_global_adapters_info_all_pp_ranks", lambda *_: _split_qkv_adapters_info(base)
    )
    monkeypatch.setattr(bridge, "_get_adapter_wrap_module", lambda *_: (adapter, None))
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(bridge, "mapping_registry", _gdn_kda_registry)

    tasks_by_base = bridge.build_adapter_conversion_tasks([Mock()])
    assert list(tasks_by_base) == [base]
    tasks = tasks_by_base[base]
    assert len(tasks) == 3, "one task per face (not per face per entry)"
    by_face = {task.qkv_face: task for task in tasks}
    assert sorted(by_face) == ["k", "q", "v"]
    assert by_face["q"].linear_in_task.param_weight is adapter.linear_in.weight
    assert by_face["k"].linear_in_task.param_weight is adapter.k.linear_in.weight
    assert by_face["v"].linear_out_task.param_weight is adapter.v.linear_out.weight
    # ``_resolve_hf_adapter_param_name`` resolves ONE placeholder name per fused
    # base (the first registry value, i.e. q_proj); the face-specific name is
    # chosen by the ``.{face}_proj.`` marker filter in the streaming loop, which
    # the export test below pins for all three faces.
    assert by_face["q"].linear_in_task.mapping.hf_param == f"{_GDN_HF_LAYER}.self_attn.q_proj.lora_A.weight"
    assert by_face["v"].linear_out_task.mapping.hf_param == f"{_GDN_HF_LAYER}.self_attn.q_proj.lora_B.weight"


def test_stream_adapter_weights_megatron_to_hf_gdn_in_proj_faces(monkeypatch):
    """An ``in_proj`` split adapter streams under the three per-face HF names.

    Driven end-to-end through the REAL GLM-5.3-Flash KDA registry: the faces are
    selected by the ``.{face}_proj.`` marker, so the ``in_proj`` base name needs
    no special casing, and the fused-slicing path must not run.
    """
    bridge = DummyBridge()
    bridge.hf_pretrained = SimpleNamespace()
    bridge.hf_config = bridge.hf_pretrained

    base = f"{_GDN_ATTN_LAYER}.inner_layer.self_attention.in_proj"
    face_tensors = {
        "q": torch.full((2, 2), 1.0),
        "k": torch.full((2, 2), 2.0),
        "v": torch.full((2, 2), 3.0),
    }

    adapter = _fake_blocked_split_qkv_adapter()
    monkeypatch.setattr(
        bridge, "_megatron_global_adapters_info_all_pp_ranks", lambda *_: _split_qkv_adapters_info(base)
    )
    monkeypatch.setattr(bridge, "_get_adapter_wrap_module", lambda *_: (adapter, None))
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(bridge, "mapping_registry", _gdn_kda_registry)
    monkeypatch.setattr(
        bridge,
        "materialize_adapter_weights",
        lambda tasks: [
            AdapterWeight(
                global_base_prefix=base,
                adapter_key=None,
                alpha=2,
                dim=4,
                linear_in_weight=MegatronWeightTuple("local_in", torch.ones(2, 2), vp_stage=0),
                linear_out_weight=MegatronWeightTuple("local_out", face_tensors[tasks[0].qkv_face], vp_stage=0),
            )
        ],
    )

    def _no_fused_slicing(*_args, **_kwargs):
        raise AssertionError("split q/k/v faces must not go through the fused slicing path")

    monkeypatch.setattr(bridge, "_get_fused_adapter_linear_out_slices", _no_fused_slicing)

    weights = list(
        bridge.stream_adapter_weights_megatron_to_hf(
            [SimpleNamespace(config=SimpleNamespace(num_moe_experts=0))],
            cpu=False,
            show_progress=False,
        )
    )

    assert [w.param_name for w in weights] == [
        f"{_GDN_HF_LAYER}.self_attn.q_proj.lora_A.weight",
        f"{_GDN_HF_LAYER}.self_attn.q_proj.lora_B.weight",
        f"{_GDN_HF_LAYER}.self_attn.k_proj.lora_A.weight",
        f"{_GDN_HF_LAYER}.self_attn.k_proj.lora_B.weight",
        f"{_GDN_HF_LAYER}.self_attn.v_proj.lora_A.weight",
        f"{_GDN_HF_LAYER}.self_attn.v_proj.lora_B.weight",
    ]
    torch.testing.assert_close(weights[1].weight, face_tensors["q"])
    torch.testing.assert_close(weights[3].weight, face_tensors["k"])
    torch.testing.assert_close(weights[5].weight, face_tensors["v"])


def test_stream_adapter_weights_megatron_to_hf_gdn_out_proj_is_one_plain_task(monkeypatch):
    """``out_proj`` is a single pair and streams under HF ``o_proj``."""
    bridge = DummyBridge()
    bridge.hf_pretrained = SimpleNamespace()
    bridge.hf_config = bridge.hf_pretrained

    base = f"{_GDN_ATTN_LAYER}.inner_layer.self_attention.out_proj"
    adapter = _build_plain_adapter()
    monkeypatch.setattr(
        bridge, "_megatron_global_adapters_info_all_pp_ranks", lambda *_: plain_adapters_info(base)
    )
    monkeypatch.setattr(bridge, "_get_adapter_wrap_module", lambda *_: (adapter, None))
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(bridge, "mapping_registry", _gdn_kda_registry)

    tasks_by_base = bridge.build_adapter_conversion_tasks([Mock()])
    assert list(tasks_by_base) == [base]
    tasks = tasks_by_base[base]
    assert len(tasks) == 1
    assert tasks[0].qkv_face is None
    assert tasks[0].linear_in_task.mapping.hf_param == f"{_GDN_HF_LAYER}.self_attn.o_proj.lora_A.weight"
    assert tasks[0].linear_out_task.mapping.hf_param == f"{_GDN_HF_LAYER}.self_attn.o_proj.lora_B.weight"



def test_adapter_wrapper_delegates_output_layer_flags():
    """``GPTModel``/``HybridModel`` read and ASSIGN ``sequence_parallel`` on
    their output layer and read ``gather_output`` from it. A LoRA-wrapped
    output layer must answer both, and an assignment must reach the wrapped
    linear (a wrapper-local copy would diverge the wrapper from the module
    that actually runs the gather). Regression for the live failure
    ``AttributeError: 'LoRALinear' object has no attribute 'gather_output'``
    (GLM-5.3-Flash, unembed surface enabled)."""
    import torch.nn as nn

    from megatron.bridge.peft.lora_layers import LoRALinear

    class _FakeOutputLayer(nn.Module):
        """Stand-in for TE's ColumnParallelLinear output layer: carries the
        two flags megatron's model code touches, plus a forward."""

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, 4))
            self.sequence_parallel = True
            self.gather_output = False

        def forward(self, x, *args, **kwargs):
            return x, None

    class _FakeAdapter(nn.Module):
        def forward(self, x, *args, **kwargs):
            return x

    base = _FakeOutputLayer()
    wrapper = LoRALinear(base, _FakeAdapter())

    # Reads delegate to the wrapped linear.
    assert wrapper.sequence_parallel is True
    assert wrapper.gather_output is False

    # Assignment reaches the WRAPPED module (not a wrapper-local shadow).
    wrapper.sequence_parallel = False
    assert base.sequence_parallel is False, "the setter must write through to the wrapped linear"
    assert wrapper.sequence_parallel is False

    wrapper.sequence_parallel = True
    assert base.sequence_parallel is True

    # A wrapped module without the flags still raises AttributeError rather
    # than silently answering something (same class of error as before the
    # properties existed).
    plain = nn.Linear(4, 4)
    plain_wrapper = LoRALinear(plain, _FakeAdapter())
    try:
        plain_wrapper.gather_output
    except AttributeError:
        pass
    else:
        raise AssertionError("a wrapped module without gather_output must not silently answer")


def test_dequantize_fp8_blockwise_matches_the_ragged_loop():
    """The vectorized exact-block path must agree bit-for-bit with the
    reference loop (the ragged path is the same code the fast path replaced).
    Guards the GLM-5.3-Flash import: 37,338 tensors ride this function, and a
    silent numeric drift here would move every weight in the model."""
    import torch

    from megatron.bridge.models.conversion.quantization_utils import (
        dequantize_fp8_blockwise,
    )

    torch.manual_seed(0)

    def ragged_reference(weight, scale_inv, block_size=128, dtype=torch.bfloat16):
        M, N = weight.shape
        w = weight.float()
        out = torch.empty_like(w)
        sM, sN = scale_inv.shape
        for bi in range(sM):
            for bj in range(sN):
                r0, r1 = bi * block_size, min((bi + 1) * block_size, M)
                c0, c1 = bj * block_size, min((bj + 1) * block_size, N)
                out[r0:r1, c0:c1] = w[r0:r1, c0:c1] * scale_inv[bi, bj]
        return out.to(dtype)

    # Exact-block geometry (the checkpoint's): both dims multiples of 128.
    for (M, N) in [(256, 128), (512, 256), (4096, 2048)]:
        w = torch.randn(M, N).to(torch.float8_e4m3fn)
        s = torch.rand(M // 128, N // 128, dtype=torch.float32) * 0.5 + 0.1
        fast = dequantize_fp8_blockwise(w, s)
        ref = ragged_reference(w, s)
        assert fast.shape == ref.shape
        assert torch.equal(fast, ref), f"fast path diverged at {(M, N)}"

    # Ragged geometry must still work (loop path).
    w = torch.randn(300, 200).to(torch.float8_e4m3fn)
    s = torch.rand(3, 2, dtype=torch.float32) * 0.5 + 0.1
    out = dequantize_fp8_blockwise(w, s)
    assert out.shape == (300, 200)
    assert torch.equal(out, ragged_reference(w, s))


def test_shard_readahead_is_best_effort(tmp_path):
    """The readahead helper must never break a load: a missing file, a
    directory, or a platform without posix_fadvise all fall through silently.
    The win it buys on the real mount is 4.5-7.6x on cold shards (measured),
    and the load must not trade that for a new failure mode."""
    from megatron.bridge.models.hf_pretrained import state as state_mod

    # A real file: call must succeed and leave the file readable.
    f = tmp_path / "shard.safetensors"
    f.write_bytes(b"\0" * 4096)
    state_mod._issue_shard_readahead(f)
    assert f.read_bytes() == b"\0" * 4096

    # Missing path and a directory: silent no-ops, no exception.
    state_mod._issue_shard_readahead(tmp_path / "nope.safetensors")
    state_mod._issue_shard_readahead(tmp_path)

    # Platform without fadvise: monkeypatch it away and confirm the helper
    # still swallows the AttributeError.
    import os

    had = hasattr(os, "posix_fadvise")
    if had:
        orig = os.posix_fadvise
        del os.posix_fadvise
        try:
            state_mod._issue_shard_readahead(f)
        finally:
            os.posix_fadvise = orig
