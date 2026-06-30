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
"""H100 performance recipes for GLM-5.1 and GLM-5.2 SFT."""

from megatron.bridge import AutoBridge
from megatron.bridge.perf_recipes._common import _benchmark_common, _perf_precision
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import ConfigContainer


def glm51_sft_416gpu_h100_bf16_config() -> ConfigContainer:
    """GLM-5.1 SFT: 416x H100, BF16, 128K packed THD, CP=4, cuDNN DSA."""
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained("zai-org/GLM-5.1").to_megatron_provider(load_weights=False)
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.seq_length = 131072
    cfg.dataset.num_workers = 1
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
    cfg.dataset.offline_packing_specs.packed_sequence_size = 131072
    cfg.dataset.offline_packing_specs.pad_seq_to_mult = 8
    cfg.dataset.offline_packing_specs.tokenizer_model_name = "glm5"

    cfg.model.seq_length = 131072
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 26
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 4
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    cfg.train.global_batch_size = 520
    cfg.train.micro_batch_size = 1

    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.optimizer.use_distributed_optimizer = True

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cp_comm_type = "allgather"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_router_dtype = "fp32"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.deallocate_pipeline_outputs = True
    cfg.model.persist_layer_norm = True
    cfg.model.bias_dropout_fusion = True
    cfg.model.bias_activation_fusion = True
    cfg.model.calculate_per_token_loss = True
    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.dsa_indexer_n_heads = 32
    cfg.model.dsa_indexer_head_dim = 128
    cfg.model.dsa_indexer_topk = 2048
    cfg.model.dsa_indexer_rope_interleaved = True
    cfg.model.dsa_indexer_rotate_activation = False
    cfg.model.dsa_indexer_k_norm_epsilon = 1e-6
    cfg.model.dsa_indexer_loss_coeff = 0.001
    cfg.model.dsa_indexer_use_sparse_loss = True
    cfg.model.mtp_num_layers = 1

    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    _benchmark_common(cfg, cross_entropy_impl="native")
    cfg.model.apply_rope_fusion = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    return cfg


def glm52_sft_416gpu_h100_bf16_config() -> ConfigContainer:
    """GLM-5.2 SFT: 416x H100, BF16, 128K packed THD, CP=4, cuDNN DSA."""
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained("zai-org/GLM-5.2").to_megatron_provider(load_weights=False)
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.seq_length = 131072
    cfg.dataset.num_workers = 1
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
    cfg.dataset.offline_packing_specs.packed_sequence_size = 131072
    cfg.dataset.offline_packing_specs.pad_seq_to_mult = 8
    cfg.dataset.offline_packing_specs.tokenizer_model_name = "glm5"

    cfg.model.seq_length = 131072
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 26
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 4
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    cfg.train.global_batch_size = 520
    cfg.train.micro_batch_size = 1

    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.optimizer.use_distributed_optimizer = True

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cp_comm_type = "allgather"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_router_dtype = "fp32"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.deallocate_pipeline_outputs = True
    cfg.model.persist_layer_norm = True
    cfg.model.bias_dropout_fusion = True
    cfg.model.bias_activation_fusion = True
    cfg.model.calculate_per_token_loss = True
    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.dsa_indexer_n_heads = 32
    cfg.model.dsa_indexer_head_dim = 128
    cfg.model.dsa_indexer_topk = 2048
    cfg.model.dsa_indexer_topk_freq = 4
    cfg.model.dsa_indexer_skip_topk_offset = 3
    cfg.model.dsa_indexer_rope_interleaved = True
    cfg.model.dsa_indexer_rotate_activation = False
    cfg.model.dsa_indexer_k_norm_epsilon = 1e-6
    cfg.model.dsa_indexer_loss_coeff = 0.001
    cfg.model.dsa_indexer_use_sparse_loss = True
    cfg.model.mtp_num_layers = 1

    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    _benchmark_common(cfg, cross_entropy_impl="native")
    cfg.model.apply_rope_fusion = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    return cfg
