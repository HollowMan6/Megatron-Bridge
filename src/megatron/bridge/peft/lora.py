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

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.optimizer import OptimizerConfig, ParamKey, get_standard_config_overrides
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.utils import unwrap_model

from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora_layers import (
    LinearAdapter,
    LoRALinear,
    LoRATopKRouter,
    TEFusedLoRALinear,
    TEFusedLoRAMergeLinear,
)
from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.peft.utils import (
    GroupedExpertLinearAdapter,
    ParallelLinearAdapter,
    SharedOuterGroupedExpertAdapter,
    SplitGateUpAdapter,
    SplitQKVAdapter,
    align_expert_dim_for_tp,
    get_adapter_attributes_from_linear,
    get_effective_lora_dim,
    is_expert_linear,
    is_grouped_expert_linear,
    is_modelopt_linear,
    wildcard_match,
)


logger = logging.getLogger(__name__)


def get_lora_plus_config_overrides(
    config: OptimizerConfig,
    lora_plus_ratio: float,
) -> Dict[ParamKey, ParamGroupOverride]:
    """Build Megatron-Core optimizer ``config_overrides`` for LoRA+.

    LoRA+ (https://arxiv.org/abs/2402.12354) trains ``B`` (``linear_out``) at
    ``lora_plus_ratio`` times the LR of ``A`` (``linear_in``); the ratio is fixed
    and only the base LR is tuned. Standard overrides (bias/1-D ``wd_mult=0.0``,
    decoupled LR) are preserved as the starting point.

    Args:
        config: base optimizer config; ``lr`` / ``min_lr`` are A's learning rates.
        lora_plus_ratio: ``lr_B / lr_A`` (e.g. 16.0). Must be > 0.

    Returns:
        Per-group overrides ready for ``get_megatron_optimizer(config_overrides=...)``.
    """
    if lora_plus_ratio <= 0:
        raise ValueError(f"lora_plus_ratio must be > 0, got {lora_plus_ratio}")

    config_overrides = get_standard_config_overrides(config)

    base_lr = config.lr
    base_min_lr = config.min_lr if config.min_lr is not None else 0.0

    config_overrides[ParamKey(name="*.linear_in.weight")] = ParamGroupOverride(max_lr=base_lr, min_lr=base_min_lr)
    config_overrides[ParamKey(name="*.linear_out.weight")] = ParamGroupOverride(
        max_lr=base_lr * lora_plus_ratio, min_lr=base_min_lr * lora_plus_ratio
    )

    logger.info(
        "LoRA+ enabled: lora_plus_ratio=%s (linear_in/A max_lr=%s, linear_out/B max_lr=%s)",
        lora_plus_ratio,
        base_lr,
        base_lr * lora_plus_ratio,
    )
    return config_overrides


def gdn_in_proj_block_layout(
    config: object,
    out_features: int,
    tp_size: int = 1,
) -> Optional[Tuple[int, int, int]]:
    """Derive the blocked q/k/v geometry of a GDN (KDA / linear-attention) ``in_proj``.

    ``megatron/core/ssm/gated_delta_net`` fuses q, k and v into ONE
    block-concatenated ``in_proj`` whose sections are read straight off the
    config (``common.py``: ``self.qk_dim = config.linear_key_head_dim *
    config.linear_num_key_heads`` and ``self.v_dim =
    config.linear_value_head_dim * config.linear_num_value_heads``, then
    ``in_proj_split_sections`` starts ``[qk, qk, v, ...]``). The rows after the
    v block (``z`` / ``f`` / ``b`` / ``w`` ... depending on the variant) are NOT
    platform LoRA targets, so they are passed through as a zero delta.

    Args:
        config: The base linear's config. The geometry is read as
            ``linear_key_head_dim`` / ``linear_num_key_heads`` /
            ``linear_value_head_dim`` / ``linear_num_value_heads`` with the
            mixer's own shorter spellings (``key_head_dim`` / ``num_key_heads``
            / ``value_head_dim`` / ``num_value_heads``) accepted as fallbacks.
        out_features: The fused ``in_proj`` output width reported by
            ``get_adapter_attributes_from_linear`` — GLOBAL on both base-linear
            families, with the per-rank reading accepted as a fallback.
        tp_size: Tensor-parallel size, for that per-rank fallback.

    Returns:
        ``(qk_dim, v_dim, extra_features_dim)`` with GLOBAL widths, or ``None``
        when the config exposes no GDN geometry (so the module is not a GDN
        ``in_proj`` and the caller must not select the split adapter).

    Raises:
        ValueError: The config DOES expose the GDN geometry but neither the
            global nor the per-rank reading of ``out_features`` leaves room for
            the q/k/v blocks. That is a real mismatch, not a non-GDN module, so
            it fails loudly rather than silently degrading to a fused adapter.
    """

    def _geom(primary: str, fallback: str) -> Optional[int]:
        value = getattr(config, primary, None)
        if value is None:
            value = getattr(config, fallback, None)
        return None if value is None else int(value)

    key_heads = _geom("linear_num_key_heads", "num_key_heads")
    key_head_dim = _geom("linear_key_head_dim", "key_head_dim")
    value_heads = _geom("linear_num_value_heads", "num_value_heads")
    value_head_dim = _geom("linear_value_head_dim", "value_head_dim")
    if key_heads is None or key_head_dim is None or value_heads is None or value_head_dim is None:
        return None

    qk_dim = key_heads * key_head_dim
    v_dim = value_heads * value_head_dim
    qkv_dim = 2 * qk_dim + v_dim
    if out_features >= qkv_dim:
        extra_features_dim = out_features - qkv_dim
    elif out_features * max(tp_size, 1) >= qkv_dim:
        extra_features_dim = out_features * max(tp_size, 1) - qkv_dim
    else:
        raise ValueError(
            "GDN in_proj split: the config's q/k/v geometry "
            f"(2 * qk_dim + v_dim = 2 * {qk_dim} + {v_dim} = {qkv_dim}) does not fit the fused "
            f"in_proj out_features={out_features} under either the global or the "
            f"tensor_model_parallel_size={tp_size} reading. Set "
            "split_qkv_adapters=False for this model, or fix the GDN head geometry on the config."
        )
    return (qk_dim, v_dim, extra_features_dim)


@dataclass
class LoRA(PEFT, ModuleMatcher):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

    LoRA uses a low-rank projection to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of LoRA to specific modules within the model architecture.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections
                                in self-attention.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv
                on the first two layers.
        exclude_modules (List[str], optional): A list of module names not to apply LoRa to. It will
            match all nn.Linear & nn.Linear-adjacent modules whose name does not match any string in
            exclude_modules. If used, will require target_modules to be empty list or None.
        dim (int): Dimension of the low-rank projection space. Defaults to 32.
        alpha (int): Weighting factor for the low-rank projection. Defaults to 32.
        dropout (float): Dropout rate for the low-rank projection. Defaults to 0.0.
        dropout_position (Literal['pre', 'post'], optional): Position for applying dropout.
            Can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'pre'.
        sequence_parallel_input_regather (bool): Reduce retained activation memory for eligible
            column-parallel LoRA-A projections. The full LayerNorm output consumed by LoRA-A is
            gathered temporarily in forward, released after the LoRA-A computation, and gathered
            again during backward for the LoRA-A weight gradient. MCore overlaps the backward
            all-gather with dgrad computation when possible. This has no effect when sequence
            parallelism is disabled and falls back for unsupported adapters or overlapping
            activation recompute. Defaults to False.
        a2a_experimental (bool): Enables the experimental All-to-All (A2A) communication strategy. Defaults to False.
        lora_A_init_method (str): Initialization method for the low-rank matrix A. Defaults to "xavier".
        lora_B_init_method (str): Initialization method for the low-rank matrix B. Defaults to "zero".
        lora_dtype (torch.dtype): Parameter data type for LoRA weights. Default None (will use model's dtype).
        normalize_moe_lora (bool): When True, expert linear layers use dim // moe_router_topk as the LoRA rank
            while non-expert layers keep the full dim. This normalizes the total adapter capacity for MoE models
            so it is comparable to a dense model. Defaults to False.
        share_expert_adapters (bool): When True, grouped MoE expert linears share one adapter across all local
            experts on the EP rank. Set to False to create one adapter per local expert instead. Defaults to True.
        experts_shared_outer_loras (bool): When True, grouped-expert LoRA
            (``TE*ParallelGroupedLinear`` base modules) uses
            :class:`SharedOuterGroupedExpertAdapter` — ``gate_up`` lora_A and
            ``down`` lora_B are shared across experts (expert_dim=1), matching
            SGLang's ``experts_shared_outer_loras=True`` serving contract (PR
            #21466). Default False preserves the adapter layout selected by
            ``share_expert_adapters``.
        split_gate_up_adapters (bool): When True, a fused gate-up linear
            (megatron ``linear_fc1`` with ``gated_linear_unit=True``, dense MLP
            and shared-expert surfaces) gets TWO independent rank-``dim``
            adapter pairs — one per half of the fused output — instead of one
            pair shared by both halves. Matches the reference platform's
            independent ``gate_proj`` / ``up_proj`` adapters
            (tinker-official-reference models/glm-5.3.md; checkpoint-verified
            ``gateA != upA``). Default False keeps the fused single-pair
            topology (reference behaviour for the routed grouped-expert
            surfaces, where the platform also fuses).
        split_qkv_adapters (bool): When True, a fused q/k/v projection gets
            THREE independent rank-``dim`` adapter pairs — one per q / k / v
            face — instead of one pair shared by the whole fused output.
            Matches the reference platform's GLM-5.3-Flash adapter, which
            carries SEPARATE ``q_proj`` / ``k_proj`` / ``v_proj`` faces (each
            with its own ``lora_A`` / ``lora_B``) on both of its attention
            families (tinker-official-reference models/glm-5.3-flash.md, live
            r32 platform probe; the conversion emits the same three faces for
            either). Two fused surfaces are covered:
              * ``linear_qkv`` (self-attention), where the fused rows are
                packed per query group and the adapter rebuilds that
                interleaving;
              * ``in_proj`` (the GDN / KDA linear-attention mixer), where the
                fused rows are the block concatenation ``[q; k; v; extra]``
                and the non-target ``extra`` rows (``z`` / ``f`` / ``b`` /
                ``w`` ...) receive a zero delta. The q/k/v widths come from the
                GDN head geometry on the module config (see
                :func:`gdn_in_proj_block_layout`).
            ``out_proj`` (the GDN mixer's output projection, HF ``o_proj``)
            needs no flag: it is a plain single-pair target, so listing it in
            ``target_modules`` is enough. Default False keeps the fused
            single-pair topology.
        use_transformer_engine_op_fuser (bool): Enable Transformer Engine fusion only for LoRA wrappers.
    """

    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    sequence_parallel_input_regather: bool = False
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"
    a2a_experimental: bool = False
    lora_dtype: torch.dtype = None
    normalize_moe_lora: bool = False
    share_expert_adapters: bool = True
    experts_shared_outer_loras: bool = False
    split_gate_up_adapters: bool = False
    split_qkv_adapters: bool = False
    use_transformer_engine_op_fuser: bool = False

    def __call__(self, model, training: bool = True):
        """Apply LoRA, promoting targeted parameter-backed projections first.

        Parameter-backed projections use the same explicit ``target_modules``
        patterns as ordinary modules. Turning such a parameter into the regular
        LoRA module only when it is targeted lets the existing LoRA wrapper and
        adapter export path handle it without changing the base model's
        checkpoint layout for non-LoRA runs.
        """

        self._promote_parameter_targets(model)
        return super().__call__(model, training=training)

    def _promote_parameter_targets(self, model) -> None:
        """Expose supported bare parameters as linear modules before walking.

        The promoted module keeps the original ``nn.Parameter`` object as its
        weight, so initialization, freezing, and optimizer ownership remain
        unchanged. Parameters that do not match ``target_modules`` retain the
        existing PEFT behavior.
        """

        target_patterns = self.target_modules
        if isinstance(target_patterns, str):
            target_patterns = (target_patterns,)
        else:
            target_patterns = tuple(target_patterns or ())
        if not target_patterns:
            return

        models = model if isinstance(model, list) else [model]
        for root in models:
            if isinstance(root, torch.nn.parallel.DistributedDataParallel):
                root = root.module
            for prefix, parent in list(root.named_modules()):
                for parameter_name, parameter in list(parent.named_parameters(recurse=False)):
                    full_name = f"{prefix}.{parameter_name}" if prefix else parameter_name
                    if not any(
                        parameter_name == pattern or wildcard_match(pattern, full_name) for pattern in target_patterns
                    ):
                        continue
                    if parameter.ndim != 2:
                        raise ValueError(
                            f"{full_name} must be a 2-D matrix for LoRA, got shape {tuple(parameter.shape)}"
                        )
                    promoted = nn.Linear(
                        parameter.shape[1],
                        parameter.shape[0],
                        bias=False,
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    promoted.weight = parameter
                    delattr(parent, parameter_name)
                    setattr(parent, parameter_name, promoted)
                    logger.info("Promoted parameter-backed LoRA target: %s", full_name)

    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """
        Applies LoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply LoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.
            prefix (str, optional): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with LoRA applied, or the original module if not a target.
        """
        # Skip already transformed modules
        adapter_types = (LoRALinear, LoRATopKRouter)
        if isinstance(module, adapter_types):
            return module

        if (ans := self.match(module, name, prefix)) is not None:
            _, full_name = ans
            if isinstance(module, nn.Linear) and not is_modelopt_linear(module):
                adapter = LinearAdapter(
                    module,
                    dim=self.dim,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    lora_A_init_method=self.lora_A_init_method,
                    lora_dtype=self.lora_dtype,
                )
                return LoRALinear(module, adapter)

            is_expert = is_expert_linear(full_name)
            attrs = get_adapter_attributes_from_linear(
                module,
                is_expert=is_expert,
                sequence_parallel_input_regather=self.sequence_parallel_input_regather and not is_expert,
            )

            dim = get_effective_lora_dim(
                module, dim=self.dim, normalize_moe_lora=self.normalize_moe_lora, is_expert=is_expert
            )
            dim = align_expert_dim_for_tp(
                module,
                dim,
                normalize_moe_lora=self.normalize_moe_lora,
                is_expert=is_expert,
                input_is_parallel=attrs.input_is_parallel,
            )
            is_grouped_expert_name = is_grouped_expert_linear(full_name)
            use_shared_outer_adapter = self.experts_shared_outer_loras and is_grouped_expert_name
            use_per_expert_adapter = (
                is_grouped_expert_name and not self.share_expert_adapters and not use_shared_outer_adapter
            )
            use_grouped_expert_adapter = use_shared_outer_adapter or use_per_expert_adapter
            # Fused gate-up surfaces (megatron ``linear_fc1`` of a gated MLP:
            # dense and shared-expert) get independent gate/up pairs when the
            # caller opts in — the reference platform's per-projection
            # adapters. Grouped routed-expert fc1 stays fused on both sides.
            use_split_gate_up_adapter = (
                self.split_gate_up_adapters
                and not is_grouped_expert_name
                and str(full_name).endswith("linear_fc1")
                and bool(getattr(module.config, "gated_linear_unit", False))
            )
            # Fused q/k/v surfaces (megatron ``linear_qkv``, i.e. the attention
            # q/k/v projection as one TE/MCore column-parallel linear) get three
            # independent pairs when the caller opts in — the reference
            # platform's per-face ``q_proj`` / ``k_proj`` / ``v_proj``
            # adapters. The head layout must be readable off the module config
            # because the adapter has to rebuild megatron's per-query-group
            # interleaved packing.
            qkv_layout_available = getattr(module.config, "num_attention_heads", None) is not None
            use_split_qkv_adapter = (
                self.split_qkv_adapters
                and not is_grouped_expert_name
                and str(full_name).endswith("linear_qkv")
                and qkv_layout_available
            )
            # The GDN / KDA linear-attention mixer fuses q, k and v into
            # ``in_proj`` with a BLOCK-CONCATENATED output ([q; k; v; extra],
            # no per-query-group packing), and the reference platform carries the
            # same three ``q_proj`` / ``k_proj`` / ``v_proj`` faces on it. The
            # geometry comes from the GDN head config, so the gate is "the config
            # exposes that geometry", not a fixed architecture name.
            gdn_block_layout = (
                gdn_in_proj_block_layout(
                    module.config,
                    attrs.out_features,
                    getattr(module.config, "tensor_model_parallel_size", 1) or 1,
                )
                if self.split_qkv_adapters and not is_grouped_expert_name and str(full_name).endswith("in_proj")
                else None
            )
            use_split_in_proj_adapter = gdn_block_layout is not None

            enable_op_fuser = (
                not use_grouped_expert_adapter
                and not use_split_gate_up_adapter
                and not use_split_qkv_adapter
                and not use_split_in_proj_adapter
                and (
                    self.use_transformer_engine_op_fuser
                    or getattr(module.config, "use_transformer_engine_op_fuser", False)
                )
                # TP not yet supported
                and parallel_state.get_tensor_model_parallel_world_size() == 1
            )

            logger.info(f"Adding lora to: {full_name}")
            if use_shared_outer_adapter:
                adapter_cls = SharedOuterGroupedExpertAdapter
            elif use_per_expert_adapter:
                adapter_cls = GroupedExpertLinearAdapter
            elif use_split_gate_up_adapter:
                adapter_cls = SplitGateUpAdapter
            elif use_split_qkv_adapter or use_split_in_proj_adapter:
                adapter_cls = SplitQKVAdapter
            else:
                adapter_cls = ParallelLinearAdapter
            adapter_kwargs = dict(
                base_linear_name=full_name,
                activation="identity",
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                input_is_parallel=attrs.input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=module.config,
                alpha=self.alpha,
                base_linear_is_parallel=attrs.base_linear_is_parallel,
            )
            if use_grouped_expert_adapter:
                first_param = next(module.parameters())
                adapter_kwargs.update(
                    num_local_experts=module.num_gemms,
                    params_device=first_param.device,
                    params_dtype=first_param.dtype,
                )
            else:
                adapter_kwargs.update(
                    is_expert=is_expert,
                    a2a_experimental=self.a2a_experimental,
                    sequence_parallel_input_regather=self.sequence_parallel_input_regather,
                    disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
                    disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
                    replicate_adapter=attrs.replicate_adapter,
                )
            if use_split_qkv_adapter:
                # ``qkv_head_layout`` is keyword-only on SplitQKVAdapter, and the
                # kwargs dict is shared with the plain path, so the key is only
                # added on the branch that selected the split adapter.
                # ``num_query_groups=None`` means full MHA; the adapter normalises
                # it. ``kv_channels`` may be absent on some configs — the adapter
                # then infers the head size from the fused output dim.
                adapter_kwargs.update(
                    qkv_head_layout=(
                        getattr(module.config, "num_attention_heads", None),
                        getattr(module.config, "num_query_groups", None),
                        getattr(module.config, "kv_channels", None),
                    )
                )
            elif use_split_in_proj_adapter:
                # Same class, blocked mode: ``qkv_block_layout`` carries the GDN
                # geometry and is mutually exclusive with ``qkv_head_layout``.
                adapter_kwargs.update(qkv_block_layout=gdn_block_layout)
            adapter = adapter_cls(attrs.in_features, attrs.out_features, dim, **adapter_kwargs)
            if isinstance(module, TopKRouter):
                return LoRATopKRouter(module, adapter)
            if enable_op_fuser:
                if is_expert:
                    return TEFusedLoRAMergeLinear(module, adapter)
                return TEFusedLoRALinear(module, adapter)
            else:
                return LoRALinear(module, adapter)
        return module


@dataclass
class VLMLoRA(LoRA):
    """
    Implements the LoRA for Vision-Language Models.
    VLMLoRA additionally allows the user to specify whether the language or vision
    models should be frozen.
    For example, a common finetuning workload for multimodal models is to apply adapters to language model and fully
    finetune the vision model.

    """

    freeze_vision_model: bool = True
    freeze_vision_projection: bool = True
    freeze_language_model: bool = True

    def freeze_model(self, model: nn.Module, training: bool = True) -> None:
        modules_to_freeze = []

        model = unwrap_model(model)[0]
        if hasattr(model, "llava_model"):
            model = model.llava_model

        if self.freeze_vision_model and model.vision_model is not None:
            modules_to_freeze.append(model.vision_model)
        if self.freeze_vision_projection and model.vision_projection is not None:
            modules_to_freeze.append(model.vision_projection)
        if self.freeze_language_model and model.language_model is not None:
            modules_to_freeze.append(model.language_model)

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        if training:
            if isinstance(model, list):
                for model_chunk in model:
                    model_chunk.train(mode=True)
            elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.train(mode=True)
            else:
                model.train(mode=True)
