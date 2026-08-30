"""Stateless batched multi-LoRA execution mode (pool-driven, per-token routing).

Distinct from :mod:`megatron.bridge.peft.multi_lora` (the upstream
Multi-LoRA transform): that mode keeps per-adapter weights ON the layer
and routes CONTIGUOUS token spans per adapter; this package is the
complementary execution mode driven by an external pool — the modules
are stateless, all adapter state lives in pool tensors
(``lora_extras = {lora_a_pool, lora_b_pool, lora_indices, lora_scale}``)
owned by the training harness, and ``lora_indices: [T]`` may interleave
adapters per token in any order. Fused BGMV kernels serve both dense and
grouped-expert surfaces (per-surface sharing directions per the pool
contract: fc1 A-shared + B-per-expert, fc2 A-per-expert + B-shared).

Consumers: the mint multilora megatron backend (pool lifecycle, LRU
rotation, snapshot/restore); the MoE token dispatcher patches in the
companion Megatron-LM tree import :class:`ExpertSlotRelay` from here.
"""

from megatron.bridge.peft.batched.lora import (
    BatchedExpertLoraAdapter,
    BatchedLoraAdapter,
    ExpertSlotRelay,
)

__all__ = [
    "BatchedLoraAdapter",
    "BatchedExpertLoraAdapter",
    "ExpertSlotRelay",
]
