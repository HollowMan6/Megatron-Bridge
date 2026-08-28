# Copyright (c) 2026, MindLab.  All rights reserved.

"""Batched (multi-tenant) LoRA delta ops for the verl-mint/glm52 mcore fork.

Multi-adapter counterpart of mbridge's ``ParallelLinearAdapter``: one forward
computes the LoRA delta for N adapters, routing each token to its adapter via
per-token ``lora_indices``. The op is owned by this Megatron-LM fork because it
must reproduce the fork's sequence-parallel semantics (gather/scatter are
pushed inside the parallel linears, see ``megatron/core/tensor_parallel/layers.py``
``linear_with_*_async_allreduce(sequence_parallel=True)``).

I/O contract (verified against mbridge ``ParallelLinearAdapter`` on GPU, TP=2):

- column-parallel surface (``input_is_parallel=False``):
    x ``[T_local, in]`` (SP-local) -> delta ``[T, out/tp]`` (full sequence,
    no trailing scatter). Pool A ``[N, rank/tp, in]`` (rank dim TP-sharded),
    pool B ``[N, out/tp, rank]`` (out dim TP-sharded).
- row-parallel surface (``input_is_parallel=True``):
    x ``[T, in/tp]`` (full sequence, in-sharded) -> delta ``[T/tp, out]``
    (SP-local after scatter). Pool A ``[N, rank, in/tp]`` (in-sharded),
    pool B ``[N, out/tp, rank]`` (out-sharded); hidden is TP-all-reduced,
    B output is out-gathered before the sequence scatter.
- non-SP (tp=1): column surface is the plain fused BGMV path
  (``_BgmvLinearLoRA``), row surface reduces to the same math without comms.

Ordering contract: ``lora_indices`` may be in any order. Forward uses
``index_select`` / sorted group offsets (order-agnostic); the plan/format layer
that produces ``lora_indices`` must NOT pre-sort (sorting is op-internal).

Pool grads are NOT all-reduced inside this op: each TP rank's A/B shards are
disjoint, and the CP/DP reduction is owned by the engine's optimizer step.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.bridge.peft.batched.kernels.bgmv import (
    _bgmv_expand_kernel,
    _bgmv_shrink_kernel,
    _bgmv_shrink_transpose_kernel,
)
from megatron.bridge.peft.batched.kernels.bgmv import _l2_prefetch as _l2_prefetch
from megatron.bridge.peft.batched.kernels.bgmv import bgmv_bwd
from megatron.bridge.peft.batched.kernels.bgmv import bgmv_fwd
from megatron.bridge.peft.batched.kernels.bgmv import compute_group_offsets as _compute_group_offsets
from megatron.bridge.peft.batched.kernels.bgmv import is_triton_available as _bgmv_available

__all__ = [
    "BatchedLoraAdapter",
    "batched_lora_delta",
    "_BatchedColumnLoRA",
    "_BatchedRowLoRA",
    "_BgmvColumnLoRA",
    "_BgmvRowLoRA",
    "_BgmvLinearLoRA",
]

# Hopper tl.dot requires every dot dim >= 16; rank/tp < 16 falls back to torch.
_MIN_BGMV_RANK = 16
_LORA_DEBUG = bool(int(os.environ.get("MINT_LORA_DEBUG", "0")))
# Raw shrink/expand kernels mask K/N with runtime bounds; only the BLOCK
# dims (>= 32 from the autotune configs) feed tl.dot, so a rank-local
# shard of 8 is legal. The fused bgmv_fwd path pads to 16 internally
# (_pad_rank) and never sees a sharded rank. This gate used to be 16,
# which silently routed every TP=4 column surface (rank/tp = 8 — the
# lm_head/unembed surface included) onto the torch fallback whose
# per-token [T, out_local, rank] fp32 temporaries OOM at >= 96K packed
# tokens ([12288, 36320, 32] fp32 = 53 GiB against ~26 GiB free).
_MIN_BGMV_LOCAL_RANK = 8
_MAX_BGMV_RANK = 64


# ── sequence-parallel helpers (raw dist: called inside custom autograd fns) ──


def _all_gather_sequence_forward(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    out = torch.empty((x.shape[0] * world_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _reduce_scatter_sequence_forward(x: torch.Tensor, group, local_seq: int) -> torch.Tensor:
    out = torch.empty((local_seq, *x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, x.contiguous(), group=group)
    return out


def _all_gather_last_dim_forward(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return x
    local_width = x.shape[-1]
    flat = x.movedim(-1, 0).contiguous().view(local_width, -1)
    gathered = torch.empty(
        (local_width * world_size, flat.shape[1]), dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(gathered, flat, group=group)
    return gathered.view(local_width * world_size, *x.shape[:-1]).movedim(0, -1).contiguous()


def _split_last_dim(x: torch.Tensor, group_rank: int, local_width: int) -> torch.Tensor:
    start = int(group_rank) * local_width
    return x.narrow(-1, start, local_width).contiguous()


def _segment_scatter_sorted(
    per_token: torch.Tensor, indices: torch.Tensor, num_slots: int, trailing_shape: tuple[int, ...]
) -> torch.Tensor:
    """Scatter-sum per_token rows into [num_slots, *trailing] (order-agnostic).

    index_add_ (atomic) for small T; sort + segment_reduce for large T.
    """
    out = per_token.new_zeros((num_slots, *trailing_shape))
    if per_token.numel() == 0:
        return out
    if per_token.shape[0] < 4096:
        out.index_add_(0, indices, per_token)
        return out
    sort_idx = torch.argsort(indices, stable=True)
    sorted_indices = indices[sort_idx]
    sorted_per_token = per_token.index_select(0, sort_idx)
    lengths = torch.bincount(sorted_indices, minlength=num_slots)
    reduced = torch.segment_reduce(sorted_per_token, reduce="sum", lengths=lengths, axis=0)
    out.copy_(reduced)
    return out


def _gather_groups(x: torch.Tensor, lora_indices: torch.Tensor, tp_group, world_size: int):
    """Sequence-gather x and lora_indices together (rank-ordered concat).

    Returns (gathered_x, gathered_indices). No-op when world_size == 1.
    """
    if world_size <= 1:
        return x, lora_indices
    gathered = _all_gather_sequence_forward(x, tp_group, world_size)
    gathered_indices = _all_gather_sequence_forward(
        lora_indices.unsqueeze(-1), tp_group, world_size
    ).squeeze(-1)
    return gathered, gathered_indices


# ── torch reference path ───────────────────────────────────────────────────


class _BatchedColumnLoRA(torch.autograd.Function):
    """Torch path for column-parallel surfaces (input_is_parallel=False).

    forward: x [T_local, in] (SP-local) -> delta [T, out_local] (full seq)
    backward: returns grad_x [T_local, in] (reduce-scattered), grad_A
    [N, rank_local, in], grad_B [N, out_local, rank]. Pool grads are NOT
    TP-reduced (shards disjoint); CP/DP reduce owned by the engine.
    """

    @staticmethod
    def forward(ctx, x, lora_a_pool, lora_b_pool, lora_indices, scale, tp_group,
                sequence_parallel, rank_partitioned):
        world_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        gathered, gathered_indices = _gather_groups(x, lora_indices, tp_group, world_size) \
            if sequence_parallel else (x, lora_indices)

        a_gathered = lora_a_pool.index_select(0, gathered_indices)  # [T, rank_local, in]
        b_gathered = lora_b_pool.index_select(0, gathered_indices)  # [T, out_local, rank]

        hidden_local = torch.einsum("tri,ti->tr", a_gathered, gathered)  # [T, rank_local]
        if rank_partitioned and world_size > 1:
            hidden = _all_gather_last_dim_forward(hidden_local, tp_group, world_size)  # [T, rank]
        else:
            hidden = hidden_local

        delta = torch.einsum("tor,tr->to", b_gathered, hidden)  # [T, out_local]

        ctx.save_for_backward(x, lora_indices, lora_a_pool, lora_b_pool)
        ctx.tp_group = tp_group
        ctx.world_size = world_size
        ctx.sequence_parallel = bool(sequence_parallel)
        ctx.rank_partitioned = bool(rank_partitioned)
        ctx.local_seq = x.shape[0]
        ctx.local_rank_width = hidden_local.shape[-1]
        ctx.scale = float(scale)
        return scale * delta

    @staticmethod
    def backward(ctx, grad_out):
        x, lora_indices, lora_a_pool, lora_b_pool = ctx.saved_tensors
        tp_group = ctx.tp_group
        world_size = ctx.world_size

        gathered, gathered_indices = _gather_groups(x, lora_indices, tp_group, world_size) \
            if ctx.sequence_parallel else (x, lora_indices)
        # forward output is full-sequence [T, out_local]; grad_out needs no gather.

        a_gathered = lora_a_pool.index_select(0, gathered_indices)
        b_gathered = lora_b_pool.index_select(0, gathered_indices)

        hidden_local = torch.einsum("tri,ti->tr", a_gathered, gathered)
        if ctx.rank_partitioned and world_size > 1:
            hidden = _all_gather_last_dim_forward(hidden_local, tp_group, world_size)
        else:
            hidden = hidden_local

        grad_out = grad_out * ctx.scale
        grad_out_f = grad_out.float()

        # grad_hidden [T, rank]; split to the local rank shard when partitioned.
        # fp32 accumulation: grad scatter/sums over up to T tokens in bf16
        # loses ~3% (2^-8 mantissa per partial); accumulate in fp32, cast back.
        grad_hidden = torch.einsum("to,tor->tr", grad_out_f, b_gathered.float())
        if ctx.rank_partitioned and world_size > 1:
            # grad_hidden is a partial sum over each rank's B out-shard; the TP
            # all-reduce must cover the FULL rank dim BEFORE splitting to the
            # local rank shard (the partial is along o, not along the rank dim).
            dist.all_reduce(grad_hidden, op=dist.ReduceOp.SUM, group=tp_group)
            grad_hidden_local = _split_last_dim(
                grad_hidden, dist.get_rank(tp_group), ctx.local_rank_width
            )
            # grad_x partial: sum over the local rank shard only, then TP all-reduce.
            grad_x_gathered = torch.einsum("tr,tri->ti", grad_hidden_local, a_gathered.float())
            dist.all_reduce(grad_x_gathered, op=dist.ReduceOp.SUM, group=tp_group)
            grad_a_per_token = torch.einsum("tr,ti->tri", grad_hidden_local, gathered.float())
        else:
            grad_x_gathered = torch.einsum("tr,tri->ti", grad_hidden, a_gathered.float())
            grad_a_per_token = torch.einsum("tr,ti->tri", grad_hidden, gathered.float())

        grad_lora_a = _segment_scatter_sorted(
            grad_a_per_token, gathered_indices, lora_a_pool.shape[0],
            (lora_a_pool.shape[1], lora_a_pool.shape[2]),
        ).to(lora_a_pool.dtype)
        grad_b_per_token = torch.einsum("to,tr->tor", grad_out_f, hidden.float())
        grad_lora_b = _segment_scatter_sorted(
            grad_b_per_token, gathered_indices, lora_b_pool.shape[0],
            (lora_b_pool.shape[1], lora_b_pool.shape[2]),
        ).to(lora_b_pool.dtype)

        if ctx.sequence_parallel and world_size > 1:
            # grad_x_gathered is already TP-complete (all-reduced above); the
            # forward input was SP-local, so take this rank's plain seq slice.
            # (reduce_scatter would all-reduce AGAIN -> TPx scale.)
            start = dist.get_rank(tp_group) * ctx.local_seq
            grad_x = grad_x_gathered[start:start + ctx.local_seq]
        else:
            grad_x = grad_x_gathered
        grad_x = grad_x.to(x.dtype)

        return grad_x, grad_lora_a, grad_lora_b, None, None, None, None, None


class _BatchedRowLoRA(torch.autograd.Function):
    """Torch path for row-parallel surfaces (input_is_parallel=True).

    forward: x [T, in_local] (full seq, in-sharded) -> delta [T/tp, out] (SP-local)
    A [N, rank, in_local], B [N, out_local, rank]; hidden = TP-all-reduce of
    the per-rank partial; delta is out-gathered then sequence-scattered.
    """

    @staticmethod
    def forward(ctx, x, lora_a_pool, lora_b_pool, lora_indices, scale, tp_group,
                sequence_parallel):
        world_size = dist.get_world_size(tp_group) if tp_group is not None else 1

        a_gathered = lora_a_pool.index_select(0, lora_indices)  # [T, rank, in_local]
        b_gathered = lora_b_pool.index_select(0, lora_indices)  # [T, out_local, rank]

        hidden_local = torch.einsum("tri,ti->tr", a_gathered, x)  # [T, rank] partial
        if world_size > 1:
            dist.all_reduce(hidden_local, op=dist.ReduceOp.SUM, group=tp_group)
        hidden = hidden_local

        delta_local = torch.einsum("tor,tr->to", b_gathered, hidden)  # [T, out_local]
        delta_full = _all_gather_last_dim_forward(delta_local, tp_group, world_size)  # [T, out]
        delta = delta_full * scale
        if sequence_parallel and world_size > 1:
            # delta_full is replicated across TP (out-gather); the SP-local
            # output is this rank's plain seq slice (reduce_scatter would
            # all-reduce AGAIN -> TPx scale).
            start = dist.get_rank(tp_group) * (x.shape[0] // world_size)
            delta = delta[start:start + x.shape[0] // world_size]

        ctx.save_for_backward(x, lora_indices, lora_a_pool, lora_b_pool)
        ctx.tp_group = tp_group
        ctx.world_size = world_size
        ctx.sequence_parallel = bool(sequence_parallel)
        ctx.local_seq = x.shape[0] // world_size
        ctx.scale = float(scale)
        return delta

    @staticmethod
    def backward(ctx, grad_out):
        x, lora_indices, lora_a_pool, lora_b_pool = ctx.saved_tensors
        tp_group = ctx.tp_group
        world_size = ctx.world_size

        if ctx.sequence_parallel and world_size > 1:
            grad_out = _all_gather_sequence_forward(grad_out, tp_group, world_size)  # [T, out]

        a_gathered = lora_a_pool.index_select(0, lora_indices)  # [T, rank, in_local]
        b_gathered = lora_b_pool.index_select(0, lora_indices)  # [T, out_local, rank]
        # B full (out-gathered along dim 1) for grad_hidden, matching forward's
        # delta_full gather. _all_gather_last_dim_forward gathers dim -1, so
        # transpose (out <-> rank), gather, transpose back.
        b_full = _all_gather_last_dim_forward(
            b_gathered.transpose(1, 2).contiguous(), tp_group, world_size
        ).transpose(1, 2).contiguous()  # [T, out, rank] (kernels need row-major)

        hidden_local = torch.einsum("tri,ti->tr", a_gathered, x)
        if world_size > 1:
            dist.all_reduce(hidden_local, op=dist.ReduceOp.SUM, group=tp_group)
        hidden = hidden_local

        grad_out = grad_out * ctx.scale
        grad_out_f = grad_out.float()

        # fp32 accumulation (see column-path comment).
        grad_hidden = torch.einsum("to,tor->tr", grad_out_f, b_full.float())  # [T, rank]
        grad_a_per_token = torch.einsum("tr,ti->tri", grad_hidden, x.float())
        grad_lora_a = _segment_scatter_sorted(
            grad_a_per_token, lora_indices, lora_a_pool.shape[0],
            (lora_a_pool.shape[1], lora_a_pool.shape[2]),
        ).to(lora_a_pool.dtype)
        # grad_B uses this rank's out shard of grad_out (B's out dim is sharded).
        o_width = lora_b_pool.shape[1]
        start_o = dist.get_rank(tp_group) * o_width if world_size > 1 else 0
        go_local = grad_out_f[:, start_o:start_o + o_width]
        grad_b_per_token = torch.einsum("to,tr->tor", go_local, hidden.float())
        grad_lora_b = _segment_scatter_sorted(
            grad_b_per_token, lora_indices, lora_b_pool.shape[0],
            (lora_b_pool.shape[1], lora_b_pool.shape[2]),
        ).to(lora_b_pool.dtype)

        # grad_x: A's rank dim is full (row surface), so the sum over r is
        # complete per rank; each rank's grad_x covers its own in-shard and
        # needs NO TP all-reduce. Output is full seq [T, in_local].
        grad_x = torch.einsum("tr,tri->ti", grad_hidden, a_gathered.float())
        grad_x = grad_x.to(x.dtype)

        return grad_x, grad_lora_a, grad_lora_b, None, None, None, None


# ── Triton BGMV paths ──────────────────────────────────────────────────────


class _BgmvColumnLoRA(torch.autograd.Function):
    """Triton BGMV for the column-parallel surface (TP>1, rank_local >= 16).

    Mirrors `_BatchedColumnLoRA`'s comm structure but replaces the per-token
    `index_select`+`einsum` (which materializes [T,rank,in]/[T,out,rank]) with
    grouped `_bgmv_shrink_kernel` / `_bgmv_expand_kernel` calls.
    """

    @staticmethod
    def forward(ctx, x, lora_a_pool, lora_b_pool, lora_indices, scale, tp_group,
                sequence_parallel, rank_partitioned):
        import triton

        world_size = dist.get_world_size(tp_group)
        local_seq = x.shape[0]
        G, rank_local, in_features = lora_a_pool.shape
        out_features = lora_b_pool.shape[1]

        gathered, gathered_indices = _gather_groups(x, lora_indices, tp_group, world_size) \
            if sequence_parallel else (x, lora_indices)
        T = gathered.shape[0]

        sorted_indices, perm = torch.sort(gathered_indices, stable=True)
        x_sorted = gathered[perm]
        group_offsets, group_sizes, G = _compute_group_offsets(sorted_indices, num_loras=G)
        max_g_size = T
        id_perm = torch.arange(T, device=x.device)
        _l2_prefetch(lora_a_pool, lora_b_pool)

        # gemm1: hidden_local = x_sorted @ A_shard.T -> [T, rank_local]
        hidden_local = torch.empty(T, rank_local, device=x.device, dtype=x.dtype)
        _bgmv_shrink_kernel[lambda m: (
            triton.cdiv(max_g_size, m["BLOCK_M"]),
            triton.cdiv(rank_local, m["BLOCK_N"]), G)](
            x_sorted, lora_a_pool, hidden_local,
            group_offsets, group_sizes, id_perm,
            T, K=in_features, N=rank_local, G=G, scale=1.0,
        )

        if rank_partitioned and world_size > 1:
            hidden = _all_gather_last_dim_forward(hidden_local, tp_group, world_size)
        else:
            hidden = hidden_local

        # gemm2: delta = scale * hidden @ B.T -> [T, out_local]
        delta_full = torch.empty(T, out_features, device=x.device, dtype=x.dtype)
        _bgmv_shrink_kernel[lambda m: (
            triton.cdiv(max_g_size, m["BLOCK_M"]),
            triton.cdiv(out_features, m["BLOCK_N"]), G)](
            hidden, lora_b_pool, delta_full,
            group_offsets, group_sizes, id_perm,
            T, K=hidden.shape[-1], N=out_features, G=G, scale=scale,
        )

        # unsort delta back to gathered order (output is full sequence, no scatter)
        delta = torch.empty_like(delta_full).index_copy_(0, perm, delta_full)

        ctx.save_for_backward(x, lora_indices, lora_a_pool, lora_b_pool)
        ctx.tp_group = tp_group
        ctx.world_size = world_size
        ctx.sequence_parallel = bool(sequence_parallel)
        ctx.rank_partitioned = bool(rank_partitioned)
        ctx.local_seq = local_seq
        ctx.local_rank_width = rank_local
        ctx.scale = float(scale)
        ctx.rank = hidden.shape[-1]
        return delta

    @staticmethod
    def backward(ctx, grad_out):
        import triton

        x, lora_indices, lora_a_pool, lora_b_pool = ctx.saved_tensors
        tp_group = ctx.tp_group
        world_size = ctx.world_size
        scale = ctx.scale
        G, rank_local, in_features = lora_a_pool.shape
        out_features = lora_b_pool.shape[1]

        gathered, gathered_indices = _gather_groups(x, lora_indices, tp_group, world_size) \
            if ctx.sequence_parallel else (x, lora_indices)
        T = gathered.shape[0]

        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        sorted_indices, perm = torch.sort(gathered_indices, stable=True)
        x_sorted = gathered[perm]
        grad_sorted = grad_out[perm]
        group_offsets, group_sizes, G = _compute_group_offsets(sorted_indices, num_loras=G)
        max_g_size = T
        id_perm = torch.arange(T, device=x.device)

        # gemm1 (expand): grad_hidden = grad_out @ (scale*B) -> [T, full_rank]
        full_rank = ctx.rank
        grad_hidden = torch.empty(T, full_rank, device=x.device, dtype=x.dtype)
        _bgmv_expand_kernel[lambda m: (
            triton.cdiv(max_g_size, m["BLOCK_M"]),
            triton.cdiv(full_rank, m["BLOCK_N"]), G)](
            grad_sorted, lora_b_pool, grad_hidden,
            group_offsets, group_sizes, id_perm,
            T, K=out_features, N=full_rank, G=G, scale=scale,
        )

        if ctx.rank_partitioned and world_size > 1:
            # grad_hidden is a partial sum over each rank's B out-shard; the TP
            # all-reduce must cover the FULL rank dim BEFORE splitting to the
            # local rank shard (the partial is along o, not along the rank dim).
            dist.all_reduce(grad_hidden, op=dist.ReduceOp.SUM, group=tp_group)
            grad_hidden_local = _split_last_dim(
                grad_hidden, dist.get_rank(tp_group), ctx.local_rank_width)
            # grad_x uses only the local A shard; TP all-reduce completes the sum.
            gh_for_gx = grad_hidden_local
            rank_for_gx = rank_local
        else:
            gh_for_gx = grad_hidden
            rank_for_gx = full_rank

        # gemm2 (expand): grad_x_sorted = grad_hidden_local @ A_shard -> [T, in]
        grad_x_sorted = torch.empty(T, in_features, device=x.device, dtype=x.dtype)
        _bgmv_expand_kernel[lambda m: (
            triton.cdiv(max_g_size, m["BLOCK_M"]),
            triton.cdiv(in_features, m["BLOCK_N"]), G)](
            gh_for_gx, lora_a_pool, grad_x_sorted,
            group_offsets, group_sizes, id_perm,
            T, K=rank_for_gx, N=in_features, G=G, scale=1.0,
        )
        if ctx.rank_partitioned and world_size > 1:
            dist.all_reduce(grad_x_sorted, op=dist.ReduceOp.SUM, group=tp_group)

        # grad_A = grad_hidden_local.T @ x -> [G, rank_local, in]
        grad_A = torch.zeros(G, rank_local, in_features, device=x.device, dtype=x.dtype)
        _bgmv_shrink_transpose_kernel[lambda m: (
            triton.cdiv(rank_local, m["BLOCK_M"]),
            triton.cdiv(in_features, m["BLOCK_N"]), G)](
            gh_for_gx, x_sorted, grad_A,
            group_offsets, group_sizes, id_perm,
            T, M=rank_local, N=in_features, G=G,
        )

        # grad_B = grad_out.T @ hidden -> [G, out_local, full_rank]; hidden recomputed.
        hidden = torch.empty(T, rank_local, device=x.device, dtype=x.dtype)
        _bgmv_shrink_kernel[lambda m: (
            triton.cdiv(max_g_size, m["BLOCK_M"]),
            triton.cdiv(rank_local, m["BLOCK_N"]), G)](
            x_sorted, lora_a_pool, hidden, group_offsets, group_sizes, id_perm,
            T, K=in_features, N=rank_local, G=G, scale=1.0,
        )
        if ctx.rank_partitioned and world_size > 1:
            hidden = _all_gather_last_dim_forward(hidden, tp_group, world_size)
        grad_B = torch.zeros(G, out_features, full_rank, device=x.device, dtype=x.dtype)
        _bgmv_shrink_transpose_kernel[lambda m: (
            triton.cdiv(out_features, m["BLOCK_M"]),
            triton.cdiv(full_rank, m["BLOCK_N"]), G)](
            grad_sorted, hidden, grad_B,
            group_offsets, group_sizes, id_perm,
            T, M=out_features, N=full_rank, G=G, scale=scale,
        )

        # unsort grad_x to gathered order, then take this rank's plain SP slice
        # (grad_x_sorted is TP-complete after all-reduce; reduce_scatter would
        # all-reduce AGAIN -> TPx scale).
        grad_x_gathered = torch.empty_like(grad_x_sorted).index_copy_(0, perm, grad_x_sorted)
        if ctx.sequence_parallel and world_size > 1:
            start = dist.get_rank(tp_group) * ctx.local_seq
            grad_x = grad_x_gathered[start:start + ctx.local_seq]
        else:
            grad_x = grad_x_gathered

        return grad_x, grad_A, grad_B, None, None, None, None, None


class _BgmvRowLoRA(torch.autograd.Function):
    """Triton BGMV for the row-parallel surface (TP>1, rank >= 16).

    Mirrors `_BatchedRowLoRA`: gemm1 partial + TP all-reduce, gemm2 with
    out-gather + sequence scatter.
    """

    @staticmethod
    def forward(ctx, x, lora_a_pool, lora_b_pool, lora_indices, scale, tp_group,
                sequence_parallel):
        import triton

        world_size = dist.get_world_size(tp_group)
        T = x.shape[0]
        G, rank, in_features = lora_a_pool.shape
        out_local = lora_b_pool.shape[1]

        sorted_indices, perm = torch.sort(lora_indices, stable=True)
        x_sorted = x[perm]
        group_offsets, group_sizes, G = _compute_group_offsets(sorted_indices, num_loras=G)
        max_g_size = T
        id_perm = torch.arange(T, device=x.device)
        _l2_prefetch(lora_a_pool, lora_b_pool)

        # gemm1: hidden_local = x_sorted @ A.T -> [T, rank] partial, then TP all-reduce
        hidden = torch.empty(T, rank, device=x.device, dtype=x.dtype)
        _bgmv_shrink_kernel[lambda m: (
            triton.cdiv(max_g_size, m["BLOCK_M"]),
            triton.cdiv(rank, m["BLOCK_N"]), G)](
            x_sorted, lora_a_pool, hidden,
            group_offsets, group_sizes, id_perm,
            T, K=in_features, N=rank, G=G, scale=1.0,
        )
        if world_size > 1:
            dist.all_reduce(hidden, op=dist.ReduceOp.SUM, group=tp_group)

        # gemm2: delta_local = scale * hidden @ B.T -> [T, out_local]
        delta_local_sorted = torch.empty(T, out_local, device=x.device, dtype=x.dtype)
        _bgmv_shrink_kernel[lambda m: (
            triton.cdiv(max_g_size, m["BLOCK_M"]),
            triton.cdiv(out_local, m["BLOCK_N"]), G)](
            hidden, lora_b_pool, delta_local_sorted,
            group_offsets, group_sizes, id_perm,
            T, K=rank, N=out_local, G=G, scale=scale,
        )
        delta_local = torch.empty_like(delta_local_sorted).index_copy_(0, perm, delta_local_sorted)

        delta_full = _all_gather_last_dim_forward(delta_local, tp_group, world_size)
        if sequence_parallel and world_size > 1:
            # delta_full is replicated across TP (out-gather); the SP-local
            # output is this rank's plain seq slice (reduce_scatter would
            # all-reduce AGAIN -> TPx scale).
            start = dist.get_rank(tp_group) * (T // world_size)
            delta = delta_full[start:start + T // world_size]
        else:
            delta = delta_full

        ctx.save_for_backward(x, lora_indices, lora_a_pool, lora_b_pool)
        ctx.tp_group = tp_group
        ctx.world_size = world_size
        ctx.sequence_parallel = bool(sequence_parallel)
        ctx.local_seq = T // world_size
        ctx.scale = float(scale)
        return delta

    @staticmethod
    def backward(ctx, grad_out):
        import triton

        x, lora_indices, lora_a_pool, lora_b_pool = ctx.saved_tensors
        tp_group = ctx.tp_group
        world_size = ctx.world_size
        scale = ctx.scale
        T = x.shape[0]
        G, rank, in_features = lora_a_pool.shape
        out_local = lora_b_pool.shape[1]

        if ctx.sequence_parallel and world_size > 1:
            grad_out = _all_gather_sequence_forward(grad_out, tp_group, world_size)
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        sorted_indices, perm = torch.sort(lora_indices, stable=True)
        x_sorted = x[perm]
        grad_sorted = grad_out[perm]
        group_offsets, group_sizes, G = _compute_group_offsets(sorted_indices, num_loras=G)
        max_g_size = T
        id_perm = torch.arange(T, device=x.device)

        # gemm1 (expand): grad_hidden = grad_out @ (scale*B) with the OUT dim
        # of the B pool gathered across TP. The kernel indexes the pool per
        # group (w[g] = [K, N]); per-token tensors are NOT a valid input here.
        b_pool_full = _all_gather_last_dim_forward(
            lora_b_pool.transpose(1, 2).contiguous(), tp_group, world_size
        ).transpose(1, 2).contiguous()  # [G, out, rank], row-major
        out_full = b_pool_full.shape[1]
        grad_hidden = torch.empty(T, rank, device=x.device, dtype=x.dtype)
        _bgmv_expand_kernel[lambda m: (
            triton.cdiv(max_g_size, m["BLOCK_M"]),
            triton.cdiv(rank, m["BLOCK_N"]), G)](
            grad_sorted, b_pool_full, grad_hidden,
            group_offsets, group_sizes, id_perm,
            T, K=out_full, N=rank, G=G, scale=scale,
        )

        # gemm2 (expand): grad_x_sorted = grad_hidden @ A -> [T, in_local]
        # A's rank dim is full (row surface): no TP all-reduce needed.
        grad_x_sorted = torch.empty(T, in_features, device=x.device, dtype=x.dtype)
        _bgmv_expand_kernel[lambda m: (
            triton.cdiv(max_g_size, m["BLOCK_M"]),
            triton.cdiv(in_features, m["BLOCK_N"]), G)](
            grad_hidden, lora_a_pool, grad_x_sorted,
            group_offsets, group_sizes, id_perm,
            T, K=rank, N=in_features, G=G, scale=1.0,
        )

        # grad_A = grad_hidden.T @ x -> [G, rank, in_local]
        grad_A = torch.zeros(G, rank, in_features, device=x.device, dtype=x.dtype)
        _bgmv_shrink_transpose_kernel[lambda m: (
            triton.cdiv(rank, m["BLOCK_M"]),
            triton.cdiv(in_features, m["BLOCK_N"]), G)](
            grad_hidden, x_sorted, grad_A,
            group_offsets, group_sizes, id_perm,
            T, M=rank, N=in_features, G=G,
        )

        # grad_B = grad_out.T @ hidden -> [G, out_local, rank]; hidden recomputed
        # (post all-reduce, matching forward's hidden).
        hidden = torch.empty(T, rank, device=x.device, dtype=x.dtype)
        _bgmv_shrink_kernel[lambda m: (
            triton.cdiv(max_g_size, m["BLOCK_M"]),
            triton.cdiv(rank, m["BLOCK_N"]), G)](
            x_sorted, lora_a_pool, hidden, group_offsets, group_sizes, id_perm,
            T, K=in_features, N=rank, G=G, scale=1.0,
        )
        if world_size > 1:
            dist.all_reduce(hidden, op=dist.ReduceOp.SUM, group=tp_group)
        grad_B = torch.zeros(G, out_local, rank, device=x.device, dtype=x.dtype)
        # grad_B uses this rank's out shard of grad_out (B's out dim is sharded).
        start_o = dist.get_rank(tp_group) * out_local if world_size > 1 else 0
        grad_sorted_local = grad_sorted[:, start_o:start_o + out_local].contiguous()
        _bgmv_shrink_transpose_kernel[lambda m: (
            triton.cdiv(out_local, m["BLOCK_M"]),
            triton.cdiv(rank, m["BLOCK_N"]), G)](
            grad_sorted_local, hidden, grad_B,
            group_offsets, group_sizes, id_perm,
            T, M=out_local, N=rank, G=G, scale=scale,
        )

        grad_x = torch.empty_like(grad_x_sorted).index_copy_(0, perm, grad_x_sorted)
        return grad_x, grad_A, grad_B, None, None, None, None


class _BgmvLinearLoRA(torch.autograd.Function):
    """Triton fused BGMV for the non-SP (tp=1) path.

    Replaces index_select+einsum+scale with the fused 1-kernel BGMV path;
    backward reuses saved hidden (no recompute).
    """

    @staticmethod
    def forward(ctx, x, lora_a_pool, lora_b_pool, lora_indices, scale):
        # R8: grid M dim = max group size, not T — per-group blocks beyond
        # the largest group are no-ops (over-launch = G-1/G of the grid when
        # groups are balanced). One host sync per call, amortized over the
        # kernels it saves.
        _max_g = int(torch.bincount(lora_indices, minlength=lora_a_pool.shape[0]).max().item())
        delta, hidden, perm = bgmv_fwd(
            x, lora_a_pool, lora_b_pool, lora_indices, scale,
            max_g_size_hint=max(_max_g, 1),
        )
        ctx.save_for_backward(x, lora_a_pool, lora_b_pool, lora_indices)
        ctx.hidden = hidden
        ctx.perm = perm
        ctx.scale = scale
        return delta

    @staticmethod
    def backward(ctx, grad_out):
        x, lora_a_pool, lora_b_pool, lora_indices = ctx.saved_tensors
        grad_x, grad_a, grad_b = bgmv_bwd(
            x, grad_out, lora_a_pool, lora_b_pool, lora_indices, ctx.scale,
            hidden=ctx.hidden, perm=ctx.perm,
        )
        return grad_x, grad_a, grad_b, None, None


class _BatchedLinearNoComm(torch.autograd.Function):
    """Torch path for duplicated (non-parallel) surfaces: plain fused BGMV
    math without any TP/SP comms (each rank holds the full weights)."""

    @staticmethod
    def forward(ctx, x, lora_a_pool, lora_b_pool, lora_indices, scale):
        a_gathered = lora_a_pool.index_select(0, lora_indices)
        b_gathered = lora_b_pool.index_select(0, lora_indices)
        hidden = torch.einsum("tri,ti->tr", a_gathered.float(), x.float())
        delta = torch.einsum("tor,tr->to", b_gathered.float(), hidden)
        ctx.save_for_backward(x, lora_indices, lora_a_pool, lora_b_pool)
        ctx.scale = float(scale)
        return (scale * delta).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        x, lora_indices, lora_a_pool, lora_b_pool = ctx.saved_tensors
        go = grad_out.float() * ctx.scale
        a_gathered = lora_a_pool.index_select(0, lora_indices)
        b_gathered = lora_b_pool.index_select(0, lora_indices)
        hidden = torch.einsum("tri,ti->tr", a_gathered.float(), x.float())
        grad_hidden = torch.einsum("to,tor->tr", go, b_gathered.float())
        grad_a = _segment_scatter_sorted(
            torch.einsum("tr,ti->tri", grad_hidden, x.float()),
            lora_indices, lora_a_pool.shape[0],
            (lora_a_pool.shape[1], lora_a_pool.shape[2]),
        ).to(lora_a_pool.dtype)
        grad_b = _segment_scatter_sorted(
            torch.einsum("to,tr->tor", go, hidden),
            lora_indices, lora_b_pool.shape[0],
            (lora_b_pool.shape[1], lora_b_pool.shape[2]),
        ).to(lora_b_pool.dtype)
        grad_x = torch.einsum("tr,tri->ti", grad_hidden, a_gathered.float()).to(x.dtype)
        return grad_x, grad_a, grad_b, None, None


# ── stateless module ───────────────────────────────────────────────────────


class BatchedLoraAdapter(torch.nn.Module):
    """Multi-tenant LoRA delta module for one sharded linear surface.

    Holds no Parameters: the caller supplies the 3D pool slices
    (``lora_a_pool`` / ``lora_b_pool``) and per-token ``lora_indices`` at
    forward time. The pool lives in the mint engine.

    Constructor args mirror mbridge's ``ParallelLinearAdapter`` so a
    ``CanonicalLoRA`` transform subclass can construct this module from
    ``get_adapter_attributes_from_linear`` output:

        input_is_parallel=False -> column-parallel surface
        input_is_parallel=True  -> row-parallel surface
        disable_sequence_parallel_comm -> skip SP gather/scatter when the base
            config has sequence_parallel off.

    Args:
        in_features: full input dim of the base linear.
        out_features: full output dim of the base linear.
        dim: full LoRA rank.
        input_is_parallel: base is row-parallel.
        is_expert: expert linear (not supported on GLM-5.2; kept for signature
            parity with mbridge).
        disable_sequence_parallel_comm: base config sequence_parallel is off.
        model_parallel_config: used only to read sequence_parallel.
        alpha: scale = alpha / dim.
        dropout: must be 0 (batched dropout is not part of the contract).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        base_linear_name: str = "",
        *,
        input_is_parallel: bool = False,
        is_expert: bool = False,
        disable_sequence_parallel_comm: bool = True,
        base_linear_is_parallel: bool = True,
        model_parallel_config=None,
        alpha: float | None = None,
        dropout: float = 0.0,
        tp_group=None,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError("LoRA rank must be positive for BatchedLoraAdapter.")
        # The fused BGMV path compiles tl.arange(0, RANK) — reject ranks above
        # its ceiling only where the fused path would actually run (rank >=
        # _MIN_BGMV_RANK, and local rank >= _MIN_BGMV_RANK for partitioned-A
        # surfaces). The torch fallback materializes [T, rank/tp, in], which
        # TP>1 sharding keeps bounded — a large rank on TP>1 is legal there.
        if dim >= _MIN_BGMV_RANK and dim > _MAX_BGMV_RANK:
            raise ValueError(
                f"LoRA rank {dim} exceeds the BGMV kernel limit {_MAX_BGMV_RANK} "
                "on a path where the fused kernel would run; use a smaller "
                "rank or TP such that the torch fallback's [T, rank/tp, in] "
                "materialization stays acceptable."
            )
        if dropout != 0.0:
            raise NotImplementedError("batched LoRA dropout is not supported.")
        if is_expert:
            raise NotImplementedError(
                "expert LoRA surfaces are not part of the GLM-5.2 batch-LoRA contract."
            )
        self.base_linear_name = base_linear_name
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(dim)
        self.input_is_parallel = bool(input_is_parallel)
        # Duplicated (non-parallel) base linears (e.g. AbsorbedMLA q/kv down
        # projections with parallel_mode="duplicated") hold the FULL weight on
        # every TP rank: no sharding, no gather, no TP comm for the delta.
        self.base_linear_is_parallel = bool(base_linear_is_parallel)
        self.alpha = alpha if alpha is not None else self.rank
        self.tp_group = tp_group
        self.world_size = (
            dist.get_world_size(tp_group) if tp_group is not None else 1
        )
        # CP context for per-token routing alignment: the engine passes
        # GLOBAL-order lora_indices ([T_total] in datum order); the op slices
        # this rank's TP (and, for row surfaces, CP) shard internally.
        try:
            from megatron.core import parallel_state as _ps
            self.cp_world = _ps.get_context_parallel_world_size()
            self.cp_rank = _ps.get_context_parallel_rank()
            self.tp_rank = _ps.get_tensor_model_parallel_rank()
        except Exception:
            self.cp_world = 1
            self.cp_rank = 0
            self.tp_rank = 0
        if model_parallel_config is not None:
            sequence_parallel = getattr(model_parallel_config, "sequence_parallel", False)
        else:
            sequence_parallel = False
        self.sequence_parallel = bool(sequence_parallel) and not bool(
            disable_sequence_parallel_comm
        )
        # Stable pool key; assigned by the engine's surface scan.
        self._pool_name: str = ""
        if self.world_size > 1:
            if self.rank % self.world_size != 0:
                raise ValueError(
                    f"LoRA rank {self.rank} must be divisible by tp size {self.world_size}."
                )
            if self.out_features % self.world_size != 0:
                raise ValueError(
                    f"LoRA out_features {self.out_features} must be divisible by "
                    f"tp size {self.world_size}."
                )
            if self.input_is_parallel and self.in_features % self.world_size != 0:
                raise ValueError(
                    f"LoRA in_features {self.in_features} must be divisible by "
                    f"tp size {self.world_size} for row-parallel surfaces (A is "
                    "in-sharded; a non-divisible size truncates silently)."
                )
        # Local shard shapes (what the engine must allocate):
        # column: A [N, rank/tp, in], B [N, out/tp, rank]
        # row:    A [N, rank, in/tp], B [N, out/tp, rank]
        # duplicated: A [N, rank, in], B [N, out, rank] (full, per rank)
        if not self.base_linear_is_parallel:
            self.local_a_shape = (self.rank, self.in_features)
            self.local_b_shape = (self.out_features, self.rank)
            self.local_out_features = self.out_features
            self.local_rank = self.rank
        elif self.input_is_parallel:
            self.local_a_shape = (self.rank, self.in_features // self.world_size)
            self.local_b_shape = (self.out_features // self.world_size, self.rank)
            self.local_out_features = self.out_features
            self.local_rank = self.rank
        else:
            self.local_a_shape = (self.rank // self.world_size, self.in_features)
            self.local_b_shape = (self.out_features // self.world_size, self.rank)
            self.local_out_features = self.out_features // self.world_size
            self.local_rank = self.rank // self.world_size

    @property
    def rank_partitioned_a(self) -> bool:
        return (
            (not self.input_is_parallel)
            and self.world_size > 1
            and self.base_linear_is_parallel
        )

    def _align_indices(
        self,
        lora_indices: torch.Tensor,
        t: int,
        lora_indices_padded: Optional[torch.Tensor] = None,
        cu_seqlens_padded: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Align engine-supplied GLOBAL-order indices to this rank's x layout.

        With ``lora_indices_padded`` + ``cu_seqlens_padded`` (the padded
        global stream verl's preprocess_thd_engine builds: every datum padded
        to a multiple of tp*cp*2, cp_layout="zigzag"), this rank's rows are
        recovered from the true layout: per datum, the rank-local front chunk
        followed by the mirrored back chunk (datums in stream order). Column
        surfaces additionally hold a contiguous SP shard of that CP chunk.

        The legacy path (no padded metadata) assumes each CP rank holds a
        CONTIGUOUS chunk of the stream — wrong for the zigzag THD layout; it
        misassigns adapter slots whenever a merged stream mixes sessions.
        Kept only for callers without CP or without padded metadata.
        """
        if (
            lora_indices_padded is not None
            and cu_seqlens_padded is not None
            and self.cp_world > 1
        ):
            cu = cu_seqlens_padded.to(device=lora_indices_padded.device, dtype=torch.int64)
            seq_lens = cu[1:] - cu[:-1]
            half = seq_lens // (2 * self.cp_world)
            # Per datum: [front_start, front_end), [back_start, back_end) —
            # the local stream is front_0, back_0, front_1, back_1, ...
            seg = torch.stack(
                [
                    torch.stack([cu[:-1] + self.cp_rank * half,
                                  cu[:-1] + (self.cp_rank + 1) * half], dim=1),
                    torch.stack([cu[1:] - (self.cp_rank + 1) * half,
                                 cu[1:] - self.cp_rank * half], dim=1),
                ],
                dim=1,
            ).reshape(-1, 2)
            lens = seg[:, 1] - seg[:, 0]
            keep = lens > 0
            seg = seg[keep]
            lens = lens[keep]
            if seg.numel() == 0:
                return lora_indices_padded[:0]
            max_len = int(lens.max().item())
            ar = torch.arange(max_len, device=lens.device).unsqueeze(0).expand(seg.shape[0], -1)
            mask = ar < lens.unsqueeze(1)
            positions = (seg[:, 0:1] + ar)[mask].reshape(-1)
            cp_chunk = lora_indices_padded.index_select(0, positions)
            if t == cp_chunk.shape[0]:
                return cp_chunk
            # Column surface: this TP rank's contiguous SP shard of the CP
            # chunk (the op re-gathers the full chunk across TP internally).
            if t * self.world_size == cp_chunk.shape[0]:
                base = self.tp_rank * t
                return cp_chunk[base : base + t]
            raise ValueError(
                f"zigzag lora_indices alignment mismatch: surface={self._pool_name} "
                f"t={t} cp_chunk={cp_chunk.shape[0]} tp={self.world_size} "
                f"cp={self.cp_world} T_pad={int(cu[-1].item())}"
            )

        tp_world = self.world_size
        cp_world = max(1, self.cp_world)
        if lora_indices.shape[0] == t:
            return lora_indices
        if self.input_is_parallel:
            # row surface: x is the full CP chunk (t tokens); the global stream
            # is t * cp_world tokens.
            T_pad = t * cp_world
            if lora_indices.shape[0] != T_pad:
                if lora_indices.shape[0] > T_pad:
                    raise ValueError(
                        f"lora_indices len {lora_indices.shape[0]} exceeds "
                        f"row-surface padded stream {T_pad}"
                    )
                lora_indices = torch.nn.functional.pad(
                    lora_indices, (0, T_pad - lora_indices.shape[0]), value=0
                )
            base = self.cp_rank * t
            return lora_indices[base:base + t]
        # column surface: x is the SP-local shard of the CP chunk when SP is
        # on; with SP off (disable_sequence_parallel_comm + TP>1) x is the
        # FULL CP chunk (t = T_pad / cp, no TP sharding).
        if self.sequence_parallel:
            seg = t * tp_world
            T_pad = seg * cp_world
            if lora_indices.shape[0] != T_pad:
                if lora_indices.shape[0] > T_pad:
                    raise ValueError(
                        f"lora_indices len {lora_indices.shape[0]} exceeds "
                        f"column-surface padded stream {T_pad} "
                        f"(surface={self._pool_name} t={t} tp={tp_world} "
                        f"cp={cp_world} sp={self.sequence_parallel})"
                    )
                lora_indices = torch.nn.functional.pad(
                    lora_indices, (0, T_pad - lora_indices.shape[0]), value=0
                )
            base = self.cp_rank * seg + self.tp_rank * t
            return lora_indices[base:base + t]
        # SP off. Two x layouts exist:
        #   (a) full CP chunk (t = T/cp, e.g. TE gathered layernorm output)
        #       -> global stream is t * cp;
        #   (b) SP-local shard (t = T/(tp*cp), DSA surfaces whose mbridge
        #       attrs say comm-disabled but whose base module still feeds
        #       the SP-local input) -> global stream is t * tp * cp.
        # Probe at runtime: if the non-SP padded stream cannot hold the
        # global indices, x must be the SP-local shard.
        T_pad = t * cp_world
        sp_local = lora_indices.shape[0] > T_pad
        if sp_local:
            T_pad = t * tp_world * cp_world
        if lora_indices.shape[0] != T_pad:
            if lora_indices.shape[0] > T_pad:
                raise ValueError(
                    f"lora_indices len {lora_indices.shape[0]} exceeds "
                    f"column-surface padded stream {T_pad} "
                    f"(surface={self._pool_name} t={t} tp={tp_world} "
                    f"cp={cp_world} sp={self.sequence_parallel} "
                    f"sp_local_probe={sp_local})"
                )
            lora_indices = torch.nn.functional.pad(
                lora_indices, (0, T_pad - lora_indices.shape[0]), value=0
            )
        if sp_local:
            base = self.cp_rank * (t * tp_world) + self.tp_rank * t
            return lora_indices[base:base + t]
        base = self.cp_rank * t
        return lora_indices[base:base + t]

    def forward(
        self,
        x: torch.Tensor,
        lora_a_pool: torch.Tensor,
        lora_b_pool: torch.Tensor,
        lora_indices: torch.Tensor,
        scale: float | None = None,
        *,
        lora_indices_padded: Optional[torch.Tensor] = None,
        cu_seqlens_padded: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the batched LoRA delta.

        Args:
            x: [..., in] input; leading dims reshaped to 2D internally. Layout
               per surface kind (SP-local for column, full-seq for row).
            lora_a_pool / lora_b_pool: [N, ...] 3D pool slices (local shards).
            lora_indices: [T] int64, adapter per token (any order).
            scale: alpha / rank; defaults to self.alpha / self.rank.
            lora_indices_padded / cu_seqlens_padded: zigzag-CP padded stream
               metadata (see ``_align_indices``); required for correct
               per-token slot alignment when cp > 1 on packed THD streams.

        Returns:
            delta with the mbridge-matching layout (full-seq for column,
            SP-local for row when sequence_parallel).
        """
        if scale is None:
            scale = self.alpha / self.rank
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.size(-1))
        t = x_2d.shape[0]
        lora_indices = self._align_indices(
            lora_indices, t, lora_indices_padded, cu_seqlens_padded
        )
        num_slots = lora_a_pool.shape[0]
        if lora_indices.numel() and (
            int(lora_indices.min().item()) < 0
            or int(lora_indices.max().item()) >= num_slots
        ):
            raise ValueError(
                f"lora_indices out of range [0, {num_slots}): min="
                f"{lora_indices.min().item()}, max={lora_indices.max().item()}"
            )

        use_bgmv = (
            x.is_cuda
            and _bgmv_available()
            and (self.rank >= _MIN_BGMV_RANK)
            and (self.local_rank >= _MIN_BGMV_LOCAL_RANK if self.rank_partitioned_a else True)
        )
        if not self.base_linear_is_parallel:
            # Duplicated surface: full weights on every rank, no TP comm.
            if use_bgmv:
                delta = _BgmvLinearLoRA.apply(
                    x_2d, lora_a_pool, lora_b_pool, lora_indices, scale
                )
            else:
                delta = _BatchedLinearNoComm.apply(
                    x_2d, lora_a_pool, lora_b_pool, lora_indices, scale
                )
        elif use_bgmv:
            if self.input_is_parallel:
                if self.world_size > 1:
                    delta = _BgmvRowLoRA.apply(
                        x_2d, lora_a_pool, lora_b_pool, lora_indices, scale,
                        self.tp_group, self.sequence_parallel,
                    )
                else:
                    delta = _BgmvLinearLoRA.apply(
                        x_2d, lora_a_pool, lora_b_pool, lora_indices, scale
                    )
            else:
                if self.world_size > 1 or self.sequence_parallel:
                    delta = _BgmvColumnLoRA.apply(
                        x_2d, lora_a_pool, lora_b_pool, lora_indices, scale,
                        self.tp_group, self.sequence_parallel, self.rank_partitioned_a,
                    )
                else:
                    delta = _BgmvLinearLoRA.apply(
                        x_2d, lora_a_pool, lora_b_pool, lora_indices, scale
                    )
        else:
            if self.input_is_parallel:
                delta = _BatchedRowLoRA.apply(
                    x_2d, lora_a_pool, lora_b_pool, lora_indices, scale,
                    self.tp_group, self.sequence_parallel,
                )
            else:
                delta = _BatchedColumnLoRA.apply(
                    x_2d, lora_a_pool, lora_b_pool, lora_indices, scale,
                    self.tp_group, self.sequence_parallel, self.rank_partitioned_a,
                )
        return delta.view(*orig_shape[:-1], -1) if x.dim() > 2 else delta


def batched_lora_delta(lora_module: BatchedLoraAdapter, x: torch.Tensor, lora_extras: dict):
    """Resolve pool slices from lora_extras and call the adapter.

    lora_extras keys: ``lora_a_pool`` (dict[name -> [N, ...]]),
    ``lora_b_pool`` (dict), ``lora_indices`` ([T] int64), ``lora_scale`` (float).
    """
    assert lora_extras is not None, "multi-tenant LoRA requires lora_extras."
    name = getattr(lora_module, "_pool_name")
    return lora_module(
        x,
        lora_extras["lora_a_pool"][name],
        lora_extras["lora_b_pool"][name],
        lora_extras["lora_indices"],
        lora_extras["lora_scale"],
    )


# ── Expert batched LoRA (grouped expert linears) ───────────────────────────
#
# Sharing-direction contract (tinker-ref #1092 / SGLang PR #21466
# experts_shared_outer_loras): the HIDDEN-side factor is shared across
# experts and the INTERMEDIATE-side factor is per-expert. Per surface
# (ETP=1, N = pool slots, E = num_local_experts):
#
#   fc2 / down_proj  (in=inter, out=hidden):
#     A_pool [N, E, rank, in]    per-(slot, expert)   (intermediate side)
#     B_pool [N, out, rank]      shared               (hidden side)
#     delta[t] = scale * B[slot_t] @ (A[slot_t, e_t] @ x[t])
#
#   fc1 / gate_up    (in=hidden, out=2*inter):
#     A_pool [N, rank, in]       shared               (hidden side)
#     B_pool [N, E, out, rank]   per-(slot, expert)   (intermediate side)
#     delta[t] = scale * B[slot_t, e_t] @ (A[slot_t] @ x[t])
#
# The expert linears see rows in the MoE dispatch layout: grouped by local
# expert (tokens_per_expert block boundaries), where each row belongs to a
# (token, expert) pair of some rank in the CP/EP world. Per-row slots are
# relayed by ExpertSlotRelay: the engine arms it with this rank's CP-chunk
# slot tensor; the token dispatcher mirrors its permutation and A2A onto the
# slots; the expert forward then reads per-row slots aligned with x.
#

class ExpertSlotRelay:
    """Relay of per-token LoRA slots through the MoE token dispatcher.

    The dense surfaces read per-token slots directly from the batch context
    (the op slices its own CP/TP shard). The expert path cannot: after the
    dispatcher's permutation + expert-parallel A2A, a row's originating
    token (and therefore its slot) lives on another rank. The engine arms
    the relay with this rank's CP-chunk slots; MoEAlltoAllTokenDispatcher
    mirrors its exact permutation and A2A onto the slot vector; the expert
    adapter reads the result aligned 1:1 with its input rows.

    Slots are int64 routing metadata computed under no_grad; they never
    participate in autograd (the delta op's grad_x flows through its own
    math, not through slot selection).
    """

    ARMED: bool = False
    LOCAL_SLOTS: Optional[torch.Tensor] = None  # [t_cp] slots of this rank's CP chunk
    # Relay state written by the dispatcher hooks (per MoE layer forward):
    _PERM1: Optional[torch.Tensor] = None       # local slots after permutation 1
    CURRENT: Optional[torch.Tensor] = None      # [S] slots aligned with expert rows

    @staticmethod
    def derive_perm1_from_fused_map(
        row_id_map: torch.Tensor, local_slots: torch.Tensor, num_out: int
    ) -> torch.Tensor:
        """Derive the permuted slot vector from the fused permute's map.

        TE's moe_permute_mask_map (fused_permute_with_probs) returns
        row_id_map [S, 2E+1] int32. Per token t: [t, 2E] = n_routed,
        [t, :n_routed] = destination rows in the permuted tensor,
        [t, E:E+n_routed] = the expert ids for those rows. Entries beyond
        n_routed are UNDEFINED and never read. The routed pairs fully
        describe the permutation, so no row-ordering assumption is needed.

        The full-coverage assert fail-fasts if a future fused variant
        (e.g. a padded layout where permuted rows exceed routed pairs)
        reaches this code.
        """
        S, width = row_id_map.shape
        num_experts = (width - 1) // 2
        n_routed = row_id_map[:, 2 * num_experts]
        take = (
            torch.arange(num_experts, device=row_id_map.device).unsqueeze(0)
            < n_routed.unsqueeze(1)
        )  # [S, E] k-th routed slot of each token
        rows = row_id_map[:, :num_experts][take].long()
        toks = take.nonzero(as_tuple=False)[:, 0]
        if (
            rows.numel() != num_out
            or (rows.numel() and (rows.min() < 0 or rows.max() >= num_out))
        ):
            raise RuntimeError(
                "expert slot relay: fused row_id_map does not describe a "
                f"valid permutation of {num_out} rows (padded fused layouts "
                "and map-contract violations fail fast here)"
            )
        perm1 = local_slots.new_full((num_out,), -1)
        perm1.index_copy_(0, rows, local_slots[toks])
        if bool((perm1 < 0).any()):
            raise RuntimeError(
                "expert slot relay: fused row_id_map does not cover all "
                f"{num_out} permuted rows"
            )
        return perm1

    @staticmethod
    def arm(local_slots: torch.Tensor) -> None:
        ExpertSlotRelay.ARMED = True
        ExpertSlotRelay.LOCAL_SLOTS = local_slots
        ExpertSlotRelay._PERM1 = None
        ExpertSlotRelay.CURRENT = None

    @staticmethod
    def disarm() -> None:
        ExpertSlotRelay.ARMED = False
        ExpertSlotRelay.LOCAL_SLOTS = None
        ExpertSlotRelay._PERM1 = None
        ExpertSlotRelay.CURRENT = None

    @staticmethod
    def take_current(num_rows: int) -> Optional[torch.Tensor]:
        """Consume the relayed slots for the expert forward (fail-fast)."""
        slots = ExpertSlotRelay.CURRENT
        if slots is None:
            return None
        if slots.shape[0] != num_rows:
            raise RuntimeError(
                f"expert slot relay misaligned: {slots.shape[0]} slots vs "
                f"{num_rows} expert rows — dispatcher relay and expert forward "
                "disagree (dispatcher type or fused-path mismatch?)"
            )
        return slots


def _cp_chunk_row_indices(
    lora_indices_padded: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    cp_rank: int,
    cp_world: int,
) -> torch.Tensor:
    """This rank's full-CP-chunk view of the padded global slot stream.

    Mirrors the row-surface branch of ``BatchedLoraAdapter._align_indices``:
    per datum the local front chunk followed by the mirrored back chunk,
    datums in stream order. The MoE layer consumes exactly this stream.
    """
    cu = cu_seqlens_padded.to(device=lora_indices_padded.device, dtype=torch.int64)
    seq_lens = cu[1:] - cu[:-1]
    half = seq_lens // (2 * cp_world)
    seg = torch.stack(
        [
            torch.stack([cu[:-1] + cp_rank * half, cu[:-1] + (cp_rank + 1) * half], dim=1),
            torch.stack([cu[1:] - (cp_rank + 1) * half, cu[1:] - cp_rank * half], dim=1),
        ],
        dim=1,
    ).reshape(-1, 2)
    lens = seg[:, 1] - seg[:, 0]
    keep = lens > 0
    seg, lens = seg[keep], lens[keep]
    if seg.numel() == 0:
        return lora_indices_padded[:0]
    max_len = int(lens.max().item())
    ar = torch.arange(max_len, device=lens.device).unsqueeze(0).expand(seg.shape[0], -1)
    positions = (seg[:, 0:1] + ar)[ar < lens.unsqueeze(1)].reshape(-1)
    return lora_indices_padded.index_select(0, positions)


class _ExpertSharedOuterLoRA(torch.autograd.Function):
    """fc2/down expert LoRA delta over dispatch-layout rows (B shared).

    Rows are grouped by local expert (tpe block boundaries). Within a block
    the delta for row t is scale * B[slot_t] @ A[slot_t, e] @ x_t. The math
    is evaluated per expert block with an N-broadcast einsum + diagonal
    select (intermediates [S_e, N, rank] stay small — N is the pool size),
    avoiding any [S, rank, in] materialization.
    """

    @staticmethod
    def forward(ctx, x, a_pool, b_pool, slot_rows, expert_rows, scale):
        # x: [S, in]; a_pool: [N, E, rank, in]; b_pool: [N, out, rank];
        # slot_rows/expert_rows: [S] int64; returns delta [S, out].
        S, _ = x.shape
        N, E, rank, _ = a_pool.shape
        out = b_pool.shape[1]
        delta = torch.empty(S, out, device=x.device, dtype=x.dtype)
        hidden_all = torch.empty(S, rank, device=x.device, dtype=torch.float32)
        x_f = x.float()
        a_f = a_pool.float()
        b_f = b_pool.float()
        for e in range(E):
            rows = expert_rows == e
            if not bool(rows.any()):
                continue
            xe = x_f[rows]                      # [S_e, in]
            se = slot_rows[rows]                # [S_e]
            a_e = a_f[:, e]                     # [N, rank, in]
            h_n = torch.einsum("nri,bi->bnr", a_e, xe)  # [S_e, N, rank]
            h = h_n.gather(1, se.view(-1, 1, 1).expand(-1, 1, rank)).squeeze(1)
            hidden_all[rows] = h
            b_n = torch.einsum("nor,bnr->bno", b_f, h_n)  # [S_e, N, out]
            d = b_n.gather(1, se.view(-1, 1, 1).expand(-1, 1, out)).squeeze(1)
            delta[rows] = (d * scale).to(x.dtype)
        ctx.save_for_backward(x, slot_rows, expert_rows, hidden_all, a_pool, b_pool)
        ctx.scale = float(scale)
        return delta

    @staticmethod
    def backward(ctx, grad_out):
        x, slot_rows, expert_rows, hidden_all, a_pool, b_pool = ctx.saved_tensors
        scale = ctx.scale
        N, E, rank, _ = a_pool.shape
        out = b_pool.shape[1]
        grad_a = torch.zeros_like(a_pool) if a_pool.requires_grad else None
        grad_b = torch.zeros_like(b_pool) if b_pool.requires_grad else None
        grad_x = None
        go_f = grad_out.float() * scale
        b_f = b_pool.float()
        gx_f = torch.zeros_like(x, dtype=torch.float32) if ctx.needs_input_grad[0] else None
        # Chunk budget for row-temporaries: [c, out, rank] fp32 must stay
        # ~<= 1 GiB (at 96K packed tokens the unchunked [S_e, out, rank]
        # reach tens of GiB and OOM the pass).
        chunk = max(128, (1 << 30) // max(1, out * rank * 4))
        if _LORA_DEBUG:
            print(f"[expert-outer.bwd] S={x.shape[0]} in={x.shape[1]} "
                  f"N={a_pool.shape[0]} E={E} out={out} rank={rank} "
                  f"chunk={chunk}", flush=True)
        for e in range(E):
            rows_idx = (expert_rows == e).nonzero().flatten()
            S_e = rows_idx.shape[0]
            if S_e == 0:
                continue
            xe = x[rows_idx].float()               # [S_e, in]
            se = slot_rows[rows_idx]
            he = hidden_all[rows_idx]              # [S_e, rank]
            # gh[t] = B[slot_t] @ go_f[t], chunked over rows so the gather
            # b_f[se] never materializes [S_e, out, rank] in one piece.
            gh = torch.empty(S_e, rank, device=x.device, dtype=torch.float32)
            for s0 in range(0, S_e, chunk):
                sl = slice(s0, min(s0 + chunk, S_e))
                gh[sl] = torch.einsum(
                    "sor,so->sr", b_f[se[sl]], go_f[rows_idx[sl]]
                )
            if _LORA_DEBUG:
                print(f"[expert-outer.bwd] e={e} S_e={S_e}", flush=True)
            if grad_b is not None:
                for s0 in range(0, S_e, chunk):
                    idx = rows_idx[s0 : s0 + chunk]
                    gb = torch.einsum(
                        "so,sr->sor", go_f[idx], hidden_all[idx]
                    )  # [c, out, rank]
                    grad_b.index_add_(0, slot_rows[idx], gb.to(grad_b.dtype))
            if grad_a is not None:
                # d(hidden)/d(A[slot,e]) = grad_hidden ⊗ x; grad_hidden per row
                # is B[slot] @ grad_out^T (same for all experts — shared B).
                for s0 in range(0, S_e, chunk):
                    idx = rows_idx[s0 : s0 + chunk]
                    ga = torch.einsum("sr,si->sri", gh[s0 : s0 + chunk], xe[s0 : s0 + chunk])
                    flat_slot = slot_rows[idx] * E + e
                    grad_a.view(-1, *grad_a.shape[2:]).index_add_(
                        0, flat_slot, ga.to(grad_a.dtype)
                    )
            if gx_f is not None:
                a_e = a_pool.float()[:, e]  # [N, rank, in]
                for s0 in range(0, S_e, chunk):
                    idx = rows_idx[s0 : s0 + chunk]
                    sl = slice(s0, min(s0 + chunk, S_e))
                    gx_n = torch.einsum("nri,br->bni", a_e, gh[sl])  # [c, N, in]
                    in_dim = x.size(-1)
                    gx_f[idx] = gx_n.gather(
                        1, se[sl].view(-1, 1, 1).expand(-1, 1, in_dim)
                    ).squeeze(1)
        if gx_f is not None:
            grad_x = gx_f.to(x.dtype)
        return grad_x, grad_a, grad_b, None, None, None


class _ExpertSharedInnerLoRA(torch.autograd.Function):
    """fc1/gate_up expert LoRA delta over dispatch-layout rows (A shared).

    Sharing-direction contract (tinker-ref #1092): the hidden-side factor A
    is shared across experts, the intermediate-side factor B is per-expert.
    Within an expert block the delta for row t is
    scale * B[slot_t, e] @ (A[slot_t] @ x_t). Same N-broadcast einsum +
    diagonal-select structure as the shared-B op; grad_a accumulates over
    SLOTS only (A is shared across expert blocks).
    """

    @staticmethod
    def forward(ctx, x, a_pool, b_pool, slot_rows, expert_rows, scale):
        # x: [S, in]; a_pool: [N, rank, in]; b_pool: [N, E, out, rank];
        # slot_rows/expert_rows: [S] int64; returns delta [S, out].
        S, _ = x.shape
        N, rank, _ = a_pool.shape
        E, out = b_pool.shape[1], b_pool.shape[2]
        delta = torch.empty(S, out, device=x.device, dtype=x.dtype)
        hidden_all = torch.empty(S, rank, device=x.device, dtype=torch.float32)
        x_f = x.float()
        a_f = a_pool.float()
        b_f = b_pool.float()
        for e in range(E):
            rows = expert_rows == e
            if not bool(rows.any()):
                continue
            xe = x_f[rows]                      # [S_e, in]
            se = slot_rows[rows]                # [S_e]
            h_n = torch.einsum("nri,bi->bnr", a_f, xe)  # [S_e, N, rank]
            h = h_n.gather(1, se.view(-1, 1, 1).expand(-1, 1, rank)).squeeze(1)
            hidden_all[rows] = h
            b_e = b_f[:, e]                     # [N, out, rank]
            d_n = torch.einsum("nor,bnr->bno", b_e, h_n)  # [S_e, N, out]
            d = d_n.gather(1, se.view(-1, 1, 1).expand(-1, 1, out)).squeeze(1)
            delta[rows] = (d * scale).to(x.dtype)
        ctx.save_for_backward(x, slot_rows, expert_rows, hidden_all, a_pool, b_pool)
        ctx.scale = float(scale)
        return delta

    @staticmethod
    def backward(ctx, grad_out):
        x, slot_rows, expert_rows, hidden_all, a_pool, b_pool = ctx.saved_tensors
        scale = ctx.scale
        N, rank, _ = a_pool.shape
        E, out = b_pool.shape[1], b_pool.shape[2]
        grad_a = torch.zeros_like(a_pool) if a_pool.requires_grad else None
        grad_b = torch.zeros_like(b_pool) if b_pool.requires_grad else None
        grad_x = None
        go_f = grad_out.float() * scale
        a_f = a_pool.float()
        b_f = b_pool.float()
        gx_f = torch.zeros_like(x, dtype=torch.float32) if ctx.needs_input_grad[0] else None
        chunk = max(128, (1 << 30) // max(1, out * rank * 4))
        if _LORA_DEBUG:
            print(f"[expert-inner.bwd] S={x.shape[0]} in={x.shape[1]} "
                  f"N={N} E={E} out={out} rank={rank} chunk={chunk}", flush=True)
        for e in range(E):
            rows_idx = (expert_rows == e).nonzero().flatten()
            S_e = rows_idx.shape[0]
            if S_e == 0:
                continue
            xe = x[rows_idx].float()               # [S_e, in]
            se = slot_rows[rows_idx]
            he = hidden_all[rows_idx]              # [S_e, rank]
            # gh[t] = B[slot_t, e] @ go_f[t], chunked over rows so the gather
            # b_f[se, e] never materializes [S_e, out, rank] in one piece.
            gh = torch.empty(S_e, rank, device=x.device, dtype=torch.float32)
            for s0 in range(0, S_e, chunk):
                sl = slice(s0, min(s0 + chunk, S_e))
                gh[sl] = torch.einsum(
                    "sor,so->sr", b_f[se[sl], e], go_f[rows_idx[sl]]
                )
            if grad_b is not None:
                # d/dB[slot, e] = grad_out ⊗ h — only rows routed to expert e
                for s0 in range(0, S_e, chunk):
                    idx = rows_idx[s0 : s0 + chunk]
                    gb = torch.einsum(
                        "so,sr->sor", go_f[idx], hidden_all[idx]
                    )  # [c, out, rank]
                    flat_slot = slot_rows[idx] * E + e
                    grad_b.view(-1, out, rank).index_add_(
                        0, flat_slot, gb.to(grad_b.dtype)
                    )
            if grad_a is not None:
                # grad_hidden[t] = B[slot_t, e_t] @ grad_out[t] (per-expert B)
                for s0 in range(0, S_e, chunk):
                    idx = rows_idx[s0 : s0 + chunk]
                    ga = torch.einsum("sr,si->sri", gh[s0 : s0 + chunk], xe[s0 : s0 + chunk])
                    # A is SHARED across experts: accumulate over slots only
                    grad_a.index_add_(0, slot_rows[idx], ga.to(grad_a.dtype))
            if gx_f is not None:
                for s0 in range(0, S_e, chunk):
                    idx = rows_idx[s0 : s0 + chunk]
                    sl = slice(s0, min(s0 + chunk, S_e))
                    gx_n = torch.einsum("nri,br->bni", a_f, gh[sl])  # [c, N, in]
                    in_dim = x.size(-1)
                    gx_f[idx] = gx_n.gather(
                        1, se[sl].view(-1, 1, 1).expand(-1, 1, in_dim)
                    ).squeeze(1)
        if gx_f is not None:
            grad_x = gx_f.to(x.dtype)
        return grad_x, grad_a, grad_b, None, None, None


class BatchedExpertLoraAdapter(torch.nn.Module):
    """Pooled shared-outer LoRA adapter for one grouped expert surface.

    Counterpart of mbridge's grouped-expert adapters for the multi-tenant
    pool. ``forward`` receives the dispatch-layout rows and the MoE block
    sizes (``tokens_per_expert``) exactly like the base grouped linear;
    per-row slots come from the engine-armed ``ExpertSlotRelay``.

    ETP must be 1 (expert tensor parallelism unsupported): with ETP=1 the
    expert linears hold full in/out dims and the delta needs no TP comm.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        base_linear_name: str = "",
        *,
        num_local_experts: int,
        alpha: float | None = None,
        shared_a: bool = False,
    ):
        super().__init__()
        if num_local_experts <= 0:
            raise ValueError("num_local_experts must be positive")
        self.base_linear_name = base_linear_name
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(dim)
        self.num_local_experts = int(num_local_experts)
        self.alpha = alpha if alpha is not None else self.rank
        self._pool_name: str = ""
        # Sharing-direction contract (tinker-ref #1092): the hidden-side
        # factor is shared, the intermediate-side factor is per-expert.
        #   fc1/gate_up (shared_a=True):  A [N, rank, in],  B [N, E, out, rank]
        #   fc2/down    (shared_a=False): A [N, E, rank, in], B [N, out, rank]
        self.shared_a = bool(shared_a)
        if self.shared_a:
            self.local_a_shape = (self.rank, self.in_features)
            self.local_b_shape = (
                self.num_local_experts, self.out_features, self.rank,
            )
        else:
            self.local_a_shape = (self.num_local_experts, self.rank, self.in_features)
            self.local_b_shape = (self.out_features, self.rank)

    def forward(
        self,
        x: torch.Tensor,
        lora_a_pool: torch.Tensor,
        lora_b_pool: torch.Tensor,
        lora_indices: torch.Tensor,
        scale: float | None = None,
        tokens_per_expert=None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the batched expert LoRA delta.

        Args:
            x: [S, in] dispatch-layout rows (grouped by local expert).
            lora_a_pool / lora_b_pool: the surface's pool tensors —
                shared_a=True: A [N, rank, in], B [N, E, out, rank];
                shared_a=False: A [N, E, rank, in], B [N, out, rank].
            lora_indices: ignored here (dense-path signature parity); the
                expert path reads slots from ExpertSlotRelay.
            tokens_per_expert: per-local-expert row counts (list/tensor),
                the same argument the base grouped linear receives.
        """
        if scale is None:
            scale = self.alpha / self.rank
        if tokens_per_expert is None:
            raise RuntimeError(
                f"expert LoRA surface {self._pool_name!r} called without "
                "tokens_per_expert — expected the MoE grouped-linear call "
                "signature (x, tokens_per_expert)"
            )
        if not ExpertSlotRelay.ARMED:
            raise RuntimeError(
                f"expert LoRA surface {self._pool_name!r} forward without an "
                "armed slot relay — the engine must arm ExpertSlotRelay for "
                "every batched forward/backward"
            )
        S = x.reshape(-1, x.size(-1)).shape[0]
        slot_rows = ExpertSlotRelay.take_current(S)
        if slot_rows is None:
            raise RuntimeError(
                f"expert LoRA surface {self._pool_name!r}: slot relay empty at "
                "forward — dispatcher hooks did not run (dispatcher type not "
                "alltoall?)"
            )
        tpe = tokens_per_expert
        if torch.is_tensor(tpe):
            tpe = tpe.tolist()
        expert_rows = torch.repeat_interleave(
            torch.arange(self.num_local_experts, device=x.device),
            torch.tensor(tpe, device=x.device, dtype=torch.long),
        )
        if expert_rows.shape[0] != S:
            raise RuntimeError(
                f"expert LoRA surface {self._pool_name!r}: tokens_per_expert "
                f"sums to {expert_rows.shape[0]} rows but x has {S}"
            )
        op = _ExpertSharedInnerLoRA if self.shared_a else _ExpertSharedOuterLoRA
        delta = op.apply(
            x, lora_a_pool, lora_b_pool, slot_rows, expert_rows, scale
        )
        return delta if x.dim() == 2 else delta.view(*x.shape[:-1], -1)
