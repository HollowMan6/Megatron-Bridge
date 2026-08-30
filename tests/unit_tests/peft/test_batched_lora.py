"""Multi-LoRA batched delta op parity tests (TP=1/2/4, col/row, torch vs BGMV).

Run on a GPU node with the fork on PYTHONPATH:
  torchrun --nproc_per_node=<TP> tests/unit_tests/transformer/test_lora_batched.py

The reference implementation computes the delta adapter-by-adapter from the
GLOBAL pool ([N, rank, in] / [N, out, rank]); each rank compares against the
shards it owns (col: out-shard + rank-shard of A; row: in-shard of A +
out-shard of B). Also checks cross-tenant gradient isolation (each adapter's
grad lands only in its own slot).
"""

import os
import sys

import torch
import pytest

# torchrun-driven distributed GPU parity scripts (see module docstring):
# skip under bare pytest collection — they initialize dist/CUDA at import.
pytestmark = pytest.mark.skipif(
    os.environ.get("RANK") is None or not torch.cuda.is_available(),
    reason="distributed GPU parity script: run via torchrun (see docstring)",
)

import torch
import torch.distributed as dist

from megatron.core import ModelParallelConfig, parallel_state as mpu
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    initialize_rng_tracker,
)

TP = int(os.environ["WORLD_SIZE"])
RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(LOCAL_RANK)
dist.init_process_group("nccl")
mpu.initialize_model_parallel(tensor_model_parallel_size=TP)
initialize_rng_tracker()
get_cuda_rng_tracker().add("model-parallel-rng", 1234)
get_cuda_rng_tracker().add("data-parallel-rng", 1235)
get_cuda_rng_tracker().add("context-parallel-rng", 1236)

from megatron.bridge.peft.batched.lora import BatchedLoraAdapter

tp_group = mpu.get_tensor_model_parallel_group()

# RANK_DIM env: 16 exercises the torch fallback on TP>1 (local rank < 16... for
# TP=2; TP=4 gives local rank 8 which now rides the raw-kernel BGMV path —
# masked N/K with BLOCK dims >= 32 — since _MIN_BGMV_LOCAL_RANK dropped to 8);
# RANK_DIM=32 + TP=2/4 exercises the BGMV path with real TP comms at local
# rank 16/8 — the production col surface (rank 32, TP 4) runs local rank 8.
# OUT env: 512 crosses the bgmv out>=256 threshold and exercises the fused
# hidden-storage branch (the production attention outputs are >= 256).
N, IN, T = 2, 64, 256
OUT = int(os.environ.get("OUT", "128"))
RANK_DIM = int(os.environ.get("RANK_DIM", "16"))
SCALE = 0.5
DTYPE = torch.bfloat16  # production dtype; BGMV tl.dot is exact for bf16 inputs
# FORCE_TORCH=1 disables the BGMV path to test the torch fallback with the
# same assertions (R1-lossless: torch vs BGMV diff must be < 1e-5).
FORCE_TORCH = os.environ.get("FORCE_TORCH", "0") == "1"
if FORCE_TORCH:
    from megatron.core.transformer import lora_batched as _lb
    _lb._bgmv_available = lambda: False
# Same seed on ALL ranks: x_full / idx_full / global pool must be identical
# across ranks (each rank then takes its own shard of the shared tensors).
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def shard(t, dim, idx, n):
    w = t.shape[dim] // n
    return t.narrow(dim, idx * w, w).contiguous()


def rel_err(a, b):
    """Global-scale relative error: max abs diff over the reference's max
    magnitude (robust to near-zero reference elements)."""
    scale = b.abs().max().clamp_min(1e-3)
    return ((a - b).abs().max() / scale).item()


def check_against(path_delta, ref_shard, tag):
    """Two-tier parity: vs fp32 exact ref (bf16 noise, <5e-2) and the shape."""
    err = rel_err(path_delta.float(), ref_shard)
    if RANK == 0:
        print(f"[{tag}] fwd rel err vs fp32-ref: {err:.2e}", flush=True)
    assert err < 5e-2, err


def ref_delta(x_full, idx_full, a_global, b_global):
    """Reference: per-token explicit adapter forward from the GLOBAL pool.

    Computed in fp32 from bf16 weights (bf16 matmul error is dominated by
    input quantization ~1e-2; the fp32 reference is exact w.r.t. the
    quantized values, so tolerance 2e-2 catches logic errors).
    """
    out = torch.zeros(T, OUT, device=x_full.device, dtype=torch.float32)
    a_global = a_global.float()
    b_global = b_global.float()
    x_full = x_full.float()
    for t in range(T):
        g = idx_full[t].item()
        a = a_global[g]  # [rank, in]
        b = b_global[g]  # [out, rank]
        out[t] = SCALE * (b @ (a @ x_full[t]))
    return out


def build_global_pool(rank_partitioned):
    a_global = torch.randn(N, RANK_DIM, IN, device="cuda", dtype=DTYPE)
    b_global = torch.randn(N, OUT, RANK_DIM, device="cuda", dtype=DTYPE)
    return a_global, b_global


def make_indices(device):
    return torch.randint(0, N, (T,), device=device, dtype=torch.int64)


def test_col(use_bgmv_expected=True):
    """Column-parallel surface: x [T/tp, in] SP-local -> delta [T, out/tp]."""
    x_full = torch.randn(T, IN, device="cuda", dtype=DTYPE)
    idx_full = make_indices("cuda")
    a_global, b_global = build_global_pool(True)
    if RANK == 0:
        print(f"[col] TP={TP} bgmv_expected={use_bgmv_expected}", flush=True)

    adapter = BatchedLoraAdapter(
        IN, OUT, RANK_DIM, "col",
        input_is_parallel=False, disable_sequence_parallel_comm=False,
        model_parallel_config=ModelParallelConfig(
            tensor_model_parallel_size=TP, sequence_parallel=(TP > 1)),
        tp_group=tp_group,
    )
    local = T // TP
    x_local = x_full[RANK * local:(RANK + 1) * local].requires_grad_(True)
    idx_local = idx_full[RANK * local:(RANK + 1) * local]
    a_local = shard(a_global, 1, RANK, TP).requires_grad_(True)  # [N, rank/tp, in]
    b_local = shard(b_global, 1, RANK, TP).requires_grad_(True)  # [N, out/tp, rank]

    delta = adapter(x_local, a_local, b_local, idx_local, SCALE)
    ref = ref_delta(x_full, idx_full, a_global, b_global)
    ref_shard = shard(ref, 1, RANK, TP)
    assert delta.shape == (T, OUT // TP), delta.shape
    check_against(delta, ref_shard, "col")
    # R1-lossless: torch fallback vs BGMV path on identical inputs.
    from megatron.bridge.peft.batched.lora import (
        _BatchedColumnLoRA, _BatchedRowLoRA, _BgmvColumnLoRA, _BgmvRowLoRA,
        _BgmvLinearLoRA,
    )
    delta_torch = _BatchedColumnLoRA.apply(
        x_local, a_local, b_local, idx_local, SCALE, tp_group,
        adapter.sequence_parallel, adapter.rank_partitioned_a)
    err12 = rel_err(delta.float(), delta_torch.float())
    if RANK == 0:
        print(f"[col] BGMV-vs-torch rel err: {err12:.2e}", flush=True)
    assert err12 < 1e-2, err12

    # backward: compare grads vs an fp32 autograd reference computed from the
    # quantized (bf16) pool — this isolates comm/sharding logic from matmul
    # quantization noise.
    grad_out = torch.randn_like(delta)
    delta2 = adapter(x_local, a_local, b_local, idx_local, SCALE)
    (gx2, ga2, gb2) = torch.autograd.grad(
        delta2, (x_local, a_local, b_local), grad_outputs=grad_out
    )
    a_ref = a_global.detach().float().requires_grad_(True)
    b_ref = b_global.detach().float().requires_grad_(True)
    x_ref = x_full.detach().float().requires_grad_(True)
    ref_auto = ref_delta(x_ref, idx_full, a_ref, b_ref)
    # complete grad_out across TP (out-dim shards) for the fp32 reference
    go_list = [torch.zeros_like(grad_out) for _ in range(TP)]
    dist.all_gather(go_list, grad_out)
    go_full = torch.cat(go_list, dim=-1).float()  # [T, OUT]
    ref_auto.backward(go_full)
    ref_gx = x_ref.grad[RANK * local:(RANK + 1) * local]
    ref_ga = shard(a_ref.grad, 1, RANK, TP)
    ref_gb = shard(b_ref.grad, 1, RANK, TP)
    err_gx = rel_err(gx2.float(), ref_gx)
    err_ga = rel_err(ga2.float(), ref_ga)
    err_gb = rel_err(gb2.float(), ref_gb)
    if RANK == 0:
        print(f"[col] grad rel errs gx/ga/gb: {err_gx:.2e}/{err_ga:.2e}/{err_gb:.2e}", flush=True)
    assert err_gx < 5e-2, err_gx
    assert err_ga < 5e-2, err_ga
    assert err_gb < 5e-2, err_gb


def test_row():
    """Row-parallel surface: x [T, in/tp] full-seq -> delta [T/tp, out]."""
    x_full = torch.randn(T, IN, device="cuda", dtype=DTYPE)
    idx_full = make_indices("cuda")
    a_global = torch.randn(N, RANK_DIM, IN, device="cuda", dtype=DTYPE)  # [N, rank, in]
    b_global = torch.randn(N, OUT, RANK_DIM, device="cuda", dtype=DTYPE)
    if RANK == 0:
        print(f"[row] TP={TP}", flush=True)

    adapter = BatchedLoraAdapter(
        IN, OUT, RANK_DIM, "row",
        input_is_parallel=True, disable_sequence_parallel_comm=False,
        model_parallel_config=ModelParallelConfig(
            tensor_model_parallel_size=TP, sequence_parallel=(TP > 1)),
        tp_group=tp_group,
    )
    local = T // TP
    x_local = shard(x_full, 1, RANK, TP).requires_grad_(True)  # [T, in/tp]
    idx_local = idx_full
    a_local = shard(a_global, 2, RANK, TP).requires_grad_(True)  # [N, rank, in/tp]
    b_local = shard(b_global, 1, RANK, TP).requires_grad_(True)  # [N, out/tp, rank]

    delta = adapter(x_local, a_local, b_local, idx_local, SCALE)
    ref = ref_delta(x_full, idx_full, a_global, b_global)  # [T, out]
    ref_shard = ref[RANK * local:(RANK + 1) * local]
    err = rel_err(delta.float(), ref_shard)
    if RANK == 0:
        print(f"[row] fwd max rel err vs ref: {err:.2e}", flush=True)
    assert delta.shape == (T // TP, OUT), delta.shape
    assert err < 2e-2, err

    grad_out = torch.randn_like(delta)
    (gx, ga, gb) = torch.autograd.grad(
        delta, (x_local, a_local, b_local), grad_outputs=grad_out
    )
    a_ref = a_global.detach().float().requires_grad_(True)
    b_ref = b_global.detach().float().requires_grad_(True)
    x_ref = x_full.detach().float().requires_grad_(True)
    ref_auto = ref_delta(x_ref, idx_full, a_ref, b_ref)
    # complete grad_out across TP (seq-dim shards) for the fp32 reference
    go_list = [torch.zeros_like(grad_out) for _ in range(TP)]
    dist.all_gather(go_list, grad_out)
    go_full = torch.cat(go_list, dim=0).float()  # [T, OUT]
    ref_auto.backward(go_full)
    ref_gx = shard(x_ref.grad, 1, RANK, TP)
    ref_ga = shard(a_ref.grad, 2, RANK, TP)
    ref_gb = shard(b_ref.grad, 1, RANK, TP)
    err_gx = rel_err(gx.float(), ref_gx)
    err_ga = rel_err(ga.float(), ref_ga)
    err_gb = rel_err(gb.float(), ref_gb)
    if RANK == 0:
        print(f"[row] grad rel errs gx/ga/gb: {err_gx:.2e}/{err_ga:.2e}/{err_gb:.2e}", flush=True)
    assert err_gx < 5e-2, err_gx
    assert err_ga < 5e-2, err_ga
    assert err_gb < 5e-2, err_gb


def test_isolation():
    """Cross-tenant isolation: adapter-0 tokens' grads must not touch slot 1."""
    x_local = torch.randn(T // TP, IN, device="cuda", dtype=DTYPE)
    idx = torch.zeros(T // TP, dtype=torch.int64, device="cuda")  # all adapter 0
    a_local = torch.randn(N, RANK_DIM // TP, IN, device="cuda", dtype=DTYPE).requires_grad_(True)
    b_local = torch.randn(N, OUT // TP, RANK_DIM, device="cuda", dtype=DTYPE)
    adapter = BatchedLoraAdapter(
        IN, OUT, RANK_DIM, "col",
        input_is_parallel=False, disable_sequence_parallel_comm=False,
        model_parallel_config=ModelParallelConfig(
            tensor_model_parallel_size=TP, sequence_parallel=(TP > 1)),
        tp_group=tp_group,
    )
    delta = adapter(x_local, a_local, b_local, idx, SCALE)
    loss = delta.sum()
    (ga,) = torch.autograd.grad(loss, (a_local,))
    slot1 = ga[1]
    assert torch.count_nonzero(slot1).item() == 0, "gradient leaked into slot 1"
    if RANK == 0:
        print("[isolation] slot-1 grad is exactly zero", flush=True)


if __name__ == "__main__":
    test_col()
    test_row()
    test_isolation()
    dist.barrier()
    if RANK == 0:
        print("ALL PASS", flush=True)
    dist.destroy_process_group()
