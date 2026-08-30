"""Triton BGMV (Batched Gather MatMul-Vector) kernels for multi-LoRA training.

Ported from the mlite runtime reference implementation (mindlab
feat/mlite-batch-lora-op-rebase, experimental/lite/.../kernels/bgmv.py);
expert shared-outer kernels omitted (GLM-5.2 experts carry no LoRA).

Design (C1): tokens stay in ORIGINAL order. One ``torch.sort`` produces
``perm`` (sorted-position -> original token). Every kernel gathers its
per-token inputs via ``perm`` and scatter-stores per-token outputs to
``perm`` (original order). This fuses the gather+gemm+scatter into one pass
per tensor (halves x/delta traffic vs the old sort+gather+kernel+scatter
sequence) and removes the backward re-gather of x/grad_out/lora_indices.

Forward: 1 fused kernel -> delta [T, out] + hidden [T, rank]
Backward: 4 kernels (grad_hidden, grad_x, grad_A, grad_B) — hidden saved, no recompute

Grid: (M_blocks, G) — 3rd dim = adapter index (free parallelism across adapters)
No padding: tl.load(mask=...) handles boundaries natively
"""


import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    """Return True if Triton is available (CUDA kernels usable)."""
    return _TRITON_AVAILABLE


_BLOCK_N = 64

# When N >= this threshold, use the fused forward kernel (saves 1 launch
# but pays an A-transpose cost per block).  Below it, the 3-kernel path
# (shrink + shrink) is faster because the transpose cost exceeds the
# saved launch overhead.
_FUSED_N_THRESHOLD = 256

# LoRA rank constraints (by design, all adapters in one forward share a rank):
#   - rank must be a power of two in [2, 64]
#   - rank < 16 is padded to 16 (Hopper tl.dot requires every dot dim >= 16)
#   - rank > 64 is unsupported (would need K-loop tiling in fused_fwd)
_MIN_RANK = 16
_MAX_RANK = 64

_TUNING_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    # BLOCK_N=32 variants for small-N outputs (e.g. rank=16/32)
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_K": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_K": 64, "BLOCK_N": 32}, num_warps=8, num_stages=4),
    # BLOCK_M=32 variant for small-M dims in shrink_transpose (e.g. rank=32)
    triton.Config({"BLOCK_M": 32, "BLOCK_K": 32, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_K": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_K": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_K": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
]


# ── L2 prefetch ────────────────────────────────────────────────────


def _l2_prefetch(*tensors: torch.Tensor) -> None:
    # view(-1) requires contiguity; pool shards arriving from narrow()/pad()
    # are contiguous by construction — enforce it instead of silently
    # prefetching a stale layout (L5).
    for t in tensors:
        if not t.is_contiguous():
            raise ValueError(
                f"_l2_prefetch requires contiguous tensors, got shape "
                f"{tuple(t.shape)} strides {t.stride()}"
            )
    """Touch tensors to bring them into L2 cache."""
    for t in tensors:
        if t.numel() > 0:
            flat = t.view(-1)
            idx = torch.arange(0, flat.numel(), 64, device=t.device)
            _ = flat[idx]


def _identity_perm(T: int, device) -> torch.Tensor:
    """arange(T) — identity permutation for callers that pre-gather x."""
    return torch.arange(T, dtype=torch.int64, device=device)


def _pad_rank(a_pool: torch.Tensor, b_pool: torch.Tensor, rank: int):
    """Pad the rank dim to _MIN_RANK when rank < 16 (Hopper mma minimum).

    a_pool[..., rank, :] and b_pool[..., :, rank] get zero-padded along rank to
    _MIN_RANK. Returns (a_pool, b_pool, eff_rank) where eff_rank is the padded
    rank kernels compile against. No-op (same tensors, same rank) when rank >= 16.
    """
    if rank >= _MIN_RANK:
        return a_pool, b_pool, rank
    pad = _MIN_RANK - rank
    # a_pool: rank is dim -2 of [..., rank, in]  -> pad dim -2
    a = torch.nn.functional.pad(a_pool, (0, 0, 0, pad))
    # b_pool: rank is dim -1 of [..., out, rank]  -> pad dim -1
    b = torch.nn.functional.pad(b_pool, (0, pad))
    return a, b, _MIN_RANK


# ── Host-side helpers ──────────────────────────────────────────────


def compute_group_offsets(
    lora_indices: torch.Tensor,
    num_loras: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute per-group offsets and sizes from **sorted** lora_indices.

    Requires ``lora_indices`` to be monotonically non-decreasing.
    ``bgmv_fwd`` and ``bgmv_bwd`` sort internally before calling this;
    callers must not pass unsorted indices directly.

    ``num_loras`` is REQUIRED (the pool first dim, = max_loras). The old
    ``None`` fallback used ``lora_indices.max().item()+1``. That value always
    covers every observed index (so scatter_add_ stays in bounds), BUT under
    slot gaps (evicted slots -> lora_indices sparse, e.g. {0,3,5} -> max+1=6)
    the returned G (6) is SMALLER than the pool first dim (max_loras=8), which
    undersizes downstream grad buffers and the kernel grid (G groups) ->
    silent corruption / wrong pool-slot indexing. Every caller passes the
    pool shape (G = lora_a_pool.shape[0]); making it required hardens the
    load-bearing assumption (group index = direct pool slot, grid = max_loras
    groups, empty groups early-exit in kernels).
    """
    assert num_loras is not None and num_loras > 0, (
        "num_loras is required (pool first dim / max_loras); the None fallback "
        "undersizes group tables under slot gaps (evicted slots)"
    )
    # No sortedness check: bgmv_fwd/bgmv_bwd always sort before calling
    # this function. A runtime assert here would require .all().item()
    # (CPU-GPU sync), which defeats the sync-avoidance design.
    # Use int32 directly on the correct device (CUDA-graph safe, no host transfer)
    group_sizes = torch.zeros(num_loras, dtype=torch.int32, device=lora_indices.device)
    group_sizes.scatter_add_(0, lora_indices, torch.ones_like(lora_indices, dtype=torch.int32))
    group_offsets = torch.zeros(num_loras, dtype=torch.int32, device=lora_indices.device)
    if num_loras > 1:
        group_offsets[1:] = torch.cumsum(group_sizes[:-1], dim=0)
    return group_offsets, group_sizes, num_loras


# ── Fused forward kernel: gemm1 + gemm2 in 1 launch ───────────────
#   grid = (M_blocks, G) — each block computes [BLOCK_M, N] of delta
#   hidden [BLOCK_M, RANK] stays in register between gemm1 and gemm2
#   perm maps sorted-position -> original token (inputs gathered, delta scattered)


@triton.autotune(configs=_TUNING_CONFIGS, key=["K", "N", "G"])
@triton.jit
def _bgmv_fused_fwd_kernel(
    x_ptr, A_ptr, B_ptr, delta_ptr,
    group_offsets_ptr, group_sizes_ptr,
    perm_ptr,
    T,
    K: tl.constexpr,     # in_features
    RANK: tl.constexpr,  # rank
    N: tl.constexpr,     # out_features
    G: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    hidden_ptr,           # optional: store gemm1 hidden here for bwd
    BLOCK_N: tl.constexpr = _BLOCK_N,
    HIDDEN_PTR: tl.constexpr = 0,  # nonzero => store gemm1 hidden for bwd
):
    pid_m = tl.program_id(0)
    g = tl.program_id(1)

    g_size = tl.load(group_sizes_ptr + g)
    if g_size == 0:
        return
    m_start = tl.load(group_offsets_ptr + g)
    m_offset = pid_m * BLOCK_M
    m_len = tl.minimum(BLOCK_M, g_size - m_offset)
    if m_len <= 0:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, RANK)
    mask_m = offs_m < m_len
    x_row = m_start + m_offset + offs_m
    tok = tl.load(perm_ptr + x_row, mask=mask_m, other=0)  # sorted-pos -> original token

    # ── gemm1: hidden = x @ A[g].T → [BLOCK_M, RANK] (in register) ──
    # A[g] is [RANK, K], element [r, k] at offset r*K + k
    # Load as [RANK, BK]: a_ptrs = A + g*(RANK*K) + r*K + k
    # Then dot(x, a.T) = [BM, BK] @ [BK, RANK] = [BM, RANK]
    x_ptrs = x_ptr + tok[:, None] * K + offs_k[None, :]
    a_ptrs = A_ptr + g * (RANK * K) + offs_r[:, None] * K + offs_k[None, :]

    hidden = tl.zeros([BLOCK_M, RANK], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k * BLOCK_K + offs_k) < K
        x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)  # [BM, BK]
        a = tl.load(a_ptrs, mask=(offs_r[:, None] < RANK) & k_mask[None, :], other=0.0)  # [RANK, BK]
        hidden += tl.dot(x, a.T)  # [BM, BK] @ [BK, RANK] → [BM, RANK]
        x_ptrs += BLOCK_K
        a_ptrs += BLOCK_K

    # H5: store hidden for bwd before gemm2 consumes it (fuses out the separate
    # _bgmv_shrink_kernel that recomputed the same x@A[g].T). Only when HIDDEN_PTR.
    if HIDDEN_PTR != 0:
        h_ptrs = hidden_ptr + tok[:, None] * RANK + offs_r[None, :]
        tl.store(h_ptrs, hidden.to(hidden_ptr.dtype.element_ty),
                 mask=mask_m[:, None] & (offs_r[None, :] < RANK))

    # ── gemm2: delta = scale * (hidden @ B[g].T) → [BLOCK_M, N] ──
    # B[g] is [N, RANK]; B[g].T is [RANK, N]
    # Load B[g].T as [RANK, BN]: b_ptrs = B + g*(N*RANK) + r + n*RANK
    b_base = B_ptr + g * (N * RANK)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        b_ptrs = b_base + offs_r[:, None] + offs_n[None, :] * RANK  # [RANK, BN]
        b = tl.load(b_ptrs, mask=(offs_r[:, None] < RANK) & mask_n[None, :])  # [RANK, BN]
        delta = tl.dot(hidden.to(b.dtype), b)  # [BM, RANK] @ [RANK, BN] → [BM, BN]
        delta = delta * scale
        out_ptrs = delta_ptr + tok[:, None] * N + offs_n[None, :]
        tl.store(out_ptrs, delta.to(delta_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


# ── Shrink kernel: out = scale * (x @ w.T) ─────────────────────────
#   w[g] is [N, K]; compute x[T_g, K] @ w[g].T → out[T_g, N]
#   perm: gather x via perm, scatter out via perm (original token order)


@triton.autotune(configs=_TUNING_CONFIGS, key=["K", "N", "G"])
@triton.jit
def _bgmv_shrink_kernel(
    x_ptr, w_ptr, out_ptr,
    group_offsets_ptr, group_sizes_ptr,
    perm_ptr,
    T,
    K: tl.constexpr, N: tl.constexpr, G: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr = _BLOCK_N,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2)

    g_size = tl.load(group_sizes_ptr + g)
    if g_size == 0:
        return
    m_start = tl.load(group_offsets_ptr + g)
    m_offset = pid_m * BLOCK_M
    m_len = tl.minimum(BLOCK_M, g_size - m_offset)
    if m_len <= 0:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m_len
    mask_n = offs_n < N
    x_row = m_start + m_offset + offs_m
    tok = tl.load(perm_ptr + x_row, mask=mask_m, other=0)

    x_ptrs = x_ptr + tok[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + g * (N * K) + offs_n[:, None] * K + offs_k[None, :]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k * BLOCK_K + offs_k) < K
        x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)  # [BM, BK]
        w = tl.load(w_ptrs, mask=mask_n[:, None] & k_mask[None, :], other=0.0)  # [BN, BK]
        acc += tl.dot(x, w.T)  # [BM, BK] @ [BK, BN] → [BM, BN]
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    acc = acc * scale
    out_ptrs = out_ptr + tok[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


# ── Expand kernel: out = scale * (x @ w) ───────────────────────────
#   w[g] is [K, N]; compute x[T_g, K] @ w[g] → out[T_g, N]
#   perm: gather x via perm, scatter out via perm (original token order)


@triton.autotune(configs=_TUNING_CONFIGS, key=["K", "N", "G"])
@triton.jit
def _bgmv_expand_kernel(
    x_ptr, w_ptr, out_ptr,
    group_offsets_ptr, group_sizes_ptr,
    perm_ptr,
    T,
    K: tl.constexpr, N: tl.constexpr, G: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr = _BLOCK_N,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2)

    g_size = tl.load(group_sizes_ptr + g)
    if g_size == 0:
        return
    m_start = tl.load(group_offsets_ptr + g)
    m_offset = pid_m * BLOCK_M
    m_len = tl.minimum(BLOCK_M, g_size - m_offset)
    if m_len <= 0:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m_len
    mask_n = offs_n < N
    x_row = m_start + m_offset + offs_m
    tok = tl.load(perm_ptr + x_row, mask=mask_m, other=0)

    x_ptrs = x_ptr + tok[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + g * (K * N) + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k * BLOCK_K + offs_k) < K
        x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)  # [BM, BK]
        w = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)  # [BK, BN]
        acc += tl.dot(x, w)  # [BM, BK] @ [BK, BN] → [BM, BN]
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * N

    acc = acc * scale
    out_ptrs = out_ptr + tok[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


# ── Expand-loop kernel: out = scale * (x @ w), N tiled in-loop ──────────
#   For the expert grad_x path: G=N*E=2048 groups, ~8 tokens each. Tiling N
#   across the grid (like _bgmv_expand_kernel) explodes programs
#   (cdiv(T,BM)*cdiv(N,BN)*G). Here N is tiled in a python loop, so grid =
#   cdiv(T,BM)*G only — 16x fewer programs. K=rank is small (<=BLOCK_K), so
#   the full [BM,K] dot is computed once per program, then N is emitted in tiles.
#   perm: gather x via perm, scatter out via perm (original token order)


# ── Shrink-transpose kernel: out[g] = x[g].T @ w[g] (reduce over tokens) ──
#   perm: gather x and w (both per-token) via perm. out is per-group (no perm).


@triton.autotune(configs=_TUNING_CONFIGS, key=["M", "N", "G"], reset_to_zero=["out_ptr"])
@triton.jit
def _bgmv_shrink_transpose_kernel(
    x_ptr, w_ptr, out_ptr,
    group_offsets_ptr, group_sizes_ptr,
    perm_ptr,
    T,
    M: tl.constexpr, N: tl.constexpr, G: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr = _BLOCK_N,
    scale: tl.constexpr = 1.0,
    SPLIT_K: tl.constexpr = 1,  # H1: strided split over token-reduce dim
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    gs = tl.program_id(2)  # encodes (group, split-K)
    g = gs // SPLIT_K
    sk = gs % SPLIT_K

    g_size = tl.load(group_sizes_ptr + g)
    if g_size == 0:
        return
    m_start = tl.load(group_offsets_ptr + g)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # H1: strided split-K. Block sk reduces tokens at positions sk, sk+SK, ...
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k_base in range(0, g_size, BLOCK_K * SPLIT_K):
        kk = k_base + sk + offs_k * SPLIT_K  # token positions for this split
        k_mask = kk < g_size
        # clamp perm index to valid range so the gather never derefs past perm[T]
        # (out-of-range tokens contribute 0 via k_mask=False on the x/w loads)
        kk_clamped = tl.minimum(kk, g_size - 1)
        tok_k = tl.load(perm_ptr + m_start + kk_clamped, mask=k_mask, other=0)
        x_ptrs = x_ptr + tok_k[None, :] * M + offs_m[:, None]
        w_ptrs = w_ptr + tok_k[:, None] * N + offs_n[None, :]
        x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)

    res = acc * scale
    out_ty = out_ptr.dtype.element_ty
    out_ptrs = out_ptr + g * (M * N) + offs_m[:, None] * N + offs_n[None, :]
    if SPLIT_K > 1:
        # atomic single-buffer: SK split partials accumulate to the (zero-init)
        # out. autotune's reset_to_zero=["out_ptr"] prevents the first-call
        # corruption (autotune benchmarks N configs on the same buffer without
        # re-zeroing -> atomic accumulation across configs = garbage).
        tl.atomic_add(out_ptrs, res.to(out_ty), mask=mask_m[:, None] & mask_n[None, :])
    else:
        tl.store(out_ptrs, res.to(out_ty), mask=mask_m[:, None] & mask_n[None, :])


# ── Forward: 1 fused kernel ────────────────────────────────────────


def bgmv_fwd(
    x: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    lora_indices: torch.Tensor,
    scale: float,
    max_g_size_hint: int | None = None,
    need_hidden: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fused BGMV forward: 1 kernel, hidden stays in register.

    Tokens stay in original order; ``perm`` (sorted->original) is returned for
    backward. Kernels gather x via perm and scatter delta to original order.

    Args:
        need_hidden: If False (forward-only / no_grad), skip computing
            ``hidden`` and return ``None`` — saves 1 kernel launch and
            a [T, rank] allocation per layer.

    Returns:
        delta: [T, out] — LoRA delta (in original token order)
        hidden: [T, rank] — saved for backward (original order), or None
        perm: [T] int64 — sort permutation (sorted-position -> original token)
    """
    T = x.shape[0]
    G, rank, in_features = lora_A.shape
    out_features = lora_B.shape[1]
    # Pad rank to >=16 so tl.dot's rank dim satisfies the Hopper mma minimum.
    lora_A, lora_B, rank = _pad_rank(lora_A, lora_B, rank)

    # ── Sort tokens by lora_indices (kernels require contiguous groups) ──
    # perm[i] = original token index of the i-th sorted token. x is NOT
    # gathered — kernels gather x via perm themselves (fused, 1 traffic pass).
    if T > 1:
        lora_indices, perm = torch.sort(lora_indices)
    else:
        perm = _identity_perm(T, x.device)

    group_offsets, group_sizes, G = compute_group_offsets(lora_indices, num_loras=G)
    # No host sync: T is a safe upper bound for the largest group (over-launched
    # blocks hit m_len<=0 and early-exit). Avoids group_sizes.amax().item().
    max_g_size = max_g_size_hint if max_g_size_hint is not None else T

    # L2 prefetch: bring pool into L2 before kernel
    _l2_prefetch(lora_A, lora_B)

    # Hidden is needed for backward (saves 1 recompute kernel).
    # Skip entirely in forward-only mode (no_grad) to avoid wasted compute.
    # H5: when the fused path runs (out>=256) AND need_hidden, the fused kernel
    # stores hidden in-register -> no separate shrink needed. The 2-kernel path
    # (out<256) still needs the shrink to materialize hidden.
    hidden = None
    need_shrink_hidden = need_hidden and out_features < _FUSED_N_THRESHOLD
    if need_shrink_hidden:
        hidden = torch.empty(T, rank, device=x.device, dtype=x.dtype)
        grid_h = lambda meta: (
            triton.cdiv(max_g_size, meta["BLOCK_M"]),
            triton.cdiv(rank, meta["BLOCK_N"]),
            G,
        )
        _bgmv_shrink_kernel[grid_h](
            x, lora_A, hidden,
            group_offsets, group_sizes, perm,
            T, K=in_features, N=rank, G=G, scale=1.0,
        )

    # Forward path: choose fused (1 kernel) vs 2-kernel based on out_features.
    # Fused saves 1 launch but pays an A-transpose cost per block.
    # Break-even is out_features≈256; below that the 2-kernel path is faster.
    delta = torch.empty(T, out_features, device=x.device, dtype=x.dtype)
    if out_features >= _FUSED_N_THRESHOLD or not need_hidden:
        # Fused kernel computes hidden in-register; use it when:
        # - out_features is large (transpose cost < saved launch), or
        # - need_hidden=False (2-kernel path needs hidden as input)
        # H5: if need_hidden, also pass hidden so fused stores it (kills the
        # separate shrink). hidden was allocated above only for the 2-kernel path.
        if need_hidden and hidden is None:
            hidden = torch.empty(T, rank, device=x.device, dtype=x.dtype)
        h_ptr = hidden if hidden is not None else x  # dummy (HIDDEN_PTR=0 guards)
        _bgmv_fused_fwd_kernel[lambda meta: (triton.cdiv(max_g_size, meta["BLOCK_M"]), G)](
            x, lora_A, lora_B, delta,
            group_offsets, group_sizes, perm,
            T, K=in_features, RANK=rank, N=out_features, G=G,
            scale=scale,
            hidden_ptr=h_ptr, HIDDEN_PTR=1 if need_hidden else 0,
        )
    else:
        # 2-kernel path: shrink gemm1 (hidden) already computed above,
        # now shrink gemm2 (delta = hidden @ B.T)
        grid_d = lambda meta: (
            triton.cdiv(max_g_size, meta["BLOCK_M"]),
            triton.cdiv(out_features, meta["BLOCK_N"]),
            G,
        )
        _bgmv_shrink_kernel[grid_d](
            hidden, lora_B, delta,
            group_offsets, group_sizes, perm,
            T, K=rank, N=out_features, G=G, scale=scale,
        )

    return delta, hidden, perm


# ── Backward: 4 kernels (hidden saved, no recompute) ──────────────


def bgmv_bwd(
    x: torch.Tensor,
    grad_out: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    lora_indices: torch.Tensor,
    scale: float,
    hidden: torch.Tensor | None = None,
    max_g_size_hint: int | None = None,
    perm: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BGMV backward: 4 kernels, hidden saved from forward.

    x, hidden, grad_out stay in original order; ``perm`` (from forward) maps
    sorted-position -> original token. Kernels gather via perm and scatter
    per-token outputs (grad_x) to original order. No re-gather of x/grad_out.

    Returns:
        grad_x: [T, in] — in original token order
        grad_A: [G, rank, in]
        grad_B: [G, out, rank]
    """
    T = x.shape[0]
    G, rank, in_features = lora_A.shape
    out_features = lora_B.shape[1]
    orig_rank = rank
    # Pad rank to >=16 (matches forward's _pad_rank; grad_A/B come back padded).
    lora_A, lora_B, rank = _pad_rank(lora_A, lora_B, rank)

    # Re-sort lora_indices to match the forward sort (for group offsets).
    # x, grad_out, hidden are NOT gathered — kernels gather them via perm.
    if perm is not None:
        lora_indices = lora_indices[perm]

    group_offsets, group_sizes, G = compute_group_offsets(lora_indices, num_loras=G)
    # No host sync (see bgmv_fwd): T is a safe upper bound for the largest group.
    max_g_size = max_g_size_hint if max_g_size_hint is not None else T

    # grad_out may be non-contiguous (e.g. broadcasted from .sum() → stride=(0,0)).
    # The expand kernel assumes row-major contiguous input (stride == K).
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    # gemm1 (expand): grad_hidden = grad_out @ B[g] → [T, rank]
    # scale folded into GEMM epilogue (avoids 336MB alloc + read at T=64K)
    grad_hidden = torch.empty(T, rank, device=x.device, dtype=x.dtype)
    grid_gh = lambda meta: (
        triton.cdiv(max_g_size, meta["BLOCK_M"]),
        triton.cdiv(rank, meta["BLOCK_N"]),
        G,
    )
    _bgmv_expand_kernel[grid_gh](
        grad_out, lora_B, grad_hidden,
        group_offsets, group_sizes, perm,
        T, K=out_features, N=rank, G=G, scale=scale,
    )

    # gemm2 (expand): grad_x = grad_hidden @ A[g] → [T, in]
    grad_x = torch.empty(T, in_features, device=x.device, dtype=x.dtype)
    grid_gx = lambda meta: (
        triton.cdiv(max_g_size, meta["BLOCK_M"]),
        triton.cdiv(in_features, meta["BLOCK_N"]),
        G,
    )
    _bgmv_expand_kernel[grid_gx](
        grad_hidden, lora_A, grad_x,
        group_offsets, group_sizes, perm,
        T, K=rank, N=in_features, G=G, scale=1.0,
    )

    # gemm3 (shrink_transpose): grad_A = grad_hidden.T @ x → [G, rank, in]
    # zero-init: idle adapters (g_size==0) must have zero gradient, not garbage
    # H1: split-K over the token-reduce dim when few groups (SM-starved: grid
    # was 256, Waves 0.47). atomic single-buffer (fp32) + SK=4 for G<=16;
    # autotune reset_to_zero handles the atomic-vs-autotune interaction.
    use_split_a = G <= 16
    SK_A = 4 if use_split_a else 1
    grad_A = torch.zeros(G, rank, in_features, device=x.device, dtype=torch.float32 if use_split_a else x.dtype)
    grid_gA = lambda meta: (
        triton.cdiv(rank, meta["BLOCK_M"]),
        triton.cdiv(in_features, meta["BLOCK_N"]),
        G * SK_A,
    )
    _bgmv_shrink_transpose_kernel[grid_gA](
        grad_hidden, x, grad_A,
        group_offsets, group_sizes, perm,
        T, M=rank, N=in_features, G=G, SPLIT_K=SK_A,
    )
    if use_split_a:
        grad_A = grad_A.to(x.dtype)

    # gemm4 (shrink_transpose): grad_B = grad_out.T @ hidden → [G, out, rank]
    if hidden is None:
        hidden = torch.empty(T, rank, device=x.device, dtype=x.dtype)
        _bgmv_shrink_kernel[grid_gh](
            x, lora_A, hidden, group_offsets, group_sizes, perm,
            T, K=in_features, N=rank, G=G, scale=1.0,
        )

    use_split_b = G <= 16
    SK_B = 4 if use_split_b else 1
    grad_B = torch.zeros(G, out_features, rank, device=x.device, dtype=torch.float32 if use_split_b else x.dtype)
    grid_gB = lambda meta: (
        triton.cdiv(out_features, meta["BLOCK_M"]),
        triton.cdiv(rank, meta["BLOCK_N"]),
        G * SK_B,
    )
    _bgmv_shrink_transpose_kernel[grid_gB](
        grad_out, hidden, grad_B,
        group_offsets, group_sizes, perm,
        T, M=out_features, N=rank, G=G, scale=scale, SPLIT_K=SK_B,
    )
    if use_split_b:
        grad_B = grad_B.to(x.dtype)

    # Unpad grad_A/grad_B back to the original rank (the padded rows are zeros
    # in A/B, so their gradients are valid but unwanted by the caller's shape).
    if orig_rank < rank:
        grad_A = grad_A[:, :orig_rank, :]
        grad_B = grad_B[:, :, :orig_rank]
    return grad_x, grad_A, grad_B


# ── Expert shared-outer: per-(adapter,expert) A, shared-outer B ──────────
#   A_pool [N, E, rank, in]  -> flat [N*E, rank, in], group g = adapter*E + expert
#   B_pool [N, out, rank]    -> B indexed by adapter = g // E (shared across experts)
#   delta[t] = scale * B[adapter_t] @ (A[adapter_t, expert_t] @ x[t])
#   perm: gather x via perm, scatter delta via perm (original token order)


