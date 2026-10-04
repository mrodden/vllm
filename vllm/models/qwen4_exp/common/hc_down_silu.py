# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fused Qwen4Exp HC down projection + SiLU (triton).

Computes the skinny GEMM ``C = bf16(fp32acc(A @ W^T))`` for the merged
down+inject weight, then applies the mHC SiLU epilogue: columns
< ``rank`` get ``silu(bf16(acc) / hc)`` (production rounding boundary:
the GEMM output is materialized in bf16 before SiLU, matching the eager
``MergedColumnParallelLinear -> split -> hc_silu`` path), and the
``hc`` injection-logit columns pass through unchanged. Unlike the eager
path the two outputs are written to separate tensors directly, so no
split or copy is materialized.

Two backends:
  - FMA broadcast for M <= 4 (decode hot case; no ``tl.dot``).
  - ``tl.dot`` with a fixed split-K table for 5 <= M <= MAX_FUSED_M.

Beyond ``MAX_FUSED_M`` (or on odd shapes) callers fall back to the
unfused eager path; the dispatch boundary mirrors the NVIDIA cute_dsl
kernel's so both paths fuse the same token counts.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# The fused kernel stops winning past M ~ 64 on Qwen3.8-Next-Flash;
# kept identical to the cute_dsl variant so both paths fuse the same Ms.
MAX_FUSED_M = 48

# (split_k, block_m, block_n) keyed by M bucket for the tl.dot backend.
# Fixed configs (no autotune) so CUDA-graph capture never recompiles.
_SPLITK_TABLE: dict[int, tuple[int, int, int]] = {
    **{m: (6, 16, 32) for m in range(5, 17)},
    **{m: (6, 16, 64) for m in range(17, 33)},
    **{m: (6, 16, 64) for m in range(33, MAX_FUSED_M + 1)},
}


@triton.jit
def _hc_down_silu_fma_kernel(
    x_ptr,
    w_ptr,
    lora_ptr,
    inj_ptr,
    M,
    K,
    RANK: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """C[n] = bf16(sum_k x[k] * w[n, k]); SiLU on lora columns."""
    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < RANK + HC
    k_offs = tl.arange(0, BLOCK_K)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # Broadcast the (single or few) activation rows; each program handles
    # one row x one BLOCK_N of outputs, looping K in BLOCK_K chunks.
    row = tl.program_id(1)
    acc = tl.zeros([BLOCK_N], tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            x = tl.load(x_ptr + row * K + k0 * BLOCK_K + k_offs).to(tl.float32)
            w = tl.load(
                w_ptr + n_offs[:, None] * K + k0 * BLOCK_K + k_offs[None, :],
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            k_mask = k0 * BLOCK_K + k_offs < K
            x = tl.load(
                x_ptr + row * K + k0 * BLOCK_K + k_offs,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            w = tl.load(
                w_ptr + n_offs[:, None] * K + k0 * BLOCK_K + k_offs[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        acc += tl.sum(x[None, :] * w, axis=1)

    # Production rounding boundary: bf16 GEMM output, fp32 SiLU.
    acc_bf16 = acc.to(tl.bfloat16).to(tl.float32)
    is_lora = n_offs < RANK
    silu_in = acc_bf16 / HC
    epilog = tl.where(is_lora, silu_in * tl.sigmoid(silu_in), acc_bf16)
    epilog = epilog.to(tl.bfloat16)

    tl.store(
        lora_ptr + row * RANK + n_offs,
        epilog,
        mask=n_mask & is_lora,
    )
    tl.store(
        inj_ptr + row * HC + (n_offs - RANK),
        epilog,
        mask=n_mask & (~is_lora),
    )
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _hc_down_silu_dot_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    M,
    K,
    RANK: tl.constexpr,
    HC: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Split-K tl.dot GEMM; partials reduced in a second pass."""
    pid = tl.program_id(0)
    k_id = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(RANK + HC, BLOCK_N)
    num_pid_in_m = tl.cdiv(num_pid_m, 1)
    pid_m = pid % num_pid_in_m
    pid_n = (pid // num_pid_in_m) % num_pid_n

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < RANK + HC

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    k_per_split = tl.cdiv(K, SPLIT_K * BLOCK_K)
    k_start = k_id * k_per_split * BLOCK_K

    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for kk in range(k_per_split):
        k_offs = k_start + kk * BLOCK_K + tl.arange(0, BLOCK_K)
        if EVEN_K:
            x = tl.load(
                x_ptr + m_offs[:, None] * K + k_offs[None, :],
                mask=m_mask[:, None],
                other=0.0,
            )
            w = tl.load(
                w_ptr + n_offs[:, None] * K + k_offs[None, :],
                mask=n_mask[:, None],
                other=0.0,
            )
        else:
            k_mask = k_offs < K
            x = tl.load(
                x_ptr + m_offs[:, None] * K + k_offs[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                w_ptr + n_offs[:, None] * K + k_offs[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
        acc = tl.dot(x, tl.trans(w), acc, out_dtype=tl.float32)

    # Stash fp32 partials; the epilogue kernel reduces across split_k.
    part_ptr += (
        (k_id * M + m_offs[:, None]) * (RANK + HC) + n_offs[None, :]
    )
    tl.store(part_ptr, acc, mask=m_mask[:, None] & n_mask[None, :])
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _hc_down_silu_epilogue_kernel(
    part_ptr,
    lora_ptr,
    M,
    RANK: tl.constexpr,
    HC: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Reduce split-K partials and write the SiLU'd lora columns."""
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < RANK
    N: tl.constexpr = RANK + HC

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k in tl.static_range(SPLIT_K):
        p = tl.load(
            part_ptr + (k * M + m_offs[:, None]) * N + n_offs[None, :],
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += p

    # Production rounding boundary: bf16 GEMM output, fp32 SiLU.
    acc_bf16 = acc.to(tl.bfloat16).to(tl.float32)
    silu_in = acc_bf16 / HC
    epilog = (silu_in * tl.sigmoid(silu_in)).to(tl.bfloat16)

    tl.store(
        lora_ptr + m_offs[:, None] * RANK + n_offs[None, :],
        epilog,
        mask=m_mask[:, None] & n_mask[None, :],
    )
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _hc_down_silu_inj_kernel(
    part_ptr,
    inj_ptr,
    M,
    RANK: tl.constexpr,
    HC: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Reduce split-K partials for the injection-logit columns (passthrough).

    Kept as its own tiny kernel: a [BLOCK_M, BLOCK_N]-wide masked store into
    a [M, HC] tensor with HC << BLOCK_N proved unreliable across triton
    layouts (rows silently unwritten), while the row-program formulation is
    trivially correct.
    """
    row = tl.program_id(0)
    offs_hc = tl.arange(0, BLOCK_H)
    mask_hc = offs_hc < HC
    N: tl.constexpr = RANK + HC

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    acc = tl.zeros([BLOCK_H], tl.float32)
    for k in tl.static_range(SPLIT_K):
        p = tl.load(
            part_ptr + (k * M + row) * N + RANK + offs_hc,
            mask=mask_hc,
            other=0.0,
        )
        acc += p

    tl.store(inj_ptr + row * HC + offs_hc, acc.to(tl.bfloat16), mask=mask_hc)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()


def _hc_down_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    lora: torch.Tensor,
    injection: torch.Tensor,
) -> None:
    """Fused HC down projection + SiLU. See the module docstring."""
    M, K = x.shape
    n_compute = lora.shape[1] + injection.shape[1]
    rank = lora.shape[1]
    hc = injection.shape[1]
    w = weight[:n_compute]
    launch_pdl = current_platform.is_arch_support_pdl()

    if M <= 4:
        BLOCK_N = 32
        BLOCK_K = 256
        _hc_down_silu_fma_kernel[(triton.cdiv(n_compute, BLOCK_N), M)](
            x,
            w,
            lora,
            injection,
            M,
            K,
            RANK=rank,
            HC=hc,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            EVEN_K=(K % BLOCK_K == 0),
            launch_pdl=launch_pdl,
            num_warps=4,
        )
        return

    split_k, block_m, block_n = _SPLITK_TABLE[min(M, MAX_FUSED_M)]
    block_k = 64
    grid = (triton.cdiv(M, block_m) * triton.cdiv(n_compute, block_n), split_k)
    partials = torch.empty(
        (split_k, M, n_compute), dtype=torch.float32, device=x.device
    )
    # Unmasked K loads require the whole split-K tiling to divide K evenly;
    # otherwise the tail chunk of the last split would read past K.
    _hc_down_silu_dot_kernel[grid](
        x,
        w,
        partials,
        M,
        K,
        RANK=rank,
        HC=hc,
        SPLIT_K=split_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        EVEN_K=(K % (split_k * block_k) == 0),
        launch_pdl=launch_pdl,
        num_warps=4,
        num_stages=3,
    )
    _hc_down_silu_epilogue_kernel[(triton.cdiv(M, block_m),)](
        partials,
        lora,
        M,
        RANK=rank,
        HC=hc,
        SPLIT_K=split_k,
        BLOCK_M=block_m,
        BLOCK_N=triton.next_power_of_2(n_compute),
        launch_pdl=launch_pdl,
        num_warps=4,
    )
    _hc_down_silu_inj_kernel[(M,)](
        partials,
        injection,
        M,
        RANK=rank,
        HC=hc,
        SPLIT_K=split_k,
        BLOCK_H=triton.next_power_of_2(hc),
        launch_pdl=launch_pdl,
        num_warps=1,
    )


def _hc_down_silu_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    lora: torch.Tensor,
    injection: torch.Tensor,
) -> None:
    return None


direct_register_custom_op(
    op_name="qwen4_exp_hc_down_silu",
    op_func=_hc_down_silu,
    mutates_args=["lora", "injection"],
    fake_impl=_hc_down_silu_fake,
)


def hc_down_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused HC down projection + SiLU (triton).

    Args:
        x: Normalized hyper-hidden input, [M, K] bf16.
        weight: Merged down+inject weight, [N, K] bf16 (only the first
            ``rank + hc_count`` rows are computed).
        rank: Number of low-rank output columns.
        hc_count: Number of injection-logit output columns.

    Returns:
        LoRA activations [M, rank] and injection logits [M, hc_count].
    """
    lora = torch.empty(
        (x.shape[0], rank), dtype=torch.bfloat16, device=x.device
    )
    injection = torch.empty(
        (x.shape[0], hc_count), dtype=torch.bfloat16, device=x.device
    )
    torch.ops.vllm.qwen4_exp_hc_down_silu(x, weight, lora, injection)
    return lora, injection


__all__ = [
    "MAX_FUSED_M",
    "hc_down_silu",
]
