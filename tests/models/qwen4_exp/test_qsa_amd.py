# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.models.qwen4_exp.amd import (
    model as _qwen4_exp_model,  # noqa: F401
)
from vllm.models.qwen4_exp.amd import ple_layer as ple_layer_module
from vllm.models.qwen4_exp.amd.indexer_qsa import (
    apply_qsa_rmsnorm,
    apply_qsa_rope,
)
from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="AMD QSA tests run on CUDA and ROCm",
)

requires_qsa_kernels = pytest.mark.skipif(
    not HAS_TRITON,
    reason="AMD QSA kernels require Triton",
)


def test_ple_ngram_embedding_custom_op_uses_resident_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_name = "model.layers.0.ple"
    layer = ple_layer_module.Qwen4ExpPLELayer.__new__(ple_layer_module.Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = torch.nn.Module()
    layer.ple_embedding.ngram_embedding = torch.nn.Embedding(8, 3)
    context = SimpleNamespace(no_compile_layers={layer_name: layer})
    monkeypatch.setattr(ple_layer_module, "get_forward_context", lambda: context)

    ngram_ids = torch.tensor([[0, 1], [2, 3]])
    output = torch.empty(2, 6)
    ple_layer_module.qwen4_exp_amd_ple_ngram_embedding(
        ngram_ids,
        output,
        layer_name,
    )

    expected = layer.ple_embedding.ngram_embedding(ngram_ids).flatten(-2)
    torch.testing.assert_close(output, expected)


def _qsa_sparse_paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    output = torch.zeros_like(q)
    repeats = q.shape[1] // k_cache.shape[2]
    page_size = k_cache.shape[1]
    # Mirror the kernel's page-table-width guard: indices resolving to a
    # logical page beyond the block table are dropped, not clamped.
    max_logical = block_table.shape[1] * page_size
    for row in range(q.shape[0]):
        logical = logical_indices[row]
        logical = logical[(logical >= 0) & (logical < max_logical)].long()
        if not logical.numel():
            continue
        request = token_to_req[row].long()
        pages = block_table[request, logical // page_size].long()
        offsets = logical % page_size
        keys = k_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        values = v_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float())
        probabilities = torch.softmax(scores * softmax_scale, dim=-1)
        output[row] = torch.einsum("hk,khd->hd", probabilities, values.float()).to(
            q.dtype
        )
    return output


def test_qsa_rope_uses_platform_dispatch() -> None:
    tensor = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    positions = torch.tensor([0, 1])
    calls = []

    def apply_rotary_emb(
        rotary_input: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        calls.append((rotary_input, cos, sin))
        return rotary_input + 1

    rotary_emb = SimpleNamespace(
        rotary_dim=2,
        apply_rotary_emb=apply_rotary_emb,
        _match_cos_sin_cache_dtype=lambda _: torch.zeros(2, 4),
    )

    output = apply_qsa_rope(rotary_emb, positions, tensor)

    assert len(calls) == 1
    torch.testing.assert_close(output[..., :2], tensor[..., :2] + 1)
    torch.testing.assert_close(output[..., 2:], tensor[..., 2:])


def test_qsa_rmsnorm_uses_portable_implementation(default_vllm_config) -> None:
    norm = GemmaRMSNorm(4, eps=1e-6)
    norm.weight.data.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    output = apply_qsa_rmsnorm(norm, tensor)

    torch.testing.assert_close(output, norm.forward_native(tensor))


def test_qsa_selection_uses_portable_topk_on_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 2
    token_topk = 8
    compress_ratio = 4
    block_topk = token_topk // compress_ratio
    blocks = torch.empty((rows, block_topk), dtype=torch.int32)
    visible_blocks = torch.tensor([4, 6], dtype=torch.int32)
    logits = torch.empty((rows, 8), dtype=torch.float32)
    selection_output = torch.empty(
        (rows, token_topk + compress_ratio - 1), dtype=torch.int32
    )
    topk_call: dict[str, Any] = {}

    monkeypatch.setattr(
        qsa_ops,
        "qsa_mqa_paged",
        lambda *args, **kwargs: (logits, visible_blocks),
    )
    monkeypatch.setattr(qsa_ops.current_platform, "is_cuda", lambda: False)

    def top_k_per_row_decode(
        topk_logits,
        next_n,
        seq_lens,
        raw_topk_indices,
        num_rows,
        stride0,
        stride1,
        topk_tokens,
    ) -> None:
        topk_call.update(
            logits=topk_logits,
            next_n=next_n,
            seq_lens=seq_lens,
            raw_topk_indices=raw_topk_indices,
            num_rows=num_rows,
            strides=(stride0, stride1),
            topk_tokens=topk_tokens,
        )
        raw_topk_indices.zero_()

    def expand_qsa_block_indices(*args) -> None:
        args[-1].fill_(-1)

    monkeypatch.setattr(qsa_ops.ops, "top_k_per_row_decode", top_k_per_row_decode)
    monkeypatch.setattr(
        qsa_ops,
        "expand_qsa_block_indices_cuda",
        expand_qsa_block_indices,
    )

    output = qsa_ops.qsa_select_paged_tokens(
        torch.empty((rows, 1, 1)),
        torch.empty((1, 4, 1, 1)),
        torch.empty((1, 2), dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.arange(rows, dtype=torch.int32),
        torch.tensor([8], dtype=torch.int32),
        token_topk,
        compress_ratio,
        selection_output,
    )

    assert output is selection_output
    assert torch.all(output == -1)
    assert topk_call["logits"] is logits
    assert topk_call["next_n"] == 1
    assert topk_call["seq_lens"] is visible_blocks
    assert topk_call["raw_topk_indices"].shape == blocks.shape
    assert topk_call["num_rows"] == rows
    assert topk_call["strides"] == (logits.stride(0), logits.stride(1))
    assert topk_call["topk_tokens"] == block_topk


@requires_qsa_kernels
def _run_sparse_paged_case(
    num_rows: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    selection_width: int,
    num_requests: int = 2,
    invalid_tail: int = 0,
    out_of_range: bool = False,
) -> None:
    """One differential case of qsa_sparse_paged_attention vs reference.

    Args:
        num_rows: query rows (drives the split-K profile dispatch).
        num_query_heads / num_kv_heads: grouped-query geometry.
        head_dim: per-head dimension.
        page_size: KV cache page size.
        selection_width: logical selection width per row.
        num_requests: request count driving the block table.
        invalid_tail: trailing columns of logical_indices set to -1.
        out_of_range: include indices beyond the block-table width.

    """
    torch.manual_seed(num_rows * 31 + num_query_heads + head_dim)
    pages_per_request = 8
    num_cache_blocks = num_requests * pages_per_request
    k_cache = torch.randn(
        num_cache_blocks, page_size, num_kv_heads, head_dim,
        dtype=torch.bfloat16, device="cuda",
    )
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(
        num_rows, num_query_heads, head_dim,
        dtype=torch.bfloat16, device="cuda",
    )
    gate = torch.randn_like(q)
    token_to_req = (
        torch.arange(num_rows, device="cuda", dtype=torch.int32) % num_requests
    )
    block_table = torch.randperm(
        num_cache_blocks, device="cuda", dtype=torch.int64
    ).reshape(num_requests, pages_per_request).to(torch.int32)
    max_logical = pages_per_request * page_size
    logical_indices = torch.randint(
        0, max_logical, (num_rows, selection_width), device="cuda",
        dtype=torch.int32,
    )
    if invalid_tail:
        logical_indices[:, -invalid_tail:] = -1
    if out_of_range:
        logical_indices[:, 0] = max_logical + page_size

    ungated = qsa_ops.qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req
    )
    expected = _qsa_sparse_paged_attention_reference(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req,
        q.shape[-1] ** -0.5,
    )
    torch.testing.assert_close(ungated, expected, rtol=2e-2, atol=2e-2)

    gated = qsa_ops.qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req,
        output_gate=gate,
    )
    gated_expected = (expected.float() * torch.sigmoid(gate.float())).to(
        torch.bfloat16
    )
    torch.testing.assert_close(gated, gated_expected, rtol=2e-2, atol=2e-2)


@requires_qsa_kernels
@pytest.mark.parametrize(
    ("num_rows", "num_kv_heads"),
    [
        pytest.param(1, 8, id="profile_tiny"),
        pytest.param(2, 4, id="profile_8"),
        pytest.param(3, 8, id="profile_8b"),
        pytest.param(4, 2, id="profile_8c"),
        pytest.param(8, 1, id="profile_8_limit"),
        pytest.param(9, 2, id="profile_32"),
        pytest.param(16, 1, id="profile_32b"),
        pytest.param(32, 1, id="profile_32_limit"),
        pytest.param(33, 1, id="profile_256"),
        pytest.param(64, 4, id="profile_256b"),
        pytest.param(128, 2, id="profile_256c"),
        pytest.param(256, 1, id="profile_256_limit"),
        pytest.param(257, 1, id="profile_512"),
        pytest.param(512, 1, id="profile_512_limit"),
        pytest.param(513, 1, id="profile_prefill"),
        pytest.param(2048, 1, id="profile_prefill_large"),
    ],
)
def test_qsa_sparse_paged_attention_profile_boundaries(
    num_rows: int, num_kv_heads: int
) -> None:
    _run_sparse_paged_case(
        num_rows=num_rows,
        num_query_heads=24,
        num_kv_heads=num_kv_heads,
        head_dim=64,
        page_size=16,
        selection_width=64,
    )


@requires_qsa_kernels
@pytest.mark.parametrize(
    ("num_query_heads", "num_kv_heads"),
    [
        pytest.param(8, 8, id="group1_mqa"),
        pytest.param(9, 3, id="group3_nonpow2"),
        pytest.param(24, 1, id="group24"),
    ],
)
def test_qsa_sparse_paged_attention_group_sizes(
    num_query_heads: int, num_kv_heads: int
) -> None:
    _run_sparse_paged_case(
        num_rows=17,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=64,
        page_size=16,
        selection_width=64,
    )


@requires_qsa_kernels
@pytest.mark.parametrize("head_dim", [16, 32, 64, 256])
def test_qsa_sparse_paged_attention_head_dims(head_dim: int) -> None:
    _run_sparse_paged_case(
        num_rows=33,
        num_query_heads=12,
        num_kv_heads=2,
        head_dim=head_dim,
        page_size=16,
        selection_width=64,
    )


@requires_qsa_kernels
@pytest.mark.parametrize(
    "selection_width",
    [1, 16, 17, 128, 2048],
    ids=["below_block", "exact_block", "block_plus1", "multi_tile", "deep_splitk"],
)
def test_qsa_sparse_paged_attention_selection_widths(selection_width: int) -> None:
    _run_sparse_paged_case(
        num_rows=17,
        num_query_heads=24,
        num_kv_heads=4,
        head_dim=64,
        page_size=16,
        selection_width=selection_width,
    )


@requires_qsa_kernels
def test_qsa_sparse_paged_attention_all_invalid_indices() -> None:
    """All -1 indices produce zero outputs without touching the cache."""
    torch.manual_seed(11)
    num_rows, num_kv_heads, head_dim, page_size = 9, 2, 64, 16
    k_cache = torch.randn(
        16, page_size, num_kv_heads, head_dim,
        dtype=torch.bfloat16, device="cuda",
    )
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(
        num_rows, 8, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    logical_indices = torch.full(
        (num_rows, 32), -1, dtype=torch.int32, device="cuda"
    )
    token_to_req = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    block_table = torch.arange(16, dtype=torch.int32, device="cuda").reshape(1, 16)

    actual = qsa_ops.qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req
    )
    assert torch.all(actual == 0)


@requires_qsa_kernels
def test_qsa_sparse_paged_attention_out_of_range_indices() -> None:
    """Indices beyond the block table are masked, matching the reference."""
    _run_sparse_paged_case(
        num_rows=9,
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=64,
        page_size=16,
        selection_width=48,
        out_of_range=True,
    )


@requires_qsa_kernels
def test_qsa_sparse_paged_attention_single_request() -> None:
    """All rows share one request; exercises intra-request gather only."""
    _run_sparse_paged_case(
        num_rows=48,
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=64,
        page_size=16,
        selection_width=64,
        num_requests=1,
    )


@requires_qsa_kernels
def test_qsa_sparse_paged_attention_invalid_tail() -> None:
    """Trailing -1 columns exercise the per-tile validity mask."""
    _run_sparse_paged_case(
        num_rows=17,
        num_query_heads=24,
        num_kv_heads=4,
        head_dim=64,
        page_size=16,
        selection_width=64,
        invalid_tail=13,
    )
