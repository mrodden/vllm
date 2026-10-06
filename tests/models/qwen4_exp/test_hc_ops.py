# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.cute_dsl.hc_down_silu import hc_down_silu
from vllm.models.qwen4_exp.nvidia.ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="HC kernels require CUDA and Triton",
)

HC = 4
HIDDEN_SIZE = 2560
HYPER_HIDDEN_SIZE = HC * HIDDEN_SIZE
EPS = 1e-6
LORA_RANK = 320
DOWN_N = LORA_RANK + HC + 12  # merged down+inject weight, 16-row padded

requires_sm90 = pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="fused HC down+SiLU requires SM90+",
)


def test_grouped_gemma_rmsnorm() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = grouped_gemma_rmsnorm(x, weight, EPS, HC)

    grouped = x.float().unflatten(-1, (HC, HIDDEN_SIZE))
    variance = grouped.square().mean(-1, keepdim=True)
    expected = grouped * torch.rsqrt(variance + EPS)
    expected = expected.flatten(-2) * (1.0 + weight.float())
    torch.testing.assert_close(actual, expected.to(torch.bfloat16))


def test_hc_gate_mix() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = hc_gate_mix(x, gate, HC)
    expected = (
        torch.sigmoid(gate.float().unflatten(-1, (HC, HIDDEN_SIZE)))
        * x.float().unflatten(-1, (HC, HIDDEN_SIZE))
    ).mean(-2)

    torch.testing.assert_close(actual, expected.to(torch.bfloat16))


def test_hc_combine() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    injection = torch.randn(2, HC, dtype=torch.bfloat16, device="cuda")

    actual = hc_combine(residual, block_output, injection, HC)
    injection_weight = 2.0 * torch.sigmoid(injection.float() / HC)
    expected = residual.float().unflatten(-1, (HC, HIDDEN_SIZE))
    expected = expected + block_output.float().unsqueeze(
        -2
    ) * injection_weight.unsqueeze(-1)

    torch.testing.assert_close(actual, expected.flatten(-2).to(torch.bfloat16))


def test_hc_combine_unit_injection() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = hc_combine(residual, block_output, None, HC)
    expected = residual.unflatten(-1, (HC, HIDDEN_SIZE))
    expected = expected + block_output.unsqueeze(-2)

    assert torch.equal(actual, expected.flatten(-2))


def test_hc_combine_norm() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    injection = torch.randn(2, HC, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual, actual_norm = hc_combine_norm(
        residual, block_output, injection, weight, EPS, HC
    )

    injection_weight = 2.0 * torch.sigmoid(injection.float() / HC)
    expected = residual.float().unflatten(-1, (HC, HIDDEN_SIZE))
    expected = expected + block_output.float().unsqueeze(
        -2
    ) * injection_weight.unsqueeze(-1)
    expected = expected.flatten(-2).to(residual.dtype)
    grouped = expected.float().unflatten(-1, (HC, HIDDEN_SIZE))
    variance = grouped.square().mean(-1, keepdim=True)
    expected_norm = grouped * torch.rsqrt(variance + EPS)
    expected_norm = expected_norm.flatten(-2) * (1.0 + weight.float())

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_norm, expected_norm.to(torch.bfloat16))


@pytest.mark.parametrize("num_tokens", [1, 17, 2048])
def test_hc_combine_norm_unit_injection(num_tokens: int) -> None:
    torch.manual_seed(0)
    embedding = torch.randn(
        num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    hidden = torch.randn(
        num_tokens, HC, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual, actual_norm = hc_combine_norm(
        hidden.flatten(1), embedding, None, weight, EPS, HC
    )

    expected = (hidden + embedding.unsqueeze(1)).flatten(1)
    expected_norm = grouped_gemma_rmsnorm(expected, weight, EPS, HC)
    assert torch.equal(actual, expected)
    torch.testing.assert_close(actual_norm, expected_norm)


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5, 17, 48, 49, 64, 128])
def test_hc_down_silu_triton(num_tokens: int) -> None:
    """Shared triton fused kernel must match the unfused eager reference.

    48/49 straddle MAX_FUSED_M: at/under it the fused kernel runs; above
    it the caller falls back, and both paths must match the reference.
    """
    from vllm.models.qwen4_exp.common.hc_down_silu import hc_down_silu as triton_fused

    torch.manual_seed(0)
    x = torch.randn(num_tokens, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(DOWN_N, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    lora, injection = triton_fused(x, weight, LORA_RANK, HC)

    # Unfused reference: GEMM (fp32 acc -> bf16) -> split -> hc_silu.
    down = (x.float() @ weight[: LORA_RANK + HC].float().t()).to(torch.bfloat16)
    ref_lora = hc_silu(down[:, :LORA_RANK], HC)
    ref_inj = down[:, LORA_RANK : LORA_RANK + HC]
    torch.testing.assert_close(lora, ref_lora, rtol=0.01, atol=0.01)
    torch.testing.assert_close(injection, ref_inj, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("lora_rank", [64, 320, 512])
@pytest.mark.parametrize("hc_count", [1, 4, 8])
def test_hc_down_silu_triton_shape_sweep(lora_rank: int, hc_count: int) -> None:
    """Vary rank/hc_count and the merged weight's row count.

    DOWN_N stays a multiple of 8 at the pad boundary; also exercise the
    un-padded rank+hc row count (the kernel must mask it correctly).
    """
    from vllm.models.qwen4_exp.common.hc_down_silu import hc_down_silu as triton_fused

    for num_tokens in (1, 17, 48):
        for down_n in (lora_rank + hc_count, lora_rank + hc_count + 12):
            torch.manual_seed(lora_rank * 100 + hc_count + num_tokens + down_n)
            x = torch.randn(
                num_tokens, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
            )
            weight = torch.randn(
                down_n, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
            )
            lora, injection = triton_fused(x, weight, lora_rank, hc_count)
            down = (
                x.float() @ weight[: lora_rank + hc_count].float().t()
            ).to(torch.bfloat16)
            ref_lora = hc_silu(down[:, :lora_rank], hc_count)
            ref_inj = down[:, lora_rank : lora_rank + hc_count]
            torch.testing.assert_close(lora, ref_lora, rtol=0.01, atol=0.01)
            torch.testing.assert_close(injection, ref_inj, rtol=0.01, atol=0.01)


def _build_hc_module(path: str, use_combine: bool = True):
    """Instantiate a HyperConnection module from either path (CPU weights)."""
    from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig

    mod = importlib.import_module(path)
    cfg = HyperConnectionConfig(
        hidden_size=HIDDEN_SIZE,
        hc_count=HC,
        hc_lowrank=LORA_RANK,
        rms_norm_eps=EPS,
        params_dtype=torch.bfloat16,
    )
    m = mod.GatedResidual(cfg, use_combine=use_combine, prefix="test_hc")
    return m.to("cuda")


@pytest.mark.parametrize("num_tokens", [1, 3, 17, 47, 48, 49])
@pytest.mark.parametrize("mod_path", ["vllm.models.qwen4_exp.amd.hyperconnection",
                                      "vllm.models.qwen4_exp.nvidia.hyperconnection"])
def test_down_and_inject_caller(num_tokens: int, mod_path: str) -> None:
    """_down_and_inject (fused) must match the module's unfused branch.

    Exercises the caller contract (weight layout, split sizes, output
    dtypes) — the level that direct op tests miss. Each path runs in a
    subprocess because importing both hyperconnection modules into one
    process double-registers the shared custom-op names.
    """
    import os
    import subprocess
    import sys

    code = f"""
import importlib
import os

import torch

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29617")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import vllm.distributed.parallel_state as pstate
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig

mod = importlib.import_module({mod_path!r})
cfg = HyperConnectionConfig(
    hidden_size={HIDDEN_SIZE},
    hc_count={HC},
    hc_lowrank={LORA_RANK},
    rms_norm_eps={EPS},
    params_dtype=torch.bfloat16,
)
with set_current_vllm_config(VllmConfig()):
    pstate.init_distributed_environment(backend="gloo")
    pstate.initialize_model_parallel(tensor_model_parallel_size=1)
    m = mod.GatedResidual(cfg, use_combine=True, prefix="test_hc").to("cuda")
    torch.manual_seed(0)
    xn = torch.randn(
        {num_tokens}, {HYPER_HIDDEN_SIZE}, dtype=torch.bfloat16, device="cuda"
    )
    lora, injection = m._down_and_inject(xn)
    assert lora.shape == ({num_tokens}, {LORA_RANK}), lora.shape
    assert injection is not None and injection.shape == ({num_tokens}, {HC})
    lora_ref, injection_ref = m._down_and_inject(torch.cat([xn] * 2, dim=0))
    lora_ref = lora_ref[:{num_tokens}]
    injection_ref = injection_ref[:{num_tokens}]
    torch.testing.assert_close(lora, lora_ref, rtol=0.02, atol=0.02)
    torch.testing.assert_close(injection, injection_ref, rtol=0.02, atol=0.02)
print("OK")
"""
    env = dict(os.environ)
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=repo_root,
        env=env,
    )
    assert proc.returncode == 0, (
        f"caller check failed for {mod_path} M={num_tokens}:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


@requires_sm90
@pytest.mark.parametrize("num_tokens", [1, 3, 5, 17, 48])
def test_hc_down_silu_fused(num_tokens: int) -> None:
    # Compare computed columns with the unfused ll_bf16 + hc_silu reference.
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm

    torch.manual_seed(0)
    x = torch.randn(num_tokens, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(DOWN_N, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    lora, injection = hc_down_silu(x, weight, LORA_RANK, HC)

    down = ll_bf16_gemm(x, weight).to(torch.bfloat16)
    torch.testing.assert_close(
        lora, hc_silu(down[:, :LORA_RANK], HC), rtol=0.01, atol=0.01
    )
    torch.testing.assert_close(
        injection, down[:, LORA_RANK : LORA_RANK + HC], rtol=0.01, atol=0.01
    )
