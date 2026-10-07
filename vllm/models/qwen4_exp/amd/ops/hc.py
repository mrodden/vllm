# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD HyperConnection ops; thin re-export of the shared registrations."""

from vllm.models.qwen4_exp.common.hc_ops import (  # noqa: F401
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
