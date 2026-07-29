# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QPhiScore:
    """Sequence probability of an exploration-memory target under q_phi."""

    log_prob: float
    prob: float
    token_count: int


@dataclass(frozen=True)
class QPhiRLLoss:
    """REINFORCE-style q_phi loss for the variational EAPO objective."""

    loss: torch.Tensor
    policy_gradient_loss: torch.Tensor
    kl_loss: torch.Tensor
    approximate_kl: torch.Tensor


def build_explore_memory_target(explore: str, memory: str) -> str:
    """Serialize [e, m] in the same structured format used by the agent."""

    return f"<explore>{explore}</explore>\n<memory>{memory}</memory>"


def compute_target_log_prob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_length: int,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log q_phi(target | prompt) for one prompt-target sequence.

    Args:
        model: Causal language model used as q_phi.
        input_ids: Tensor of shape [1, prompt_len + target_len].
        prompt_length: Number of prompt tokens, i.e. the state s.
        attention_mask: Optional attention mask aligned with input_ids.

    Returns:
        Tuple of ``(log_prob_sum, target_token_count)``.
    """

    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError(f"input_ids must have shape [1, seq_len], got {tuple(input_ids.shape)}")
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive so target tokens have a prefix")
    if prompt_length >= input_ids.size(1):
        raise ValueError("prompt_length must be smaller than the full prompt-target sequence length")

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # Shift position k predicts token k+1. Target tokens begin at original
    # position prompt_length, so their predicting logits begin at prompt_length - 1.
    target_mask = torch.zeros_like(labels, dtype=torch.bool)
    target_mask[:, prompt_length - 1 :] = True
    if attention_mask is not None:
        target_mask &= attention_mask[:, 1:].bool()

    return log_probs[target_mask].sum(), target_mask.sum()


def log_prob_to_probability(log_prob: torch.Tensor, token_count: torch.Tensor, mode: str = "sequence") -> torch.Tensor:
    """Convert a q_phi target log-probability to a reward-scale probability."""

    if mode == "sequence":
        return torch.exp(torch.clamp(log_prob, min=-60.0, max=0.0))
    if mode == "token_mean":
        denom = torch.clamp(token_count.to(dtype=log_prob.dtype), min=1.0)
        return torch.exp(torch.clamp(log_prob / denom, min=-60.0, max=0.0))
    if mode == "log_prob":
        return log_prob
    raise ValueError(f"Unknown q_phi probability mode: {mode}")


def compute_q_phi_reinforce_loss(
    q_log_prob: torch.Tensor,
    q_value: torch.Tensor | float,
    ref_log_prob: torch.Tensor | None = None,
    beta: float = 1.0,
    kl_coef: float = 1.0,
) -> QPhiRLLoss:
    """Compute the RL loss for q_phi from the EAPO variational objective.

    The paper optimizes
    ``max_q beta * E_{e,m~q(.|s)}[Q(s,e,m)] - KL(q(e,m|s) || p(e,m|s))``.
    For a sampled ``[e, m]`` this uses the score-function estimator with
    learning signal ``beta * Q - kl_coef * (log q_phi - log p)``.
    """

    if not torch.is_tensor(q_value):
        q_value = torch.tensor(float(q_value), dtype=q_log_prob.dtype, device=q_log_prob.device)
    else:
        q_value = q_value.to(dtype=q_log_prob.dtype, device=q_log_prob.device)

    if ref_log_prob is None or float(kl_coef) == 0.0:
        approximate_kl = q_log_prob.new_zeros(())
        kl_loss = q_log_prob.new_zeros(())
    else:
        approximate_kl = q_log_prob.detach() - ref_log_prob.detach().to(dtype=q_log_prob.dtype, device=q_log_prob.device)
        kl_loss = float(kl_coef) * approximate_kl

    expected_q_loss = -float(beta) * q_value.detach() * q_log_prob
    learning_signal = float(beta) * q_value.detach() - kl_loss.detach()
    loss = -learning_signal * q_log_prob
    return QPhiRLLoss(
        loss=loss,
        policy_gradient_loss=expected_q_loss,
        kl_loss=kl_loss,
        approximate_kl=approximate_kl,
    )
