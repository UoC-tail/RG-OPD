# Copyright 2026 The RG-OPD Authors
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
"""Reward-Gated On-Policy Distillation (RG-OPD) loss.

Implements Equations 1-4 of Akhondzadeh et al., "Reward-Gated On-Policy Distillation",
arXiv:2607.04037 (https://arxiv.org/abs/2607.04037).

Setting. At each training step the student samples trajectories on-policy; a verifier
assigns each trajectory a scalar reward/advantage; a fixed teacher provides token-level
top-K log-probabilities over the tokens the student actually sampled. On-policy
distillation (OPD) applies a reverse-KL distillation loss unconditionally to every
trajectory. RG-OPD instead *gates* each trajectory's contribution to that loss by
whether the teacher is directionally informative given the observed reward: for a
reward-positive trajectory the teacher is only trusted if it assigns the trajectory
higher likelihood than the student does; for a reward-negative trajectory, only if it
assigns lower likelihood. This bridges the sparse, trajectory-level verifier reward with
the dense, token-level teacher signal.

This module has two independent halves that compose into the full loss (`compute_rgopd_loss`):
  - `compute_topk_reverse_kl`: the distillation loss itself (Eq. 4).
  - `compute_reward_teacher_gate`: the trajectory-level keep/drop mask (Eq. 1-2).
Setting `gate.enabled: False` in config recovers plain (ungated) top-K reverse-KL
on-policy distillation -- this is the "Reverse-KL" baseline reported in the paper.
"""

from typing import Any, Optional

import torch
import torch.nn.functional as F

from verl.trainer.ppo.core_algos import agg_loss

__all__ = [
    "compute_topk_reverse_kl",
    "compute_reward_teacher_gate",
    "compute_rgopd_loss",
]


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    """Read `key` from `config`, whether it's a plain dict, an OmegaConf DictConfig, or a dataclass."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _add_tail_bucket(topk_log_probs: torch.Tensor) -> torch.Tensor:
    """Append the aggregated out-of-top-K probability mass as one extra "tail" bucket.

    `topk_log_probs` holds log pi(y | s) for the K highest-probability tokens at each
    position -- not a normalized distribution on its own. Appending a bucket for
    "everything outside the top-K" turns it into a proper (K+1)-way distribution, and
    keeps that residual mass differentiable instead of just discarding it. This is the
    "tail correction" term of Eq. 4 (the second summand); Appendix A.2 shows removing it
    measurably hurts, even though its contribution to the total KL is small, because it
    still carries gradient.
    """
    log_topk_mass = torch.logsumexp(topk_log_probs, dim=-1, keepdim=True)
    log_topk_mass = torch.clamp(log_topk_mass, max=-1e-7)  # keep 1 - mass > 0 so log1p(-mass) is finite
    log_tail_mass = torch.log(-torch.expm1(log_topk_mass))
    return torch.cat([topk_log_probs, log_tail_mass], dim=-1)


def compute_topk_reverse_kl(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    add_tail: bool = True,
) -> torch.Tensor:
    """Top-K reverse-KL distillation loss (Eq. 4).

    `student_topk_log_probs` and `teacher_topk_log_probs` are both `(batch, response_len,
    K)` log-probabilities of the *same* K token ids at each position -- i.e. student and
    teacher must already be evaluated at a shared index set (see
    `recipe/rgopd/dp_actor.py` for how that index set is constructed for the external and
    self-teacher cases). With `add_tail=True` (default; K=50 in the paper) this computes
    the top-K explicit sum plus tail-mass correction of Eq. 4; with `add_tail=False` it
    renormalizes the top-K support into its own K-way distribution and omits the residual
    term (the "no-tail" ablation in Appendix Table 2).

    Returns per-token loss, shape `(batch, response_len)`.
    """
    student_lp, teacher_lp = student_topk_log_probs, teacher_topk_log_probs
    if add_tail:
        student_lp = _add_tail_bucket(student_lp)
        teacher_lp = _add_tail_bucket(teacher_lp)
    else:
        student_lp = student_lp - torch.logsumexp(student_lp, dim=-1, keepdim=True)
        teacher_lp = teacher_lp - torch.logsumexp(teacher_lp, dim=-1, keepdim=True)

    # torch.nn.functional.kl_div(input, target, log_target=True) computes KL(target || exp(input)).
    # We want the reverse-KL KL(student || teacher) that the paper distills with (Sec. 3,
    # "we ... apply the top-k reverse-KL ... distillation loss"), so target=student, input=teacher.
    per_token_kl = F.kl_div(teacher_lp, student_lp, reduction="none", log_target=True)
    return per_token_kl.sum(-1)


def compute_reward_teacher_gate(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    margin: float = 0.0,
    chunk_size: int = 1,
    failure_includes_zero: bool = False,
    zero_adv: str = "mask",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reward-teacher likelihood gate (Eq. 1-2), applied over token chunks.

    Over a span of response tokens, define the teacher/student log-likelihoods of the
    sampled tokens (Eq. 1):
        L_T = sum_t m_t log pi_T(y_t | s_t),   L_S = sum_t m_t log pi_theta(y_t | s_t)
    and let A_i be the trajectory's (GRPO) advantage. The span is kept iff the
    teacher-vs-student likelihood gap is directionally consistent with the reward (Eq. 2):
        g = 1[ (A_i > 0 and L_T > L_S + margin) or (A_i < 0 and L_T < L_S - margin) ]

    `chunk_size` sets the span the comparison is aggregated over, matching the
    `rt_gate_chunk_size` knob in the original TTT-Reflex implementation this recipe was
    ported from (verl/verl/trainer/ppo/core_algos.py there):
      - `1` (default): per-token. Each token is gated on its own log-prob difference.
      - `k > 1`: contiguous k-token chunks share one decision.
      - `<= 0`: one decision for the whole trajectory.

    Per-token is the default because the whole-trajectory form is degenerate in practice.
    L_T - L_S summed over a trajectory is dominated by a minority of tokens where the
    student is very peaked and the teacher is not, so it is essentially always negative:
    measured on Qwen2.5-1.5B rollouts scored by a Qwen2.5-14B teacher, 0 of 43
    reward-positive trajectories had L_T > L_S, and the gate kept 0% of reward-positive
    and 100% of reward-negative trajectories -- the inverse of the intent. The signal is
    there at the token level, where the teacher assigns the higher log-prob on ~69% of
    tokens in reward-positive trajectories vs ~53% in reward-negative ones; per-token
    gating recovers it (68% of reward-positive tokens kept, 42% of reward-negative).

    Zero-advantage trajectories (A_i == 0) come from degenerate GRPO groups where every
    sample scored identically -- ~56% of trajectories in the run measured above. They
    carry no reward signal to be consistent *with*, so by default they are masked out
    entirely (`zero_adv="mask"`), as in TTT-Reflex. Set `failure_includes_zero=True` to
    treat them as failures instead, or `zero_adv="keep"` to pass them through ungated.

    `teacher_log_probs` / `student_log_probs` are `(batch, response_len)` log-probs of the
    tokens actually sampled (distinct from the top-K support used by the distillation
    loss). `advantages` is `(batch, response_len)`, the (typically token-broadcast)
    per-trajectory advantage.

    Returns:
        gate_mask: `(batch, response_len)` in {0, 1}.
        metrics: scalar logging metrics (kept-token fraction, kept counts by reward sign).
    """
    with torch.no_grad():
        mask_f = response_mask.to(teacher_log_probs.dtype)
        B, T = student_log_probs.shape

        seq_adv = (advantages * response_mask).sum(-1) / response_mask.sum(-1).clamp(min=1)
        is_positive = seq_adv > 0
        if failure_includes_zero:
            is_failure = seq_adv <= 0
        else:
            is_failure = seq_adv < 0
        is_zero = (~is_positive) & (~is_failure)

        diff = (teacher_log_probs - student_log_probs) * mask_f
        if chunk_size is None or chunk_size <= 0:
            span = diff.sum(-1, keepdim=True).expand(B, T)
        elif chunk_size == 1:
            span = diff
        else:
            pad = (-T) % chunk_size
            padded = torch.nn.functional.pad(diff, (0, pad))
            span = padded.view(B, -1, chunk_size).sum(-1, keepdim=True)
            span = span.expand(-1, -1, chunk_size).reshape(B, -1)[:, :T]

        keep = (is_positive.unsqueeze(1) & (span > margin)) | (is_failure.unsqueeze(1) & (span < -margin))
        if zero_adv == "keep" and not failure_includes_zero:
            keep = keep | is_zero.unsqueeze(1)

        gate_mask = keep.to(response_mask.dtype) * response_mask

        # Logging-only reductions are float: verl's response_mask is an int64 slice of
        # attention_mask, and Tensor.mean() raises on integer dtypes.
        kept_per_seq = (gate_mask.sum(-1) > 0).to(torch.float32)
        pos_f, fail_f = is_positive.to(torch.float32), is_failure.to(torch.float32)
        total_tok = response_mask.sum().clamp(min=1)
        metrics = {
            "rgopd/gate_kept_token_frac": (gate_mask.sum() / total_tok).item(),
            "rgopd/gate_kept_traj_frac": kept_per_seq.mean().item(),
            "rgopd/gate_kept_positive": (pos_f * kept_per_seq).sum().item(),
            "rgopd/gate_kept_nonpositive": (fail_f * kept_per_seq).sum().item(),
            "rgopd/gate_total_positive": pos_f.sum().item(),
            "rgopd/gate_total_nonpositive": fail_f.sum().item(),
            "rgopd/gate_total_zero_adv": is_zero.to(torch.float32).sum().item(),
        }
        # Token-level keep rate within each reward class -- the quantity that actually
        # differs between per-token and whole-trajectory gating.
        for name, sel in (("positive", pos_f), ("negative", fail_f)):
            denom = (sel.unsqueeze(1) * mask_f).sum().clamp(min=1)
            metrics[f"rgopd/gate_kept_tok_frac_{name}"] = (
                (sel.unsqueeze(1) * gate_mask.to(mask_f.dtype)).sum() / denom
            ).item()
    return gate_mask, metrics


def compute_rgopd_loss(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    config: Any,
    old_log_probs: Optional[torch.Tensor] = None,
    student_log_probs: Optional[torch.Tensor] = None,
    teacher_log_probs: Optional[torch.Tensor] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Full RG-OPD training loss (Eq. 3): gated top-K reverse-KL distillation.

    L_RG-OPD = [ sum_i g_i sum_t m_t KL(pi_theta(.|s_t^(i)) || pi_T(.|s_t^(i))) ] / [ sum_i g_i sum_t m_t ]

    Args:
        student_topk_log_probs, teacher_topk_log_probs: see `compute_topk_reverse_kl`.
        response_mask: `(B, T)`, 1 for valid response tokens (this is m_t in Eq. 1/3).
        advantages: `(B, T)` trajectory-level advantage broadcast over response tokens
            (A_i in Eq. 1-2; the paper uses the GRPO advantage).
        config: distillation/gate settings, e.g. the `actor_rollout_ref.actor.recipe.rgopd`
            subtree of `recipe/rgopd/config/rgopd_trainer.yaml`. Read with `.get(key,
            default)`/`getattr`, so a plain dict, DictConfig, or dataclass all work. Relevant
            keys: `distillation.add_tail` (bool), `distillation.is_clip` (float | None,
            importance-sampling ratio clip for off-policy correction), `gate.enabled` (bool
            -- False recovers the ungated "Reverse-KL" baseline), `gate.margin` (float, delta
            in Eq. 2).
        old_log_probs, student_log_probs: `(B, T)` sampled-token log-probs under the
            behavior policy and the current student, respectively. `student_log_probs` is
            required whenever the gate is enabled (for L_S in Eq. 1); `old_log_probs` is
            required only if `distillation.is_clip` is set.
        teacher_log_probs: `(B, T)` teacher log-prob of the actually-sampled token (for
            L_T in Eq. 1). Required whenever the gate is enabled. Distinct from
            `teacher_topk_log_probs`, which covers the top-K *support*, not necessarily the
            sampled token itself.
        rollout_is_weights: optional `(B, T)` off-policy rollout-correction weights.

    Returns:
        (scalar loss for backprop, metrics dict for logging)
    """
    distill_cfg = _cfg(config, "distillation", {})
    gate_cfg = _cfg(config, "gate", {})

    metrics: dict[str, Any] = {}
    loss_mask = response_mask

    per_token_loss = compute_topk_reverse_kl(
        student_topk_log_probs,
        teacher_topk_log_probs,
        add_tail=_cfg(distill_cfg, "add_tail", True),
    )

    is_clip = _cfg(distill_cfg, "is_clip", None)
    if is_clip is not None:
        if old_log_probs is None or student_log_probs is None:
            raise ValueError("distillation.is_clip requires old_log_probs and student_log_probs.")
        negative_approx_kl = torch.clamp((student_log_probs - old_log_probs).detach(), min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio

    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    if _cfg(gate_cfg, "enabled", True):
        if student_log_probs is None or teacher_log_probs is None:
            raise ValueError("gate.enabled=True requires student_log_probs and teacher_log_probs.")
        gate_mask, gate_metrics = compute_reward_teacher_gate(
            teacher_log_probs=teacher_log_probs,
            student_log_probs=student_log_probs,
            response_mask=response_mask,
            advantages=advantages,
            margin=_cfg(gate_cfg, "margin", 0.0),
            chunk_size=int(_cfg(gate_cfg, "chunk_size", 1)),
            failure_includes_zero=bool(_cfg(gate_cfg, "failure_includes_zero", False)),
            zero_adv=_cfg(gate_cfg, "zero_adv", "mask"),
        )
        per_token_loss = per_token_loss * gate_mask
        # Fold the gate into loss_mask so the agg_loss denominator matches the numerator.
        loss_mask = loss_mask * gate_mask
        metrics.update(gate_metrics)

    loss = agg_loss(
        loss_mat=per_token_loss,
        loss_mask=loss_mask,
        loss_agg_mode=loss_agg_mode,
        batch_num_tokens=loss_mask.sum().clamp(min=1.0),
    )
    metrics["rgopd/loss"] = loss.detach().item()
    return loss, metrics
