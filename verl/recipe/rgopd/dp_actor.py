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
"""RG-OPD actor: a `DataParallelPPOActor` whose `update_policy` trains on the gated
top-K reverse-KL distillation loss of `recipe/rgopd/core_algos.py` instead of a
PPO/GRPO policy-gradient loss.

Two teacher sources are supported (`recipe.rgopd.teacher.source` in config):
  - "external": a separate, frozen, larger model served over the network (the paper's
    setup: a dedicated vLLM endpoint reused from `recipe/gkd`'s teacher server). The
    trainer (`recipe/rgopd/ray_trainer.py`) queries it once per step and attaches its
    top-K log-probs to the batch; this module only needs to gather the student's own
    log-probs at that same top-K index set.
  - "ema" / "trust-region": the student's own frozen or slowly-updated copy scores its
    own rollout directly (no reprompting, no second server) -- the self-distillation
    variant discussed in the paper's related work. `TrustRegionTeacher` mixes a frozen
    reference copy with the live student; `_update_teacher` implements the EMA update.

Only the padded (non-remove-padding, non-Ulysses-SP) forward path is implemented; see
`_forward_topk_micro_batch`.
"""

import logging
import os
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import ActorConfig

from .core_algos import _cfg, compute_rgopd_loss

__all__ = ["TrustRegionTeacher", "RGOPDActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _has_multi_modal_inputs(field) -> bool:
    """True only if `field` actually carries multi-modal content.

    Testing `"multi_modal_inputs" in batch` is not enough: verl's rollout path
    unconditionally materializes the key for *every* sample, defaulting it to an empty
    dict for pure-text data (`verl/workers/rollout/schemas.py`: "if not
    values.get('multi_modal_inputs'): values['multi_modal_inputs'] = {}"). A presence
    check therefore fires on ordinary text-only batches and makes this recipe
    unrunnable. `field` is a numpy object array of per-sample dicts (or None when the
    key is absent), so check whether any sample's dict is non-empty.
    """
    if field is None:
        return False
    try:
        return any(bool(sample) for sample in field)
    except TypeError:  # not iterable -- fall back to plain truthiness
        return bool(field)


class TrustRegionTeacher(nn.Module):
    """A teacher whose logits are a fixed linear mix of a frozen reference copy and the
    live student: `logits = lerp(ref_logits, student_logits, mix_coef)`.

    With `mix_coef=0` this is a plain frozen (never-updated) reference teacher; with
    `mix_coef=1` it degenerates to self-distillation against the live student (not
    useful on its own, since student==teacher). Intermediate values act as a soft trust
    region: the teacher partially tracks the student so it never drifts arbitrarily far
    behind, while `ref_module` anchors it against catastrophic forgetting.

    Known cost: `update_policy`'s self-teacher branch already forwards the student once
    to determine the shared top-K index set, then forwards `teacher_module` (this class)
    at those same indices; `forward()` below redundantly forwards `student_module` a
    second time with identical inputs to get the same logits, since it has no way to
    accept the already-computed ones through the generic `module(*args, **kwargs)` call
    `_forward_topk_micro_batch` uses for every teacher source. This only costs an extra
    full student forward pass under `teacher.source="trust-region"` specifically (not
    `"external"`, the paper's setup, or `"ema"`); avoiding it would mean threading
    precomputed logits through that generic call path, which wasn't judged worth the
    added complexity for this optional, non-default teacher source.
    """

    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class RGOPDActor(DataParallelPPOActor):
    """DataParallelPPOActor variant that trains with `compute_rgopd_loss` instead of a
    PPO/GRPO policy-gradient loss. See module docstring."""

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        super().__init__(config, actor_module, actor_optimizer)
        # Set by recipe/rgopd/fsdp_workers.py when teacher.source is "ema" or "trust-region";
        # left None for teacher.source="external" (no local teacher forward needed).
        self.teacher_module: Optional[nn.Module] = None

    def _forward_topk_micro_batch(
        self,
        micro_batch: dict,
        temperature: float,
        topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        calculate_entropy: bool = False,
        module: Optional[nn.Module] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning log-probs of the sampled tokens plus top-K log-probs.

        Exactly one of `topk` or `topk_indices` must be given:
          - `topk`: compute *this* module's own top-K tokens (and their log-probs) at each
            response position.
          - `topk_indices`: `(B, response_len, K)` token ids to gather this module's
            log-probs at -- e.g. gathering the student's log-probs at an external
            teacher's returned top-K indices, or a teacher gathering at the student's.

        Returns a dict with `log_probs` (B, response_len), `topk_logps` (B, response_len,
        K), `topk_indices` (B, response_len, K), and `entropys` (B, response_len) if
        `calculate_entropy`.
        """
        if self.use_remove_padding or self.use_ulysses_sp:
            raise NotImplementedError(
                "recipe/rgopd's top-K distillation forward only supports the padded, "
                "non-Ulysses-SP path (actor.use_remove_padding=False, "
                "actor.ulysses_sequence_parallel_size=1). Contributions welcome."
            )
        if (topk is None) == (topk_indices is None):
            raise ValueError("Exactly one of `topk` or `topk_indices` must be given.")
        if _has_multi_modal_inputs(micro_batch.get("multi_modal_inputs")):
            raise NotImplementedError("recipe/rgopd does not yet support multi-modal inputs.")

        model = module or self.actor_module
        response_length = micro_batch["responses"].size(-1)

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            output = model(
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch["attention_mask"],
                position_ids=micro_batch["position_ids"],
                use_cache=False,
            )
            logits = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_len, vocab)

            log_probs = logprobs_from_logits(logits, micro_batch["responses"])

            if topk_indices is None:
                k = min(topk, logits.size(-1))
                topk_logits, topk_indices_out = torch.topk(logits, k, dim=-1)
            else:
                topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                topk_indices_out = topk_indices
            logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
            topk_logps = topk_logits - logsumexp

            entropy = None
            if calculate_entropy:
                entropy = verl_F.entropy_from_logits(logits)

        outputs = {"log_probs": log_probs, "topk_logps": topk_logps, "topk_indices": topk_indices_out}
        if calculate_entropy:
            outputs["entropys"] = entropy
        return outputs

    def _update_teacher(self) -> None:
        """EMA-update `self.teacher_module` toward the student, if configured to do so."""
        rgopd_cfg = self.config.get("recipe", {}).get("rgopd", None)
        teacher_cfg = _cfg(rgopd_cfg, "teacher", {})
        if _cfg(teacher_cfg, "source", "external") != "ema":
            return
        update_rate = float(_cfg(teacher_cfg, "ema_update_rate", 0.0))
        if update_rate == 0.0:
            return
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("teacher.source='ema' requires a separate teacher_module (see fsdp_workers.py).")
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_module.parameters(), self.actor_module.parameters()):
                student_data = student_param.data.to(device=teacher_param.device)
                teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        rgopd_cfg = self.config.get("recipe", {}).get("rgopd", None)
        if rgopd_cfg is None:
            raise ValueError(
                "RGOPDActor requires actor_rollout_ref.actor.recipe.rgopd config "
                "(see recipe/rgopd/config/rgopd_trainer.yaml)."
            )
        distill_cfg = _cfg(rgopd_cfg, "distillation", {})
        topk = int(_cfg(distill_cfg, "topk", 50))
        use_external_teacher = data.meta_info.get("use_external_teacher", False)

        select_keys = ["responses", "response_mask", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if use_external_teacher:
            select_keys.extend(["ext_teacher_topk_logps", "ext_teacher_topk_indices"])

        if _has_multi_modal_inputs(data.non_tensor_batch.get("multi_modal_inputs")):
            raise NotImplementedError("recipe/rgopd does not yet support multi-modal inputs.")
        data = data.select(batch_keys=select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {"actor/pg_loss": 0.0}
        did_update = False
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = micro_batch.batch

                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    loss_agg_mode = self.config.loss_agg_mode
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    loss_scale_factor = (
                        response_mask.shape[0] / self.config.ppo_mini_batch_size
                        if self.config.use_dynamic_bsz
                        else 1 / self.gradient_accumulation
                    )

                    if use_external_teacher:
                        # Teacher log-probs were queried once per step by the trainer (see
                        # ray_trainer.py) and cover the *full* input sequence (prompt +
                        # response). Teacher logits at position t predict token t+1
                        # (autoregressive), so the response tokens at [prompt_len,
                        # prompt_len+resp_len) are predicted by teacher output at
                        # [prompt_len-1, prompt_len+resp_len-1).
                        ext_topk_indices_all = model_inputs["ext_teacher_topk_indices"]
                        ext_topk_logps_all = model_inputs["ext_teacher_topk_logps"]
                        prompt_len = model_inputs["input_ids"].shape[1] - model_inputs["responses"].shape[1]
                        resp_len = model_inputs["responses"].shape[1]
                        # `prompt_len - 1` below relies on there being at least one prompt
                        # token (see the autoregressive-offset comment above); a degenerate
                        # all-response, no-prompt sample would silently wrap to a
                        # from-the-end slice instead of raising, so guard it explicitly.
                        assert prompt_len > 0, f"expected at least one prompt token, got prompt_len={prompt_len}"
                        teacher_topk_indices = ext_topk_indices_all[:, prompt_len - 1 : prompt_len + resp_len - 1, :]
                        teacher_topk_logps = ext_topk_logps_all[:, prompt_len - 1 : prompt_len + resp_len - 1, :]

                        # Student gathers its own log-probs at the *teacher's* top-K indices,
                        # so both sides share one index set without a second FSDP forward.
                        student_out = self._forward_topk_micro_batch(
                            model_inputs, temperature=temperature, topk_indices=teacher_topk_indices
                        )
                        student_log_prob = student_out["log_probs"]
                        student_topk_logps = student_out["topk_logps"]

                        # Teacher log-prob of the actually-sampled token: look it up inside
                        # the teacher's returned top-K; tokens the teacher didn't rank in its
                        # top-K are floored at -20 nats ("very unlikely under the teacher").
                        resp_tokens = model_inputs["responses"]
                        match = teacher_topk_indices == resp_tokens.unsqueeze(-1)
                        teacher_log_prob = (teacher_topk_logps * match.to(teacher_topk_logps.dtype)).sum(-1)
                        teacher_log_prob = teacher_log_prob.masked_fill(~match.any(-1), -20.0)
                    else:
                        # Self-teacher (ema / trust-region): student determines the shared
                        # top-K index set (matching Eq. 4's sum over topK(pi_theta)), then the
                        # teacher is scored at those same indices -- no reprompting, same
                        # (input_ids, attention_mask, position_ids) as the student's rollout.
                        teacher_model = self.teacher_module or self.actor_module
                        student_out = self._forward_topk_micro_batch(model_inputs, temperature=temperature, topk=topk)
                        student_log_prob = student_out["log_probs"]
                        student_topk_logps = student_out["topk_logps"]
                        with torch.no_grad():
                            teacher_out = self._forward_topk_micro_batch(
                                model_inputs,
                                temperature=temperature,
                                topk_indices=student_out["topk_indices"],
                                module=teacher_model,
                            )
                        teacher_topk_logps = teacher_out["topk_logps"]
                        teacher_log_prob = teacher_out["log_probs"]

                    pg_loss, pg_metrics = compute_rgopd_loss(
                        student_topk_log_probs=student_topk_logps,
                        teacher_topk_log_probs=teacher_topk_logps,
                        response_mask=response_mask,
                        advantages=advantages,
                        config=rgopd_cfg,
                        old_log_probs=old_log_prob,
                        student_log_probs=student_log_prob,
                        teacher_log_probs=teacher_log_prob,
                        rollout_is_weights=rollout_is_weights,
                        loss_agg_mode=loss_agg_mode,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    loss = pg_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                if torch.isfinite(grad_norm).item():
                    did_update = True
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        if did_update:
            self._update_teacher()
        return metrics
