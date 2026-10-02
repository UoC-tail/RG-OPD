# RG-OPD: Reward-Gated On-Policy Distillation

Code for:

> Mohammad Sadegh Akhondzadeh, Vijay Lingam, Atula Tejaswi, Chanakya Ekbote, Sujay Sanghavi,
> Aleksandar Bojchevski. **Reward-Gated On-Policy Distillation.** arXiv:2607.04037, 2026.
> https://arxiv.org/abs/2607.04037

This is a [verl](https://github.com/volcengine/verl) recipe, structured like the other
recipes under `verl/recipe/` (`dapo`, `gkd`, `spin`, ...): it adds a training entry point,
a trainer/actor/worker subclass, a reward verifier, and data-prep scripts on top of the
unmodified `verl` package, rather than forking `verl` itself.

## The method

On-policy distillation (OPD) samples trajectories from the student's own policy and has a
teacher score them token-by-token, which gives dense, on-distribution supervision that
sparse verifier rewards can't. But the teacher isn't always right: it can assign high
likelihood to a plausible-looking wrong answer, or low likelihood to a correct answer the
student happened to phrase differently. Unconditional distillation then actively reinforces
the teacher's mistakes.

RG-OPD resolves this by using the verifier reward to decide *when* the teacher's signal is
trustworthy, not just *whether* the trajectory was good. At each step, for every sampled
trajectory `i` with response tokens `y^(i)`, states `s^(i)`, and response mask `m`:

**1. Top-K reverse-KL distillation** (Eq. 4; `recipe/rgopd/core_algos.py:compute_topk_reverse_kl`).
Computing the full-vocabulary `KL(pi_student || pi_teacher)` at every token is expensive, so
both sides are approximated over the top-K token support (K=50 by default) plus a tail
bucket for the remaining probability mass:

```
KL(pi_theta(.|s_t) || pi_T(.|s_t))
  ~= sum_{y in topK} pi_theta(y|s_t) * log(pi_theta(y|s_t) / pi_T(y|s_t))
   + (1 - sum_{y in topK} pi_theta(y|s_t)) * log((1 - sum_topK pi_theta) / (1 - sum_topK pi_T))
```

**2. Reward-teacher likelihood gate** (Eq. 1-2; `recipe/rgopd/core_algos.py:compute_reward_teacher_gate`).
Define each trajectory's teacher/student log-likelihood on its own sampled tokens,
`L_T^(i) = sum_t m_t log pi_T(y_t^(i)|s_t^(i))` and `L_S^(i)` likewise for the student, and let
`A_i` be its advantage (we use the GRPO advantage). Keep the trajectory (`g_i=1`) only when
the teacher is *directionally* informative:

```
g_i = 1[ (A_i > 0  and  L_T^(i) > L_S^(i) + delta)  or  (A_i <= 0  and  L_T^(i) < L_S^(i) - delta) ]
```

i.e. for a good trajectory, only trust the teacher if it likes that trajectory *more* than
the student already does; for a bad one, only if the teacher likes it *less*. `delta` (the
gate margin) defaults to 0.

**3. The training loss** (Eq. 3; `recipe/rgopd/core_algos.py:compute_rgopd_loss`) is the
top-K reverse-KL loss above, averaged over kept trajectories only:

```
L_RG-OPD = [ sum_i g_i * sum_t m_t * KL(pi_theta(.|s_t^(i)) || pi_T(.|s_t^(i))) ] / [ sum_i g_i * sum_t m_t ]
```

This is the *entire* training objective — there is no separate PPO/GRPO policy-gradient
term. The GRPO advantage is used only to decide the gate, not to update the policy directly.
Setting `gate.enabled: False` in config drops the gate and recovers the "Reverse-KL" OPD
baseline reported in the paper (same loss, applied unconditionally).

## Teacher sources

`actor_rollout_ref.actor.recipe.rgopd.teacher.source` selects where the teacher's
log-probabilities come from:

- **`external`** (the paper's setup): a separate, frozen, larger model served over the
  network — a dedicated vLLM endpoint (`scripts/serve_teacher.sh`, reusing
  `recipe/gkd`'s teacher server), queried once per training step
  (`recipe/rgopd/ray_trainer.py`) for its top-K log-probs on the just-sampled rollout. The
  student then gathers its own log-probs at that *same* top-K index set
  (`recipe/rgopd/dp_actor.py`), so no second FSDP forward pass is needed. Paper setup: a
  frozen Qwen2.5-14B-Instruct teacher (tensor-parallel=2) distilling into a
  Qwen2.5-1.5B-Instruct student.
- **`ema`** / **`trust-region`**: a self-teacher scores the student's own rollout directly —
  no reprompting, no second server, just a second forward pass on the same
  `(input_ids, attention_mask, position_ids)`, colocated on the actor's GPUs. `ema` is a
  slowly-updated copy of the student (`teacher <- (1-r)*teacher + r*student` after every
  update); `trust-region` mixes a frozen reference copy with the *live* student's logits
  (`logits = lerp(ref, student, mix_coef)`), so it partially tracks the student without a
  separate EMA buffer. Neither needs `scripts/serve_teacher.sh`; both require the reference
  model to be colocated with the actor, which `recipe/rgopd/main_rgopd.py` wires up
  automatically when `teacher.source != "external"`.

## Running it end-to-end

All commands below are run **from the `verl/` directory** (this repo's vendored framework
root) — that's what makes `recipe.*` imports and the Hydra config searchpath resolve, same
as every other `verl` recipe.

```bash
cd verl

# 1. Prepare training data: a math subset of UltraInteract, verified by boxed-answer
#    extraction. Read data/prepare_ultra_interact.py's module docstring before you run
#    this -- UltraInteract_sft (the public HF dataset) turns out not to carry structured
#    test cases for its Coding rows or verifier-friendly logic problems, so this script
#    can only produce a verifiable dataset for Math_CoT/Math_PoT out of the box; Coding
#    and Logic are excluded (with a printed per-task row count so this isn't silent) --
#    see the docstring for how to plug in your own code-verifiable data instead.
python -m recipe.rgopd.data.prepare_ultra_interact --output-dir ./data/rgopd

# 2. Start the external teacher server (skip this if using teacher.source=ema/trust-region).
bash recipe/rgopd/scripts/serve_teacher.sh

# 3. Train.
bash recipe/rgopd/scripts/train_rgopd.sh
# or with Hydra overrides, e.g. a quick smoke test:
bash recipe/rgopd/scripts/train_rgopd.sh trainer.total_epochs=1 data.train_batch_size=8

# 4. Evaluate a checkpoint with lm-evaluation-harness (the paper's official protocol).
# See recipe/rgopd/eval/README.md.
```

## Config reference

See `config/rgopd_trainer.yaml` for the full, commented config (it extends verl's stock
`ppo_trainer.yaml`). The RG-OPD-specific settings live under
`actor_rollout_ref.actor.recipe.rgopd`:

| Key | Meaning | Paper default |
|---|---|---|
| `distillation.topk` | K in Eq. 4 | 50 |
| `distillation.add_tail` | tail-mass correction term in Eq. 4 | `True` |
| `distillation.is_clip` | optional IS-ratio clip for off-policy correction (not used in the paper; it trains on-policy) | `null` (off) |
| `gate.enabled` | apply the reward-teacher gate (Eq. 1-2); `False` = plain OPD / "Reverse-KL" baseline | `True` |
| `gate.margin` | confidence margin `delta` in Eq. 2 | `0.0` |
| `teacher.source` | `external` \| `ema` \| `trust-region` | `external` |
| `teacher.external.{server_ip,server_port,n_server_workers}` | teacher vLLM endpoint | `localhost:15555` |
| `teacher.ema_update_rate` | EMA rate `r` (only if `teacher.source=ema`) | `0.05` |
| `teacher.trust_region_mix_coef` | mix coefficient (only if `teacher.source=trust-region`) | `0.05` |

Nothing in this config is a secret or machine-specific path — every path/host is either a
repo-relative default or an `${oc.env:VAR,default}` override (`RGOPD_DATA_DIR`,
`RGOPD_TEACHER_HOST`, `RGOPD_TEACHER_PORT`, `RGOPD_STUDENT_MODEL`, `RGOPD_GPUS_PER_NODE`,
`RGOPD_CKPT_DIR`, `RGOPD_EXPERIMENT`, `WANDB_PROJECT`).

## Limitations

- `recipe/rgopd/dp_actor.py`'s top-K distillation forward only implements the padded path
  (`actor.use_remove_padding=False`, `actor.ulysses_sequence_parallel_size=1`). Long-context
  / very-large-model training would want these; contributions welcome.
- No multi-modal inputs.
- No standard KL-to-reference penalty (`actor.use_kl_loss`) — RG-OPD's only regularization
  toward a reference distribution is the distillation loss itself, so this recipe doesn't
  wire up the generic ref-policy machinery that penalty needs.

## Patches to verl

`verl/` is vendored close to upstream. The only changes outside `verl/recipe/rgopd/` are a
handful of small, generic, backward-compatible additions used by this recipe (all
opt-in / no-op unless a config field enables them):

- `verl/workers/config/actor.py`: one extra `recipe: dict` field on `ActorConfig`, a generic
  escape hatch for recipe-specific config (same pattern as the existing `profile`/
  `global_batch_info` dict fields) — this is what makes
  `actor_rollout_ref.actor.recipe.rgopd.*` resolvable.
- `verl/utils/dataset/rl_dataset.py`: optional `data.system_prompt` setting.
- `verl/workers/reward_manager/{naive,batch,dapo}.py`: sets `extra_info["truncated"]` when a
  response fills the whole response buffer (consumed by `reward_score/math.py`).
- `verl/workers/config/rollout.py` + `verl/experimental/agent_loop/*.py`: optional per-call
  `response_length` override (used for a shorter generation length at validation time).
- `verl/recipe/gkd/teacher/{vllm_engine.py,worker.py}` + `teacher_utils.py`: small
  compatibility fixes to the existing GKD teacher-server infra, which this recipe's
  `teacher.source=external` path reuses as-is.

No other file under `verl/` was modified.
