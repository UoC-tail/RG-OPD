# RG-OPD Evaluation

This directory intentionally does **not** bundle an evaluation harness. RG-OPD's
reported numbers (Tables 1/2/3 of Akhondzadeh et al., "Reward-Gated On-Policy
Distillation", arXiv:2607.04037) come from the external, community-maintained
[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) project,
and this recipe evaluates trained checkpoints with that tool directly rather than
reimplementing it.

## Why use `lm-evaluation-harness` instead of a recipe-local eval script

- **Reproducibility.** It is the exact tool and benchmark suite the paper's numbers
  were produced with. Re-deriving scores with a different prompt format, few-shot
  sampling, or answer-extraction logic would not reproduce Table 1/2/3, even if the
  underlying benchmark data were identical.
- **Avoiding a second, drifting reimplementation.** The paper evaluates across 10
  benchmarks spanning generation-based and likelihood-based tasks. Maintaining a
  parallel in-repo implementation of ~10 benchmark harnesses (prompt templates,
  metrics, few-shot configs) would be a substantial, easy-to-drift-out-of-sync
  liability compared to depending on the actively maintained upstream project that
  already gets this right.

## Install

```bash
pip install lm-eval[vllm]
```

The exact extras name/syntax can change between `lm-evaluation-harness` releases --
check the install instructions in that project's own README
(https://github.com/EleutherAI/lm-evaluation-harness) if the above doesn't work for
the version you're installing.

## Benchmarks and protocol (paper Appendix A.1)

The paper evaluates on 10 benchmarks, split into two categories by task type:

**Generation-based** (the model free-generates a response, which is then parsed/scored):
- GSM8K
- GSM-Plus
- MATH
- MMLU-Pro-Math
- MBPP
- IFEval

**Likelihood-based** (the model scores candidate continuations; no free generation):
- SciQ
- MMLU-STEM
- MuSR
- BBH

Protocol notes:
- Inference backend: vLLM (`--model vllm` in `lm_eval`).
- Sampling: temperature=0.6, averaged over 3 seeds.
- **Generation-based tasks are evaluated twice**, once with a max generation length of
  1024 tokens and once with 8096 tokens. The paper repeats generation-based eval under
  both budgets because distilled models tend to get more verbose over the course of
  training, and a single fixed generation budget can silently truncate (and therefore
  under-score) a model that would otherwise have answered correctly -- reporting both
  budgets separates "got it wrong" from "ran out of tokens."
- Likelihood-based tasks don't free-generate, so the generation-length budget is not
  applicable to them.

## Example invocations

The commands below are illustrative -- **check exact task names and supported flags
against your installed `lm-eval` version** with:

```bash
lm_eval --tasks list
```

before relying on them; task names/aliases (e.g. `gsm8k` vs `gsm8k_cot`,
`mmlu_pro_math` naming, etc.) have changed across `lm-evaluation-harness` releases.

Generation-based example (GSM8K, 1024-token budget):

```bash
# Illustrative -- verify task name/flags with `lm_eval --tasks list` first.
lm_eval \
    --model vllm \
    --model_args pretrained=/path/to/merged/hf_model,tensor_parallel_size=1 \
    --tasks gsm8k \
    --batch_size auto \
    --gen_kwargs temperature=0.6,max_gen_toks=1024 \
    --seed 0
```

Likelihood-based example (SciQ):

```bash
# Illustrative -- verify task name/flags with `lm_eval --tasks list` first.
lm_eval \
    --model vllm \
    --model_args pretrained=/path/to/merged/hf_model,tensor_parallel_size=1 \
    --tasks sciq \
    --batch_size auto \
    --seed 0
```

Notes on the flags above:
- `--model_args pretrained=...,tensor_parallel_size=...`: point `pretrained` at a
  local HF-format checkpoint directory (see "Merging a checkpoint" below) or a HF hub
  repo id; set `tensor_parallel_size` to however many GPUs you want vLLM to shard
  across.
- `--batch_size auto`: let vLLM pick a batch size; adjust if you hit an OOM.
- `--num_fewshot` is intentionally omitted above -- leave each task at its default
  few-shot count unless you're deliberately deviating from the paper's protocol.
- To reproduce the paper's "averaged over 3 seeds" methodology, repeat each
  invocation with `--seed 0`, `1`, `2` (or `lm-eval`'s current multi-seed flag, if one
  exists in your installed version) and average the reported metric.
- For the 8096-token generation-based re-run, repeat the generation-based commands
  with `max_gen_toks=8096`.

## `recipe/rgopd/reward_score/` is a separate thing

`recipe/rgopd/reward_score/` (e.g. `math.py`) is the **training-time verifier** used
by the RL loop's reward function (`custom_reward_function` in
`recipe/rgopd/config/rgopd_trainer.yaml`). It only needs to score math/code
correctness for a single response during training -- it is deliberately narrow, fast,
and has no relationship to (and is not a substitute for) the full 10-benchmark
`lm-evaluation-harness` protocol described above. Do not use it to produce reported
eval numbers, and do not expect this eval protocol to exercise it.

## Merging an FSDP checkpoint before evaluating

`lm_eval --model vllm` needs a single HF-format model directory, but RG-OPD trains
with FSDP and saves sharded checkpoints under
`trainer.default_local_dir` (`${RGOPD_CKPT_DIR}/${RGOPD_EXPERIMENT}/global_step_<N>/actor`
by default -- see `recipe/rgopd/config/rgopd_trainer.yaml`). Merge a checkpoint into a
plain HF directory with verl's built-in model merger:

```bash
# From the verl/ directory:
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/<experiment_name>/global_step_<N>/actor \
    --target_dir /path/to/merged_hf_model
```

Then point `lm_eval`'s `--model_args pretrained=...` at `/path/to/merged_hf_model`.

See `verl/model_merger/__init__.py` for the full merger docstring (including the
Megatron-backend and distributed-merge variants, not applicable to this FSDP-trained
recipe) and https://verl.readthedocs.io/en/latest/advance/checkpoint.html for more
detail.
