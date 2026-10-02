#!/usr/bin/env bash
# train_rgopd.sh -- launch RG-OPD training.
#
# Thin wrapper around `python -m recipe.rgopd.main_rgopd` (see
# recipe/rgopd/config/rgopd_trainer.yaml and recipe/rgopd/main_rgopd.py). Any arguments
# passed to this script are forwarded verbatim as Hydra config overrides.
#
# CWD REQUIREMENT: must be run with CWD = the `verl/` directory (this repo's
# vendored framework root), so the hydra searchpath and `recipe.*` imports in
# rgopd_trainer.yaml / main_rgopd.py resolve.
#
# Usage (from the `verl/` directory):
#   ./recipe/rgopd/scripts/train_rgopd.sh
#   ./recipe/rgopd/scripts/train_rgopd.sh trainer.total_epochs=1   # smoke test
#
# Example: point at a custom student model / data dir / GPU count, and a teacher
# server started by serve_teacher.sh on another host:
#   RGOPD_STUDENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
#   RGOPD_DATA_DIR=/data/rgopd \
#   RGOPD_GPUS_PER_NODE=4 \
#   RGOPD_TEACHER_HOST=10.0.0.5 RGOPD_TEACHER_PORT=15555 \
#   ./recipe/rgopd/scripts/train_rgopd.sh
#
# Env vars honored (all optional; values shown are rgopd_trainer.yaml's defaults):
#   RGOPD_DATA_DIR        Directory holding train.parquet/test.parquet.
#                         Default: ./data/rgopd
#   RGOPD_TEACHER_HOST    Host of the external teacher server (see
#                         recipe/rgopd/scripts/serve_teacher.sh). Only used when
#                         teacher.source=external (the default). Default: localhost
#   RGOPD_TEACHER_PORT    Port of the external teacher server. Default: 15555
#   RGOPD_STUDENT_MODEL   Student checkpoint path or HF repo id.
#                         Default: Qwen/Qwen2.5-1.5B-Instruct
#   RGOPD_GPUS_PER_NODE   GPUs per node for the (single-node) trainer.
#                         Default: 2
#   RGOPD_CKPT_DIR        Root directory checkpoints are written under (final path is
#                         "${RGOPD_CKPT_DIR}/${RGOPD_EXPERIMENT}"). Default: ./checkpoints
#   RGOPD_EXPERIMENT      Experiment / run name. Default: rgopd-qwen2.5-1.5b
#   WANDB_PROJECT        W&B / trainer project name. Default: rgopd
#
# All of the above are consumed via `${oc.env:VAR,default}` in
# recipe/rgopd/config/rgopd_trainer.yaml -- pass any other Hydra override as a plain
# CLI arg to this script instead (e.g. actor_rollout_ref.actor.recipe.rgopd.gate.enabled=False).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"

if [[ "$(pwd)" != "${VERL_ROOT}" ]]; then
    echo "warning: this script expects CWD to be the verl/ directory (${VERL_ROOT})," >&2
    echo "         but it is currently $(pwd). Continuing anyway -- Hydra may fail to" >&2
    echo "         resolve recipe.* imports / the config searchpath if this isn't right." >&2
fi

DATA_DIR="${RGOPD_DATA_DIR:-./data/rgopd}"
if [[ -z "${RGOPD_DATA_DIR:-}" && ( ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/test.parquet" ) ]]; then
    echo "warning: RGOPD_DATA_DIR is not set and ${DATA_DIR}/{train,test}.parquet were not" >&2
    echo "         found. Prepare the data first, e.g.:" >&2
    echo "           python -m recipe.rgopd.data.prepare_ultra_interact --output-dir ./data/rgopd" >&2
    echo "         Continuing anyway -- training will fail immediately if the files are" >&2
    echo "         genuinely missing." >&2
fi

exec python -m recipe.rgopd.main_rgopd "$@"
