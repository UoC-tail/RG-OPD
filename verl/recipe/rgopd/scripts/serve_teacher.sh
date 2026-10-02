#!/usr/bin/env bash
# serve_teacher.sh -- start the frozen RG-OPD teacher server.
#
# This launches the ZMQ proxy + one vLLM worker process from recipe/gkd/teacher/
# (proxy.py, worker.py), the same "external teacher" pattern the upstream GKD recipe
# uses. It serves the frozen teacher policy (paper default: Qwen2.5-14B-Instruct,
# tensor-parallel=2 -- Appendix A.1 of arXiv:2607.04037) that
# rgopd_trainer.yaml's `actor_rollout_ref.actor.recipe.rgopd.teacher.external` points
# the student at when `teacher.source=external` (the paper's setup, and the default in
# rgopd_trainer.yaml).
#
# You only need this script for `teacher.source=external`. The self-teacher modes,
# `teacher.source=ema` and `teacher.source=trust-region`, score the student's own
# rollouts in-process and never talk to a teacher server at all -- skip this script
# entirely for those.
#
# CWD REQUIREMENT: like recipe/gkd's own start_server.sh/join_server.sh, this must be
# run with CWD = the `verl/` directory (this repo's vendored framework root), the same
# working directory training itself requires. It cd's into recipe/gkd/teacher/ to run
# proxy.py/worker.py (those scripts import local helper modules by relative name), then
# returns.
#
# Usage (from the `verl/` directory):
#   ./recipe/rgopd/scripts/serve_teacher.sh
#
# Env vars (all optional; defaults reproduce the paper's teacher setup):
#   RGOPD_TEACHER_MODEL  Teacher checkpoint path or HF repo id.
#                         Default: Qwen/Qwen2.5-14B-Instruct
#   RGOPD_TEACHER_PORT   Proxy frontend port -- must match the training config's
#                         `actor_rollout_ref.actor.recipe.rgopd.teacher.external.server_port`
#                         (RGOPD_TEACHER_PORT there too). Default: 15555
#   RGOPD_TEACHER_TP     vLLM tensor-parallel size for the teacher. Default: 2
#   RGOPD_TEACHER_TOPK   Number of top-k logprobs the teacher returns per token
#                         (worker.py's --n-logprobs). Default: 256
#
# Note: RGOPD_TEACHER_HOST is read by the *trainer* (rgopd_trainer.yaml) to know which
# host to connect to -- it's not used here. Run this script on that host, or point
# RGOPD_TEACHER_HOST at wherever you do run it.
set -euo pipefail

RGOPD_TEACHER_MODEL="${RGOPD_TEACHER_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
RGOPD_TEACHER_PORT="${RGOPD_TEACHER_PORT:-15555}"
RGOPD_TEACHER_TP="${RGOPD_TEACHER_TP:-2}"
RGOPD_TEACHER_TOPK="${RGOPD_TEACHER_TOPK:-256}"

# proxy.py reads its ports from these env vars (see recipe/gkd/teacher/proxy.py).
export PROXY_FRONTEND_PORT="${RGOPD_TEACHER_PORT}"
export PROXY_BACKEND_PORT="${RGOPD_TEACHER_BACKEND_PORT:-$((RGOPD_TEACHER_PORT + 1))}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
TEACHER_DIR="${VERL_ROOT}/recipe/gkd/teacher"

if [[ ! -f "${TEACHER_DIR}/proxy.py" || ! -f "${TEACHER_DIR}/worker.py" ]]; then
    echo "error: could not find ${TEACHER_DIR}/{proxy.py,worker.py}." >&2
    echo "       run this script with CWD = the verl/ directory." >&2
    exit 1
fi

wait_server_ready() {
    local server="$1" ip="$2" port="$3"
    while true; do
        echo "wait ${server} server ready at ${ip}:${port}..."
        # Bash builtin /dev/tcp probe (avoids depending on `telnet` being installed,
        # unlike recipe/gkd/teacher's own start_server.sh/join_server.sh).
        if (exec 3<>"/dev/tcp/${ip}/${port}") 2>/dev/null; then
            break
        fi
        sleep 1
    done
}

cd "${TEACHER_DIR}"

# Kill any stale proxy/worker processes from a previous run.
ps -ef | grep "python proxy.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9
ps -ef | grep "python worker.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9

echo "starting teacher proxy on ports ${PROXY_FRONTEND_PORT} (frontend) / ${PROXY_BACKEND_PORT} (backend)..."
nohup python proxy.py &> proxy.log &

wait_server_ready proxy localhost "${PROXY_BACKEND_PORT}"
echo "teacher proxy is ready"

echo "starting teacher worker: backend=vllm tp-size=${RGOPD_TEACHER_TP} n-logprobs=${RGOPD_TEACHER_TOPK} ckpt-path=${RGOPD_TEACHER_MODEL}"
nohup python worker.py \
    --backend vllm \
    --proxy-addr "localhost:${PROXY_BACKEND_PORT}" \
    --tp-size "${RGOPD_TEACHER_TP}" \
    --n-logprobs "${RGOPD_TEACHER_TOPK}" \
    --ckpt-path "${RGOPD_TEACHER_MODEL}" \
    &> worker.log &

echo "teacher server is ready -- verify with: telnet localhost ${RGOPD_TEACHER_PORT}"
echo "point the trainer at it with RGOPD_TEACHER_HOST=<this host> RGOPD_TEACHER_PORT=${RGOPD_TEACHER_PORT}"
