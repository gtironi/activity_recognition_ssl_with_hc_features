#!/usr/bin/env bash
# Resilient parallel job runner for the hybrid finetuning grid.
#
# Reads a job file (default: runs/hybrid/_jobs.txt) where each line is:
#   <run_name> <hc_framework.main CLI args...>
# and runs MAX_PARALLEL jobs at a time via `python -m hc_framework.main`.
#
# Idempotent: skips jobs whose run_dir already has test_stage1_classification_metrics.json
# (or .done). Resilient: a failing job writes .failed and never stops the queue.
#
# Usage:
#   MAX_PARALLEL=2 EPOCHS=30 bash configs/hybrid_queue_runner.sh runs/hybrid/_jobs.txt
set -uo pipefail   # NOTE: no -e -- a failing job must not kill the queue.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

PYTHON="${PYTHON:-${REPO}/venv/bin/python}"
RUNS_DIR="runs/hybrid"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
EPOCHS="${EPOCHS:-30}"
LOG_CENTRAL="${LOG_CENTRAL:-logs/hybrid_all.log}"
mkdir -p "$(dirname "$LOG_CENTRAL")"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_CENTRAL"; }

JOB_FILE="${1:-${RUNS_DIR}/_jobs.txt}"
if [ ! -f "$JOB_FILE" ]; then
    log "No job file: $JOB_FILE"
    exit 0
fi

run_one() {
    local run_name="$1"; shift
    local args=("$@")
    local run_dir="${RUNS_DIR}/${run_name}"

    mkdir -p "$run_dir"

    if [ -f "${run_dir}/test_stage1_classification_metrics.json" ] || [ -f "${run_dir}/.done" ]; then
        log "SKIP  ${run_name} (already complete)"
        return 0
    fi

    rm -f "${run_dir}/.failed"
    log "START ${run_name}"

    if "$PYTHON" -m hc_framework.main \
            --epochs "$EPOCHS" --output_dir "$run_dir" "${args[@]}" \
            > "${run_dir}/run.log" 2>&1; then
        : > "${run_dir}/.done"
        log "DONE  ${run_name}"
    else
        : > "${run_dir}/.failed"
        log "FAIL  ${run_name} (see ${run_dir}/run.log)"
        tail -5 "${run_dir}/run.log" | sed 's/^/      /' | tee -a "$LOG_CENTRAL"
    fi
}

log "=== hybrid_queue_runner start | MAX_PARALLEL=${MAX_PARALLEL} | epochs=${EPOCHS} ==="
log "job file: ${JOB_FILE}"

running=0
while IFS= read -r line || [ -n "$line" ]; do
    [ -z "${line// }" ] && continue
    case "$line" in \#*) continue ;; esac

    # shellcheck disable=SC2206
    parts=($line)
    rn="${parts[0]}"
    args=("${parts[@]:1}")

    run_one "$rn" "${args[@]}" &
    running=$((running + 1))

    if [ "$running" -ge "$MAX_PARALLEL" ]; then
        wait -n 2>/dev/null || true
        running=$((running - 1))
    fi
done < "$JOB_FILE"

wait
log "=== hybrid_queue_runner finished ==="

"$PYTHON" scripts/core/summarize_hybrid.py --runs_dir "$RUNS_DIR" --out results/hybrid_summary_all.csv \
    2>&1 | tee -a "$LOG_CENTRAL" || true
