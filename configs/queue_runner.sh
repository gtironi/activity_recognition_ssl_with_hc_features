#!/usr/bin/env bash
# Resilient parallel job runner for the multi-dataset SSL ablation.
#
# Reads one or more job files (default: all runs/*/_jobs.txt),
# runs MAX_PARALLEL jobs at a time on the GPU, and is:
#   - idempotent : skips jobs whose run already has eval/metrics_test.json (or .done)
#   - resilient  : a failing job writes .failed and NEVER stops the queue
#
# Each job line format (whitespace-separated):
#   <run_name> <override> <override> ...
# where run_name is like "<dataset>/<combo>" and overrides are dot-list
# (e.g. data.dataset_id=uci_har encoder.name=cnn_tfc pretext.method=tfc ...).
#
# Usage:
#   bash configs/queue_runner.sh                  # all runs/*/_jobs.txt
#   bash configs/queue_runner.sh path/to/_jobs.txt ...
#   MAX_PARALLEL=2 CONFIG=... bash .../queue_runner.sh
set -uo pipefail   # NOTE: no -e — a failing job must not kill the queue.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

PYTHON="${PYTHON:-${REPO}/venv/bin/python}"
CONFIG="${CONFIG:-configs/multidataset_base.yaml}"
RUNS_DIR="${RUNS_DIR:-runs}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LOG_CENTRAL="${LOG_CENTRAL:-logs/ablation_all.log}"
mkdir -p "$(dirname "$LOG_CENTRAL")"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_CENTRAL"; }

# Collect job files.
if [ "$#" -gt 0 ]; then
    JOB_FILES=("$@")
else
    mapfile -t JOB_FILES < <(find "$RUNS_DIR" -maxdepth 2 -name "_jobs.txt" 2>/dev/null | sort)
fi
if [ "${#JOB_FILES[@]}" -eq 0 ]; then
    log "No _jobs.txt files found. Nothing to do."
    exit 0
fi

# Run a single job (subshell-safe). Args: run_name + overrides...
run_one() {
    local run_name="$1"; shift
    local overrides=("$@")
    local run_dir="${RUNS_DIR}/${run_name}"
    local metrics="${run_dir}/eval/metrics_test.json"

    mkdir -p "$run_dir"

    if [ -f "$metrics" ] || [ -f "${run_dir}/.done" ]; then
        log "SKIP  ${run_name} (already complete)"
        return 0
    fi

    rm -f "${run_dir}/.failed"
    : > "${run_dir}/.running"
    log "START ${run_name} :: ${overrides[*]}"

    if "$PYTHON" -m pretrain_ablations.experiment \
            --config "$CONFIG" \
            --override "run_name=${run_name}" "${overrides[@]}" \
            > "${run_dir}/run.log" 2>&1; then
        rm -f "${run_dir}/.running"
        : > "${run_dir}/.done"
        log "DONE  ${run_name}"
    else
        rm -f "${run_dir}/.running"
        : > "${run_dir}/.failed"
        log "FAIL  ${run_name} (see ${run_dir}/run.log)"
        tail -5 "${run_dir}/run.log" | sed 's/^/      /' | tee -a "$LOG_CENTRAL"
    fi
}

log "=== queue_runner start | MAX_PARALLEL=${MAX_PARALLEL} | config=${CONFIG} ==="
log "job files: ${JOB_FILES[*]}"

running=0
for jf in "${JOB_FILES[@]}"; do
    [ -f "$jf" ] || { log "WARN missing job file: $jf"; continue; }
    while IFS= read -r line || [ -n "$line" ]; do
        # skip blanks/comments
        [ -z "${line// }" ] && continue
        case "$line" in \#*) continue ;; esac

        # shellcheck disable=SC2206
        parts=($line)
        rn="${parts[0]}"
        ovr=("${parts[@]:1}")

        run_one "$rn" "${ovr[@]}" &
        running=$((running + 1))

        # throttle to MAX_PARALLEL concurrent jobs
        if [ "$running" -ge "$MAX_PARALLEL" ]; then
            wait -n 2>/dev/null || true
            running=$((running - 1))
        fi
    done < "$jf"
done

# drain remaining
wait
log "=== queue_runner finished ==="

# (Re)generate per-dataset summaries.
"$PYTHON" -m pretrain_ablations.results.summarize --per_dataset \
    --runs_dir "$RUNS_DIR" \
    --out "results/summary_all.csv" 2>&1 | tee -a "$LOG_CENTRAL" || true
log "Summaries written under results/"
