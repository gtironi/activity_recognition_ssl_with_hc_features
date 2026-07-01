#!/usr/bin/env bash
#
# For each dataset x encoder x method, runs the hybrid model
# (src/hc_framework) on the canonical .pt tensors (--from_pt), loading that
# dataset's per-dataset SSL checkpoint ("_self") at native channels/window length.
#
# The grid generator (scripts/core/hybrid_grid.py) supports ALL encoders/methods;
# this script's defaults restrict the *emitted* jobs to ENCODERS x METHODS below.
# Override via env, e.g.:
#   ENCODERS="patchtst cnn_tfc resnet1d" METHODS="simclr mae tfc" bash configs/run_hybrid_all.sh
#
# Idempotent (skips runs with test_stage1_classification_metrics.json or .done) and
# resilient (a failing job writes .failed, queue continues).
#
# Usage (the nohup base):
#   cd activity_recognition_ssl_with_hc_features
#   nohup bash configs/run_hybrid_all.sh > logs/hybrid_all.log 2>&1 &
#
# Env knobs:
#   ENCODERS="patchtst cnn_tfc"   # default
#   METHODS="simclr mae"          # default (mae restricted to patchtst by hybrid_grid.py)
#   DATASETS="..."                 # subset of the 7 (default: all)
#   VARIANTS="full frozen"         # default; "full"=normal finetune, "frozen"=--freeze_encoder,
#                                   # "lora"=PatchTST-only LoRA adapters (mutually exclusive with full/frozen)
#   MAX_PARALLEL=2
#   EPOCHS=30
#
# WARNING: never run two invocations of this script concurrently with the
# default OUT_JOBS path -- even with hybrid_grid.py's atomic-replace, a
# queue_runner.sh that is still mid-loop reading the *old* file handle will
# simply stop early once it hits EOF on its original (now unlinked) inode,
# silently dropping any jobs it hadn't reached yet. If you need two grids at
# once (e.g. VARIANTS=frozen and VARIANTS=lora in parallel), give each run
# its own OUT_JOBS path, e.g.:
#   OUT_JOBS=runs/hybrid/_jobs_lora.txt VARIANTS=lora bash configs/run_hybrid_all.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

PYTHON="${PYTHON:-${REPO}/venv/bin/python}"
LOG_CENTRAL="${LOG_CENTRAL:-logs/hybrid_all.log}"
OUT_JOBS="${OUT_JOBS:-runs/hybrid/_jobs.txt}"
mkdir -p "$(dirname "$LOG_CENTRAL")" "$(dirname "$OUT_JOBS")"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_CENTRAL"; }

log "=== run_hybrid_all start | ENCODERS=${ENCODERS:-patchtst cnn_tfc} | METHODS=${METHODS:-simclr mae} | VARIANTS=${VARIANTS:-full frozen} | OUT_JOBS=${OUT_JOBS} ==="

ENCODERS="${ENCODERS:-patchtst cnn_tfc}" METHODS="${METHODS:-simclr mae}" DATASETS="${DATASETS:-}" \
    VARIANTS="${VARIANTS:-full frozen}" OUT_JOBS="$OUT_JOBS" \
    "$PYTHON" scripts/core/hybrid_grid.py | tee -a "$LOG_CENTRAL"

MAX_PARALLEL="${MAX_PARALLEL:-2}" EPOCHS="${EPOCHS:-30}" \
    bash configs/hybrid_queue_runner.sh "$OUT_JOBS"

log "=== run_hybrid_all finished ==="
