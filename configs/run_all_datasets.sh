#!/usr/bin/env bash
# Master orchestrator for the multi-dataset SSL ablation.
#
# For every dataset, idempotently:
#   1. windowing  : raw -> dataset/processed/<NAME>/windowed_{train,test}.parquet  (scripts/datasets/<adapter>.py)
#   2. export     : windowed parquet -> dataset/processed/<id>/{train,val,test}.pt
#   3. jobs       : write runs/<id>/_jobs.txt (32 runs: 4 enc x 5 methods x {freeze,full},
#                   mae only on patchtst/patchtsmixer, supervised only full)
#   4. smoke      : one fast 1+1 run per dataset; abort THIS dataset if it fails (others continue)
# Then hands the queue to queue_runner.sh (2 parallel, resilient, idempotent).
#
# Resumable: re-run the same command to continue where it stopped.
#
# Usage (the nohup base):
#   cd activity_recognition_ssl_with_hc_features
#   nohup bash configs/run_all_datasets.sh > logs/ablation_all.log 2>&1 &
#
# Env knobs:
#   DATASETS="uci_har actbecalf"   # subset (default: all 7 paper datasets)
#   MAX_PARALLEL=2                 # passed to queue_runner
#   SKIP_SMOKE=1                   # skip the per-dataset smoke check
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

PYTHON="${PYTHON:-${REPO}/venv/bin/python}"
CONFIG="${CONFIG:-configs/multidataset_base.yaml}"
RUNS_DIR="runs"
LOG_CENTRAL="${LOG_CENTRAL:-logs/ablation_all.log}"
mkdir -p "$(dirname "$LOG_CENTRAL")" "$RUNS_DIR"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_CENTRAL"; }

# Per-dataset config: parallel arrays (index-aligned).
# Paths with spaces are safe here because bash arrays preserve them.
DS_IDS=(    actbecalf                vehkaoja                marinara                horse                uci_har                            pamap2                              wisdm                            )
DS_ADAPT=(  actbecalf                vehkaoja                marinara                horse                uci_har                            pamap2                              wisdm                            )
DS_RAW=(    "dataset/raw/AcTBeCalf"  "dataset/raw/vehkaoja"  "dataset/raw/marinara"  "dataset/raw/horse/csv"  "dataset/raw/UCI/UCI HAR Dataset"  "dataset/raw/PAMAP/PAMAP2_Dataset"  "dataset/raw/WISDM/wisdm-dataset"  )
DS_PROC=(   "dataset/processed/AcTBeCalf"  "dataset/processed/vehkaoja"  "dataset/processed/marinara"  "dataset/processed/horse"  "dataset/processed/HAR_UCI"  "dataset/processed/PAMAP2"  "dataset/processed/WISDM"  )

DATASETS="${DATASETS:-actbecalf vehkaoja marinara horse uci_har pamap2 wisdm}"

# Write the 32-run grid for one dataset to its _jobs.txt
write_jobs() {
    local dsid="$1"
    local jobs_file="${RUNS_DIR}/${dsid}/_jobs.txt"
    mkdir -p "${RUNS_DIR}/${dsid}"
    "$PYTHON" - "$dsid" "$jobs_file" <<'PY'
import sys
dsid, out = sys.argv[1], sys.argv[2]
encoders = ["cnn_tfc", "resnet1d", "patchtst", "patchtsmixer"]
mae_ok   = {"patchtst", "patchtsmixer"}
small_bs_encoders = {"patchtst", "patchtsmixer"}
small_bs = "32"
lines = []
def add(combo, enc, method, mode):
    rn = f"{dsid}/{combo}"
    extra = f" data.batch_size={small_bs}" if enc in small_bs_encoders else ""
    lines.append(f"{rn} data.dataset_id={dsid} encoder.name={enc} "
                 f"pretext.method={method} finetune.mode={mode}{extra}")
# supervised: only full
for enc in encoders:
    add(f"sup_{enc}_full", enc, "supervised", "full")
# mae: only patchtst/patchtsmixer, freeze+full
for enc in encoders:
    if enc in mae_ok:
        add(f"mae_{enc}_freeze", enc, "mae", "freeze")
        add(f"mae_{enc}_full",   enc, "mae", "full")
# simclr / tfc / tstcc: all encoders, freeze+full
for method in ("simclr", "tfc", "tstcc"):
    for enc in encoders:
        add(f"{method}_{enc}_freeze", enc, method, "freeze")
        add(f"{method}_{enc}_full",   enc, method, "full")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {len(lines)} jobs -> {out}")
PY
}

# One fast 1+1 smoke run for a dataset (cnn_tfc supervised, tiny batch/epochs).
smoke_dataset() {
    local dsid="$1"
    local rn="_smoke/${dsid}"
    local rd="${RUNS_DIR}/${rn}"
    if [ -f "${rd}/eval/metrics_test.json" ]; then
        log "smoke ${dsid}: already passed"; return 0
    fi
    rm -rf "$rd"; mkdir -p "$rd"
    log "smoke ${dsid}: running cnn_tfc supervised 1+1 ..."
    if "$PYTHON" -m pretrain_ablations.experiment --config "$CONFIG" \
            --override "run_name=${rn}" "data.dataset_id=${dsid}" \
            data.batch_size=64 pretext.epochs=1 finetune.epochs=1 \
            encoder.name=cnn_tfc pretext.method=supervised finetune.mode=full \
            > "${rd}/run.log" 2>&1; then
        log "smoke ${dsid}: PASS"; return 0
    else
        log "smoke ${dsid}: FAIL (see ${rd}/run.log)"
        tail -8 "${rd}/run.log" | sed 's/^/      /' | tee -a "$LOG_CENTRAL"
        return 1
    fi
}

log "=== run_all_datasets start | datasets: ${DATASETS} ==="

ALL_JOB_FILES=()
for i in "${!DS_IDS[@]}"; do
    dsid="${DS_IDS[$i]}"
    adapter="${DS_ADAPT[$i]}"
    raw="${DS_RAW[$i]}"
    procdir="${DS_PROC[$i]}"

    # filter by DATASETS
    case " $DATASETS " in *" $dsid "*) ;; *) continue ;; esac

    log "--- dataset: ${dsid} ---"

    # 1) windowing (idempotent)
    if [ ! -f "${procdir}/windowed_train.parquet" ]; then
        log "windowing ${dsid}: ${adapter}.py --raw '${raw}' --out '${procdir}'"
        if ! "$PYTHON" "scripts/datasets/${adapter}.py" --raw "$raw" --out "$procdir" \
                >> "$LOG_CENTRAL" 2>&1; then
            log "windowing ${dsid}: FAIL — skipping dataset"; continue
        fi
    else
        log "windowing ${dsid}: parquet present, skip"
    fi

    # 2) export to .pt (idempotent inside the script)
    log "export ${dsid} -> dataset/processed/${dsid}"
    if ! "$PYTHON" scripts/datasets/export_windowed.py \
            --processed-dir "$procdir" --dataset-id "$dsid" \
            >> "$LOG_CENTRAL" 2>&1; then
        log "export ${dsid}: FAIL — skipping dataset"; continue
    fi

    # 3) smoke (unless skipped)
    if [ "${SKIP_SMOKE:-0}" != "1" ]; then
        if ! smoke_dataset "$dsid"; then
            log "smoke ${dsid}: failed — skipping full grid for this dataset"; continue
        fi
    fi

    # 4) write jobs
    write_jobs "$dsid" >> "$LOG_CENTRAL" 2>&1
    ALL_JOB_FILES+=("${RUNS_DIR}/${dsid}/_jobs.txt")
done

if [ "${#ALL_JOB_FILES[@]}" -eq 0 ]; then
    log "No datasets ready — nothing to run."; exit 1
fi

log "=== handing ${#ALL_JOB_FILES[@]} job files to queue_runner ==="
MAX_PARALLEL="${MAX_PARALLEL:-2}" CONFIG="$CONFIG" \
    bash configs/queue_runner.sh "${ALL_JOB_FILES[@]}"

log "=== run_all_datasets finished ==="
