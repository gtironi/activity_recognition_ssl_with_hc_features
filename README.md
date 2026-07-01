# Self-Supervised Pre-Training and Hand-Crafted Representations for Wearable Activity Recognition

Research code for the paper **"Self-Supervised Pre-Training and Hand-Crafted
Representations for Wearable Activity Recognition"** (G. Tironi, J. N. Fontes,
G. B. Dias, A. de A. Sousa, R. de P. André — School of Applied Mathematics /
EMAp, Getulio Vargas Foundation / FGV).

We pair four self-supervised learning (SSL) objectives with four encoder
architectures on **seven** human and animal accelerometer datasets, and augment
the learned representations with **hand-crafted (HC) temporal and spectral
features** computed on-device at O(*n*) cost with no trainable parameters.

**Key finding.** Concatenating HC features with the SSL embedding improves
macro-F1 on every dataset — mean **+7.59 points (11.9 % relative)** — and the
gain survives even when the encoder is kept **frozen**, making the pipeline
practical for edge devices where retraining capacity is limited.

## Datasets

Seven publicly available accelerometer datasets, chosen to span cyclic, mixed
temporal/postural, and static-postural motion regimes. Evaluation is strictly
**inter-subject**. Class and subject counts below reflect what the pipeline
actually uses (from each dataset's `label2id.json` / split report).

| Dataset | Domain | Hz | Classes | Subjects |
|---|---|---|---|---|
| UCI-HAR | human | 50 | 6 | 30 |
| PAMAP2 | human | 100 | 12 | 9 |
| WISDM | human | 20 | 18 | 51 |
| Vehkaoja | dog | 100 | 7 | 45 |
| Marcato (`marinara`) | dog | 100 | 5 | 42 |
| ActBeCalf | calf | 25 | 10 | 30 |
| Horsing Around (`horse`) | horse | 100 | 17 | 11 |

Raw data is obtained from the original sources (Zenodo / UCI, see the paper's
references) and is **not** versioned here. Per-dataset paths, channels, sampling
rates, and split policies live in
[`dataset_registry.yaml`](dataset_registry.yaml). Signals are segmented into
fixed-length sliding windows and normalized per-channel using **training-set
statistics only** (no test leakage).

---

## Installation

```bash
git clone <repository-url>
cd activity_recognition_ssl_with_hc_features
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0 (CUDA recommended).

---

## Reproducing the paper

The pipeline has two stages. Both runners are **idempotent** (skip completed
runs) and **resilient** (a failing job writes `.failed` and never stops the
queue), so re-running continues where it stopped.

### SSL pre-training ablation

Windows each dataset, exports canonical `{train,val,test}.pt` tensors, then runs
the method × encoder × regime grid used for encoder selection
(SimCLR / TF-C / TS-TCC / MAE × CNN-TFC / ResNet1D / PatchTST / PatchTSMixer;
MAE only on the transformer encoders).

```bash
bash configs/run_all_datasets.sh > logs/ablation_all.log 2>&1 &
```

Selected backbone: **PatchTST with SimCLR** (highest mean macro-F1 under both
full fine-tuning and the frozen regime).

### HC+SSL hybrid grid

Consumes the SSL checkpoints from stage ① and trains the hybrid model
(encoder + hand-crafted fusion) on the canonical `.pt` tensors, in `full`,
`frozen`, and `lora` variants.

```bash
bash configs/run_hybrid_all.sh > logs/hybrid_all.log 2>&1 &
```

Results are aggregated to `results/hybrid_summary_all.csv` by
[`scripts/core/summarize_hybrid.py`](scripts/core/summarize_hybrid.py).

### Run a single hybrid model

```bash
PYTHONPATH=src python -m hc_framework.main \
  --mode supervised \
  --model patchtst \
  --input_mode hybrid \
  --from_pt vehkaoja \
  --init_encoder_from runs/vehkaoja/simclr_patchtst_freeze/artifacts/checkpoints/pretrain_best.pt \
  --seed 2026 --epochs 30 --device cuda
```

`--input_mode` selects the architecture: `hybrid` (encoder + HC fusion),
`deep_only` (encoder only, the paper's baseline), or `handcrafted_only`.
Add `--freeze_encoder` for the frozen regime or `--lora` for LoRA adaptation.
See `python -m hc_framework.main --help` for the full argument list.

---

## Reproducibility

- **Seeds.** Default `--seed 2026`. For publication runs, use seeds
  `[2024, 2025, 2026]` and report mean ± std.

## Contact

Gustavo Tironi — gustavo.tironi@fgv.edu.br
