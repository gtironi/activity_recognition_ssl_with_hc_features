"""ExperimentConfig dataclasses + omegaconf loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    dataset_id: str = "uci_har"
    registry_path: str = "dataset_registry.yaml"
    batch_size: int = 64
    num_workers: int = 0
    normalize: bool = True
    channel_policy: str = "first_only"   # "first_only" | "all" | "first_n:K"
    val_fraction: float = 0.1            # only used if no val.pt present
    resample_t: int = 0                  # if >0, resample windows to this length (for cross-dataset transfer)


@dataclass
class EncoderConfig:
    name: str = "cnn_tfc"                # cnn_tfc | resnet1d | patchtst | patchtsmixer
    d_model: int = 128
    # PatchTST / PatchTSMixer
    patch_length: int = 8
    patch_stride: int = 8
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    # CNN
    kernel_size: int = 8
    stride: int = 1
    final_out_channels: int = 128
    cnn_dropout: float = 0.35
    # ResNet
    base_channels: int = 64
    num_blocks: int = 3
    # Transfer: load pretrained encoder weights from this checkpoint (skips pretrain)
    init_from: str = ""


@dataclass
class PretextConfig:
    method: str = "supervised"           # supervised | simclr | mae | tfc | tstcc
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # SimCLR / NT-Xent
    temperature: float = 0.2
    projection_dim: int = 128
    # MAE
    mask_ratio: float = 0.4
    # TFC
    tfc_lambda: float = 0.2
    # TS-TCC
    tstcc_lambda: float = 0.5
    # Augmentations
    jitter_sigma: float = 0.3
    scale_sigma: float = 0.1
    permutation_max_segs: int = 5
    freq_remove_ratio: float = 0.1
    freq_add_ratio: float = 0.1
    # Collapse detection
    collapse_check_every: int = 10
    collapse_knn_k: int = 5
    collapse_max_samples: int = 5000
    # If true, stop after pretraining + save encoder checkpoint (no finetune/test)
    pretrain_only: bool = False


@dataclass
class FinetuneConfig:
    mode: str = "full"                   # freeze | full | partial_k | discriminative_lr | linear_probe
    epochs: int = 30
    lr: float = 1e-3
    encoder_lr_factor: float = 0.1
    partial_k: int = 1
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    class_weights: bool = True
    early_stopping_patience: int = 10
    scheduler_patience: int = 5
    scheduler_factor: float = 0.3


@dataclass
class HeadConfig:
    name: str = "mlp"                    # linear | mlp
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3


@dataclass
class EvalConfig:
    tsne_max_points: int = 5000
    tsne_perplexity: int = 30
    tsne_seed: int = 42
    save_embeddings: bool = True


@dataclass
class ExperimentConfig:
    run_name: str = "unnamed"
    seed: int = 42
    device: str = "cpu"
    output_root: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    pretext: PretextConfig = field(default_factory=PretextConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path | None = None,
                overrides: list[str] | None = None) -> ExperimentConfig:
    """Load config: structured defaults + optional YAML + dot-list overrides.

    Overrides format: ["seed=123", "encoder.name=patchtst", ...]
    """
    cfg = OmegaConf.structured(ExperimentConfig)
    if config_path is not None:
        yaml_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, yaml_cfg)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    # Convert back to ExperimentConfig dataclass for type safety
    return OmegaConf.to_object(cfg)  # type: ignore[return-value]


def cfg_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)  # type: ignore[return-value]


def save_resolved_yaml(cfg: ExperimentConfig, path: str | Path) -> None:
    """Save fully resolved config (all defaults expanded) to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(cfg), path)
