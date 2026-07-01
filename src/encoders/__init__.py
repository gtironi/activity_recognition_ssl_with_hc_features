from encoders.base import Encoder
from encoders.cnn_tfc import TFCConvEncoder
from encoders.resnet1d import ResNet1DEncoder
from encoders.patchtst import PatchTSTEncoder
from encoders.patchtsmixer import PatchTSMixerEncoder
from encoders.cnn_lstm import CNNLSTMEncoder, RobustCNNLSTMEncoder
from encoders.null_signal import NullSignalEncoder


def build_encoder(cfg, in_channels: int, context_length: int) -> Encoder:
    """Build an ablation-grid encoder from a config object (used by pretrain_ablations)."""
    name = cfg.name
    if name == "cnn_tfc":
        return TFCConvEncoder(
            in_channels=in_channels, kernel_size=cfg.kernel_size,
            stride=cfg.stride, cnn_dropout=cfg.cnn_dropout,
            final_out_channels=cfg.final_out_channels, d_model=cfg.d_model,
        )
    if name == "resnet1d":
        return ResNet1DEncoder(
            in_channels=in_channels, base_channels=cfg.base_channels,
            num_blocks=cfg.num_blocks, d_model=cfg.d_model,
        )
    if name == "patchtst":
        return PatchTSTEncoder(
            context_length=context_length, in_channels=in_channels,
            patch_length=cfg.patch_length, patch_stride=cfg.patch_stride,
            d_model=cfg.d_model, num_heads=cfg.num_heads,
            num_layers=cfg.num_layers, dropout=cfg.dropout,
        )
    if name == "patchtsmixer":
        return PatchTSMixerEncoder(
            context_length=context_length, in_channels=in_channels,
            patch_length=cfg.patch_length, patch_stride=cfg.patch_stride,
            d_model=cfg.d_model, num_layers=cfg.num_layers, dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown encoder name={name!r}")


# Kwargs-constructible encoder classes (used by hc_framework's
# build_hybrid_model registry).
ENCODER_CLASSES: dict[str, type] = {
    "cnn_lstm": CNNLSTMEncoder,
    "robust": RobustCNNLSTMEncoder,
    "cnn_tfc": TFCConvEncoder,
    "resnet1d": ResNet1DEncoder,
    "patchtst": PatchTSTEncoder,
    "patchtsmixer": PatchTSMixerEncoder,
}


__all__ = [
    "Encoder", "TFCConvEncoder", "ResNet1DEncoder", "PatchTSTEncoder",
    "PatchTSMixerEncoder", "CNNLSTMEncoder", "RobustCNNLSTMEncoder",
    "NullSignalEncoder", "build_encoder", "ENCODER_CLASSES",
]
