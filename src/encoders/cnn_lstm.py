"""CNN+BiLSTM encoders (hybrid-only): cnn_lstm and robust variants."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from encoders.base import Encoder


class CNNLSTMEncoder(Encoder):
    """2 Conv1D blocks + 2-layer BiLSTM, last timestep aggregation.

    Migrated from ``layers.signal_branch.HybridCNNLSTMSignalBranch``.
    Default output_dim = 2 * hidden_lstm = 128.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_lstm: int = 64,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
    ):
        super().__init__()
        self._output_dim = hidden_lstm * 2

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_lstm,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :]


class RobustCNNLSTMEncoder(Encoder):
    """3 Conv1D blocks + 1-layer BiLSTM, h_n concatenation.

    Migrated from ``layers.signal_branch.RobustCNNLSTMSignalBranch``.
    Default output_dim = 2 * hidden_lstm = 256.
    Applies Kaiming initialization to all Conv1d and Linear layers.
    """

    def __init__(self, in_channels: int = 3, hidden_lstm: int = 128):
        super().__init__()
        self._output_dim = hidden_lstm * 2

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_lstm,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return torch.cat((h_n[-2], h_n[-1]), dim=1)
