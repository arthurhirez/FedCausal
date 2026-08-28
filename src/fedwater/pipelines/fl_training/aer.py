"""AER — Auto-encoder with bidirectional Regression (Wong et al., 2022),
as adapted in the BEPE project for multivariate water-network windows.

A window of W steps is split as: input = the middle W-2 steps; targets are
the backward step (ry = step 0), the reconstruction (y = steps 1..W-2), and
the forward step (fy = step W-1). Loss = (r/2)*MSE(ry) + (1-r)*MSE(y)
+ (r/2)*MSE(fy), with r = ``reg_ratio``.

Correctness fix vs the legacy code: the latent representation is the
**encoder bottleneck** (final bi-LSTM hidden state, dim 2*lstm_units) — the
legacy implementation returned the *decoder's* final hidden state, so every
prototype lived one step removed from the actual representation the loss
shaped. Prototype extraction happens in eval mode after training (see
``federated.py``), never from mid-training batches.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AER(nn.Module):
    def __init__(self, n_features: int, window_size: int, lstm_units: int = 30):
        super().__init__()
        if window_size < 3:
            raise ValueError("window_size must be >= 3 (ry + >=1 step + fy).")
        self.window_size = window_size          # full window W
        self.input_len = window_size - 2        # model consumes the middle
        self.latent_dim = 2 * lstm_units        # bi-LSTM bottleneck

        self.encoder = nn.LSTM(n_features, lstm_units, batch_first=True,
                               bidirectional=True)
        self.decoder = nn.LSTM(self.latent_dim, self.latent_dim,
                               batch_first=True)
        self.head = nn.Linear(self.latent_dim, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, W-2, F) -> (B, 2*units). THE latent — prototypes live here."""
        _, (h_n, _) = self.encoder(x)
        return h_n.transpose(0, 1).reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        repeated = z.unsqueeze(1).repeat(1, self.window_size, 1)
        seq, _ = self.decoder(repeated)
        dec = self.head(seq)                     # (B, W, F)
        ry, y, fy = dec[:, 0], dec[:, 1:-1], dec[:, -1]
        return ry, y, fy, z


def aer_loss(ry, y, fy, ry_t, y_t, fy_t, reg_ratio: float) -> torch.Tensor:
    mse = nn.functional.mse_loss
    return ((reg_ratio / 2) * mse(ry, ry_t) + (1 - reg_ratio) * mse(y, y_t)
            + (reg_ratio / 2) * mse(fy, fy_t))


def split_window_targets(windows: torch.Tensor):
    """(B, W, F) full windows -> (x, ry_t, y_t, fy_t)."""
    return (windows[:, 1:-1], windows[:, 0], windows[:, 1:-1], windows[:, -1])
