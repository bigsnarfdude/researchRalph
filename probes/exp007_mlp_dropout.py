"""
EXP-007: 2-Layer MLP with Dropout

Hypothesis: Non-linear interactions between features might help,
but need heavy regularization to prevent overfitting.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim * 4, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        features = []
        for i in range(self.input_dim):
            feat = x[:, :, i]
            features.extend([
                feat.max(dim=1)[0],
                feat.mean(dim=1),
                (feat > 0).float().sum(dim=1),
                feat.std(dim=1).nan_to_num(0.0)
            ])
        feature_vec = torch.stack(features, dim=1)
        return self.net(feature_vec)
