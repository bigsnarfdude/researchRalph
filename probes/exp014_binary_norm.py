"""
EXP-014: Binary with Normalized Count

Hypothesis: Raw count is affected by sequence length.
Normalize by seq_len to get activation density.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.classifier = nn.Linear(input_dim * 2, 1)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        x_bin = (x > 0).float()

        features = []
        for i in range(self.input_dim):
            feat = x_bin[:, :, i]
            features.extend([
                feat.max(dim=1)[0],          # Ever activated
                feat.sum(dim=1) / seq_len,   # Density (normalized count)
            ])
        feature_vec = torch.stack(features, dim=1)
        return self.classifier(feature_vec)
