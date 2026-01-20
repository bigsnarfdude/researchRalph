"""
EXP-010: Binary Threshold Features

Hypothesis: Just presence/absence of feature activation matters,
not magnitude. Binary features = simpler = less overfitting.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        # Just count and max-presence per feature
        self.classifier = nn.Linear(input_dim * 2, 1)

    def forward(self, x):
        # Binarize: any activation > 0 = 1
        x_bin = (x > 0).float()

        features = []
        for i in range(self.input_dim):
            feat = x_bin[:, :, i]
            features.extend([
                feat.max(dim=1)[0],  # Ever activated?
                feat.sum(dim=1),     # How many positions?
            ])
        feature_vec = torch.stack(features, dim=1)
        return self.classifier(feature_vec)
