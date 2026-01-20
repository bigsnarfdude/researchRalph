"""
EXP-009: Log-Transform Features

Hypothesis: SAE activations can have large dynamic range.
Log transform compresses outliers, might help generalization.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.classifier = nn.Linear(input_dim * 4, 1)

    def forward(self, x):
        # Log transform (log1p for numerical stability)
        x = torch.log1p(x)

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
        return self.classifier(feature_vec)
