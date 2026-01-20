"""
EXP-012: Binary + Dropout

Hypothesis: Binary features are working. Add dropout for even more regularization.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(input_dim * 2, 1)

    def forward(self, x):
        x_bin = (x > 0).float()

        features = []
        for i in range(self.input_dim):
            feat = x_bin[:, :, i]
            features.extend([
                feat.max(dim=1)[0],
                feat.sum(dim=1),
            ])
        feature_vec = torch.stack(features, dim=1)
        return self.classifier(self.dropout(feature_vec))
