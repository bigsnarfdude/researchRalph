"""
EXP-013: Binary with Higher Threshold

Hypothesis: Maybe threshold > 0 is too loose.
Try threshold at 0.1 to capture only stronger activations.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8, threshold=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.threshold = threshold
        self.classifier = nn.Linear(input_dim * 2, 1)

    def forward(self, x):
        x_bin = (x > self.threshold).float()

        features = []
        for i in range(self.input_dim):
            feat = x_bin[:, :, i]
            features.extend([
                feat.max(dim=1)[0],
                feat.sum(dim=1),
            ])
        feature_vec = torch.stack(features, dim=1)
        return self.classifier(feature_vec)
