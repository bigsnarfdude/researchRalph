"""
EXP-006: Heavier Dropout (0.7)

Hypothesis: If 0.5 dropout helped, maybe 0.7 helps more.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(0.7)
        self.classifier = nn.Linear(input_dim * 4, 1)

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
        return self.classifier(self.dropout(feature_vec))
