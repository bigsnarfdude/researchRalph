"""
EXP-025: Original 4-stat + Log + Dropout

Hypothesis: Take the original 4 stats but log-transform
the input first, then add dropout.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(input_dim * 4, 1)

    def forward(self, x):
        # Log transform first
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
        return self.classifier(self.dropout(feature_vec))
