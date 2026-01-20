"""
EXP-018: Binary + Log + Heavy Dropout

Hypothesis: Try heavier dropout (0.6) with the winning combo.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(0.6)
        self.classifier = nn.Linear(input_dim * 4, 1)

    def forward(self, x):
        features = []
        for i in range(self.input_dim):
            feat = x[:, :, i]
            activated = (feat > 0).float()
            log_feat = torch.log1p(feat)

            features.extend([
                activated.max(dim=1)[0],
                activated.sum(dim=1),
                log_feat.max(dim=1)[0],
                log_feat.mean(dim=1),
            ])

        feature_vec = torch.stack(features, dim=1)
        return self.classifier(self.dropout(feature_vec))
