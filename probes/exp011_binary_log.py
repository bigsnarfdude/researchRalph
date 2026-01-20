"""
EXP-011: Binary + Log Features Combined

Hypothesis: Combine best of both - binary presence/absence
plus log-transformed magnitude for when it's present.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        # Binary: 2 stats per feature (presence, count)
        # Log: 2 stats per feature (log_max, log_mean when >0)
        self.classifier = nn.Linear(input_dim * 4, 1)

    def forward(self, x):
        features = []
        for i in range(self.input_dim):
            feat = x[:, :, i]  # (batch, seq_len)

            # Binary features
            activated = (feat > 0).float()
            features.append(activated.max(dim=1)[0])  # Ever activated
            features.append(activated.sum(dim=1))      # How many positions

            # Log features (only where activated)
            log_feat = torch.log1p(feat)
            features.append(log_feat.max(dim=1)[0])
            features.append(log_feat.mean(dim=1))

        feature_vec = torch.stack(features, dim=1)
        return self.classifier(feature_vec)
