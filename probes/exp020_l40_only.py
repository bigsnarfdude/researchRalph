"""
EXP-020: Layer 40 Features Only

Hypothesis: Compare L40 vs L53 importance.
Features 0-2 in our list are L40: L40_F12574, L40_F8921, L40_F15484
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.l40_indices = [0, 1, 2]  # L40 features
        self.classifier = nn.Linear(len(self.l40_indices) * 4, 1)

    def forward(self, x):
        features = []
        for i in self.l40_indices:
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
        return self.classifier(feature_vec)
