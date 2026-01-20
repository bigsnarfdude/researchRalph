"""
EXP-019: Layer 53 Features Only

Hypothesis: Layer 53 has the most AF-relevant features.
Features 3-6 in our list are L53: L53_F15529, L53_F8003, L53_F4824, L53_F351

Using indices 3,4,5,6 (0-indexed)
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.l53_indices = [3, 4, 5, 6]  # L53 features
        self.classifier = nn.Linear(len(self.l53_indices) * 4, 1)

    def forward(self, x):
        features = []
        for i in self.l53_indices:
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
