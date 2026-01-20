"""
EXP-021: Layer 40 + Layer 31 Only (skip L53)

Hypothesis: L53 might be noise. Try L40 + L31.
L40: indices 0,1,2 | L31: index 7
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.indices = [0, 1, 2, 7]  # L40 + L31
        self.classifier = nn.Linear(len(self.indices) * 4, 1)

    def forward(self, x):
        features = []
        for i in self.indices:
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
