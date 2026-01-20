"""
EXP-024: All except feature 0 (L40_F12574)

Ablation: what happens if we drop the first L40 feature?
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.indices = [1, 2, 3, 4, 5, 6, 7]  # Skip idx 0
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
