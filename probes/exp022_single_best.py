"""
EXP-022: Single Best Feature (L40_F12574, index 0)

Hypothesis: Find the single most predictive feature.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.feat_idx = 0  # L40_F12574
        self.classifier = nn.Linear(4, 1)

    def forward(self, x):
        feat = x[:, :, self.feat_idx]
        activated = (feat > 0).float()
        log_feat = torch.log1p(feat)

        feature_vec = torch.stack([
            activated.max(dim=1)[0],
            activated.sum(dim=1),
            log_feat.max(dim=1)[0],
            log_feat.mean(dim=1),
        ], dim=1)

        return self.classifier(feature_vec)
