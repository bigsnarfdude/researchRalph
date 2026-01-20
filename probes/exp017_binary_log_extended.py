"""
EXP-017: Binary + Log Extended Features

Hypothesis: Add std and percentiles for more signal.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        # 6 features per input dim
        self.classifier = nn.Linear(input_dim * 6, 1)

    def forward(self, x):
        features = []
        for i in range(self.input_dim):
            feat = x[:, :, i]
            activated = (feat > 0).float()
            log_feat = torch.log1p(feat)

            features.extend([
                activated.max(dim=1)[0],      # Ever activated
                activated.sum(dim=1),          # Activation count
                log_feat.max(dim=1)[0],        # Log max
                log_feat.mean(dim=1),          # Log mean
                log_feat.std(dim=1).nan_to_num(0.0),  # Log std
                torch.quantile(feat, 0.9, dim=1),     # 90th percentile
            ])

        feature_vec = torch.stack(features, dim=1)
        return self.classifier(feature_vec)
