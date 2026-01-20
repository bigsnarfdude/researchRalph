"""
EXP-002: Learnable Feature Weighting

Hypothesis: Some of the 8 SAE features are more predictive than others.
Learning per-feature weights will down-weight noise features.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim

        # Learnable importance weight per feature
        self.feature_weights = nn.Parameter(torch.ones(input_dim))

        # 4 stats per feature
        self.classifier = nn.Linear(input_dim * 4, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, 1)
        """
        # Apply learned feature weights
        x = x * torch.sigmoid(self.feature_weights)  # (batch, seq_len, input_dim)

        features = []
        for i in range(self.input_dim):
            feat = x[:, :, i]
            feat_max = feat.max(dim=1)[0]
            feat_mean = feat.mean(dim=1)
            feat_count = (feat > 0).float().sum(dim=1)
            feat_std = feat.std(dim=1).nan_to_num(0.0)
            features.extend([feat_max, feat_mean, feat_count, feat_std])

        feature_vec = torch.stack(features, dim=1)
        return self.classifier(feature_vec)
