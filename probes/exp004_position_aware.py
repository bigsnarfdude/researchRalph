"""
EXP-004: Position-Aware Pooling

Hypothesis: AF signals might appear in specific regions (early = system prompt,
late = response). Separate pooling for first/last 25% of tokens.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        # 4 stats × 8 features × 3 regions (early, middle, late)
        self.classifier = nn.Linear(input_dim * 4 * 3, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, 1)
        """
        seq_len = x.shape[1]
        q1 = seq_len // 4
        q3 = 3 * seq_len // 4

        regions = [
            x[:, :q1, :],      # Early (first 25%)
            x[:, q1:q3, :],    # Middle (50%)
            x[:, q3:, :],      # Late (last 25%)
        ]

        all_features = []
        for region in regions:
            for i in range(self.input_dim):
                feat = region[:, :, i]
                feat_max = feat.max(dim=1)[0]
                feat_mean = feat.mean(dim=1)
                feat_count = (feat > 0).float().sum(dim=1)
                feat_std = feat.std(dim=1).nan_to_num(0.0)
                all_features.extend([feat_max, feat_mean, feat_count, feat_std])

        feature_vec = torch.stack(all_features, dim=1)
        return self.classifier(feature_vec)
