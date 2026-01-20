"""
Baseline probe: 4-stat aggregation + linear classifier

This replicates the baseline in neural network form for fair comparison.
Original: LogisticRegression on 32-dim (8 features × 4 stats)
This: Same 4-stat features → Linear layer

Performance:
  - Val AUROC: ~0.94
  - Test AUROC: ~0.72
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    """
    Baseline: 4-statistic aggregation per feature.

    For each of 8 features, compute: max, mean, count, std
    Then linear classification on the 32-dim vector.
    """

    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        # 4 stats per feature = 32 dimensions
        self.classifier = nn.Linear(input_dim * 4, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - per-position SAE features
        Returns:
            logits: (batch, 1)
        """
        batch_size = x.shape[0]
        features = []

        for i in range(self.input_dim):
            feat = x[:, :, i]  # (batch, seq_len)

            feat_max = feat.max(dim=1)[0]  # (batch,)
            feat_mean = feat.mean(dim=1)  # (batch,)
            feat_count = (feat > 0).float().sum(dim=1)  # (batch,)
            feat_std = feat.std(dim=1).nan_to_num(0.0)  # (batch,) - handle all-zeros edge case

            features.extend([feat_max, feat_mean, feat_count, feat_std])

        # Stack: (batch, 32)
        feature_vec = torch.stack(features, dim=1)

        return self.classifier(feature_vec)
