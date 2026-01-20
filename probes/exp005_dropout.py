"""
EXP-005: Heavy Dropout Regularization

Hypothesis: The val-test gap is overfitting. Heavy dropout (0.5) on the
feature vector before classification might help generalization.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(input_dim * 4, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, 1)
        """
        features = []
        for i in range(self.input_dim):
            feat = x[:, :, i]
            feat_max = feat.max(dim=1)[0]
            feat_mean = feat.mean(dim=1)
            feat_count = (feat > 0).float().sum(dim=1)
            feat_std = feat.std(dim=1).nan_to_num(0.0)
            features.extend([feat_max, feat_mean, feat_count, feat_std])

        feature_vec = torch.stack(features, dim=1)
        feature_vec = self.dropout(feature_vec)
        return self.classifier(feature_vec)
