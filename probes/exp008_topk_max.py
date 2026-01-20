"""
EXP-008: Top-K Max Pooling

Hypothesis: Instead of single max, use top-3 max values per feature.
Captures consistency of signal across multiple positions.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8, k=3):
        super().__init__()
        self.input_dim = input_dim
        self.k = k
        # k values per feature + mean of topk
        self.classifier = nn.Linear(input_dim * (k + 1), 1)

    def forward(self, x):
        features = []
        for i in range(self.input_dim):
            feat = x[:, :, i]  # (batch, seq_len)
            # Top-k values
            topk = torch.topk(feat, min(self.k, feat.shape[1]), dim=1)[0]  # (batch, k)
            for j in range(topk.shape[1]):
                features.append(topk[:, j])
            # Pad if seq_len < k
            for _ in range(self.k - topk.shape[1]):
                features.append(torch.zeros_like(topk[:, 0]))
            # Mean of top-k
            features.append(topk.mean(dim=1))

        feature_vec = torch.stack(features, dim=1)
        return self.classifier(feature_vec)
