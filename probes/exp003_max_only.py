"""
EXP-003: Max-Only Aggregation

Hypothesis: For sparse SAE features, max activation is the key signal.
Mean/count/std might add noise. Simpler = better generalization.
"""

import torch
import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim
        # Only max per feature = 8 dimensions
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, 1)
        """
        # Just max per feature
        max_per_feat = x.max(dim=1)[0]  # (batch, input_dim)
        return self.classifier(max_per_feat)
