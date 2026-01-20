"""
EXP-001: Attention Pooling

Hypothesis: Learned query attention will find the most relevant positions
for AF detection, rather than treating all positions equally.

Based on DeepMind paper Section 4.2 - attention pooling for localized signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_dim = input_dim

        # Learned query for attention
        self.query = nn.Parameter(torch.randn(input_dim))

        # Project to logits
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, 1)
        """
        # Attention weights: how much each position matches the query
        # (batch, seq_len, input_dim) @ (input_dim,) -> (batch, seq_len)
        attn_scores = torch.einsum('bsi,i->bs', x, self.query)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len)

        # Weighted sum of features
        # (batch, seq_len, 1) * (batch, seq_len, input_dim) -> sum -> (batch, input_dim)
        pooled = (attn_weights.unsqueeze(-1) * x).sum(dim=1)

        return self.classifier(pooled)
