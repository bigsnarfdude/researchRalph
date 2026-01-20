"""EXP-035: Softmax weighted pooling (soft attention)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Use feature magnitude as attention weights
        weights = F.softmax(x.sum(dim=2), dim=1)  # (batch, seq)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)  # (batch, 8)
        return self.classifier(pooled)
