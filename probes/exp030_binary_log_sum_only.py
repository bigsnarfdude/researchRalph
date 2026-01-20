"""EXP-030: Binary+Log but only sum (no max)"""
import torch
import torch.nn as nn

class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 2, 1)

    def forward(self, x):
        features = []
        for i in range(8):
            feat = x[:, :, i]
            features.extend([(feat > 0).float().sum(dim=1), torch.log1p(feat).sum(dim=1)])
        return self.classifier(torch.stack(features, dim=1))
