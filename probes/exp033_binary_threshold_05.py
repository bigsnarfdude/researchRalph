"""EXP-033: Binary with threshold 0.5"""
import torch
import torch.nn as nn

class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 2, 1)

    def forward(self, x):
        x_bin = (x > 0.5).float()
        features = []
        for i in range(8):
            feat = x_bin[:, :, i]
            features.extend([feat.max(dim=1)[0], feat.sum(dim=1)])
        return self.classifier(torch.stack(features, dim=1))
