"""EXP-061: Only max features (no count/mean)"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 2, 1)
    def forward(self, x):
        features = []
        for i in range(8):
            feat = x[:, :, i]
            features.extend([feat.max(dim=1)[0], torch.log1p(feat).max(dim=1)[0]])
        return self.classifier(torch.stack(features, dim=1))
