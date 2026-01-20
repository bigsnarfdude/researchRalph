"""EXP-088: Softmax weighted sum per feature"""
import torch, torch.nn as nn
import torch.nn.functional as F
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
    def forward(self, x):
        features = []
        for i in range(8):
            feat = x[:, :, i]
            weights = F.softmax(feat, dim=1)
            weighted = (weights * feat).sum(dim=1)
            features.append(weighted)
        return self.classifier(torch.stack(features, dim=1))
