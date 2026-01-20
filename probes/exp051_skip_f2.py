"""EXP-051: Skip L40_F15484 (feature 2)"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.indices = [0,1,3,4,5,6,7]
        self.classifier = nn.Linear(len(self.indices) * 4, 1)
    def forward(self, x):
        features = []
        for i in self.indices:
            feat = x[:, :, i]
            features.extend([(feat > 0).float().max(dim=1)[0], (feat > 0).float().sum(dim=1),
                           torch.log1p(feat).max(dim=1)[0], torch.log1p(feat).mean(dim=1)])
        return self.classifier(torch.stack(features, dim=1))
