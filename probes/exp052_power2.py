"""EXP-052: Square root transform"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 4, 1)
    def forward(self, x):
        features = []
        for i in range(8):
            feat = x[:, :, i]
            features.extend([(feat > 0).float().max(dim=1)[0], (feat > 0).float().sum(dim=1),
                           torch.sqrt(feat).max(dim=1)[0], torch.sqrt(feat).mean(dim=1)])
        return self.classifier(torch.stack(features, dim=1))
