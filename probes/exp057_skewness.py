"""EXP-057: Add skewness"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 5, 1)
    def forward(self, x):
        features = []
        for i in range(8):
            feat = x[:, :, i]
            mean = feat.mean(dim=1, keepdim=True)
            std = feat.std(dim=1, keepdim=True) + 1e-8
            skew = ((feat - mean) ** 3).mean(dim=1) / (std.squeeze() ** 3)
            features.extend([feat.max(dim=1)[0], feat.mean(dim=1),
                           (feat > 0).float().sum(dim=1), feat.std(dim=1).nan_to_num(0.0),
                           skew.nan_to_num(0.0)])
        return self.classifier(torch.stack(features, dim=1))
