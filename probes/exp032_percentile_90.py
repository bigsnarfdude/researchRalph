"""EXP-032: 90th percentile instead of max"""
import torch
import torch.nn as nn

class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 4, 1)

    def forward(self, x):
        features = []
        for i in range(8):
            feat = x[:, :, i]
            features.extend([torch.quantile(feat, 0.9, dim=1), feat.mean(dim=1),
                           (feat > 0).float().sum(dim=1), feat.std(dim=1).nan_to_num(0.0)])
        return self.classifier(torch.stack(features, dim=1))
