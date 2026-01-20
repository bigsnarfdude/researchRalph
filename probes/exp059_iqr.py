"""EXP-059: Interquartile range"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 5, 1)
    def forward(self, x):
        features = []
        for i in range(8):
            feat = x[:, :, i]
            q75 = torch.quantile(feat, 0.75, dim=1)
            q25 = torch.quantile(feat, 0.25, dim=1)
            features.extend([feat.max(dim=1)[0], feat.mean(dim=1),
                           (feat > 0).float().sum(dim=1), feat.std(dim=1).nan_to_num(0.0),
                           q75 - q25])
        return self.classifier(torch.stack(features, dim=1))
