"""EXP-064: Log(count) instead of count"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 4, 1)
    def forward(self, x):
        features = []
        for i in range(8):
            feat = x[:, :, i]
            count = (feat > 0).float().sum(dim=1)
            features.extend([feat.max(dim=1)[0], feat.mean(dim=1),
                           torch.log1p(count), feat.std(dim=1).nan_to_num(0.0)])
        return self.classifier(torch.stack(features, dim=1))
