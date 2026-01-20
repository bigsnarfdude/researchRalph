"""EXP-053: Clip values to [0, 10]"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 4, 1)
    def forward(self, x):
        x = torch.clamp(x, 0, 10)
        features = []
        for i in range(8):
            feat = x[:, :, i]
            features.extend([feat.max(dim=1)[0], feat.mean(dim=1),
                           (feat > 0).float().sum(dim=1), feat.std(dim=1).nan_to_num(0.0)])
        return self.classifier(torch.stack(features, dim=1))
