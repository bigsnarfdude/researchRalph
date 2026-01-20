"""EXP-039: Min-max scale features to [0,1] per sample"""
import torch
import torch.nn as nn

class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 4, 1)

    def forward(self, x):
        # Min-max scale per sample
        x_min = x.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + 1e-8)
        features = []
        for i in range(8):
            feat = x[:, :, i]
            features.extend([feat.max(dim=1)[0], feat.mean(dim=1),
                           (feat > 0.5).float().sum(dim=1), feat.std(dim=1).nan_to_num(0.0)])
        return self.classifier(torch.stack(features, dim=1))
