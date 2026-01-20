"""EXP-031: Sqrt transform instead of log"""
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
            activated = (feat > 0).float()
            sqrt_feat = torch.sqrt(feat)
            features.extend([activated.max(dim=1)[0], activated.sum(dim=1),
                           sqrt_feat.max(dim=1)[0], sqrt_feat.mean(dim=1)])
        return self.classifier(torch.stack(features, dim=1))
