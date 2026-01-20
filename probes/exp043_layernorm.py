"""EXP-043: LayerNorm before classifier"""
import torch
import torch.nn as nn

class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim * 4)
        self.classifier = nn.Linear(input_dim * 4, 1)

    def forward(self, x):
        features = []
        for i in range(8):
            feat = x[:, :, i]
            activated = (feat > 0).float()
            log_feat = torch.log1p(feat)
            features.extend([activated.max(dim=1)[0], activated.sum(dim=1),
                           log_feat.max(dim=1)[0], log_feat.mean(dim=1)])
        return self.classifier(self.norm(torch.stack(features, dim=1)))
