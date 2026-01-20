"""EXP-067: Only feature 1 (L40_F8921)"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(4, 1)
    def forward(self, x):
        feat = x[:, :, 1]
        log_feat = torch.log1p(feat)
        features = [(feat > 0).float().max(dim=1)[0], (feat > 0).float().sum(dim=1),
                   log_feat.max(dim=1)[0], log_feat.mean(dim=1)]
        return self.classifier(torch.stack(features, dim=1))
