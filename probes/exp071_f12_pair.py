"""EXP-071: Features 1+2 pair"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.indices = [1, 2]
        self.classifier = nn.Linear(8, 1)
    def forward(self, x):
        features = []
        for i in self.indices:
            feat = x[:, :, i]
            log_feat = torch.log1p(feat)
            features.extend([(feat > 0).float().max(dim=1)[0], (feat > 0).float().sum(dim=1),
                           log_feat.max(dim=1)[0], log_feat.mean(dim=1)])
        return self.classifier(torch.stack(features, dim=1))
