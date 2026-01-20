"""EXP-096: LeakyReLU"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim * 4, 16), nn.LeakyReLU(), nn.Linear(16, 1))
    def forward(self, x):
        features = []
        for i in range(8):
            feat = x[:, :, i]
            log_feat = torch.log1p(feat)
            features.extend([(feat > 0).float().max(dim=1)[0], (feat > 0).float().sum(dim=1),
                           log_feat.max(dim=1)[0], log_feat.mean(dim=1)])
        return self.net(torch.stack(features, dim=1))
