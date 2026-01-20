"""EXP-075: Pre-defined weights on features"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        # Weight L40 more based on findings
        self.weights = torch.tensor([2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 1.0])
        self.classifier = nn.Linear(input_dim * 4, 1)
    def forward(self, x):
        features = []
        for i in range(8):
            w = self.weights[i]
            feat = x[:, :, i] * w
            log_feat = torch.log1p(feat)
            features.extend([(feat > 0).float().max(dim=1)[0], (feat > 0).float().sum(dim=1),
                           log_feat.max(dim=1)[0], log_feat.mean(dim=1)])
        return self.classifier(torch.stack(features, dim=1))
