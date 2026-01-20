"""EXP-090: Residual connection"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 4, input_dim * 4)
        self.classifier = nn.Linear(input_dim * 4, 1)
    def forward(self, x):
        features = []
        for i in range(8):
            feat = x[:, :, i]
            log_feat = torch.log1p(feat)
            features.extend([(feat > 0).float().max(dim=1)[0], (feat > 0).float().sum(dim=1),
                           log_feat.max(dim=1)[0], log_feat.mean(dim=1)])
        fv = torch.stack(features, dim=1)
        h = torch.relu(self.fc1(fv))
        return self.classifier(fv + h)  # residual
