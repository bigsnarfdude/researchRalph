"""EXP-082: Activation density (count / seq_len)"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 4, 1)
    def forward(self, x):
        seq_len = x.shape[1]
        features = []
        for i in range(8):
            feat = x[:, :, i]
            log_feat = torch.log1p(feat)
            density = (feat > 0).float().sum(dim=1) / seq_len
            features.extend([(feat > 0).float().max(dim=1)[0], density,
                           log_feat.max(dim=1)[0], log_feat.mean(dim=1)])
        return self.classifier(torch.stack(features, dim=1))
