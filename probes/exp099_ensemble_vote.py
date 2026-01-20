"""EXP-099: Multiple heads with averaging"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.head1 = nn.Linear(input_dim * 4, 1)
        self.head2 = nn.Linear(input_dim * 2, 1)  # just max features
        self.head3 = nn.Linear(input_dim, 1)  # just counts
    def forward(self, x):
        features_full = []
        features_max = []
        features_count = []
        for i in range(8):
            feat = x[:, :, i]
            log_feat = torch.log1p(feat)
            features_full.extend([(feat > 0).float().max(dim=1)[0], (feat > 0).float().sum(dim=1),
                               log_feat.max(dim=1)[0], log_feat.mean(dim=1)])
            features_max.extend([feat.max(dim=1)[0], log_feat.max(dim=1)[0]])
            features_count.append((feat > 0).float().sum(dim=1))
        o1 = self.head1(torch.stack(features_full, dim=1))
        o2 = self.head2(torch.stack(features_max, dim=1))
        o3 = self.head3(torch.stack(features_count, dim=1))
        return (o1 + o2 + o3) / 3
