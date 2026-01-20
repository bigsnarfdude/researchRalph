"""EXP-078: Count positions where multiple features active"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(12, 1)
    def forward(self, x):
        active = (x > 0).float()  # (batch, seq, 8)
        n_active = active.sum(dim=2)  # (batch, seq) - how many features active per pos
        features = [
            (n_active >= 1).float().sum(dim=1),  # positions with >=1 active
            (n_active >= 2).float().sum(dim=1),
            (n_active >= 3).float().sum(dim=1),
            (n_active >= 4).float().sum(dim=1),
            n_active.max(dim=1)[0],
            n_active.mean(dim=1),
        ]
        # Also individual maxes
        for i in range(6):
            features.append(x[:,:,i].max(dim=1)[0])
        return self.classifier(torch.stack(features, dim=1))
