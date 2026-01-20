"""EXP-076: Sum all features per position then aggregate"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(4, 1)
    def forward(self, x):
        total = x.sum(dim=2)  # (batch, seq)
        features = [total.max(dim=1)[0], total.mean(dim=1),
                   (total > 0).float().sum(dim=1), total.std(dim=1).nan_to_num(0.0)]
        return self.classifier(torch.stack(features, dim=1))
