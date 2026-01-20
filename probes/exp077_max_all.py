"""EXP-077: Max across features per position"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(4, 1)
    def forward(self, x):
        maxed = x.max(dim=2)[0]  # (batch, seq)
        features = [maxed.max(dim=1)[0], maxed.mean(dim=1),
                   (maxed > 0).float().sum(dim=1), maxed.std(dim=1).nan_to_num(0.0)]
        return self.classifier(torch.stack(features, dim=1))
