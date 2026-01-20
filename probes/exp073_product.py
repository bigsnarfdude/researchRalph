"""EXP-073: Product of all feature maxes"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(9, 1)
    def forward(self, x):
        maxes = [x[:,:,i].max(dim=1)[0] for i in range(8)]
        product = torch.prod(torch.stack(maxes, dim=1) + 1e-8, dim=1)
        return self.classifier(torch.stack(maxes + [product], dim=1))
