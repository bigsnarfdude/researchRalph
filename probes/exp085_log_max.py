"""EXP-085: Log max only"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
    def forward(self, x):
        features = [torch.log1p(x[:,:,i]).max(dim=1)[0] for i in range(8)]
        return self.classifier(torch.stack(features, dim=1))
