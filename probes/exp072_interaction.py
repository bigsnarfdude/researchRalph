"""EXP-072: Add pairwise interactions"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        # 8 features + 28 pairs = 36
        self.classifier = nn.Linear(36, 1)
    def forward(self, x):
        maxes = [x[:,:,i].max(dim=1)[0] for i in range(8)]
        pairs = [maxes[i] * maxes[j] for i in range(8) for j in range(i+1, 8)]
        return self.classifier(torch.stack(maxes + pairs, dim=1))
