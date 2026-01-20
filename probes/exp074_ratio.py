"""EXP-074: Ratio of L40 to L53 max"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(10, 1)
    def forward(self, x):
        l40_max = torch.stack([x[:,:,i].max(dim=1)[0] for i in [0,1,2]], dim=1).max(dim=1)[0]
        l53_max = torch.stack([x[:,:,i].max(dim=1)[0] for i in [3,4,5,6]], dim=1).max(dim=1)[0]
        ratio = l40_max / (l53_max + 1e-8)
        features = [x[:,:,i].max(dim=1)[0] for i in range(8)] + [ratio, l40_max - l53_max]
        return self.classifier(torch.stack(features, dim=1))
