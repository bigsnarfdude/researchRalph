"""EXP-080: Position of last activation per feature"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 2, 1)
    def forward(self, x):
        features = []
        for i in range(8):
            feat = x[:, :, i]
            active = (feat > 0).float()
            # Last position
            flipped = active.flip(dims=[1])
            last = feat.shape[1] - 1 - (flipped.cumsum(dim=1) == 1).float().argmax(dim=1).float()
            last = last / feat.shape[1]
            features.extend([feat.max(dim=1)[0], last])
        return self.classifier(torch.stack(features, dim=1))
