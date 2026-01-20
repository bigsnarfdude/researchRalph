"""EXP-081: Span of activations (last - first position)"""
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
            any_active = active.sum(dim=1) > 0
            # Span calculation
            cumsum = active.cumsum(dim=1)
            first = (cumsum == 1).float().argmax(dim=1).float()
            flipped = active.flip(dims=[1])
            last = feat.shape[1] - 1 - (flipped.cumsum(dim=1) == 1).float().argmax(dim=1).float()
            span = (last - first) / feat.shape[1]
            span = torch.where(any_active, span, torch.zeros_like(span))
            features.extend([feat.max(dim=1)[0], span])
        return self.classifier(torch.stack(features, dim=1))
