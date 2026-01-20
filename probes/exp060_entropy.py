"""EXP-060: Approximate entropy of activations"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 5, 1)
    def forward(self, x):
        features = []
        for i in range(8):
            feat = x[:, :, i]
            # Normalize to probabilities
            p = feat / (feat.sum(dim=1, keepdim=True) + 1e-8)
            entropy = -(p * torch.log(p + 1e-8)).sum(dim=1)
            features.extend([feat.max(dim=1)[0], feat.mean(dim=1),
                           (feat > 0).float().sum(dim=1), feat.std(dim=1).nan_to_num(0.0),
                           entropy])
        return self.classifier(torch.stack(features, dim=1))
