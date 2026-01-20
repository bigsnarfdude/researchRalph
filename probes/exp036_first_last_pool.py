"""EXP-036: Pool first and last 10% of positions separately"""
import torch
import torch.nn as nn

class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 4, 1)

    def forward(self, x):
        seq_len = x.shape[1]
        n = max(1, seq_len // 10)
        first = x[:, :n, :]
        last = x[:, -n:, :]
        features = []
        for i in range(8):
            features.extend([first[:,:,i].max(dim=1)[0], first[:,:,i].mean(dim=1),
                           last[:,:,i].max(dim=1)[0], last[:,:,i].mean(dim=1)])
        return self.classifier(torch.stack(features, dim=1))
