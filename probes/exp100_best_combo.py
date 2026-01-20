"""EXP-100: Best combo - Binary+Log+L40 focus+light dropout"""
import torch, torch.nn as nn
class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        # Weight L40 features (0,1,2) more heavily
        self.feature_weights = nn.Parameter(torch.tensor([1.5, 1.5, 1.5, 0.8, 0.8, 0.8, 0.8, 1.0]))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(input_dim * 4, 1)
    def forward(self, x):
        features = []
        for i in range(8):
            w = torch.sigmoid(self.feature_weights[i])
            feat = x[:, :, i] * w
            log_feat = torch.log1p(feat)
            features.extend([(feat > 0).float().max(dim=1)[0], (feat > 0).float().sum(dim=1),
                           log_feat.max(dim=1)[0], log_feat.mean(dim=1)])
        return self.classifier(self.dropout(torch.stack(features, dim=1)))
