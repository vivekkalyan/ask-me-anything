import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super(Classifier, self).__init__()
        self.drop1 = nn.Dropout(dropout)
        self.lin1 = nn.Linear(in_features, mid_features)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(mid_features, out_features)

    def forward(self, x):
        x = self.drop1(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.lin2(x)
        return x
