import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(EdgeClassifier, self).__init__()

        # 极简 2层 MLP (无 BatchNorm，防止小样本坍塌)
        # Input -> Hidden -> ReLU -> Dropout -> Output

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, edge_feat):
        x = self.layer1(edge_feat)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x