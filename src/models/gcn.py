import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from src.config import SEED


class GCNModel(nn.Module):
    def __init__(self, in_features, hidden_size=100):
        super(GCNModel, self).__init__()

        torch.manual_seed(SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = GCNConv(in_features, hidden_size)

        self.conv2 = GCNConv(hidden_size, hidden_size)

        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index):
        # x is the feature matrix (nodes x features)
        # edge_index is the graph structure (2 x edges)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        embeddings = self.conv2(x, edge_index)
        x = F.relu(embeddings)

        out = self.classifier(x)

        return out, embeddings


def build_gcn(in_features=166, hidden_size=100):
    model = GCNModel(in_features=in_features, hidden_size=hidden_size)
    model.to(model.device)

    return model