import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import torch.nn as nn


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        # self.conv1 = GCNConv(1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv1 = nn.Linear(1, hidden_channels)
        # self.lin = Linear(hidden_channels, dataset.num_classes)
    def forward(self, x, edge_index, batch):
        # 1. 获得节点嵌入
        # x = self.conv1(x, edge_index)
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = self.conv3(x, edge_index)
        
        # 2. Readout layer
        # breakpoint()
        x = global_mean_pool(x, batch)   # [batch_size, hidden_channels]
        
        # 3. 分类器
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin(x)
        
        return x