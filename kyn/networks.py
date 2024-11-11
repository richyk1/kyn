import torch.nn as nn
import torch

from torch.nn import Linear, InstanceNorm1d, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GraphNorm, LayerNorm
from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.nn import global_max_pool


class GraphConvInstanceGlobalMaxSmall(nn.Module):
    def __init__(self, hidden_channels, in_channels=31, dropout_ratio=0.2):
        super(GraphConvInstanceGlobalMaxSmall, self).__init__()
        torch.manual_seed(12345)
        self.dropout = Dropout(p=dropout_ratio, inplace=True)
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.norm1 = InstanceNorm1d(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.norm2 = InstanceNorm1d(hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.norm3 = InstanceNorm1d(hidden_channels)
        self.lin1 = Linear(hidden_channels * 3, hidden_channels * 2)
        self.lin2 = Linear(hidden_channels * 2, hidden_channels // 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x1 = x.relu()
        x = self.conv2(x1, edge_index)
        x = self.norm2(x)
        x2 = x.relu()
        x = self.conv3(x2, edge_index)
        x = self.norm3(x)
        x3 = x.relu()

        # 2. Readout layer
        # Graph-level readout
        h1 = global_max_pool(x1, batch)
        h2 = global_max_pool(x2, batch)
        h3 = global_max_pool(x3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # 3. Apply a final classifier
        x = self.lin1(h)
        x = self.dropout(x)
        x = x.relu()
        x = self.lin2(x)
        x = F.normalize(x, p=2)

        return x


class GraphConvInstanceGlobalMaxSmallSoftMaxAggr(nn.Module):
    def __init__(self, hidden_channels, in_channels=31, dropout_ratio=0.2):
        super(GraphConvInstanceGlobalMaxSmallSoftMaxAggr, self).__init__()
        self.aggr = SoftmaxAggregation(learn=True)
        torch.manual_seed(12345)
        self.dropout = Dropout(p=dropout_ratio, inplace=True)
        self.conv1 = GraphConv(in_channels, hidden_channels, aggr=self.aggr)
        self.norm1 = InstanceNorm1d(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr=self.aggr)
        self.norm2 = InstanceNorm1d(hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels, aggr=self.aggr)
        self.norm3 = InstanceNorm1d(hidden_channels)
        self.lin1 = Linear(hidden_channels * 3, hidden_channels * 2)
        self.lin2 = Linear(hidden_channels * 2, hidden_channels // 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x1 = x.relu()
        x = self.conv2(x1, edge_index)
        x = self.norm2(x)
        x2 = x.relu()
        x = self.conv3(x2, edge_index)
        x = self.norm3(x)
        x3 = x.relu()

        # 2. Readout layer
        # Graph-level readout
        h1 = global_max_pool(x1, batch)
        h2 = global_max_pool(x2, batch)
        h3 = global_max_pool(x3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # 3. Apply a final classifier
        x = self.lin1(h)
        x = self.dropout(x)
        x = x.relu()
        x = self.lin2(x)
        x = F.normalize(x, p=2)

        return x


class GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge(nn.Module):
    def __init__(self, hidden_channels, in_channels=31, dropout_ratio=0.2):
        super(GraphConvInstanceGlobalMaxSmallSoftMaxAggrEdge, self).__init__()
        self.aggr = SoftmaxAggregation(learn=True)
        torch.manual_seed(12345)
        self.dropout = Dropout(p=dropout_ratio, inplace=True)
        self.conv1 = GraphConv(in_channels, hidden_channels, aggr=self.aggr)
        self.norm1 = InstanceNorm1d(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr=self.aggr)
        self.norm2 = InstanceNorm1d(hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels, aggr=self.aggr)
        self.norm3 = InstanceNorm1d(hidden_channels)
        self.lin1 = Linear(hidden_channels * 3, hidden_channels * 2)
        self.lin2 = Linear(hidden_channels * 2, hidden_channels // 2)

    def forward(self, x, edge_index, batch, edge_weight):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x = self.norm1(x)
        x1 = x.relu()
        x = self.conv2(x1, edge_index, edge_weight)
        x = self.norm2(x)
        x2 = x.relu()
        x = self.conv3(x2, edge_index, edge_weight)
        x = self.norm3(x)
        x3 = x.relu()

        # 2. Readout layer
        # Graph-level readout
        h1 = global_max_pool(x1, batch)
        h2 = global_max_pool(x2, batch)
        h3 = global_max_pool(x3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # 3. Apply a final classifier
        x = self.lin1(h)
        x = self.dropout(x)
        x = x.relu()
        x = self.lin2(x)
        x = F.normalize(x, p=2)

        return x


class GraphConvGraphNormGlobalMaxSmallSoftMaxAggrEdge(nn.Module):
    def __init__(self, hidden_channels, in_channels=31, dropout_ratio=0.2):
        super(GraphConvGraphNormGlobalMaxSmallSoftMaxAggrEdge, self).__init__()
        self.aggr = SoftmaxAggregation(learn=True)
        torch.manual_seed(12345)
        self.dropout = Dropout(p=dropout_ratio, inplace=True)
        self.conv1 = GraphConv(in_channels, hidden_channels, aggr=self.aggr)
        self.norm1 = GraphNorm(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr=self.aggr)
        self.norm2 = GraphNorm(hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels, aggr=self.aggr)
        self.norm3 = GraphNorm(hidden_channels)
        self.lin1 = Linear(hidden_channels * 3, hidden_channels * 2)
        self.lin2 = Linear(hidden_channels * 2, hidden_channels // 2)

    def forward(self, x, edge_index, batch, edge_weight):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x = self.norm1(x)
        x1 = x.relu()
        x = self.conv2(x1, edge_index, edge_weight)
        x = self.norm2(x)
        x2 = x.relu()
        x = self.conv3(x2, edge_index, edge_weight)
        x = self.norm3(x)
        x3 = x.relu()

        # 2. Readout layer
        # Graph-level readout
        h1 = global_max_pool(x1, batch)
        h2 = global_max_pool(x2, batch)
        h3 = global_max_pool(x3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # 3. Apply a final classifier
        x = self.lin1(h)
        x = self.dropout(x)
        x = x.relu()
        x = self.lin2(x)
        x = F.normalize(x, p=2)

        return x


class GraphConvLayerNormGlobalMaxSmallSoftMaxAggrEdge(nn.Module):
    def __init__(self, hidden_channels, in_channels=31, dropout_ratio=0.2):
        super(GraphConvLayerNormGlobalMaxSmallSoftMaxAggrEdge, self).__init__()
        self.aggr = SoftmaxAggregation(learn=True)
        torch.manual_seed(12345)
        self.dropout = Dropout(p=dropout_ratio, inplace=True)
        self.conv1 = GraphConv(in_channels, hidden_channels, aggr=self.aggr)
        self.norm1 = LayerNorm(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr=self.aggr)
        self.norm2 = LayerNorm(hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels, aggr=self.aggr)
        self.norm3 = LayerNorm(hidden_channels)
        self.lin1 = Linear(hidden_channels * 3, hidden_channels * 2)
        self.lin2 = Linear(hidden_channels * 2, hidden_channels // 2)

    def forward(self, x, edge_index, batch, edge_weight):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x = self.norm1(x)
        x1 = x.relu()
        x = self.conv2(x1, edge_index, edge_weight)
        x = self.norm2(x)
        x2 = x.relu()
        x = self.conv3(x2, edge_index, edge_weight)
        x = self.norm3(x)
        x3 = x.relu()

        # 2. Readout layer
        # Graph-level readout
        h1 = global_max_pool(x1, batch)
        h2 = global_max_pool(x2, batch)
        h3 = global_max_pool(x3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # 3. Apply a final classifier
        x = self.lin1(h)
        x = self.dropout(x)
        x = x.relu()
        x = self.lin2(x)
        x = F.normalize(x, p=2)

        return x
