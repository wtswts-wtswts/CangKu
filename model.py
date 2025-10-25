"""
二分图卷积网络/编码器
演员网络
评价网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.codegen.scipy_nodes import powm1
from torch_geometric.nn import GCNConv, global_mean_pool

'''编码器（二分图卷积算子）'''
class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers=2):
        """
        :param in_dim: 输入节点特征维度
        :param hid_dim: 隐藏层维度(超参数，描述节点的容量)
        :param num_layers: GNN层数
        hid_dim;num_layers
        """
        super().__init__()
        self.convs = nn.ModuleList()
        ##存储图卷积层的列表
        self.bns = nn.ModuleList()
        #存储批归一化层的列表
        self.convs.append(GCNConv(in_dim, hid_dim))
        #创建第一个图卷积层
        self.bns.append(nn.BatchNorm1d(hid_dim))
        #创建一维批归一化层
        for _ in range(num_layers - 1):
            #循环
            self.convs.append(GCNConv(hid_dim, hid_dim))
            self.bns.append(nn.BatchNorm1d(hid_dim))

    def forward(self, x, edge_index):
        """
        :param x: 节点特征矩阵
        :param edge_index: 图的边索引

        """
        for conv, bn in zip(self.convs, self.bns):
            #将卷积层列表self.convs和批归一化层列表self.bns配对(对应元素的组合)
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            #卷积 批归一化 激活函数
        return x  # node embeddings (N, hid_dim)

class ActorCritic(nn.Module):
    #演员、评价网络
    def __init__(self, node_feat_dim, hid_dim=128, num_layers=2):
        super().__init__()
        self.encoder = GNNEncoder(node_feat_dim, hid_dim, num_layers=num_layers)
        # 创建图编码器（二分图卷积网络）
        #演员网络
        self.policy_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim//2),
            nn.ReLU(),
            nn.Linear(hid_dim//2, 1)
        )
        #评价网络
        self.value_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim//2),
            nn.ReLU(),
            nn.Linear(hid_dim//2, 1)
        )

    def forward(self, data):
        # data.x: (N, node_feat_dim), data.edge_index
        node_emb = self.encoder(data.x, data.edge_index)  # (N, hid)
        logits = self.policy_mlp(node_emb).squeeze(-1)  # (N,)
        # value: graph-level
        # if batch not provided, assume single graph
        batch = getattr(data, 'batch', None)
        if batch is None:
            graph_emb = node_emb.mean(dim=0, keepdim=True)  # (1, hid)
        else:
            graph_emb = global_mean_pool(node_emb, batch)  # (B, hid)
        value = self.value_mlp(graph_emb).squeeze(-1)  # (B,) or (1,)
        return logits, value
