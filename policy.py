# -*- coding: utf-8 -*-
"""
GNN 策略（完全对齐旧工程的 Actor；保留门控）
- Actor：三层双向卷积 + between_gcn（item/column）+ 标量 ScaleNorm PostNorm
  * 键名与旧权重一致：feature_module_*.0.*, output_module.0/1.*, post_conv_module.0.scale
  * 旧权重中的 conv_*_3 与 between_gcn 将不再是 unexpected
- Critic：沿用两层双向卷积（与你旧权重已完全匹配）
- 保留定价门控头 item_gate_head（旧权重无此头，会 missing；用新初始化）
"""

import torch
import torch_geometric

EMB_SIZE = 128
ITEM_NFEATS = 2
EDGE_NFEATS = 1
COLUMN_NFEATS = 4
ACTIVATION_FUNCTION = torch.nn.LeakyReLU()

class ScaleNorm(torch.nn.Module):
    """标量版 ScaleNorm（与旧权重 post_conv_module.0.scale 形状一致：shape=[1]）"""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(1))  # 标量

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    二部图卷积（列-物品）带 PostNorm（ScaleNorm）
    键名与旧权重对齐：
      - feature_module_left/edge/right/final 采用 Sequential 包一层 Linear，生成 .0.weight/.0.bias
      - output_module 两层线性：.0, .1
      - post_conv_module.0.scale（标量）
    """
    def __init__(self):
        super().__init__('mean')
        emb = EMB_SIZE
        self.feature_module_left = torch.nn.Sequential(torch.nn.Linear(emb, emb))
        self.feature_module_edge = torch.nn.Sequential(torch.nn.Linear(emb, emb, bias=True))
        self.feature_module_right = torch.nn.Sequential(torch.nn.Linear(emb, emb, bias=True))
        self.feature_module_final = torch.nn.Sequential(torch.nn.Linear(emb, emb))
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb, emb),
            torch.nn.Linear(emb, emb)
        )
        self.post_conv_module = torch.nn.Sequential(ScaleNorm(emb))

    def forward(self, left_features, edge_indices, edge_features, right_features):
        out = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                             node_features=(left_features, right_features), edge_features=edge_features)
        y = self.output_module(torch.cat([out, right_features], dim=-1))
        y = self.post_conv_module(y)
        return ACTIVATION_FUNCTION(y)

    def message(self, node_features_i, node_features_j, edge_features):
        left = self.feature_module_left(node_features_i)
        edge = torch.sigmoid(self.feature_module_edge(edge_features))
        right = self.feature_module_right(node_features_j)
        return self.feature_module_final(left + edge * right)

class Actor(torch.nn.Module):
    """
    三层双向卷积的 Actor（完全对齐旧工程）：
      conv_column_to_item_1 → conv_item_to_column_1
      [between_gcn]
      conv_column_to_item_2 → conv_item_to_column_2
      [between_gcn]
      conv_column_to_item_3 → conv_item_to_column_3
    输出每列打分（logits），形状 [N_cols]
    """
    def __init__(self):
        super().__init__()
        emb = EMB_SIZE
        # 嵌入层
        self.item_embedding = torch.nn.Sequential(
            torch.nn.Linear(ITEM_NFEATS, emb),
            torch.nn.Linear(emb, emb),
            ACTIVATION_FUNCTION
        )
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.Linear(EDGE_NFEATS, emb),
            torch.nn.Linear(emb, emb),
            ACTIVATION_FUNCTION
        )
        self.column_embedding = torch.nn.Sequential(
            torch.nn.Linear(COLUMN_NFEATS, emb),
            torch.nn.Linear(emb, emb),
            ACTIVATION_FUNCTION
        )
        # 三层双向卷积
        self.conv_item_to_column_1 = BipartiteGraphConvolution()
        self.conv_column_to_item_1 = BipartiteGraphConvolution()
        self.conv_item_to_column_2 = BipartiteGraphConvolution()
        self.conv_column_to_item_2 = BipartiteGraphConvolution()
        self.conv_item_to_column_3 = BipartiteGraphConvolution()
        self.conv_column_to_item_3 = BipartiteGraphConvolution()
        # between_gcn（与旧权重键名一致：column_between_gcn.0.*, item_between_gcn.0.*）
        self.column_between_gcn = torch.nn.Sequential(torch.nn.Linear(emb, emb))
        self.item_between_gcn = torch.nn.Sequential(torch.nn.Linear(emb, emb))
        # 输出头（两层线性）
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb, emb),
            torch.nn.Linear(emb, 1, bias=True)
        )
        # 定价门控头（物品维度）
        self.item_gate_head = torch.nn.Sequential(
            torch.nn.Linear(emb, emb),
            ACTIVATION_FUNCTION,
            torch.nn.Linear(emb, 1),
            torch.nn.Sigmoid()
        )
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, item_features, edge_indices, edge_features, column_features):
        rev = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        # 嵌入
        item_features = self.item_embedding(item_features)
        edge_features = self.edge_embedding(edge_features)
        column_features = self.column_embedding(column_features)
        # 层 1
        item_features = self.conv_column_to_item_1(column_features, rev, edge_features, item_features)
        column_features = self.conv_item_to_column_1(item_features, edge_indices, edge_features, column_features)
        # between
        item_features = self.item_between_gcn(item_features)
        column_features = self.column_between_gcn(column_features)
        # 层 2
        item_features = self.conv_column_to_item_2(column_features, rev, edge_features, item_features)
        column_features = self.conv_item_to_column_2(item_features, edge_indices, edge_features, column_features)
        # between
        item_features = self.item_between_gcn(item_features)
        column_features = self.column_between_gcn(column_features)
        # 层 3
        item_features = self.conv_column_to_item_3(column_features, rev, edge_features, item_features)
        column_features = self.conv_item_to_column_3(item_features, edge_indices, edge_features, column_features)
        # 输出列打分
        return self.output_module(column_features).squeeze(-1)

    @torch.no_grad()
    def pricing_mask(self, item_features: torch.Tensor) -> torch.Tensor:
        x = self.item_embedding(item_features)
        return self.item_gate_head(x).squeeze(-1)

class Critic(torch.nn.Module):
    """
    两层双向卷积的 Critic（保持与旧权重完全匹配）
    """
    def __init__(self):
        super().__init__()
        emb = EMB_SIZE
        self.item_embedding = torch.nn.Sequential(
            torch.nn.Linear(ITEM_NFEATS, emb),
            torch.nn.Linear(emb, emb),
            ACTIVATION_FUNCTION
        )
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.Linear(EDGE_NFEATS, emb),
            torch.nn.Linear(emb, emb),
            ACTIVATION_FUNCTION
        )
        self.column_embedding = torch.nn.Sequential(
            torch.nn.Linear(COLUMN_NFEATS, emb),
            torch.nn.Linear(emb, emb),
            ACTIVATION_FUNCTION
        )
        self.conv_item_to_column_1 = BipartiteGraphConvolution()
        self.conv_column_to_item_1 = BipartiteGraphConvolution()
        self.conv_item_to_column_2 = BipartiteGraphConvolution()
        self.conv_column_to_item_2 = BipartiteGraphConvolution()
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb, emb),
            ACTIVATION_FUNCTION,
            torch.nn.Linear(emb, 1, bias=True)
        )
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, item_features, edge_indices, edge_features, column_features):
        rev = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        item_features = self.item_embedding(item_features)
        edge_features = self.edge_embedding(edge_features)
        column_features = self.column_embedding(column_features)
        column_features = self.conv_item_to_column_1(item_features, edge_indices, edge_features, column_features)
        item_features = self.conv_column_to_item_1(column_features, rev, edge_features, item_features)
        column_features = self.conv_item_to_column_2(item_features, edge_indices, edge_features, column_features)
        item_features = self.conv_column_to_item_2(column_features, rev, edge_features, item_features)
        return self.output_module(item_features).squeeze(-1)

class Policy:
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)

    @torch.no_grad()
    def pricing_mask(self, item_features: torch.Tensor) -> torch.Tensor:
        return self.actor.pricing_mask(item_features)

    @torch.no_grad()
    def action_logits(self, item_features, edge_indices, edge_features, column_features) -> torch.Tensor:
        return self.actor(item_features, edge_indices, edge_features, column_features)

    @torch.no_grad()
    def choose_action(self, logits: torch.Tensor, mode: str = 'eval') -> int:
        probs = torch.softmax(logits, dim=0)
        if mode == 'train':
            return torch.multinomial(probs, num_samples=1).item()
        return torch.argmax(probs).item()