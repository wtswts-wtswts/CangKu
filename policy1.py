# -*- coding: utf-8 -*-
"""
GNN 策略（用于 roll-out 选列）
关键修复：
- edge_index 的定义：row0=item节点索引，row1=column节点索引
- 因此：item->column 卷积用 edge_index 原样
        column->item 卷积用 edge_index.flip(0)
否则会把 column 索引当 item 索引，导致 PyG 报 index 越界（你现在遇到的错误）
"""

import torch
import torch_geometric

EMB_SIZE = 128
ITEM_NFEATS = 2
EDGE_NFEATS = 1
COLUMN_NFEATS = 4
ACTIVATION_FUNCTION = torch.nn.LeakyReLU()


class ScaleNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(1))  # 标量

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    二部图卷积：edge_index[0] 对应 left_features，edge_index[1] 对应 right_features
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
        out = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features
        )
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
    输出每列 logits，[n_cols]
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

        # 双向卷积层（方向很重要）
        self.conv_item_to_column_1 = BipartiteGraphConvolution()
        self.conv_column_to_item_1 = BipartiteGraphConvolution()
        self.conv_item_to_column_2 = BipartiteGraphConvolution()
        self.conv_column_to_item_2 = BipartiteGraphConvolution()
        self.conv_item_to_column_3 = BipartiteGraphConvolution()
        self.conv_column_to_item_3 = BipartiteGraphConvolution()

        # between（线性混合，保持你工程的简单风格）
        self.column_between_gcn = torch.nn.Sequential(torch.nn.Linear(emb, emb))
        self.item_between_gcn = torch.nn.Sequential(torch.nn.Linear(emb, emb))

        # 输出头
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb, emb),
            torch.nn.Linear(emb, 1, bias=True)
        )

        # gate head（保留，但你当前不启用门控）
        self.item_gate_head = torch.nn.Sequential(
            torch.nn.Linear(emb, emb),
            ACTIVATION_FUNCTION,
            torch.nn.Linear(emb, 1),
            torch.nn.Sigmoid()
        )

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def pricing_mask(self, item_features: torch.Tensor) -> torch.Tensor:
        item_emb = self.item_embedding(item_features)
        return self.item_gate_head(item_emb).squeeze(-1)  # [m]

    def forward(self, item_features, edge_indices, edge_features, column_features):
        """
        edge_indices: shape [2, E]
          - edge_indices[0] ∈ [0, m)  (item index)
          - edge_indices[1] ∈ [0, n_cols) (column index)
        """
        item_emb = self.item_embedding(item_features)
        edge_emb = self.edge_embedding(edge_features)
        col_emb = self.column_embedding(column_features)

        # layer 1 (item -> column) 用原 edge_index
        col_emb = self.conv_item_to_column_1(item_emb, edge_indices, edge_emb, col_emb)
        # layer 1 (column -> item) 用 flip
        item_emb = self.conv_column_to_item_1(col_emb, edge_indices.flip(0), edge_emb, item_emb)

        col_emb = self.column_between_gcn(col_emb)
        item_emb = self.item_between_gcn(item_emb)

        # layer 2
        col_emb = self.conv_item_to_column_2(item_emb, edge_indices, edge_emb, col_emb)
        item_emb = self.conv_column_to_item_2(col_emb, edge_indices.flip(0), edge_emb, item_emb)

        col_emb = self.column_between_gcn(col_emb)
        item_emb = self.item_between_gcn(item_emb)

        # layer 3
        col_emb = self.conv_item_to_column_3(item_emb, edge_indices, edge_emb, col_emb)
        item_emb = self.conv_column_to_item_3(col_emb, edge_indices.flip(0), edge_emb, item_emb)

        logits = self.output_module(col_emb).squeeze(-1)
        return logits


class Critic(torch.nn.Module):
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
            torch.nn.Linear(emb, 1, bias=True)
        )

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, item_features, edge_indices, edge_features, column_features):
        item_emb = self.item_embedding(item_features)
        edge_emb = self.edge_embedding(edge_features)
        col_emb = self.column_embedding(column_features)

        col_emb = self.conv_item_to_column_1(item_emb, edge_indices, edge_emb, col_emb)
        item_emb = self.conv_column_to_item_1(col_emb, edge_indices.flip(0), edge_emb, item_emb)

        col_emb = self.conv_item_to_column_2(item_emb, edge_indices, edge_emb, col_emb)
        item_emb = self.conv_column_to_item_2(col_emb, edge_indices.flip(0), edge_emb, item_emb)

        v = self.output_module(col_emb).mean(dim=0)
        return v.squeeze(-1)


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
        # 确保输入在同 device
        item_features = item_features.to(self.device)
        edge_indices = edge_indices.to(self.device)
        edge_features = edge_features.to(self.device)
        column_features = column_features.to(self.device)
        return self.actor(item_features, edge_indices, edge_features, column_features)

    @torch.no_grad()
    def choose_action(self, logits: torch.Tensor, mode: str = 'eval', return_probs: bool = False):
        probs = torch.softmax(logits, dim=0)
        if mode == 'train':
            a = torch.multinomial(probs, num_samples=1).item()
        else:
            a = torch.argmax(probs).item()
        return (a, probs) if return_probs else a


# ========== 兼容旧版本接口（关键） ==========
# 你的 rl_rollout_rc.py 里是：from policy import PolicyNet
# 这里提供别名，避免 ImportError
PolicyNet = Policy

__all__ = ["Policy", "PolicyNet", "Actor", "Critic"]
