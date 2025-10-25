"""
Small synthetic dataset generator for quick tests.

Generates small DAGs (or simple graphs) with random weights, and synthetic duals for training/demo.
"""
import networkx as nx
import random
import numpy as np
import torch
import Column_Generation
import env
import model
import pricing_solver
import trainer

def generate_simple_line_graph(n=6):
    G = nx.DiGraph()
    for i in range(n-1):
        # add forward edge and some shortcuts
        w = random.uniform(1.0, 5.0)
        G.add_edge(i, i+1, weight=w)
        if i+2 < n:
            G.add_edge(i, i+2, weight=random.uniform(1.0, 6.0))
    return G

def generate_training_set(num_instances=50, n_nodes=6):
    graphs = []
    for _ in range(num_instances):
        G = generate_simple_line_graph(n=n_nodes)
        # synthetic duals: positive values encourage covering nodes
        duals = np.random.uniform(0.0, 5.0, size=(G.number_of_nodes(),)).tolist()
        graphs.append((G, duals))
    return graphs


# 测试策略网络 - 修正版本
policy = model.ActorCritic(node_feat_dim=2, hid_dim=128)
test_graphs = generate_training_set(num_instances=10)

print("开始测试策略网络...")
print("=" * 50)

for i, (G, duals) in enumerate(test_graphs):
    print(
        f"\n测试实例 {i + 1}: 图有 {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")

    # 构建图数据
    data = pricing_solver.build_pyg_data(G, duals)
    print(f"节点特征维度: {data.x.shape}, 边索引维度: {data.edge_index.shape}")

    # 前向传播
    logits, value = policy(data)

    # 输出结果
    print(
        f"Logits 形状: {logits.shape}, 值范围: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"Value 估计: {value.item():.3f}")

    # 验证输出合理性
    if torch.isnan(logits).any() or torch.isnan(value).any():
        print("❌ 检测到NaN值!")
    else:
        print("✅ 输出数值正常")

print("\n测试完成!")