"""
用训练好的Actor-Critic策略网络来生成新的列（路径）
"""
import torch
from model import ActorCritic
from torch_geometric.data import Data
from env import sample_path_from_logits, reduced_cost_for_path
import networkx as nx

def build_pyg_data(graph: nx.DiGraph, duals, device='cpu'):
    #将NetworkX图转换为PyTorch Geometric所需的Data格式
    N = graph.number_of_nodes()
    deg = [graph.out_degree(n) + graph.in_degree(n) for n in range(N)]
    x = torch.tensor([[deg[n], float(duals[n])] for n in range(N)], dtype=torch.float, device=device)
    edge_list = list(graph.edges())
    if len(edge_list) == 0:
        return None
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data

def propose_column_from_policy(policy: ActorCritic, graph: nx.DiGraph, duals, device='cpu', greedy=True):
    #使用策略网络生成候选列并评估其约简成本
    data = build_pyg_data(graph, duals, device=device)
    if data is None:
        return None
    policy.to(device)
    policy.eval()
    with torch.no_grad():
        logits, _ = policy(data)
    path = sample_path_from_logits(graph, logits.cpu(), start=0, target=graph.number_of_nodes()-1, greedy=greedy)
    if path is None:
        return None
    rc = reduced_cost_for_path(graph, path, duals)
    if rc < -1e-6:
        # column vector: node coverage
        col_vector = [1 if i in path else 0 for i in range(graph.number_of_nodes())]
        return col_vector, rc
    return None