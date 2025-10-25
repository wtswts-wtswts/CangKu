"""
用训练好的Actor-Critic策略网络来生成新的列（路径）
"""
import torch
from model import ActorCritic
from torch_geometric.data import Data
from env import sample_path_from_logits, reduced_cost_for_path, path_cost
import networkx as nx

def build_pyg_data(graph: nx.DiGraph, duals, device='cpu'):
    """
    将NetworkX图转换为PyTorch Geometric所需的Data格式，并将节点重标号为0..N-1以保证索引一致性。
    返回 (data, relabeled_graph)
    duals: list or dict-like providing a dual value for each original node id
    """
    # Create a deterministic node ordering and a relabeled copy of the graph
    node_list = list(graph.nodes())
    N = len(node_list)
    if N == 0:
        return None, None
    idx_map = {orig: i for i, orig in enumerate(node_list)}
    relabeled = nx.relabel_nodes(graph, idx_map, copy=True)

    # degrees and duals aligned to relabeled indices
    deg = [relabeled.out_degree(i) + relabeled.in_degree(i) for i in range(N)]
    # duals may be list-like indexed by original node ids or dict-like
    dual_vals = []
    for orig in node_list:
        try:
            # try list/sequence access by original id
            dual_vals.append(float(duals[orig]))
        except Exception:
            # fallback: if duals is a list aligned already with relabeled indices
            try:
                dual_vals.append(float(duals[idx_map[orig]]))
            except Exception:
                dual_vals.append(0.0)

    x = torch.tensor([[deg[n], float(dual_vals[n])] for n in range(N)], dtype=torch.float, device=device)
    edge_list = list(relabeled.edges())
    if len(edge_list) == 0:
        return None, None
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data, relabeled

def propose_column_from_policy(policy: ActorCritic, graph: nx.DiGraph, duals, device='cpu', greedy=True):
    """
    使用策略网络生成候选列并评估其约简成本。

    返回 (col_vector, cost, reduced_cost) 或 None
    col_vector 是长度为 N 的节点覆盖向量（按重标号 0..N-1 的顺序）
    cost 是该列的原始成本（路径成本）
    reduced_cost = cost - sum_j dual[j] * coverage_j
    """
    data, relabeled = build_pyg_data(graph, duals, device=device)
    if data is None or relabeled is None:
        return None
    policy.to(device)
    policy.eval()
    with torch.no_grad():
        logits, _ = policy(data)
    # sample a path on the relabeled graph (nodes 0..N-1)
    path = sample_path_from_logits(relabeled, logits.cpu(), start=0, target=relabeled.number_of_nodes()-1, greedy=greedy)
    if path is None:
        return None
    # compute costs
    cost = path_cost(relabeled, path)
    rc = reduced_cost_for_path(relabeled, path, duals)
    if rc < -1e-6:
        # column vector: node coverage in relabeled index order
        col_vector = [1 if i in path else 0 for i in range(relabeled.number_of_nodes())]
        return col_vector, cost, rc
    return None
