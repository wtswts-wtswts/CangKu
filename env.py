"""
子问题求解
"""
import torch
import random
import networkx as nx

def sample_path_from_logits(graph: nx.DiGraph, logits: torch.Tensor, start: int, target: int,
                            max_len: int = 50, greedy: bool = False):
    """
    根据节点logits从起点到终点采样一条路径
    """
    N = graph.number_of_nodes()
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    cur = start
    path = [cur]
    visited = set([cur])
    for _ in range(max_len):
        if cur == target:
            return path
        neighbors = list(graph.successors(cur))
        if not neighbors:
            return None
        choices = neighbors
        # if not greedy, sample according to logits
        if greedy:
            nxt = max(choices, key=lambda n: float(probs[n]))
        else:
            weights = [float(probs[n]) for n in choices]
            total = sum(weights)
            if total <= 0:
                # fallback uniform
                nxt = random.choice(choices)
            else:
                r = random.random() * total
                cum = 0.0
                nxt = choices[-1]
                for n, w in zip(choices, weights):
                    cum += w
                    if r <= cum:
                        nxt = n
                        break
        path.append(nxt)
        cur = nxt
        visited.add(cur)
    return path if path[-1] == target else None

def path_cost(graph: nx.DiGraph, path):
    #计算给定路径的总成本
    if path is None or len(path) < 2:
        return float('inf')
    cost = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if graph.has_edge(u, v):
            cost += graph[u][v].get('weight', 1.0)
        else:
            return float('inf')
    return cost

def reduced_cost_for_path(graph: nx.DiGraph, path, duals):
    #约简成本
    """
    Default reduced cost: cost(path) - sum_j dual[j] * coverage_j
    coverage_j = 1 if node j in path else 0
    """
    c = path_cost(graph, path)
    if c == float('inf'):
        return float('inf')
    coverage_sum = 0.0
    for n in set(path):
        coverage_sum += float(duals[n])
    return c - coverage_sum