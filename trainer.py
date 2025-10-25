"""
Actor-Critic trainer for the pricing policy.

- graphs: list of (graph, duals) training instances
- policy: ActorCritic model
- optimizer: torch optimizer
- Uses simple on-policy actor-critic (one-step) update per episode
"""
import torch
import torch.optim as optim
from torch_geometric.data import Data
from env import sample_path_from_logits, reduced_cost_for_path
import numpy as np

def build_pyg_data(graph, duals, device='cpu'):
    # node features: degree and dual
    N = graph.number_of_nodes()
    deg = [graph.out_degree(n) + graph.in_degree(n) for n in range(N)]
    x = torch.tensor([[deg[n], float(duals[n])] for n in range(N)], dtype=torch.float, device=device)
    edge_list = list(graph.edges())
    if len(edge_list) == 0:
        return None
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data

def train(policy, graphs, optimizer, epochs=500, device='cpu', gamma=0.99, log_interval=50):
    policy.to(device)
    for epoch in range(epochs):
        total_loss = 0.0
        total_reward = 0.0
        for graph, duals in graphs:
            data = build_pyg_data(graph, duals, device=device)
            if data is None:
                continue
            logits, value = policy(data)  # logits: (N,), value: scalar tensor or (1,)
            # sample path
            path = sample_path_from_logits(graph, logits.detach(), start=0, target=graph.number_of_nodes()-1, greedy=False)
            rc = reduced_cost_for_path(graph, path, duals)
            reward = -rc if rc != float('inf') else -100.0  # reward: want lower reduced cost -> higher reward
            total_reward += reward
            # compute log prob (approximate by nodes in path)
            probs = torch.softmax(logits, dim=0)
            if path is None:
                # heavy penalty, update value toward negative reward
                adv = torch.tensor(reward, device=device) - value
                actor_loss = -torch.tensor(-10.0, device=device) * adv.detach()
                critic_loss = adv.pow(2)
                loss = actor_loss + critic_loss
            else:
                logp = torch.log(torch.stack([probs[n] for n in path]) + 1e-9).sum()
                # advantage
                adv = torch.tensor(reward, device=device) - value
                actor_loss = -logp * adv.detach()
                critic_loss = adv.pow(2)
                loss = actor_loss + critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1}/{epochs}: avg loss {total_loss:.4f}, avg reward {total_reward:.4f}")
    return policy