#!/usr/bin/env python3
"""
evaluate_long.py

Evaluate the learned policy (theta) for many episodes and long horizon.
Usage example:
  python evaluate_long.py --config data/patterns.json --theta_file result/theta_iters.npy --episodes 30 --horizon 1000 --N1_eval 3 --N2_eval 50

Outputs:
 - prints "Costs per episode" and "Average cost"
 - saves inventory_example.png for the last episode
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from env import CuttingStockEnv
from cross_entropy import cross_entropy_greedy
from api import APITrainer

def load_env(config_path):
    cfg = json.load(open(config_path))
    env = CuttingStockEnv(cfg['patterns'], cfg['trim_losses'], cfg['item_lengths'],
                          cfg['demand_prob'], dmin=cfg['dmin'], dmax=cfg['dmax'],
                          s_max=cfg['s_max'], x_max=cfg['x_max'],
                          h_plus_factor=cfg['h_plus_factor'],
                          h_minus_factor=cfg['h_minus_factor'],
                          g_factor=cfg['g_factor'], seed=None)
    return env

def run_evaluation(env, theta, phi_fn, basis_params, episodes, horizon, seed=123):
    rng = np.random.default_rng(seed)
    costs = []
    last_inv_traj = None
    for ep in range(episodes):
        s0 = np.full(env.m, 20, dtype=int)
        env.reset(s0)
        inv_traj = np.zeros((horizon+1, env.m), dtype=float)
        inv_traj[0] = env.s.copy()
        total_cost = 0.0
        for t in range(horizon):
            s_cur = env.s.copy()
            x, _ = cross_entropy_greedy(env, s_cur, theta,
                                        phi_fn=phi_fn,
                                        basis_params=basis_params,
                                        N1=basis_params.get('N1_eval', 3),
                                        N2=basis_params.get('N2_eval', 50),
                                        rho=basis_params.get('rho', 0.1),
                                        x_max=env.x_max, rng=rng)
            s_next, cost, d = env.step(x)
            total_cost += cost
            inv_traj[t+1] = s_next.copy()
        costs.append(total_cost)
        last_inv_traj = inv_traj
    return np.array(costs), last_inv_traj

def plot_inventory(inv_traj, out_path='result/inventory_example.png'):
    import os
    os.makedirs('result', exist_ok=True)
    timesteps = np.arange(inv_traj.shape[0])
    m = inv_traj.shape[1]
    cmap = plt.get_cmap('tab10')
    plt.figure(figsize=(10,4.5))
    for i in range(m):
        plt.plot(timesteps, inv_traj[:,i], label=f'item{i+1}', color=cmap(i % 10))
    plt.xlabel('Time steps')
    plt.ylabel('Inventory level')
    plt.title('Inventory trajectory (last episode)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='data/patterns.json')
    parser.add_argument('--theta_file', default='result/theta_iters.npy')
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--horizon', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--N1_eval', type=int, default=3)
    parser.add_argument('--N2_eval', type=int, default=50)
    parser.add_argument('--rho', type=float, default=0.1)
    args = parser.parse_args()

    env = load_env(args.config)

    # load theta
    thetas = np.load(args.theta_file, allow_pickle=True)
    theta = thetas[-1]

    # -------------------------------------------------------------------------
    # NOTE: replaced the original symmetric freq_vectors ([[1]*m, [2]*m, [3]*m])
    # with a richer set that includes one-hot-like vectors and mixed/scaled
    # vectors so phi depends on per-dimension inventory. Also increased reg.
    # -------------------------------------------------------------------------
    # generate richer, non-symmetric freq_vectors (one-hot + mixed)
    freq_vectors = []
    # one-hot like vectors for per-item sensitivity
    for i in range(env.m):
        v = [0] * env.m
        v[i] = 1
        freq_vectors.append(v)
    # add a few mixed / scaled vectors to capture interactions and scales
    for k in range(1, min(5, env.m)):
        freq_vectors.append([ (j + 1) * (k + 1) for j in range(env.m) ])

    # instantiate trainer with stronger regularization for stability (tune as needed)
    trainer_tmp = APITrainer(env,
                             basis='fourier',
                             basis_params={'params': {'freq_vectors': freq_vectors}},
                             reg=1e-2)  # increased from 1e-3 to 1e-2

    phi_fn = trainer_tmp.phi_from_sx
    basis_params = {'type':'fourier', 'params':{'freq_vectors': freq_vectors}, 'N1_eval':args.N1_eval, 'N2_eval':args.N2_eval, 'rho':args.rho}

    print(f"Evaluating learned policy: episodes={args.episodes}, horizon={args.horizon}, N1_eval={args.N1_eval}, N2_eval={args.N2_eval}")
    costs, last_inv = run_evaluation(env, theta, phi_fn, basis_params, args.episodes, args.horizon, seed=args.seed)
    print("Costs per episode:", costs)
    print("Average cost:", float(np.mean(costs)))
    plot_inventory(last_inv, out_path='result/inventory_example.png')
    print("Saved inventory_example.png")

if __name__ == '__main__':
    main()