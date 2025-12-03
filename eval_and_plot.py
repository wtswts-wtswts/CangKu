#!/usr/bin/env python3
"""
评估三种策略（learned / short-sighted / random）并绘图：
- 三张库存轨迹图： result/inventory_learned.png, result/inventory_myopic.png, result/inventory_random.png
- 一张成本比较图（对数 y 轴）： result/costs_log.png

用法示例:
  python eval_and_plot.py --theta_file result/theta_iters.npy --episodes 10 --horizon 300

可调参数: episodes, horizon, mc_samples, num_candidates (myopic), seed
"""
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from env import CuttingStockEnv
from cross_entropy import cross_entropy_greedy
from short_sighted import select_myopic_action
from random_policy import select_random_action

def load_env(config_path):
    cfg = json.load(open(config_path))
    env = CuttingStockEnv(cfg['patterns'], cfg['trim_losses'], cfg['item_lengths'],
                          cfg['demand_prob'], dmin=cfg['dmin'], dmax=cfg['dmax'],
                          s_max=cfg['s_max'], x_max=cfg['x_max'],
                          h_plus_factor=cfg['h_plus_factor'],
                          h_minus_factor=cfg['h_minus_factor'],
                          g_factor=cfg['g_factor'], seed=12345)
    return env, cfg

def run_policy_episode(env, policy_name, policy_obj, theta=None, phi_fn=None, basis_params=None,
                       horizon=300, rng=None, myopic_params=None):
    """
    返回:
      - inv_traj: array shape (horizon+1, m) 库存轨迹（包括初始 s0）
      - total_cost: scalar 总成本（累加）
    policy_obj is None for learned policy (we use cross_entropy with theta), or
    function for myopic or random that returns x.
    """
    if rng is None:
        rng = np.random.default_rng()

    m = env.m
    inv_traj = np.zeros((horizon+1, m), dtype=float)
    # init inventory as in paper (20 each) if possible
    s0 = np.full(env.m, 20, dtype=int)
    env.reset(s0)
    inv_traj[0] = env.s.copy()
    total_cost = 0.0

    for t in range(horizon):
        s_cur = env.s.copy()
        if policy_name == 'learned':
            # use cross_entropy_greedy with given theta to pick action
            basis_wrap = {"type": basis_params.get('type','fourier'), "params": basis_params.get('params', {})}
            x, _ = cross_entropy_greedy(env, s_cur, theta,
                                        phi_fn=phi_fn,
                                        basis_params=basis_wrap,
                                        N1=basis_params.get('N1_eval', 3),
                                        N2=basis_params.get('N2_eval', 50),
                                        rho=basis_params.get('rho', 0.1),
                                        x_max=env.x_max, rng=rng)
        elif policy_name == 'myopic':
            x = select_myopic_action(env, num_candidates=myopic_params.get('num_candidates',200),
                                     mc_samples=myopic_params.get('mc_samples',20),
                                     rng=rng, x_max=env.x_max)
        elif policy_name == 'random':
            x = select_random_action(env, rng=rng, x_max=env.x_max)
        else:
            raise ValueError("Unknown policy")
        s_next, cost, d = env.step(x)
        total_cost += cost
        inv_traj[t+1] = s_next.copy()
    return inv_traj, total_cost

def evaluate_policy(env_template, policy_name, policy_obj, theta=None, phi_fn=None, basis_params=None,
                    episodes=10, horizon=300, seed=0, myopic_params=None):
    rng_global = np.random.default_rng(seed)
    m = env_template.m
    all_inv = []  # will store average inventory trajectory (episodes x (horizon+1) x m)
    costs = []
    for ep in range(episodes):
        # create a fresh env copy for each episode to keep RNG and state isolated
        env = env_template  # env is stateful; env.reset will re-init internal RNG seed usage
        # Use different RNG for stochastic sampling within policies
        rng = np.random.default_rng(seed + ep)
        inv_traj, total_cost = run_policy_episode(env, policy_name, policy_obj,
                                                  theta=theta, phi_fn=phi_fn, basis_params=basis_params,
                                                  horizon=horizon, rng=rng, myopic_params=myopic_params)
        all_inv.append(inv_traj)
        costs.append(total_cost)
    all_inv = np.array(all_inv)  # episodes x (horizon+1) x m
    costs = np.array(costs)
    mean_inv = all_inv.mean(axis=0)  # (horizon+1) x m
    return mean_inv, costs

def plot_inventory(mean_inv, policy_label, out_path):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.figure(figsize=(10,4.5))
    timesteps = np.arange(mean_inv.shape[0])
    m = mean_inv.shape[1]
    cmap = plt.get_cmap('tab10')
    for i in range(m):
        plt.plot(timesteps, mean_inv[:,i], label=f'item{i+1}', color=cmap(i % 10))
    plt.xlabel('Time steps')
    plt.ylabel('Inventory level')
    plt.title(f'Inventory trajectories (example) - {policy_label}')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_costs_log(costs_dict, out_path):
    """
    costs_dict: {label: costs_array}
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.figure(figsize=(8,4.5))
    labels = list(costs_dict.keys())
    means = [np.mean(costs_dict[l]) for l in labels]
    stds = [np.std(costs_dict[l]) for l in labels]
    x = np.arange(len(labels))
    # plot mean with errorbars, but as log scale - show points with errorbars in linear scale then set yscale
    plt.errorbar(x, means, yerr=stds, fmt='o', capsize=5)
    plt.yscale('log')
    plt.xticks(x, labels)
    plt.ylabel('Average cost (log scale)')
    plt.title('Policy average costs (log scale)')
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    for xi, mval in zip(x, means):
        plt.text(xi, mval, f'{mval:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='data/patterns.json')
    parser.add_argument('--theta_file', default='result/theta_iters.npy')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=300)
    parser.add_argument('--outdir', default='result')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--myopic_candidates', type=int, default=200)
    parser.add_argument('--myopic_mc', type=int, default=20)
    args = parser.parse_args()

    env_template, cfg = load_env(args.config)

    # prepare learned policy theta (if available)
    theta = None
    phi_fn = None
    basis_params = {}
    if os.path.exists(args.theta_file):
        thetas = np.load(args.theta_file)
        theta = thetas[-1]
        # reuse a phi wrapper consistent with training (Fourier default)
        basis_params = {'type': 'fourier', 'params': {'freq_vectors': [[1]*env_template.m, [2]*env_template.m, [3]*env_template.m]},
                        'N1_eval': 3, 'N2_eval': 50, 'rho': 0.1}
        # phi_fn is APITrainer-compatible phi_from_sx: here we use env_template normalizer + features
        from api import APITrainer
        trainer_tmp = APITrainer(env_template, basis='fourier', basis_params=basis_params, reg=1e-3)
        phi_fn = trainer_tmp.phi_from_sx

    os.makedirs(args.outdir, exist_ok=True)

    # run policies
    results_costs = {}
    # 1) learned
    if theta is not None:
        print("Evaluating learned policy...")
        mean_inv_learned, costs_learned = evaluate_policy(env_template, 'learned', None,
                                                          theta=theta, phi_fn=phi_fn, basis_params=basis_params,
                                                          episodes=args.episodes, horizon=args.horizon, seed=args.seed,
                                                          myopic_params=None)
        plot_inventory(mean_inv_learned, 'learned', os.path.join(args.outdir, 'inventory_learned.png'))
        results_costs['learned'] = costs_learned
    else:
        print("No theta_file found, skipping learned policy")

    # 2) myopic
    print("Evaluating myopic (short-sighted) policy...")
    myopic_params = {'num_candidates': args.myopic_candidates, 'mc_samples': args.myopic_mc}
    mean_inv_myopic, costs_myopic = evaluate_policy(env_template, 'myopic', None,
                                                    theta=None, phi_fn=None, basis_params=None,
                                                    episodes=args.episodes, horizon=args.horizon, seed=args.seed+100,
                                                    myopic_params=myopic_params)
    plot_inventory(mean_inv_myopic, 'myopic', os.path.join(args.outdir, 'inventory_myopic.png'))
    results_costs['myopic'] = costs_myopic

    # 3) random
    print("Evaluating random policy...")
    mean_inv_random, costs_random = evaluate_policy(env_template, 'random', None,
                                                    theta=None, phi_fn=None, basis_params=None,
                                                    episodes=args.episodes, horizon=args.horizon, seed=args.seed+200,
                                                    myopic_params=None)
    plot_inventory(mean_inv_random, 'random', os.path.join(args.outdir, 'inventory_random.png'))
    results_costs['random'] = costs_random

    # plot costs on log scale
    plot_costs_log(results_costs, os.path.join(args.outdir, 'costs_log.png'))
    # also save raw costs
    np.save(os.path.join(args.outdir, 'policy_costs.npy'), results_costs)
    print("All done. Plots saved to", args.outdir)

if __name__ == '__main__':
    main()