#!/usr/bin/env python3
"""
绘制训练曲线（Policy iterations vs Average cost, log y-axis）
支持多条曲线（不同基函数或不同训练实验）。
依赖项目内 src 模块（env, features, cross_entropy）。

示例：
python plot_training_curve.py --config data/patterns.json \
    --theta_files result/theta_iters_fourier.npy result/theta_iters_poly.npy \
    --labels Fourier Polynomial \
    --basis_list fourier polynomial \
    --episodes 10 --horizon 300
"""
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from env import CuttingStockEnv
from cross_entropy import cross_entropy_greedy
from features import phi_polynomial, phi_fourier, normalize_sx

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def make_phi_fn(env, basis, basis_params):
    """
    返回一个 phi_fn(sx, **params) 接口，供 cross_entropy_greedy 使用。
    这里约定传入 sx 为 'post-decision state'（未归一化）。
    """
    if basis.lower().startswith('poly'):
        degree = basis_params.get('degree', 2)
        def phi_fn(sx, **kwargs):
            # 建议对 polynomial 基进行归一化（避免尺度问题）
            sxn = np.array(sx, dtype=float) / float(env.s_max)
            return phi_polynomial(sxn, degree=degree)
        return phi_fn
    else:
        # Fourier expects normalized sx
        freq_vectors = basis_params.get('freq_vectors', None)
        if freq_vectors is None:
            # 默认频率集（示例），可被用户通过 basis_params 覆盖
            freq_vectors = [[1]*env.m, [2]*env.m, [3]*env.m]
        def phi_fn(sx, **kwargs):
            sxn = normalize_sx(np.array(sx, dtype=float), env.s_max)
            return phi_fourier(sxn, freq_vectors)
        return phi_fn

def eval_theta_sequence(env, theta_array, basis, basis_params,
                        episodes=10, horizon=300, ce_N1=3, ce_N2=50, rho=0.1, seed_base=0):
    """
    对 theta_array（shape L x K）中每一行 theta 进行评估：
    - 在 episodes 次仿真复制上运行 horizon 步长度的样本路径
    - 每步用 cross_entropy_greedy 找近似贪婪动作（评估时用较小的 CE 参数以节省时间）
    - 返回 per-iteration 的平均 cost（平均每步 cost）以及 95% CI 下界和上界
    """
    L = theta_array.shape[0]
    mean_costs = []
    lower_ci = []
    upper_ci = []

    phi_fn = make_phi_fn(env, basis, basis_params)

    for idx in range(L):
        theta = theta_array[idx]
        ep_costs = []
        for ep in range(episodes):
            # 每个 episode 重新设种子以保证可重复性
            rng = np.random.default_rng(seed_base + ep + idx*1000)
            # reset env with initial inventory = 20 for each item (as paper)
            s0 = np.full(env.m, 20, dtype=int)
            env.reset(s0)
            total_cost = 0.0
            for t in range(horizon):
                # 在评估阶段可以使用较小的 CE 参数以加速（但保证稳定）
                basis_wrap = {"type": basis, "params": basis_params}
                x, _ = cross_entropy_greedy(env, env.s.copy(), theta,
                                            phi_fn=phi_fn,
                                            basis_params=basis_wrap,
                                            N1=ce_N1, N2=ce_N2, rho=rho,
                                            x_max=env.x_max, rng=rng)
                # step
                s_next, cost, d = env.step(x)
                total_cost += cost
            # 记录平均每步 cost（与论文表达一致）
            ep_costs.append(total_cost / float(horizon))
        ep_costs = np.array(ep_costs)
        # mean and bootstrap 95% CI (percentile)
        mean_val = ep_costs.mean()
        low = np.percentile(ep_costs, 2.5)
        high = np.percentile(ep_costs, 97.5)
        mean_costs.append(mean_val)
        lower_ci.append(low)
        upper_ci.append(high)
        print(f"iter {idx+1}/{L}: mean={mean_val:.4f}, 95%CI=({low:.4f},{high:.4f}), samples={ep_costs}")
    return np.array(mean_costs), np.array(lower_ci), np.array(upper_ci)

def plot_curves(results_list, labels, out_path='result/training_curve.png'):
    """
    results_list: list of tuples (mean_costs, lower_ci, upper_ci)
    labels: list of labels
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.figure(figsize=(8,4.5))
    for (mean_costs, low, high), label in zip(results_list, labels):
        x = np.arange(1, len(mean_costs)+1)
        plt.plot(x, mean_costs, marker='o', linewidth=1.5, label=label)
        plt.fill_between(x, low, high, alpha=0.15)
    plt.yscale('log')
    plt.xlabel('Policy iterations')
    plt.ylabel('Average cost')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='data/patterns.json')
    parser.add_argument('--theta_files', nargs='+', required=True,
                        help='one or more npy files containing theta_iters (shape L x K)')
    parser.add_argument('--labels', nargs='+', default=None, help='labels for the curves')
    parser.add_argument('--basis_list', nargs='+', required=True,
                        help='basis type for each theta file (e.g. fourier polynomial)')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=300)
    parser.add_argument('--ce_N1', type=int, default=3)
    parser.add_argument('--ce_N2', type=int, default=50)
    parser.add_argument('--rho', type=float, default=0.1)
    parser.add_argument('--out', type=str, default='result/training_curve.png')
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = CuttingStockEnv(cfg['patterns'], cfg['trim_losses'], cfg['item_lengths'],
                          cfg['demand_prob'], dmin=cfg['dmin'], dmax=cfg['dmax'],
                          s_max=cfg['s_max'], x_max=cfg['x_max'],
                          h_plus_factor=cfg['h_plus_factor'],
                          h_minus_factor=cfg['h_minus_factor'],
                          g_factor=cfg['g_factor'], seed=12345)

    if args.labels:
        labels = args.labels
    else:
        # generate default labels from filenames
        labels = [os.path.splitext(os.path.basename(p))[0] for p in args.theta_files]

    results = []
    for theta_file, basis in zip(args.theta_files, args.basis_list):
        thetas = np.load(theta_file)
        # basis params: you may want to pass actual freq_vectors for Fourier or degree for poly
        if basis.lower().startswith('poly'):
            basis_params = {'degree': 2}
        else:
            # default freq vectors (example) - better to match training freq vectors
            basis_params = {'freq_vectors': [[1]*env.m, [2]*env.m, [3]*env.m]}
        print(f"Evaluating {theta_file} with basis={basis}, {thetas.shape[0]} iterations")
        mean_costs, low, high = eval_theta_sequence(env, thetas, basis, basis_params,
                                                    episodes=args.episodes, horizon=args.horizon,
                                                    ce_N1=args.ce_N1, ce_N2=args.ce_N2, rho=args.rho)
        results.append((mean_costs, low, high))

    plot_curves(results, labels, out_path=args.out)

if __name__ == '__main__':
    main()