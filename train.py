import argparse
import json
import numpy as np
from env import CuttingStockEnv
from api import APITrainer
import os

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='data/patterns.json')
    parser.add_argument('--L1', type=int, default=30)
    parser.add_argument('--L2', type=int, default=50000)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--basis', type=str, default='fourier')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = CuttingStockEnv(cfg['patterns'], cfg['trim_losses'], cfg['item_lengths'],
                          cfg['demand_prob'], dmin=cfg['dmin'], dmax=cfg['dmax'],
                          s_max=cfg['s_max'], x_max=cfg['x_max'],
                          h_plus_factor=cfg['h_plus_factor'],
                          h_minus_factor=cfg['h_minus_factor'],
                          g_factor=cfg['g_factor'], seed=args.seed)

    if args.basis == 'polynomial':
        basis_params = {'params': {'degree': 2}}
    else:
        m = env.m
        freqs = []
        for c1 in range(1,4):
            vec = [c1]*m
            freqs.append(vec)
        basis_params = {'params': {'freq_vectors': freqs}}

    trainer = APITrainer(env, basis=args.basis, basis_params=basis_params,
                         gamma=args.gamma, L1=args.L1, L2=args.L2,
                         N1=5, N2=50, rho=0.1, rng=np.random.default_rng(args.seed))
    theta_iters = trainer.train(verbose=True)
    os.makedirs('result', exist_ok=True)
    np.save(os.path.join('result', 'theta_iters.npy'), np.array(theta_iters))
    print("Training finished. Saved result/theta_iters.npy")

if __name__ == '__main__':
    main()