#!/usr/bin/env python3
"""
记录带启发式注入评估的逐步日志并保存为 CSV，用于分析高 cost 的时刻。
Usage:
  python tools/log_eval_with_heur.py --theta_file result/theta_iters.npy --episodes 10 --horizon 300 --N1 3 --N2 300 --heuristic_num 8 --heuristic_samples 120 --out result/eval_with_heur_steps.csv
"""
import argparse
import json
import numpy as np
import csv
import copy
import logging
import os
import time
import traceback
from env import CuttingStockEnv
from cross_entropy import cross_entropy_greedy
from heuristic_candidates import generate_heuristic_candidates
from api import APITrainer

def load_env(cfg_path='data/patterns.json', seed=12345):
    cfg = json.load(open(cfg_path))
    env = CuttingStockEnv(cfg['patterns'], cfg['trim_losses'], cfg['item_lengths'],
                          cfg['demand_prob'], dmin=cfg['dmin'], dmax=cfg['dmax'],
                          s_max=cfg['s_max'], x_max=cfg['x_max'],
                          h_plus_factor=cfg['h_plus_factor'],
                          h_minus_factor=cfg['h_minus_factor'],
                          g_factor=cfg['g_factor'], seed=seed)
    return env

def _ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theta_file', default='result/theta_iters.npy')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=300)
    parser.add_argument('--N1', type=int, default=3)
    parser.add_argument('--N2', type=int, default=300)
    parser.add_argument('--heuristic_num', type=int, default=8)
    parser.add_argument('--heuristic_samples', type=int, default=120)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--out', default='result/eval_with_heur_steps.csv')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    rng = np.random.default_rng(args.seed)
    env = load_env(seed=args.seed)

    thetas = np.load(args.theta_file)
    theta = thetas[-1].copy() if thetas.ndim > 1 else thetas.copy()

    # ensure phi function consistent with training
    freq_vectors = [[1]*env.m, [2]*env.m, [3]*env.m]
    trainer = APITrainer(env, basis='fourier', basis_params={'params': {'freq_vectors': freq_vectors}},
                         L1=1, L2=1, N1=args.N1, N2=args.N2, rng=rng)

    header = ['episode','step','chosen_origin','chosen_is_initial','chosen_x','q','immediate_cost','inv_before','inv_after']

    _ensure_dir_for_file(args.out)
    debug_dir = os.path.join(os.path.dirname(args.out) or '.', 'debug_bad_q')
    os.makedirs(debug_dir, exist_ok=True)

    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ep in range(args.episodes):
            s0 = rng.integers(0, env.s_max+1, size=env.m)
            env.reset(s0)
            for t in range(args.horizon):
                s_before = env.s.copy()
                heuristics = generate_heuristic_candidates(env, rng, num=args.heuristic_num, samples_per=args.heuristic_samples)
                x, info = cross_entropy_greedy(env, env.s, theta, phi_fn=trainer.phi_from_sx,
                                               N1=args.N1, N2=args.N2, rho=0.1, x_max=env.x_max, rng=rng,
                                               initial_candidates=heuristics)
                # detect origin
                chosen_is_initial = False
                for h in heuristics:
                    if tuple(x.tolist()) == tuple(h.tolist() if hasattr(h, 'tolist') else h):
                        chosen_is_initial = True
                        break
                # compute q for chosen using phi
                q = None
                phi = None
                try:
                    sx = env.post_decision_state(env.s, x)
                    phi = trainer.phi_from_sx(sx)
                    q = float(np.dot(theta, phi))
                except Exception:
                    logging.exception("Exception computing q at ep %s step %s", ep, t)
                    q = None
                # simulate immediate cost on a copy
                immediate_cost = None
                inv_after = None
                try:
                    env_copy = copy.deepcopy(env)
                    s_tmp = env_copy.s.copy()
                    s_next, cost, _ = env_copy.step(x)
                    immediate_cost = float(cost)
                    inv_after = s_next.tolist()
                except Exception:
                    logging.exception("Exception simulating immediate cost at ep %s step %s", ep, t)
                    immediate_cost = None
                    inv_after = None

                # Defensive debug dump when q is None, non-finite, or negative (you can change threshold)
                bad_q = False
                try:
                    if q is None:
                        bad_q = True
                        reason = "q is None"
                    elif not np.isfinite(q):
                        bad_q = True
                        reason = "q not finite"
                    elif q < 0:
                        bad_q = True
                        reason = "q negative"
                    else:
                        reason = ""
                except Exception:
                    bad_q = True
                    reason = "exception checking q"

                if bad_q:
                    ctx = {
                        "ts": time.time(),
                        "episode": int(ep),
                        "step": int(t),
                        "q": (None if q is None else float(q)),
                        "q_reason": reason,
                        "chosen_origin": "CE_with_init",
                        "chosen_is_initial": int(chosen_is_initial),
                        "chosen_x": list(x.tolist()),
                        "immediate_cost": (None if immediate_cost is None else float(immediate_cost)),
                        "inv_before": list(s_before.tolist()),
                        "inv_after": inv_after,
                        "phi": (None if phi is None else (phi.tolist() if hasattr(phi, "tolist") else list(phi))),
                        "theta": (theta.tolist() if hasattr(theta, "tolist") else list(theta)),
                        "stack": traceback.format_exc()
                    }
                    fname = os.path.join(debug_dir, f"bad_q_ep{ep}_step{t}_{int(time.time())}.json")
                    try:
                        with open(fname, "w") as df:
                            json.dump(ctx, df, indent=2)
                        logging.error("Bad q detected (reason=%s) at ep %s step %s, dumped to %s, q=%s", reason, ep, t, fname, q)
                    except Exception:
                        logging.exception("Failed to write debug dump for bad q at ep %s step %s", ep, t)

                # record
                writer.writerow([ep, t, 'CE_with_init', int(chosen_is_initial), list(x.tolist()), q, immediate_cost, list(s_before.tolist()), inv_after])

    print("Saved stepwise evaluation CSV to", args.out)

if __name__ == '__main__':
    main()