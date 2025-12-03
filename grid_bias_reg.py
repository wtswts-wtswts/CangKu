#!/usr/bin/env python3
"""
Grid search over bias_scale and reg (ridge) using APITrainer (quick training).
Saves results under result/ as result_grid.json and per-run files.

Usage:
    python grid_bias_reg.py

Adjust grid, L1/L2/N1/N2 inside the script if needed.
"""
import json, os, subprocess, time
import numpy as np
from env import CuttingStockEnv
from api import APITrainer
from features import normalize_sx, phi_polynomial, phi_fourier

# ---- CONFIG ----
cfg_path = 'data/patterns.json'
results_dir = 'result'
os.makedirs(results_dir, exist_ok=True)

# grid to try
bias_scales = [1.0, 0.5, 0.1]     # try bias = 1.0 (original), 0.5, 0.1
regs = [1e-3, 1e-2]               # reg candidates
# training hyperparams (short runs for comparison)
L1 = 3
L2 = 2000
N1 = 10
N2 = 200
rho = 0.1
gamma = 0.8
# evaluation call (evaluate.py will be invoked as subprocess)
evaluate_cmd_template = "python evaluate.py --config data/patterns.json --theta_file {theta_file} --which -1"

# load env config
cfg = json.load(open(cfg_path))
env = CuttingStockEnv(cfg['patterns'], cfg['trim_losses'], cfg['item_lengths'],
                      cfg['demand_prob'], dmin=cfg['dmin'], dmax=cfg['dmax'],
                      s_max=cfg['s_max'], x_max=cfg['x_max'],
                      h_plus_factor=cfg['h_plus_factor'],
                      h_minus_factor=cfg['h_minus_factor'],
                      g_factor=cfg['g_factor'], seed=12345)

def make_custom_phi(env, basis, bias_scale):
    # returns a phi_fn(sx, **kw) that uses bias_scale
    if basis.startswith('poly'):
        def phi_fn(sx, **kw):
            sxn = np.array(sx, dtype=float) / float(env.s_max)
            feats = phi_polynomial(sxn, **kw) if callable(phi_polynomial) else phi_polynomial(sxn)
            # replace bias (first element) with scaled value
            feats = feats.copy()
            if feats.size>0:
                feats[0] = bias_scale
            return feats
        return phi_fn
    else:
        def phi_fn(sx, **kw):
            sxn = normalize_sx(sx, env.s_max)
            # phi_fourier returns [bias, cos(...)...]
            vals = phi_fourier(sxn, **kw) if callable(phi_fourier) else phi_fourier(sxn)
            vals = vals.copy()
            if vals.size>0:
                vals[0] = bias_scale
            return vals
        return phi_fn

grid_results = []
for bias in bias_scales:
    for reg in regs:
        run_name = f"bias{bias}_reg{reg:.0e}"
        print(f"\n=== RUN {run_name} ===")
        # create trainer
        basis = 'fourier'
        basis_params = {'params': {'freq_vectors': [[1]*env.m, [2]*env.m, [3]*env.m]}}
        trainer = APITrainer(env, basis=basis, basis_params=basis_params,
                             gamma=gamma, L1=L1, L2=L2, N1=N1, N2=N2, rho=rho,
                             reg=reg, rng=np.random.default_rng(1234))
        # override phi_from_sx with scaled-bias phi
        trainer.phi_from_sx = lambda sx, **kw: make_custom_phi(env, basis, bias)(sx, **basis_params.get('params', {}))

        t0 = time.time()
        thetas = trainer.train(verbose=True)
        dt = time.time() - t0
        theta_last = thetas[-1]
        theta_norm = float(np.linalg.norm(theta_last))
        # save theta
        theta_file = os.path.join(results_dir, f"theta_{run_name}.npy")
        np.save(theta_file, np.array(thetas))
        # run evaluate.py as subprocess to get quick eval
        eval_cmd = evaluate_cmd_template.format(theta_file=theta_file)
        print("Running eval:", eval_cmd)
        proc = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True)
        eval_out = proc.stdout + proc.stderr
        print(eval_out)
        # parse average cost from eval output if possible
        avg_cost = None
        for line in eval_out.splitlines():
            if 'Average cost' in line:
                try:
                    avg_cost = float(line.split(':')[-1].strip())
                except:
                    pass
        # record
        rec = {'run': run_name, 'bias': bias, 'reg': reg, 'theta_norm': theta_norm, 'time_s': dt, 'avg_cost': avg_cost, 'eval_out': eval_out}
        grid_results.append(rec)
        # write intermediate json
        with open(os.path.join(results_dir, 'grid_results.json'), 'w') as f:
            json.dump(grid_results, f, indent=2)
        print(f"Run {run_name} done: theta_norm={theta_norm:.2f}, avg_cost={avg_cost}")
print("All runs done. Results saved to", os.path.join(results_dir, 'grid_results.json'))