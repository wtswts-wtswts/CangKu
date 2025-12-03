#!/usr/bin/env python3
"""
Regenerate suspicious_q_rows using current theta and phi_fn, writing q_computed into CSV.

Usage:
  python regenerate_suspicious.py suspicious_q_rows.csv result/theta_iters.npy suspicious_q_rows_generated.csv

This script:
 - reads the input CSV to get episode/step/chosen_x/inv_before/inv_after rows,
 - recomputes phi (using APITrainer with current freq_vectors in code) and q = theta^T phi,
 - writes a new CSV with original columns plus q_computed column (and phi_len).
"""
import sys
import numpy as np
import pandas as pd
from api import APITrainer
from evaluate import load_env

def parse_list(s):
    if pd.isna(s):
        return None
    try:
        return np.array([int(x.strip()) for x in s.strip("[]").split(",")], dtype=int)
    except Exception:
        return None

def load_theta(theta_file):
    arr = np.load(theta_file, allow_pickle=True)
    try:
        last = arr[-1]
    except Exception:
        last = np.asarray(arr)
    return np.asarray(last, dtype=float).ravel()

def main():
    if len(sys.argv) < 4:
        print("Usage: python regenerate_suspicious.py suspicious_q_rows.csv result/theta_iters.npy suspicious_q_rows_generated.csv")
        return
    src_csv = sys.argv[1]
    theta_file = sys.argv[2]
    out_csv = sys.argv[3]

    df = pd.read_csv(src_csv, dtype=str)
    env = load_env('data/patterns.json')

    # Use the same richer freq_vectors we used for training
    freq_vectors = []
    for i in range(env.m):
        v = [0]*env.m
        v[i] = 1
        freq_vectors.append(v)
    for k in range(1, min(5, env.m)):
        freq_vectors.append([ (j + 1) * (k + 1) for j in range(env.m) ])

    trainer = APITrainer(env, basis='fourier', basis_params={'params':{'freq_vectors':freq_vectors}}, reg=1e-2)
    phi_fn = trainer.phi_from_sx
    theta = load_theta(theta_file)

    out_rows = []
    for idx, row in df.iterrows():
        sx_str = row.get('inv_after') or row.get('inv_before') or row.get('chosen_x')
        sx = parse_list(sx_str) if sx_str is not None else None
        if sx is None:
            q_comp = None
            phi_len = None
        else:
            phi = np.asarray(phi_fn(sx), dtype=float).ravel()
            phi_len = phi.size
            if phi_len != theta.size:
                # compute on overlapping slice but mark with NaN in full mismatch case
                q_comp = float(np.dot(theta[:min(phi_len, theta.size)], phi[:min(phi_len, theta.size)]))
            else:
                q_comp = float(np.dot(theta, phi))
        new_row = dict(row)
        new_row['q_computed'] = q_comp
        new_row['phi_len'] = phi_len
        out_rows.append(new_row)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote regenerated CSV to {out_csv} ({len(out_df)} rows)")

if __name__ == '__main__':
    main()