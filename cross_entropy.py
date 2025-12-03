#!/usr/bin/env python3
"""
cross_entropy.py

A reasonably simple Cross-Entropy sampler/optimizer adapted to this project.
Added support for `initial_candidates` input: a list of candidate actions (numpy arrays)
that will be injected into the candidate pool during each CE outer iteration so they
are considered and can influence the distribution updates.

This implementation is intentionally self-contained and robust:
 - It works without access to internal pattern definitions (reads env.n / env.x_max).
 - It keeps a distribution over total production xtotal (0..x_max) and a Dirichlet
   concentration vector for pattern probabilities (alpha). Sampling uses these two
   components to build multinomial actions.
 - initial_candidates (if provided) are evaluated alongside sampled candidates and
   count toward elite updates.

Note: This CE is a relatively lightweight implementation tuned to be compatible
with the project's CE usage (N1 outer iterations, N2 candidates per outer).
It's not a drop-in replacement for more sophisticated CE variants, but it
preserves the same external interface used across the codebase.
"""
import numpy as np

def _infer_n(env):
    # try common attribute names
    n = getattr(env, 'n', None)
    if n is None:
        n = getattr(env, 'num_patterns', None)
    if n is None:
        # last resort: try to infer from env.patterns if available
        patt = getattr(env, 'patterns', None)
        if patt is not None:
            n = len(patt)
    if n is None:
        raise RuntimeError("Unable to infer number of patterns 'n' from env")
    return int(n)

def _ensure_array(x):
    return np.array(x, dtype=int)

def cross_entropy_greedy(env, s0, theta,
                         phi_fn,
                         basis_params=None,
                         N1=10, N2=500, rho=0.1,
                         x_max=None, rng=None,
                         initial_candidates=None):
    """
    Cross-entropy greedy candidate search.

    Args:
      env: environment (must support is_feasible(x), post_decision_state(s,x))
      s0: current inventory state (array-like)
      theta: parameter vector (for scoring q = theta^T phi)
      phi_fn: function to compute phi from post-decision sx (phi = phi_fn(sx))
      basis_params: unused here but kept in signature for compatibility
      N1: number of outer CE iterations
      N2: number of sampled candidates per outer iteration
      rho: elite fraction (0 < rho <= 1)
      x_max: maximum total production (int). If None, tries env.x_max
      rng: numpy Generator. If None, a new default_rng() is used.
      initial_candidates: list of numpy arrays (each length n) to inject into candidate pool

    Returns:
      (best_x, info)
      best_x: chosen action (numpy int array)
      info: dict with diagnostic entries:
         - 'feasible_count', 'sampled_count', 'elites_found', 'initial_included'
    """
    if rng is None:
        rng = np.random.default_rng()

    n = _infer_n(env)
    if x_max is None:
        x_max = int(getattr(env, 'x_max', 10))

    # Normalize types
    s0 = _ensure_array(s0)
    theta = np.array(theta, dtype=float)

    # initialize distribution over xtotal (0..x_max)
    xtotal_probs = np.ones(x_max + 1, dtype=float)
    xtotal_probs /= xtotal_probs.sum()

    # initialize Dirichlet alpha for pattern probabilities (length n)
    alpha = np.ones(n, dtype=float)

    best_x_overall = np.zeros(n, dtype=int)
    best_q_overall = -np.inf

    info = {
        'feasible_count': 0,
        'sampled_count': 0,
        'initial_included': 0,
        'outer_stats': []
    }

    # ensure initial_candidates are arrays
    init_cands = []
    if initial_candidates:
        for ic in initial_candidates:
            try:
                xi = _ensure_array(ic)
                if xi.shape[0] != n:
                    continue
                init_cands.append(xi.copy())
            except Exception:
                continue

    # CE outer loop
    for outer in range(max(1, int(N1))):
        # sample N2 candidates from current distribution
        candidates = []
        cand_q = []
        sampled = 0
        feasible = 0

        # option: include initial candidates first so they get evaluated
        for xi in init_cands:
            try:
                # verify feasibility
                if hasattr(env, 'is_feasible'):
                    ok = env.is_feasible(xi)
                else:
                    ok = True
                if ok:
                    sx = env.post_decision_state(s0, xi)
                    phi = phi_fn(sx)
                    q = float(np.dot(theta, phi))
                    candidates.append(xi.copy())
                    cand_q.append(q)
                    feasible += 1
                    info['initial_included'] += 1
                    if q > best_q_overall:
                        best_q_overall = q
                        best_x_overall = xi.copy()
                else:
                    # even if infeasible, include with very low q to allow CE to consider constraints
                    candidates.append(xi.copy())
                    cand_q.append(-1e12)
            except Exception:
                # if post_decision or phi fails, mark very low
                candidates.append(xi.copy())
                cand_q.append(-1e12)

        # sample additional candidates to reach approximately N2 total sampled (we count inits)
        to_sample = max(0, int(N2) - len(init_cands))
        for _ in range(to_sample):
            sampled += 1
            # sample xtotal by xtotal_probs
            xt = int(rng.choice(a=np.arange(x_max + 1), p=xtotal_probs))
            if xt <= 0:
                x = np.zeros(n, dtype=int)
            else:
                # sample proportions from Dirichlet(alpha)
                p = rng.dirichlet(alpha)
                # convert to multinomial xt draws
                x = rng.multinomial(xt, p)
                x = x.astype(int)
            info['sampled_count'] = info.get('sampled_count', 0) + 1
            try:
                ok = env.is_feasible(x) if hasattr(env, 'is_feasible') else True
            except Exception:
                ok = True
            if not ok:
                # skip infeasible candidates (but still record them with very low q)
                candidates.append(x.copy())
                cand_q.append(-1e12)
            else:
                try:
                    sx = env.post_decision_state(s0, x)
                    phi = phi_fn(sx)
                    q = float(np.dot(theta, phi))
                except Exception:
                    q = -1e12
                candidates.append(x.copy())
                cand_q.append(q)
                feasible += 1
                if q > best_q_overall:
                    best_q_overall = q
                    best_x_overall = x.copy()

        info['feasible_count'] += feasible
        total_candidates = len(candidates)

        # if no feasible candidates, continue (rare)
        if feasible == 0:
            info['outer_stats'].append({'outer': outer, 'feasible': 0, 'elites': 0})
            continue

        # select elites
        q_arr = np.array(cand_q, dtype=float)
        # handle case where many are -inf: choose top by value
        m_elite = max(1, int(max(1, rho * feasible)))
        # get indices of feasible candidates only (q > -1e11)
        feasible_idxs = np.where(q_arr > -1e11)[0]
        if feasible_idxs.size == 0:
            elites_idx = np.argsort(q_arr)[-m_elite:]
        else:
            # sort feasible by q and take top m_elite among them
            feasible_q_idx_sorted = feasible_idxs[np.argsort(q_arr[feasible_idxs])]
            elites_idx = feasible_q_idx_sorted[-m_elite:]

        elites = [candidates[i] for i in elites_idx]
        elites_q = q_arr[elites_idx]

        # update xtotal_probs based on elites' xt counts
        xt_counts = np.zeros(x_max + 1, dtype=float)
        alpha_update = np.zeros(n, dtype=float)
        for e in elites:
            xt_e = int(e.sum())
            if xt_e < 0 or xt_e > x_max:
                continue
            xt_counts[xt_e] += 1.0
            alpha_update += e.astype(float)

        # smooth updates (add small epsilon to avoid zeros)
        eps = 1e-3
        xtotal_probs = (xtotal_probs + (xt_counts + eps))  # additive update
        if xtotal_probs.sum() > 0:
            xtotal_probs = xtotal_probs / xtotal_probs.sum()
        else:
            xtotal_probs = np.ones_like(xtotal_probs) / float(len(xtotal_probs))

        # update alpha (Dirichlet concentration) with elites
        alpha = alpha + (np.maximum(alpha_update, 0.0) / float(max(1.0, len(elites))))  # incremental

        info['outer_stats'].append({
            'outer': outer,
            'sampled': sampled,
            'feasible': int(feasible),
            'elites': int(len(elites)),
            'best_q': float(best_q_overall)
        })

    # Return best found action and diagnostics
    info['best_q'] = float(best_q_overall)
    info['best_x'] = best_x_overall.tolist()
    return best_x_overall, info