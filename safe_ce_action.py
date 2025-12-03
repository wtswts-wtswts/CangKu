#!/usr/bin/env python3
"""
safe_ce_action.py (robust version)

Wrapper that tries cross_entropy_greedy and falls back to a safe heuristic
if CE returns an action that would leave some monitored items below threshold.

This version avoids direct access to env.patterns and instead samples random
feasible actions and simulates them on a deepcopy(env) to choose the best
one for refill, so it works even if the env does not expose internal fields.
"""
import copy
import numpy as np
from cross_entropy import cross_entropy_greedy

def random_action_sample(env, rng, xtotal=None):
    """Sample a random multinomial action (not guaranteed feasible)."""
    n = getattr(env, 'n', None)
    if n is None:
        # fallback: try to infer n from env.action_space or env.x_max; assume env.x_max and env.n exist
        n = getattr(env, 'n', getattr(env, 'num_patterns', None))
        if n is None:
            raise RuntimeError("Environment does not expose number of patterns (n).")
    if xtotal is None:
        xtotal = int(rng.integers(0, getattr(env, 'x_max', 10) + 1))
    if xtotal == 0:
        return np.zeros(n, dtype=int)
    p = np.ones(n) / float(n)
    return rng.multinomial(int(xtotal), p)

def random_feasible(env, rng, max_attempts=500):
    """Try sampling random multinomial actions until a feasible one is found."""
    for _ in range(max_attempts):
        xtotal = int(rng.integers(0, getattr(env, 'x_max', 10) + 1))
        x = random_action_sample(env, rng, xtotal=xtotal)
        try:
            if env.is_feasible(x):
                return x
        except Exception:
            # if env.is_feasible not available or errors, accept first x
            return x
    # fallback: zero action
    n = getattr(env, 'n', getattr(env, 'num_patterns', None))
    if n is None:
        n = 1
    return np.zeros(n, dtype=int)

def heuristic_fill_low_inventory(env, rng, top_k=3, samples=300):
    """
    Heuristic fallback that samples candidate actions (random feasible) and
    simulates them on a deepcopy(env) to choose the action that increases
    inventory for the most urgent (low) items.

    - samples: number of candidate actions to sample/evaluate
    - top_k: number of most urgent items to prioritize
    """
    s_cur = env.s.copy()
    # compute urgency = demand_prob / (s + 1) if demand_prob available
    probs = np.array(getattr(env, 'demand_prob', [1.0]*len(s_cur)), dtype=float)
    urg = probs / (s_cur.astype(float) + 1.0)
    top_items = list(np.argsort(-urg)[:top_k])

    best_x = None
    best_score = -1.0

    for _ in range(samples):
        x_cand = random_feasible(env, rng, max_attempts=20)
        # simulate on a deepcopy of env to get resulting inventory after production (post-decision)
        try:
            env_copy = copy.deepcopy(env)
            # try to set its internal state to s_cur if reset accepts state
            try:
                env_copy.reset(s_cur)
            except Exception:
                try:
                    env_copy.s = s_cur.copy()
                except Exception:
                    pass
            # ensure feasibility before stepping
            if not getattr(env_copy, 'is_feasible', lambda a: True)(x_cand):
                continue
            s_next, _, _ = env_copy.step(x_cand)
        except Exception:
            # if simulation fails, skip candidate
            continue

        # score = sum increase in the top_items (s_next - s_cur), prefer larger increases
        incr = s_next[top_items].astype(float) - s_cur[top_items].astype(float)
        score = float(incr.sum())
        # as tie-breaker, prefer actions with larger total production (x.sum())
        score = score + 0.001 * float(x_cand.sum())

        if score > best_score:
            best_score = score
            best_x = x_cand.copy()

    if best_x is None:
        # fallback: try a guaranteed feasible random sample
        return random_feasible(env, rng, max_attempts=200)
    return best_x

def select_action_with_fallback(env, s_cur, theta, phi_fn, basis_params, rng,
                                N1_eval=3, N2_eval=50, rho=0.1,
                                random_attempts=200, feasible_threshold=0.3,
                                low_inventory_threshold=5, monitor_items=None):
    """
    Try CE first; if returned x is infeasible or current inventory has
    monitored items below low_inventory_threshold, use heuristic fallback.

    Returns: (x, info)
    """
    info = {'fallback': None}
    try:
        x, _ = cross_entropy_greedy(env, s_cur, theta,
                                    phi_fn=phi_fn,
                                    basis_params=basis_params,
                                    N1=N1_eval, N2=N2_eval, rho=rho,
                                    x_max=getattr(env, 'x_max', None), rng=rng)
    except Exception:
        x = None

    # If CE returned nothing or infeasible, try random feasible attempts
    if x is None:
        for _ in range(random_attempts):
            xt = random_feasible(env, rng, max_attempts=10)
            try:
                if env.is_feasible(xt):
                    info['fallback'] = 'random_feasible'
                    return xt, info
            except Exception:
                info['fallback'] = 'random_feasible'
                return xt, info
        info['fallback'] = 'heuristic_fill'
        return heuristic_fill_low_inventory(env, rng, top_k=3, samples=200), info

    try:
        feasible = env.is_feasible(x)
    except Exception:
        feasible = True

    if not feasible:
        for _ in range(random_attempts):
            xt = random_feasible(env, rng, max_attempts=10)
            try:
                if env.is_feasible(xt):
                    info['fallback'] = 'random_feasible'
                    return xt, info
            except Exception:
                info['fallback'] = 'random_feasible'
                return xt, info
        info['fallback'] = 'heuristic_fill'
        return heuristic_fill_low_inventory(env, rng, top_k=3, samples=200), info

    # CE returned a feasible x. But check low-inventory safety:
    if monitor_items is None:
        monitor_items = list(range(len(s_cur)))
    low_flags = [ (int(s_cur[i]) < low_inventory_threshold) for i in monitor_items ]
    if any(low_flags):
        # If any monitored item is below threshold, use heuristic to refill
        info['fallback'] = 'heuristic_fill_low_inv'
        xh = heuristic_fill_low_inventory(env, rng, top_k=3, samples=300)
        return xh, info

    info['fallback'] = None
    return x, info