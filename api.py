#!/usr/bin/env python3
# Updated APITrainer: integrate heuristic candidates into CE via initial_candidates param.
import os
import numpy as np
from numpy.linalg import pinv
from cross_entropy import cross_entropy_greedy
from features import phi_polynomial, phi_fourier, normalize_sx
from heuristic_candidates import generate_heuristic_candidates

class APITrainer:
    def __init__(self, env, basis='fourier', basis_params=None,
                 gamma=0.8, L1=30, L2=50000, N1=10, N2=100, rho=0.1,
                 reg=1e-3, reg_bias=None, rng=None,
                 use_heuristic_candidates=False, heuristic_params=None,
                 heuristic_force_prob=0.0, heuristic_anneal=False):
        self.env = env
        self.gamma = gamma
        self.L1 = L1
        self.L2 = L2
        self.N1 = N1
        self.N2 = N2
        self.rho = rho
        self.reg = reg
        self.reg_bias = reg_bias if reg_bias is not None else reg
        self.rng = np.random.default_rng() if rng is None else rng
        self.basis = basis
        self.basis_params = basis_params or {}

        self.use_heuristic_candidates = use_heuristic_candidates
        self.heuristic_params = heuristic_params or {'num':5, 'top_k':3, 'samples_per':100}
        self.heuristic_force_prob = float(heuristic_force_prob)
        self.heuristic_anneal = bool(heuristic_anneal)

        sx0 = np.zeros(env.m, dtype=int)
        if basis == 'polynomial':
            sxn = np.array(sx0, dtype=float) / float(env.s_max)
            phi0 = phi_polynomial(sxn, **self.basis_params.get('params', {}))
        else:
            sxn = normalize_sx(sx0, env.s_max)
            phi0 = phi_fourier(sxn, **self.basis_params.get('params', {}))
        self.K = len(phi0)

        self.theta = np.zeros(self.K, dtype=float)
        self.theta_iters = []
        self.current_outer_iter = 0

    def phi_from_sx(self, sx, **kwargs):
        if self.basis == 'polynomial':
            sxn = np.array(sx, dtype=float) / float(self.env.s_max)
            return phi_polynomial(sxn, **self.basis_params.get('params', {}))
        else:
            sxn = normalize_sx(sx, self.env.s_max)
            return phi_fourier(sxn, **self.basis_params.get('params', {}))

    def _maybe_generate_heuristics(self, s0):
        if not self.use_heuristic_candidates:
            return []
        return generate_heuristic_candidates(self.env, self.rng,
                                             num=int(self.heuristic_params.get('num', 5)),
                                             top_k=int(self.heuristic_params.get('top_k', 3)),
                                             samples_per=int(self.heuristic_params.get('samples_per', 100)))

    def train(self, init_theta=None, verbose=True):
        if init_theta is not None:
            self.theta = init_theta.copy()

        for it in range(self.L1):
            self.current_outer_iter = it
            A_hat = np.zeros((self.K, self.K), dtype=float)
            b_hat = np.zeros(self.K, dtype=float)

            # Precompute whether to force heuristics in this outer iter (if enabled with anneal)
            # Note: This is independent of the per-decision probabilistic forcing implemented earlier.
            if self.heuristic_anneal and self.L1 > 1:
                outer_force_prob = self.heuristic_force_prob * max(0.0, 1.0 - float(it) / float(self.L1 - 1))
            else:
                outer_force_prob = self.heuristic_force_prob

            for t in range(self.L2):
                s0 = self.rng.integers(0, self.env.s_max+1, size=self.env.m)
                self.env.reset(s0)
                basis_wrap = {"type": self.basis, "params": self.basis_params.get('params', {})}

                # generate heuristics to inject into CE's candidate pool
                heuristics = self._maybe_generate_heuristics(s0)

                # With probability outer_force_prob we *prefer* heuristics by injecting them with higher presence.
                # Here we simply pass them as initial_candidates so CE will always evaluate them;
                # the CE procedure will include them among candidates and may select them as elites.
                init_cands = heuristics if heuristics else None

                # main CE to select x (now with initial_candidates support)
                x, _ = cross_entropy_greedy(self.env, s0, self.theta,
                                            phi_fn=self.phi_from_sx,
                                            basis_params=basis_wrap,
                                            N1=self.N1, N2=self.N2, rho=self.rho,
                                            x_max=self.env.x_max, rng=self.rng,
                                            initial_candidates=init_cands)

                s_next, cost, d = self.env.step(x)

                sx = self.env.post_decision_state(s0, x)
                phi_t = self.phi_from_sx(sx)

                # short CE for next step (we can also inject heuristics here if desired)
                x_next, _ = cross_entropy_greedy(self.env, s_next, self.theta,
                                                 phi_fn=self.phi_from_sx,
                                                 basis_params=basis_wrap,
                                                 N1=1, N2=10, rho=self.rho,
                                                 x_max=self.env.x_max, rng=self.rng,
                                                 initial_candidates=None)
                sx_next = self.env.post_decision_state(s_next, x_next)
                phi_next = self.phi_from_sx(sx_next)

                A_hat += np.outer(phi_t, (phi_t - self.gamma * phi_next))
                b_hat += phi_t * cost

            A_hat /= float(self.L2)
            b_hat /= float(self.L2)

            reg_vec = np.full(self.K, self.reg, dtype=float)
            if self.K > 0:
                reg_vec[0] = self.reg_bias
            A_reg = A_hat + np.diag(reg_vec)

            try:
                cond = np.linalg.cond(A_reg)
            except Exception:
                cond = np.inf

            try:
                theta_new = np.linalg.solve(A_reg, b_hat)
            except Exception:
                theta_new = pinv(A_reg).dot(b_hat)

            self.theta = theta_new
            self.theta_iters.append(self.theta.copy())

            if verbose:
                print(f"Iteration {it+1}/{self.L1} done. cond(A_reg)={cond:.3e}, reg_bias={self.reg_bias}, reg_rest={self.reg}, ||theta||={np.linalg.norm(self.theta):.3e}")

        try:
            os.makedirs('result', exist_ok=True)
            np.save(os.path.join('result', 'theta_iters.npy'), np.array(self.theta_iters))
        except Exception:
            pass

        return self.theta_iters