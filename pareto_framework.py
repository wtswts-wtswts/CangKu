"""pareto_framework.py

ε-constraint Pareto frontier generator for CSP-S using Model-B.

For each y_star (maximum number of used patterns/setups), solve:
    min #objects
    s.t. #patterns_used <= y_star

The resulting list approximates the (objects, patterns) Pareto frontier.
"""

from __future__ import annotations

import numpy as np
import gurobipy as grb

from model_b import ModelB
from utils import preprocess_instance


class ParetoFramework:
    def __init__(self, lengths, demands, max_length):
        """
        Parameters
        ----------
        lengths : list[int]
            Item lengths.
        demands : list[int]
            Item demands.
        max_length : int
            Stock length.
        """
        self.lengths = list(map(int, lengths))
        self.demands = list(map(int, demands))
        self.max_length = int(max_length)
        self.items = len(self.lengths)
        self.solutions: list[dict] = []

    @staticmethod
    def estimate_max_frequency(lengths, demands, max_length, max_objects):
        """Conservative frequency upper bound used to build Jk."""
        lengths = np.asarray(lengths, dtype=int)
        demands = np.asarray(demands, dtype=int)
        return int(min(
            int(max_objects),
            int(demands.max()) if demands.size else int(max_objects),
            int(max_length // lengths.min()) if lengths.size else int(max_objects),
        ))

    def generate_pareto_front(self, max_patterns: int, max_objects: int, output_flag: int = 1):
        """Generate Pareto frontier with ε-constraint."""
        max_patterns = int(max_patterns)
        max_objects = int(max_objects)

        # Pattern slots
        K = list(range(max_patterns))

        # Frequency set (uniform for all k)
        M_freq = self.estimate_max_frequency(self.lengths, self.demands, self.max_length, max_objects)
        Jk = {k: list(range(0, M_freq + 1)) for k in K}

        self.solutions = []
        for y_star in range(1, max_patterns + 1):
            print(f"Optimizing: max patterns={y_star} ...")

            model = ModelB(
                lengths=self.lengths,
                demands=self.demands,
                max_length=self.max_length,
                items=self.items,
                output_flag=output_flag,
            )
            model.create_variables(K, Jk)
            model.add_constraints()
            model.add_max_patterns_constraint(y_star)
            model.add_max_objects_constraint(max_objects)
            model.set_objective(minimize_objects=True)

            solution = model.optimize()
            if not solution:
                continue

            num_objects = int(sum(
                j * model.lambda_kj[k, j].X for k in K for j in Jk[k] if j > 0
            ))
            num_patterns = int(sum(
                model.lambda_kj[k, j].X for k in K for j in Jk[k] if j > 0
            ))

            self.solutions.append({
                "num_objects": num_objects,
                "num_patterns": num_patterns,
                "solution": solution,
            })

        return self.solutions


if __name__ == "__main__":
    # Example input
    lengths = [54, 52, 47, 26, 25, 24, 23, 22, 20]
    demands = [4, 4, 3, 4, 8, 5, 8, 8, 5]
    max_length = 141

    lengths, demands = preprocess_instance(lengths, demands, max_length)

    framework = ParetoFramework(lengths, demands, max_length)
    results = framework.generate_pareto_front(max_patterns=7, max_objects=30, output_flag=1)

    print("\nPareto frontier:")
    for r in results:
        print(f"objects={r['num_objects']}, patterns={r['num_patterns']}")
