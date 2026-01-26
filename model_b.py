"""
model_b.py

This module contains an implementation of Model-B for the CSP-S problem based on pseudo-polynomial ILP.
The implementation is aligned with the CSP-S paper and uses Gurobi as the underlying solver.
"""

import gurobipy as grb
import numpy as np


class ModelB:
    def __init__(self, lengths, demands, max_length, items, max_freq):
        """
        Initialize Model-B for the CSP-S.

        :param lengths: List of item lengths (l_i for each i ∈ I).
        :param demands: List of item demands (d_i for each i ∈ I).
        :param max_length: Maximum allowable sheet length (L).
        :param items: Number of item types (|I| = M).
        :param max_freq: Maximum frequency for a single pattern (M^k).
        """
        self.lengths = lengths
        self.demands = demands
        self.max_length = max_length
        self.items = items
        self.max_freq = max_freq
        self.K = None  # Patterns (to be initialized with Jk)

        # Initialize the Gurobi model
        self.m = grb.Model("Model-B")
        self.m.setParam('OutputFlag', 1)

    def create_variables(self, K, Jk):
        """
        Create decision variables and initialize sets for patterns and frequencies.

        :param K: Set of patterns (integers representing pattern indices).
        :param Jk: Dictionary mapping each pattern k to its set of frequencies {0, 1, ..., M^k}.
        """
        self.K = K
        self.Jk = Jk

        # Create binary variables lambda_kj for each pattern and each allowed frequency
        self.lambda_kj = self.m.addVars(
            [(k, j) for k in self.K for j in self.Jk[k]],
            vtype=grb.GRB.BINARY,
            name="lambda_kj"
        )

        # Create integer variables alpha_kji for each pattern, frequency, and item type
        self.alpha_kji = self.m.addVars(
            [(k, j, i) for k in self.K for j in self.Jk[k] for i in range(self.items)],
            vtype=grb.GRB.INTEGER,
            name="alpha_kji"
        )

    def add_constraints(self):
        """
        Add constraints for Model-B.

        These include:
        1. Length constraints.
        2. Demand fulfillment constraints.
        3. Pattern frequency selection constraint.
        4. Frequency sorting constraints (symmetry breaking).
        5. Upper bound constraints for alpha_kji.
        """
        # 1. Constraint: Ensure the total length of items in a pattern does not exceed the sheet length
        for k in self.K:
            for j in self.Jk[k]:
                self.m.addConstr(
                    grb.quicksum(self.lengths[i] * self.alpha_kji[k, j, i] for i in range(self.items)) <=
                    self.max_length * self.lambda_kj[k, j],
                    name=f"LengthConstraint_k{k}_j{j}"
                )

        # 2. Constraint: Ensure all demands are fulfilled
        for i in range(self.items):
            self.m.addConstr(
                grb.quicksum(
                    j * self.alpha_kji[k, j, i] for k in self.K for j in self.Jk[k]
                ) >= self.demands[i],
                name=f"DemandConstraint_i_{i}"
            )

        # 3. Constraint: Each pattern can only have one selected frequency
        for k in self.K:
            self.m.addConstr(
                grb.quicksum(self.lambda_kj[k, j] for j in self.Jk[k]) == 1,
                name=f"SingleFrequency_k{k}"
            )

        # 4. Symmetry-breaking constraint: Enforce non-increasing order of frequencies
        k_list = sorted(self.K)
        for idx, k in enumerate(k_list[:-1]):
            next_k = k_list[idx + 1]
            self.m.addConstr(
                grb.quicksum(j * self.lambda_kj[k, j] for j in self.Jk[k]) >=
                grb.quicksum(j * self.lambda_kj[next_k, j] for j in self.Jk[next_k]),
                name=f"SymmetryBreak_k{k}_next_k{next_k}"
            )

        # 5. Upper bound constraints for alpha_kji
        for k in self.K:
            for j in self.Jk[k]:
                for i in range(self.items):
                    max_items_per_pattern = np.floor(self.max_length / self.lengths[i])
                    self.m.addConstr(
                        self.alpha_kji[k, j, i] <= max_items_per_pattern * self.lambda_kj[k, j],
                        name=f"AlphaBound_k{k}_j{j}_i{i}"
                    )

    def set_objective(self, minimize_objects=True):
        """
        Set the objective to either minimize the number of objects or patterns.

        :param minimize_objects: If True, minimize object count; otherwise, minimize pattern count.
        """
        if minimize_objects:
            # Minimize object count
            self.m.setObjective(
                grb.quicksum(
                    j * self.lambda_kj[k, j] for k in self.K for j in self.Jk[k] if j > 0
                ),
                grb.GRB.MINIMIZE
            )
        else:
            # Minimize pattern count (only count patterns with non-zero frequency)
            self.m.setObjective(
                grb.quicksum(
                    self.lambda_kj[k, j] for k in self.K for j in self.Jk[k] if j > 0
                ),
                grb.GRB.MINIMIZE
            )

    def optimize(self):
        """
        Optimize the model.

        :return: Solution dictionary with variable values, or empty dictionary if infeasible.
        """
        self.m.optimize()

        if self.m.status == grb.GRB.OPTIMAL:
            print("Optimal solution found!")
            return {var.varName: var.x for var in self.m.getVars()}
        elif self.m.status == grb.GRB.INFEASIBLE:
            print("Infeasible model!")
            self.m.computeIIS()
            self.m.write("infeasible.ilp")
            return {}
        else:
            print(f"Optimization failed with status: {self.m.status}")
            return {}