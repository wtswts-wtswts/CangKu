from utils import preprocess_instance
from pareto_framework import ParetoFramework

# Test input for CSP-S problem
lengths = [54, 52, 47, 26, 25, 24, 23, 22, 20]  # Item lengths
demands = [4, 4, 3, 4, 8, 5, 8, 8, 5]           # Corresponding demands
max_length = 141  # Maximum sheet length (L)

# Step 1: Preprocess instance
lengths, demands = preprocess_instance(lengths, demands, max_length)
items = len(lengths)  # Number of unique item types

# Step 2: Initialize Pareto Framework and parameters
pareto_framework = ParetoFramework(lengths, demands, max_length, items)

z_star = 20  # Minimum number of objects (assumes precomputed).
y_star = 5   # Minimum number of patterns (assumes precomputed).

# Step 3: Generate Pareto Frontier
pareto_solutions = pareto_framework.generate_pareto_front(
    z_star=z_star,
    y_star=y_star,
    max_patterns=10,    # Maximum patterns to explore
    max_objects=30,     # Maximum boards to consider
    weight_object=1,    # Weight for minimizing objects
    weight_pattern=1    # Weight for minimizing patterns
)

# Step 4: Output results
print("\nFinal Pareto Optimal Solutions:")
for idx, solution in enumerate(pareto_solutions, 1):
    print(f"Solution {idx}:")
    print(f"  Objects Used: {solution['num_objects']}")
    print(f"  Patterns Used: {solution['num_patterns']}")
    print("-" * 50)