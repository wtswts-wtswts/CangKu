"""
示例：如何把定价器接入你的列生成主循环

注意：
- 你需要把你现有的 solve_master_problem 导入或复制到本文件夹并确保其返回 (model, duals) 形式，
  且在非最优或不可行时返回 duals=None。
- duals 的期望格式：长度为节点数的列表/ndarray，dual[j] 对应约束 j 的对偶（例如节点覆盖约束）
"""
import networkx as nx
import torch
from pricing_solver import propose_column_from_policy
from model import ActorCritic
from Column_Generation import solve_master_problem
from gurobipy import GRB

# TODO: 替换为你实际RMP函数的导入/定义
# from your_rmp_module import solve_master_problem

def column_generation_with_policy(initial_columns, costs, constraints, graph, policy=None, device='cpu', max_iters=50):
    """
    initial_columns: list of column vectors (each length = num nodes)
    costs: list of costs for those columns (objective coefficients)
    constraints: list of RHS requirements (length = num nodes)
    graph: networkx.DiGraph for pricing subproblem
    policy: ActorCritic instance (if None, no policy used)
    """
    columns = initial_columns.copy()
    column_costs = costs.copy()
    model = None
    for it in range(max_iters):
        # call RMP (user should provide solve_master_problem)
        try:
            master_model, duals = solve_master_problem(columns, column_costs, constraints)
        except NameError:
            raise RuntimeError("Please provide your solve_master_problem function or import it.")
        if duals is None:
            print("RMP not optimal/infeasible. Terminate CG.")
            break
        print(f"Iteration {it}: RMP optimal, duals extracted.")
        # ask policy to propose a column
        if policy is None:
            print("No policy provided. Stopping.")
            break
        candidate = propose_column_from_policy(policy, graph, duals, device=device, greedy=True)
        if candidate is None:
            print("Policy found no improving column. Terminate.")
            break
        col_vector, rc = candidate
        print(f"Policy proposed column with reduced cost {rc:.4f}")
        columns.append(col_vector)
        column_costs.append(rc)  # or actual cost depending on formulation
        model = master_model
    # final reporting
    if model is not None and model.Status == GRB.Status.OPTIMAL:
        print("Final RMP objective:", model.ObjVal)
    else:
        print("Final RMP not optimal or not available.")
    return model