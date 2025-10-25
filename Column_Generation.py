import gurobipy as gp
from gurobipy import GRB

'''
受限主问题RMP(Gurobi求解器)
#是否适配任意种零件类型（可行，需要理解）
'''

def solve_master_problem(columns, costs, constraints):

    """
    columns: 列（方案）
    costs: 列对应的成本
    constraints: 每种零件的需求
    """

    m = gp.Model("MasterProblem")
    #创建一个Gurobi求解器模型
    x_vars = []
    # 变量（列）的列表/动作空间
    for i, col in enumerate(columns):
        #i代表列的索引 col代表具体数据
        var = m.addVar(obj=costs[i], name=f"x_{i}", lb=0)
        #添加变量 目标函数的系数（成本） lb下界（使用次数为非负数）
        x_vars.append(var)

    # Add constraints添加约束
    # addConstr（）添加必须要满足的约束规则
    for j, constraint in enumerate(constraints):
        #遍历每一种约束
        m.addConstr(gp.quicksum(
            col[j] * x_vars[i] for i, col in enumerate(columns)) >= constraint,
                    #遍历每一种列（方案），所有方案生成的零件数量之和要满足该零件的要求
                    f"constraint_{j}")

    m.optimize()
    #求解当前主问题
    duals = [c.Pi for c in m.getConstrs()]
    #获得dual变量（c.Pi是在获取c的对偶变量）
    return m, duals

"""
子问题SP（GNN+RL）【待处理】
"""

def solve_pricing_subproblem(duals):

    pass


def column_generation(initial_columns, costs, constraints):
    columns = initial_columns.copy()
    column_costs = costs.copy()
    #创建副本

    while True:
        #求解主问题
        master_model, duals = solve_master_problem(columns, column_costs,
                                                   constraints)
        new_col = solve_pricing_subproblem(duals)
        if new_col is None:
            break
            #列生成结束
        columns.append(new_col[0])
        column_costs.append(new_col[1])

    return master_model


