import time
import logging
from gurobipy import *


def solve_quadratic_knapsack(
    costs, gain_1, gain_2, budget,
    gap=None, time_limit=None, verbose=False
):
    n_items = len(costs)
    items = range(n_items)
    problem_name = "solve_quadratic_knapsack"
    logging.info("solving {}...".format(
        problem_name)
    )
    model = Model(problem_name)
    X = model.addVars(
        n_items,
        vtype=GRB.BINARY,
        name='X'
    )
    Z = model.addVars(
        n_items, n_items,
        vtype=GRB.BINARY,
        name='Z'
    )

    obj_funct = quicksum(gain_1[i] * X[i] for i in items)
    obj_funct += quicksum(gain_2[i][j] * Z[i][j] for i in items for j in items)
    model.setObjective(obj_funct, GRB.MAXIMIZE)

    model.addConstr(
         quicksum( costs[i] * X[i] for i in items ) <= budget,
        "budget_limit"
    )

    for i in items:
        for j in items:
            if j > i:
                if gain_2[i][j] > 0:
                    model.addConstr( X[i] + X[j] >= 2 * Z[i][j], "budget_limit {} {}".format(i, j))
                else:
                    model.addConstr( X[i] + X[j] <= 1 + Z[i][j], "budget_limit {} {}".format(i, j))
            else:
                model.addConstr( Z[i][j] == 0, "upper triangular {} {}".format(i, j))

    model.update()
    if gap:
        model.setParam('MIPgap', gap)
    if time_limit:
        model.setParam(GRB.Param.TimeLimit, time_limit)
    if verbose:
        model.setParam('OutputFlag', 1)
    else:
        model.setParam('OutputFlag', 0)
    model.setParam('LogFile', './logs/gurobi.log')
    # model.write("./logs/model.lp")
    start = time.time()
    model.optimize()
    end = time.time()
    comp_time = end - start
    if model.status == GRB.Status.OPTIMAL:
        sol = [0] * n_items
        for i in n_items:
            grb_var = model.getVarByName(
                "X[{}]".format(i)
            )
            sol[i] = grb_var.X
        return model.getObjective().getValue(), sol, comp_time
    else:
        return -1, [], comp_time
