import time
import logging
from gurobipy import *


def solve_polynomial_knapsack(
    gains, polynomial_gains,
    costs, budget, gap=None, time_limit=None, verbose=False
):
    n_items = len(costs)
    items = range(n_items)
    n_hog = len(polynomial_gains)
    hogs = range(n_hog)

    problem_name = "polynomial_knapsack"
    logging.info("{}".format(problem_name))

    model = Model(problem_name)
    X = model.addVars(
        n_items,
        vtype=GRB.BINARY,
        name='X'
    )
    Z = model.addVars(
        n_hog,
        vtype=GRB.BINARY,
        name='Z'
    )

    obj_funct = quicksum(gains[i] * X[i] for i in items)
    for h, key in enumerate(polynomial_gains):
        obj_funct += polynomial_gains[key] * Z[h]
    model.setObjective(obj_funct, GRB.MAXIMIZE)

    model.addConstr(
         quicksum(costs[i] * X[i] for i in items) <= budget,
        "budget_limit"
    )


    for h, key in enumerate(polynomial_gains):
        if polynomial_gains[key] > 0:
            model.addConstr(quicksum(X[i] for i in key) >= len(key) * Z[h], "hog {}".format(key))
        else:
            model.addConstr(quicksum(X[i] for i in key) <= len(key) - 1 + Z[h], "hog {}".format(key))

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
        for i in items:
            grb_var = model.getVarByName(
                "X[{}]".format(i)
            )
            sol[i] = grb_var.X
        return model.getObjective().getValue(), sol, comp_time
    else:
        return -1, [], comp_time
