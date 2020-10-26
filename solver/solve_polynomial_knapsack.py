import time
import logging
import gurobipy as gp
from gurobipy import GRB


def solve_polynomial_knapsack(
    dict_data, var_type, heuristic=False, indexes=[], gap=None, time_limit=None, verbose=False
):
    n_items = len(dict_data['costs'])
    items = range(dict_data['n_items'])
    n_hog = len(dict_data['polynomial_gains'])
    hogs = range(n_hog)
    
    if var_type == 'continuous':
        var_type = GRB.CONTINUOUS
    else:
        var_type = GRB.BINARY

    problem_name = "polynomial_knapsack"
    logging.info("{}".format(problem_name))

    model = gp.Model(problem_name)
    X = model.addVars(
        n_items,
        lb=0,
        ub=1,
        vtype=var_type,
        name='X'
    )
    Z = model.addVars(
        n_hog,
        lb=0,
        ub=1,
        vtype=var_type,
        name='Z'
    )
    Pi = model.addVars(
        n_items,
        lb=0,
        vtype=GRB.CONTINUOUS,
        name='Pi'
    )
    Rho = model.addVar(
        lb=0,
        vtype=GRB.CONTINUOUS,
        name='Rho'
    )

    #OBJECTIVE FUNCTION
    obj_funct = gp.quicksum(dict_data['profits'][0][i] * X[i] for i in items)
    for h, key in enumerate(dict_data['polynomial_gains']):
        obj_funct += dict_data['polynomial_gains'][key] * Z[h]

    obj_funct -= gp.quicksum(dict_data['costs'][i][0] * X[i] for i in items)
    obj_funct -= (dict_data['gamma']*Rho + gp.quicksum(Pi[i] for i in items))
    
    model.setObjective(obj_funct, GRB.MAXIMIZE)

    #CONSTRAINS
    model.addConstr(
         gp.quicksum(dict_data['costs'][i][0] * X[i] for i in items) + dict_data['gamma']*Rho + gp.quicksum(Pi[i] for i in items) <= dict_data['budget'],
        "budget_limit"
    )

    for i in items:
        model.addConstr(
            Rho + Pi[i] >= (dict_data['costs'][i][1]-dict_data['costs'][i][0]) * X[i],
            "duality_{}".format(i)
        )

    for h, k in enumerate(dict_data['polynomial_gains']):
        k=k.replace("(","").replace(")","").replace("'","").split(",")
        key=[]
        for i in k:
            key.append(int(i))
        key=tuple(key)
        #print("",dict_data['polynomial_gains'][str(key)],"\n",key,"\n")
        if dict_data['polynomial_gains'][str(key)] > 0:
            model.addConstr(
                gp.quicksum(X[i] for i in key) >= len(key) * Z[h],
                "hog {}".format(key)
            )
        else:
            model.addConstr(
                gp.quicksum(X[i] for i in key) <= len(key) - 1 + Z[h],
                "hog {}".format(key)
            )
    if heuristic:
        for i in indexes:
            model.addConstr(
                X[i] >= 1, "mathheur_constr{}".format(i)
            )


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
