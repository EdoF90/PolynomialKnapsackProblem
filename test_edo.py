# -*- coding: utf-8 -*-
import json
import logging
from Instance import Instance
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack


if __name__ == '__main__':
    log_name = "logs/polynomial_knapsack.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )

    fp = open("config/edo_instance.json", 'r')
    sim_setting = json.load(fp)
    fp.close()

    inst = Instance(sim_setting)
    dict_data = inst.get_data()

    var_type = 'discrete'
    of, sol, comp_time = solve_polynomial_knapsack(dict_data, var_type)

    var_type = 'continuous'
    of, sol_cont, comp_time = solve_polynomial_knapsack(dict_data, var_type)

    print(of, sol, comp_time)

    # create Training
    file_output = open(
        "./results/exp_general_table.csv",
        "w"
    )
    file_output.write("instance, relax, label\n")
    for i, ele in enumerate(sol):
        file_output.write("{}, {}, {}\n".format(
            f"inst_1", sol_cont[i], ele, 
        ))
    file_output.close()

    # create prediction (numero tra 0 e 1) e metto il

    # test ml algorithm

    
