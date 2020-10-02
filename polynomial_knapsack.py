#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import numpy as np
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack


if __name__ == '__main__':
    
    log_name = "logs/polynomial_knapsack.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )

    gamma = 2
    budget = 15
    n_obj = 4
    """
    rnd = False
    
    if rnd:
        costs = np.random.uniform(0, 1, n_obj)
        profits = np.random.uniform(-1, 1, n_obj)
    else:
        """
    costs = np.array([(5,7), (3,5), (4,5), (2,3.5)])
    profits = np.array([9.5, 9.5, 7.5, 7.5])

    print(
        "costs: {}".format(costs)
    )
    print(
        "profits: {}".format(profits)
    )
    polynomial_gains = {
        (0, 1): 1,
        (0, 2): 2,
        (0, 3):1.5,
        (0, 1, 2, 3): 0.4,
        (1, 2): 1.5,
        (2, 3): 2.5,
        (1, 3): 0.5,
        (0, 1, 2): 0.9
    }
    print("polynomial_gains:\n {}".format(
        polynomial_gains)
    )

    of, sol, comp_time = solve_polynomial_knapsack(
        profits,
        polynomial_gains,
        gamma,
        costs,
        budget
    )
    print(
        "sol: {}".format(sol)
    )
    print(
        "of: {}".format(of)
    )
