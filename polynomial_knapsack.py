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
    budget = 1
    n_obj = 5

    rnd = False
    if rnd:
        costs = np.random.uniform(0, 1, n_obj)
        gains = np.random.uniform(-1, 1, n_obj)
    else:
        costs = np.array([0.6, 0.4, 0.3, 0.2, 0.1])
        gains = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    print(
        "costs: {}".format(costs)
    )
    print(
        "gains: {}".format(gains)
    )
    polynomial_gains = {
        (1, 3): 0.1,
        (1, 2, 3): 0.5,
        (1, 2, 3, 4): -0.001
    }
    print("polynomial_gains:\n {}".format(
        polynomial_gains)
    )

    of, sol, comp_time = solve_polynomial_knapsack(
        gains,
        polynomial_gains,
        costs,
        budget
    )
    print(
        "sol: {}".format(sol)
    )
    print(
        "of: {}".format(of)
    )
