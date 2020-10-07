import logging
import numpy as np


class Instance():
    def __init__(self, sim_setting):
        """[summary]
        
        Arguments:
            sim_setting {[type]} -- [description]
        """
        logging.info("starting simulation...")
        self.max_size = sim_setting['knapsack_size']
        self.sizes = np.around(np.random.uniform(
            sim_setting['n_items']
        ))
        self.profits = sim_setting['profits']
        self.costs = sim_setting['costs']
        self.polynomial_gains = sim_setting['polynomial_gains']
        self.n_items = sim_setting['n_items']
        self.gamma = sim_setting['Gamma']
        logging.info("simulation end")


    def get_data(self):
        """[summary]
        
        Returns:
            [type] -- [description]
        """
        logging.info("getting data from instance...")
        return {
            "profits": self.profits,
            "sizes": self.sizes,
            "budget": self.max_size,
            "n_items": self.n_items,
            "costs": self.costs,
            "polynomial_gains": self.polynomial_gains,
            "gamma": self.gamma
        }