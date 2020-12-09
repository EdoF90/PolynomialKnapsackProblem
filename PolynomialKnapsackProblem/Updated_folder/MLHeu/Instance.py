import logging
import numpy as np
import random
from DataCreatorNew import create_instances

class Instance():
    def __init__(self, sim_setting = []):
        """Class of the instance

        Arguments:
            sim_setting {[type]} -- if the instance already exist, in sim_setting there are all the information

        if the instance is new, sim_setting is empty and a new configuration file is created with a random number of item

        """

        CONFIG_DIR="./config_final_3/"

        if not sim_setting:
            el=random.randint(100, 1500)
            sim_setting = create_instances(CONFIG_DIR,el)
            print(sim_setting)
        logging.info("starting simulation...")
        self.budget = sim_setting['budget']
        self.sizes = np.around(np.random.uniform(
            sim_setting['n_items']
        ))
        self.profits = sim_setting['profits']
        self.costs = sim_setting['costs']
        self.polynomial_gains = sim_setting['polynomial_gains']
        self.n_items = sim_setting['n_items']
        self.gamma = sim_setting['gamma']
        logging.info("simulation end")


    def get_data(self):
        """
        
        Returns:
            A dictionary with all the information about the instance 
        """
        logging.info("getting data from instance...")
        return {
            "profits": self.profits,
            "sizes": self.sizes,
            "budget": self.budget,
            "n_items": self.n_items,
            "costs": self.costs,
            "polynomial_gains": self.polynomial_gains,
            "gamma": self.gamma
        }