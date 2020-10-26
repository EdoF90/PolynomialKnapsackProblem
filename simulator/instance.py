import logging
import random
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


class InstanceEdo():
    def __init__(self, sim_setting):
        """[summary]
        
        Arguments:
            sim_setting {[type]} -- [description]
        """
        logging.info("starting simulation...")
        self.sizes = np.around(np.random.uniform(
            sim_setting['n_items']
        ))

        self.n_items = sim_setting['n_items']
        self.gamma = sim_setting['Gamma']

        matrix_costs = np.zeros((sim_setting['n_items'],2), dtype = float)

        d = [0.3,0.6,0.9]
        for i in range(sim_setting['n_items']):
            matrix_costs[i,0] = random.uniform(1,50)
            matrix_costs[i,1] =	(1 + random.choice(d)) * matrix_costs[i,0]

        array_profits = np.zeros((sim_setting['n_items'],1), dtype = float)

        #SCRIVERE NEL REPORT; GRUOBI CI PERDE PERò POI CI GUADAGNA
        for i in range(sim_setting['n_items']):
            array_profits[i] = random.uniform(0.8*np.max(matrix_costs[:,0]),100)	

        m = [2,3,4]
        self.max_size = np.sum(matrix_costs[:,0])/random.choice(m)

        items = list(range(sim_setting['n_items']))
        polynomial_gains = {}

        for i in range(2, sim_setting['n_items']):
            for j in range(int(sim_setting['n_items']/2**(i-1))):
                elem = str(tuple(np.random.choice(items, i, replace = False)))
                polynomial_gains[elem] = random.uniform(1, 100/i)

        array_profits = array_profits.reshape(1,sim_setting['n_items'])
        array_profits = array_profits.tolist()
        matrix_costs = matrix_costs.reshape(sim_setting['n_items'],2)
        matrix_costs = matrix_costs.tolist()

        self.profits = array_profits
        self.costs = matrix_costs
        self.polynomial_gains = polynomial_gains

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