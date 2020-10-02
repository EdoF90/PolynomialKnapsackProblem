import json
import random
import math
import numpy as np
import itertools


config = {}


#N_ITEM
config['n_items'] = input("Insert the number of items you're considering: ")

#GAMMA
config['Gamma'] = input("Insert the maximum number of elements varying: ")

while not config['Gamma'].isdecimal():
	config['Gamma'] = input("You should insert a number: ")

config['Gamma'] = int(config['Gamma'])



matrix_costs = np.zeros((config['n_items'],2), dtype = float)

d = [0.3,0.6,0.9]
for i in range(config['n_items']):
	matrix_costs[i,0] = random.uniform(1,50)
	matrix_costs[i,1] =	(1+random.choice(d)) * matrix_costs[i,0]


array_profits = np.zeros((config['n_items'],1), dtype = float)

#SCRIVERE NEL REPORT; GRUOBI CI PERDE PERò POI CI GUADAGNA
for i in range(config['n_items']):
	array_profits[i] = random.uniform(0.8*np.max(matrix_costs[:,0]),100)
		

config['profits'] = array_profits
config['costs'] = matrix_costs
m = [2,3,4]
config['knapsack_size'] = np.sum(matrix_costs[:,0])/random.choice(m)

items = list(range(config['n_items']))
polynomial_gains = {}

for i in range(2, config['n_items']):
	for j in range(config['n_items']/2**(i-1)):
		elem = list(np.random.choice(items, i, replace = False))
		polynomial_gains[elem] = random.uniform(1, 100/i)

config['polynomial_gains'] = polynomial_gains



if config['n_items'] <= 40:
	namefile = "config/Small{}_{}_{}.json".format(config['n_items'],config['Gamma'],config['knapsack_size'])
elif config['n_items'] > 40 and config['n_items']<=100:
	namefile = "config/Medium{}_{}_{}.json".format(config['n_items'],config['Gamma'],config['knapsack_size'])
else:
	namefile = "config/Large{}_{}_{}.json".format(config['n_items'],config['Gamma'],config['knapsack_size'])
file = open(namefile,"w")
json.dump(config, file, indent=4)
