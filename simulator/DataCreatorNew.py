import json
import random
import math
import numpy as np
import itertools


config = {}


#N_ITEM
config['n_items'] = input("Insert the number of items you're considering: ")

while not config['n_items'].isdecimal():
	config['n_items'] = input("You should insert a number: ")

config['n_items'] = int(config['n_items'])

#GAMMA
config['gamma'] = input("Insert the maximum number of elements varying: ")

while not config['gamma'].isdecimal():
	config['gamma'] = input("You should insert a number: ")

config['gamma'] = int(config['gamma'])


matrix_costs = np.zeros((config['n_items'],2), dtype = float)

d = [0.3,0.6,0.9]
for i in range(config['n_items']):
	matrix_costs[i,0] = random.uniform(1,50)
	matrix_costs[i,1] =	(1+random.choice(d)) * matrix_costs[i,0]

array_profits = np.zeros((config['n_items']), dtype = float)

#SCRIVERE NEL REPORT; GRUOBI CI PERDE PERò POI CI GUADAGNA
for i in range(config['n_items']):
	array_profits[i] = random.uniform(0.8*np.max(matrix_costs[:,0]),100)	

m = [2,3,4]
config['budget'] = np.sum(matrix_costs[:,0])/random.choice(m)

items = list(range(config['n_items']))
polynomial_gains = {}

for i in range(2, config['n_items']):
	for j in range(int(config['n_items']/2**(i-1))):
		elem = str(tuple(np.random.choice(items, i, replace = False)))
		polynomial_gains[elem] = random.uniform(1, 100/i)

#array_profits = array_profits.reshape(1,config['n_items'])
array_profits = list(array_profits)
matrix_costs = matrix_costs.reshape(config['n_items'],2)
matrix_costs = matrix_costs.tolist()
config['profits'] = array_profits
config['costs'] = matrix_costs

config['polynomial_gains'] = polynomial_gains	

if config['n_items'] <= 40:
	namefile = "config/S_{}_{}_{}.json".format(config['n_items'],config['gamma'],round(config['budget'],3))
elif config['n_items'] > 40 and config['n_items']<=100:
	namefile = "config/M_{}_{}_{}.json".format(config['n_items'],config['gamma'],round(config['budget'],3))
else:
	namefile = "config/L_{}_{}_{}.json".format(config['n_items'],config['gamma'],round(config['budget'],3))
file = open(namefile,"w")
json.dump(config, file, indent=4)
