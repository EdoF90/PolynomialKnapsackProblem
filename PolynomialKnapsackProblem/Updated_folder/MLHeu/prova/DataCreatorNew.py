import json
import random
import math
import numpy as np
import itertools


if __name__ == '__main__':
	CONFIG_DIR="./config_final_3/"
	n_times=200
	numerosity=np.arange(100,1500,n_times)
	for i in range(n_times):
		for el in numerosity:
			create_instances(CONFIG_DIR, el)


def create_instances(CONFIG_DIR, el):
	random.seed(44)
	print("We are at element {}".format(el))
	config = {}
	#N_ITEM
	config['n_items'] = int(el)
	#GAMMA
	config['gamma'] = int(random.uniform(0.2,0.6)*el)

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

	n_it=0

	for i in range(2, config['n_items']):
		if config['n_items']>1000:
			for j in range(int(config['n_items']/2**((i-1)))):
				n_it+=1
				elem = str(tuple(np.random.choice(items, i, replace = False)))
				polynomial_gains[elem] = random.uniform(1, 100/i)
		elif config['n_items']<=1000 and config['n_items']>300:
			for j in range(int(config['n_items']/2**(math.sqrt(i-1)))):
				n_it+=1
				elem = str(tuple(np.random.choice(items, i, replace = False)))
				polynomial_gains[elem] = random.uniform(1, 100/i)
		else:		
			for j in range(int(config['n_items']/(i-1))):
				n_it+=1
				elem = str(tuple(np.random.choice(items, i, replace = False)))
				polynomial_gains[elem] = random.uniform(1, 100/i)
		#print("We had {} iterations".format(n_it))
		#array_profits = array_profits.reshape(1,config['n_items'])
	array_profits = list(array_profits)
	matrix_costs = matrix_costs.reshape(config['n_items'],2)
	matrix_costs = matrix_costs.tolist()
	config['profits'] = array_profits
	config['costs'] = matrix_costs

	config['polynomial_gains'] = polynomial_gains	

	if config['n_items'] > 1000:
		namefile = CONFIG_DIR+"E_{}_{}_{}.json".format(config['n_items'],config['gamma'],round(config['budget'],3))
	elif config['n_items'] > 300 and config['n_items']<=1000:
		namefile = CONFIG_DIR+"S_{}_{}_{}.json".format(config['n_items'],config['gamma'],round(config['budget'],3))
	else:
		namefile = CONFIG_DIR+"L_{}_{}_{}.json".format(config['n_items'],config['gamma'],round(config['budget'],3))
	with open(namefile,'w') as f:
		json.dump(config, f, indent=4)
	return config