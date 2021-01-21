import os
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack
from functions_ml import countSynergies, create_instances
import random 
import math


""" 
Create the training set for the classification or append new lines of instances
N_INSTANCES is the number of instances to add at the train set
The output is the training file train.csv, which will be used in Train_model.py
"""

# how many instances we will add at the training set in this run, this can be changed
N_INSTANCES = 20

# the chosen features are : (1) the countinuos solution of the item
# (2) the profif it would bring to the solution
# (3) nominal and (4) upper cost (normalized to the budget of the instance)
# sum of the (5) positive and (6) negative synergies of the item
N_FEATURES = 6

CONFIG_DIR = "./config_final_3/"
file_path = "./model_data/train.csv"

# if the train is not present at all we will create it, else the new lines will be appended to the exist one
if os.path.isfile(file_path):
	file_output = open(file_path, "a")
else:
	file_output = open(file_path, "w")
	file_output.write("instance,relax,profit,nominalC,upperC,PosSinCount,NegSinCount,label,\n")

for n_instance in range(N_INSTANCES):

	# a new configuration file is created with a random number of item
	el = random.randint(100, 1500)
	dict_data = create_instances(CONFIG_DIR, el)

	# solve the model
	var_type = 'discrete'
	of, sol, comp_time = solve_polynomial_knapsack(dict_data, var_type, False, [])

	#solve the continuos relaxation
	var_type = 'continuous'
	of, sol_cont, comp_time = solve_polynomial_knapsack(dict_data, var_type, False, [])

	# create new lines of the training file
	for i in range(dict_data['n_items']):
		continuous_relaxation_i = sol_cont[i]
		profit_i = dict_data['profits'][i]
		nominal_cost =  dict_data['costs'][i][0]/dict_data['budget']
		upper_cost = dict_data['costs'][i][1]/dict_data['budget']
		positive_syn, negative_syn = countSynergies(str(i), dict_data['polynomial_gains'])
		label = sol[i]
		file_output.write("{},{},{},{},{},{},{},{}\n".format(
			f"inst_{n_instance}", 
			continuous_relaxation_i, 
			profit_i, 
			nominal_cost, 
			upper_cost, 
			positive_syn, 
			negative_syn, label
		))
file_output.close()