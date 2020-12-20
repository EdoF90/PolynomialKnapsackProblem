from pandas import *
from Instance import Instance
import logging
import numpy as np
import json
import pandas as pd
import os
import random
from copy import deepcopy
import itertools
import time as t
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack
import GAHeuristic as ga

if __name__ =='__main__':

	log_name = "logs/polynomial_knapsack.log"
	logging.basicConfig(
		filename=log_name,
		format='%(asctime)s %(levelname)s: %(message)s',
		level=logging.INFO, datefmt="%H:%M:%S",
		filemode='w'
	)

	PATH_CONFIG_FOLDER="../"
	NAME_OUTPUT_FILE="results_GAHeu_modified.txt"
	OUTPUT_FOLDER="Results/Genetic_results/Second_configs/"+NAME_OUTPUT_FILE

	list_of_files = os.listdir(PATH_CONFIG_FOLDER)
	
	#BEST PARAMS FROM TUNING -> n_chromosomes=70; penalization=0.03; weight=0.6
	n_chromosomes=70
	penalization=0.03
	weight=0.60

	for name_file in list_of_files:
		print("Doing file {}".format(name_file))
		fp = open(PATH_CONFIG_FOLDER+name_file, 'r')
		dict_data = json.load(fp)
		fp.close()

		var_type = 'continuous'
		heuristic = False
		indexes = []
		timeStart = t.time()
		#CONTINUOUS SOLUTION
		of, sol, comp_time = solve_polynomial_knapsack(dict_data, var_type, heuristic, indexes)
		#START OF THE GENETIC ALGORITHM
		g = ga.GAHeuristic(sol, dict_data, n_chromosomes, penalization, weight)
		solGA, objfun = g.run()
		timeStop = t.time()

		with open(OUTPUT_FOLDER, 'a+') as f:
			f.write('{},{},{}\n'.format(name_file,objfun,round(timeStop-timeStart,3)))