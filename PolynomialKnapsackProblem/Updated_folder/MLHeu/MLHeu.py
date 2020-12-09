# -*- coding: utf-8 -*-
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import time
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack
from functions_ml import prepare_set, fix_variables

if __name__ == '__main__':

	log_name = "logs/polynomial_knapsack.log"
	logging.basicConfig(
		filename=log_name,
		format='%(asctime)s %(levelname)s: %(message)s',
		level=logging.INFO, datefmt="%H:%M:%S",
		filemode='w'
	)

	file_model = './model_data/finalized_model_rTrees.sav'
	CONFIG_PATH = "./config_2/"
	N_FEATURES = 6
	FIXED_PERCENTAGE = 0.85
	list_of_files = os.listdir(CONFIG_PATH)
	

	for name_file in list_of_files:
		
		RESULTS_PATH = f"Results/resultsMLHeu.txt"

		print("\tDoing file {}  of {})".format(name_file,len(list_of_files)))
		fp = open(CONFIG_PATH+name_file, 'r')
		dict_data = json.load(fp)
		fp.close()
		N_ITEMS = dict_data['n_items']

		start=time.time()
		clf = pickle.load(open(file_model, 'rb'))

		#EVALUATE CONTINUOUS SOLUTION
		var_type = 'continuous'
		of, sol_cont, comp_time = solve_polynomial_knapsack(dict_data, var_type,False,[])
				
		X = prepare_set(N_ITEMS, N_FEATURES, dict_data, sol_cont)

		#PREDICT STEP
		y_mlProba = clf.predict_proba(X)

		#HANDLE THE RESULT 				
		y_ml = fix_variables(N_ITEMS, y_mlProba, FIXED_PERCENTAGE)
		
		#RUN THE DISCRETE MODEL
		var_type = 'discrete'
		of_ml, sol_ml, comp_time_ml = solve_polynomial_knapsack(dict_data, var_type, True,indexes=y_ml)
		stop=time.time()
		with open(RESULTS_PATH, 'a+') as f:
			f.write('{},{},{}\n'.format(name_file,of_ml,stop-start))