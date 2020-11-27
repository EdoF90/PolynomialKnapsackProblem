#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import numpy as np
import json
import csv
from Instance import Instance
from csv import writer
import os
import xlsxwriter
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack
import pandas as pd
import math

if __name__ == '__main__':
	
	log_name = "logs/polynomial_knapsack.log"
	logging.basicConfig(
		filename=log_name,
		format='%(asctime)s %(levelname)s: %(message)s',
		level=logging.INFO, datefmt="%H:%M:%S",
		filemode='w'
	)
	
	PATH_CONFIG_FOLDER="config_final_2"
	NAME_OUTPUT_FILE="model.txt"
	OUTPUT_FOLDER="Results/Model_results/"+NAME_OUTPUT_FILE

	list_of_files = os.listdir(PATH_CONFIG_FOLDER)

	name=[]
	oflist=[]
	time=[]
	gamma_list=[]
	n_items_list=[]

	for name_file in list_of_files:
		print("Doing file {}".format(name_file))
		fp = open(PATH_CONFIG_FOLDER+name_file, 'r')
		sim_setting = json.load(fp)
		fp.close()

		synergies=sim_setting["polynomial_gains"]
		n_items=sim_setting["n_items"]
		gamma=sim_setting["gamma"]
		dict_clustering={}
		items=list(range(n_items))
		for item in items:
			edges_per_item=0
			neighbours_per_item=[]
			for syn_k in list(synergies.keys()):
				syn_k = syn_k.replace("(","").replace(")","").replace("'","").split(",")
				syn_k = list(map(int,syn_k))
				if item in syn_k:
					edges_per_item+=len(syn_k)*(len(syn_k)-1)/2
					neighbours_per_item+=[el for el in syn_k if el!=item]
			neighbours_per_item=set(neighbours_per_item)
			try:
				dict_clustering[item]=(2*edges_per_item)/(len(neighbours_per_item)*(len(neighbours_per_item)-1))
			except:
				dict_clustering[item]=0
		
		global_clustering=sum(dict_clustering.values())/n_items

		inst = Instance(sim_setting)
		dict_data = inst.get_data()

		var_type = 'binary'
		heuristic = False
		indexes = []
		of, sol, comp_time = solve_polynomial_knapsack(dict_data, var_type, heuristic, indexes)
		of=round(of,3)
		objfun=str(of).replace(".",",")
		
		name.append(name_file)
		oflist.append(objfun)
		time.append(round(comp_time,3))

		gamma_list.append(gamma)
		n_items_list.append(n_items)
		#sollist.append(str(sol2))
		#print(str(objfun).replace('.',','))
		#print(str(round(comp_time,3)).replace('.',','))
		with open(OUTPUT_FOLDER, 'a+') as f:
			f.write('{},{},{}\n'.format(name_file,objfun,round(comp_time,3)))

