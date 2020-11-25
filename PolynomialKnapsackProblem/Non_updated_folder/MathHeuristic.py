from pandas import *
from Instance import Instance
import logging
import numpy as np
import json
import xlsxwriter
from csv import writer
import pandas as pd
import os
import time as t
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack

	
if __name__ == '__main__':

	xls = ExcelFile('results_model.xlsx')
	df = xls.parse(xls.sheet_names[0])

	log_name = "logs/polynomial_knapsack.log"
	logging.basicConfig(
		filename=log_name,
		format='%(asctime)s %(levelname)s: %(message)s',
		level=logging.INFO, datefmt="%H:%M:%S",
		filemode='w'
	)
	
	list_of_files = os.listdir("config")

	name = []
	oflist = []
	time =[]
	sollist = []

	for name_file in list_of_files:

		fp = open("config/"+name_file, 'r')
		sim_setting = json.load(fp)
		fp.close()

		inst = Instance(sim_setting)
		dict_data = inst.get_data()

		var_type = 'continuous'
		heuristic = False
		indexes = []
		timeStart = t.time()
		of, sol, comp_time = solve_polynomial_knapsack(dict_data, var_type, heuristic, indexes)

		#print("\nsolution: {}".format(sol))
		#print("objective function: {}".format(of))


		#POLICY
		for elem in sol:
			if elem > 0.5:
				indexes.append(sol.index(elem))

		#Sorting by the continuous values descending and try to add one-by-one the elements to constrain
		#At each add check through the config_file if the solution is feasible
		#HINTS: add a lot of elements if Continuous problem ~ O(Model), such that 2nd run is lean
		#HINTS: add few elements if Continuous problem is much faster than Model, such that 2nd run can explore more solutions

		#Being the value of the variables between 0 and 1, it can be seen as a probability
		#Use Bernoulli distribution as a coin toss: the probability of variable=1 is the value assumed in the continuous solution
		#We can sequentially check if the solution remains feasible
		#BUT at the same time, we need the full solution to know about which element is uppered and which one is at nominal cost
		#We can exapand this reasoning to a several-trials through a Binomial distribution

		#Prior: 0.5
		#Likelihood: the value assumed by the variable in the continuous solution
		#Posterior: Likelihood*Prior/Marginal 
		#Marginal: either variables are all equi-probable or their probability is weighted by their particular characteristcs


		#BINARY APPLICATION WITH FIXED VARIABLES
		var_type = 'binary'
		heuristic = True

		of, sol, comp_time = solve_polynomial_knapsack(dict_data, var_type, heuristic, indexes)
		timeStop = t.time()
		#print("\nsolution: {}".format(sol))
		#print("objective function: {}".format(of))
		of = round(of,3)
		objfun = str(of).replace(".",",")

		sol2 = []
		#sol with number of the items
		for i in range(0,len(sol)):
			if sol[i]==1:
				sol2.append(i)

		name.append(name_file)
		oflist.append(objfun)
		time.append(round(timeStop-timeStart,3))
		sollist.append(str(sol2))

	df['Name math'] = name
	df['O.F. math'] = oflist
	df['C.T. math'] = time
	df['Sol math'] = sollist

	perc = []
	diftime = []
	for i in range(0,len(oflist)):
		p = round(100-(float(df['O.F. math'][i].replace(',','.'))*100/float(df['O.F. model'][i].replace(',','.'))),4)
		perc.append(str(p).replace('.',','))
		t = round(float(df['C.T. model'][i])-float(df['C.T. math'][i]),4)
		diftime.append(str(t).replace('.',','))

	df['% O.F difference math'] = perc
	df['Time difference math'] = diftime
	df.to_excel("results_math.xlsx") 