# -*- coding: utf-8 -*-
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import time
import xlsxwriter
from openpyxl import load_workbook
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from Instance import  Instance
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack


if __name__ == '__main__':

	log_name = "logs/polynomial_knapsack.log"
	logging.basicConfig(
		filename=log_name,
		format='%(asctime)s %(levelname)s: %(message)s',
		level=logging.INFO, datefmt="%H:%M:%S",
		filemode='w'
	)

	file_model = './model_data/finalized_model.sav'
	CONFIG_PATH="../config_final_2/"
	RESULTS_PATH="../Results/MLHeu/resultsMLHeu.xlsx"
	N_FEATURES = 6

	list_of_files = os.listdir(CONFIG_PATH)
	count=0
	for name_file in list_of_files:
		if not name_file.startswith('.'): #filter out strange .DS_store on MacOS
			count+=1
			print("\nDoing file {} ({} of {})".format(name_file,count,len(list_of_files)))
			fp = open(CONFIG_PATH+name_file, 'r')
			sim_setting = json.load(fp)
			fp.close()
			inst = Instance(sim_setting)
			dict_data = inst.get_data()
			
			start=time.time()
			clf = pickle.load(open(file_model, 'rb'))

			#EVALUATE CONTINUOUS SOLUTION
			#start1=time.time()
			var_type = 'continuous'
			of, sol_cont, comp_time = solve_polynomial_knapsack(dict_data, var_type,False,[])
			#stop1=time.time()
			#print(f"\n\tContinuous model took {stop1-start1}")
			
			#PREPARING TEST OBJECT
			#start1=time.time()
			X = np.zeros((dict_data['n_items'], N_FEATURES))

			for i in range(dict_data['n_items']):
				sinP=0
				sinN=0
				for kpoly in dict_data['polynomial_gains'].keys():
					if str(i) in kpoly[1:-1].split(', '):        
						if dict_data['polynomial_gains'][kpoly]>0:
							sinP+=1
						else:
							sinN+=1
				X[i, 0] = sol_cont[i]
				X[i, 1] = dict_data['profits'][i]
				X[i, 2] = dict_data['costs'][i][0]/dict_data['budget']
				X[i, 3] = dict_data['costs'][i][1]/dict_data['budget']
				X[i, 4] = sinP
				X[i, 5] = sinN

			#stop1=time.time()
			#print(f"\n\tElaborating results took {stop1-start1}")

			#PREDICT STEP

			#start1=time.time()
			y_mlP = clf.predict_proba(X)
			#stop1=time.time()
			#print(f"\n\tPredicting results took {stop1-start1}")	

			#HANDLE THE RESULT 				
			list_ymlP_0=list()
			list_ymlP_1=list()
			count_0=0
			count_1=0
			for it in range(len(y_mlP)):
				if y_mlP[it][0]>0.5:
					count_0+=1
					list_ymlP_0.append((y_mlP[it][0],it))
				else:
					count_1+=1
					list_ymlP_1.append((y_mlP[it][1],it))
			#print(f"\nThe number of zero-one elements are: {count_0}-{count_1}")
			
			list_ymlP_0 = sorted(list_ymlP_0, key=lambda kv: kv[0], reverse=False)
			list_ymlP_1 = sorted(list_ymlP_1, key=lambda kv: kv[0], reverse=False)
		
			#list to pass to the solver to impose constraints on variables fixed to 0 or 1
			y_ml=np.ones(dict_data['n_items'])*(-1)   

			while len(list_ymlP_0)>0.15*count_0:
				ele=list_ymlP_0.pop()
				y_ml[ele[1]]=0

			while len(list_ymlP_1)>0.15*count_1:
				
				ele=list_ymlP_1.pop()
				y_ml[ele[1]]=1

			counter=0
			#RUN THE DISCRETE MODEL
			#start1=time.time()
			var_type = 'discrete'
			of_ml, sol_ml, comp_time_ml = solve_polynomial_knapsack(dict_data, var_type, True,indexes=y_ml)
			#stop1=time.time()
			#print(f"\n\tRunning discrete model took: {stop1-start1}")

			stop=time.time()
			print(f"\n\n\tIt took: {stop-start}\n\tThe solution was {of_ml}")
			
			#SAVING RESULTS
			df=pd.DataFrame(columns=["Instance_name", "Time_MLHeu", "Of_MLHeu"])
			df.loc[0,:]=[name_file,stop-start,of_ml]
			writer = pd.ExcelWriter(RESULTS_PATH, engine='openpyxl')
			writer.book = load_workbook(RESULTS_PATH)
			writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
			reader = pd.read_excel(RESULTS_PATH)
			df.to_excel(writer,index=False,header=False,startrow=len(reader)+1)
			
			
		writer.close() 
