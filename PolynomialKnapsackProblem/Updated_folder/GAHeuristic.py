from pandas import *
from Instance import Instance
import logging
import numpy as np
import json
import xlsxwriter
from csv import writer
import pandas as pd
import os
import random
from copy import deepcopy
import itertools
import time as t
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack
	

class GAHeuristic(object):
	def __init__(self, contSolution, data):
		self.contSolution = contSolution
		self.data = data
		self.items = list(range(data['n_items']))
		self.items.sort(key = lambda x: data['costs'][x][1] - data['costs'][x][0], reverse = True)
		self.solution = []
		self.population = []
		synWork=[key.replace("(","").replace(")","").replace("'","").split(",") for key in self.data['polynomial_gains'].keys()]
		self.synSet=[set(map(int,k)) for k in synWork]
		self.counterInf=0

	def fitnessScore(self, chromosome):
		of = 0
		investments = [i for i in range(0,len(chromosome)) if chromosome[i]=='1']	
		investments.sort(key = lambda x: self.data['costs'][x][1] - self.data['costs'][x][0], reverse = True)
		#CHECK FOR FEASIBILITY
		upperCosts = np.sum([self.data['costs'][x][1] for x in investments[:self.data['gamma']]])
		nominalCosts = np.sum([self.data['costs'][x][0] for x in investments[self.data['gamma']:]])
		if upperCosts + nominalCosts <= self.data['budget']:
			of += np.sum([self.data['profits'][x] for x in investments])
			of -= upperCosts
			of -= nominalCosts
			investments=set(investments)
			for it in range(len(self.synSet)):
				syn=self.synSet[it]
				if syn.issubset(investments):
					of += self.data['polynomial_gains'][list(self.data['polynomial_gains'].keys())[it]]
		else:
			of = -1
			self.counterInf+=1
		return of

	def createPopulation(self):
		for k in range(100):
			chromosome=""
			count=0
			for i in range(0,len(self.contSolution)):
				if random.uniform(0,1) <= self.contSolution[i]:#-0.05*int(k/int(70*0.9)):
					chromosome += '1'
					count+=1
				else:
					chromosome += '0'
			#print("Were taken {} element ".format(count))
			self.population.append(chromosome)
		#print(self.population)
		
	def hashingPopulation(self):
		return [int(chromosome,2) for chromosome in self.population]

	def parentsSelection(self, counter):
		self.population = deepcopy(list(set(self.population)))
		#tsta=t.time()
		self.counterInf=0
		self.population.sort(key = lambda x: self.fitnessScore(x), reverse = True)
		#tsto=t.time()
		#print("\tTook {} to order!".format(tsto-tsta))
		self.population = deepcopy(self.population[:int(100/(2**counter))])
		if counter==0 and len(self.population)==1:
			self.population += list(itertools.combinations(self.population[0],len(self.population[0])-1))
			self.population = map(list,self.population)

	def crossover(self):
		newpopulation = []
		couples = list(itertools.combinations(self.population,2))
		for chromosome1, chromosome2 in couples:
			crossoverPoint = random.randint(0, len(chromosome1))
			newpopulation.append(chromosome1[:crossoverPoint] + chromosome2[crossoverPoint:])
			newpopulation.append(chromosome2[:crossoverPoint] + chromosome1[crossoverPoint:])
		self.population = deepcopy(self.population + newpopulation)

	def mutation_procedure(self, chromosome):
		mutationPoint = random.randint(0, len(chromosome)-1)
		chromosome = list(chromosome)
		chromosome[mutationPoint] = str(int(not bool(int(chromosome[mutationPoint]))))
		chromosome = ''.join(chromosome)
		return chromosome

	def mutation(self):
		#SOLUTON 1: TOTALLY RANDOM
		for chromosome in self.population:
			mutationPoint = random.randint(0, len(chromosome)-1)
			chromosome = list(chromosome)
			chromosome[mutationPoint] = str(int(not bool(int(chromosome[mutationPoint]))))
			chromosome = ''.join(chromosome)

	def getOptimum(self):
		print("Best Obj.Func. : {}".format(self.fitnessScore(self.population[0])))
	
	def diffOptimum(self):
		opt = (self.population[0],self.fitnessScore(self.population[0]))
		self.sequenceOpt.append(opt)
		if len(self.sequenceOpt) > 1:
			return self.sequenceOpt[-1][1] - self.sequenceOpt[-2][1]
		return 1

	def run(self):
		counter = 0
		self.sequenceOpt = []
		#tsta=t.time()
		self.createPopulation()
		#tsto=t.time()
		#print("\nCreate population : {}".format(tsto-tsta))
		#tsta=t.time()
		self.parentsSelection(counter)
		#tsto=t.time()
		#print("Parent selection : {}".format(tsto-tsta))
		#SECONDO METHOD: DECREASING POPULATION SIZE
		it=0
		while len(self.population) != 1 :
			#print("\nIT: {}".format(it+1))
			it+=1
			counter+=1
			#tsta=t.time()
			self.crossover()
			#tsto=t.time()
			#print("Crossover : {}".format(tsto-tsta))
			#tsta=t.time()
			self.mutation()
			#tsto=t.time()
			#print("Mutation : {}".format(tsto-tsta))
			#tsta=t.time()
			self.parentsSelection(counter)
			#tsto=t.time()
			#print("Parent selection : {}".format(tsto-tsta))
		return self.population[0],self.fitnessScore(self.population[0])
		

##################################################################################################################

if __name__ =='__main__':

	log_name = "logs/polynomial_knapsack.log"
	logging.basicConfig(
		filename=log_name,
		format='%(asctime)s %(levelname)s: %(message)s',
		level=logging.INFO, datefmt="%H:%M:%S",
		filemode='w'
	)

	list_of_files = os.listdir("config_final_2")
	
	with open('Results/Model_results/model_second_configs.json') as f:
		runned_model=json.load(f)
	
	alreadyRunned=runned_model.keys()
	for name_file in list_of_files:
		if name_file in alreadyRunned:

			print("Doing file {}".format(name_file))
			fp = open("config_final_2/"+name_file, 'r')
			sim_setting = json.load(fp)
			fp.close()

			inst = Instance(sim_setting)
			dict_data = inst.get_data()

			var_type = 'continuous'
			heuristic = False
			indexes = []
			timeStart = t.time()

			of, sol, comp_time = solve_polynomial_knapsack(dict_data, var_type, heuristic, indexes)
			#START OF THE GENETIC ALGORITHM
			g = GAHeuristic(sol, dict_data)
			solGA, objfun = g.run()
			timeStop = t.time()

			sol2 = []
			#sol with number of the items
			for i in range(0,len(solGA)):
				if solGA[i]=='1':
					sol2.append(i)

			#print()
			#print(str(objfun).replace('.',','))
			#print(str(round(timeStop-timeStart,3)).replace('.',','))
			with open('Results/Genetic_results/Second_configs/results_GAHeu_non_modified.txt', 'a+') as f:
				f.write('{},{},{}\n'.format(name_file,objfun,round(timeStop-timeStart,3)))
	