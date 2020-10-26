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
import concurrent.futures
	

class GAHeuristic(object):
	def __init__(self, contSolution, data):
		self.contSolution = contSolution
		self.data = data
		self.items = list(range(data['n_items']))
		self.items.sort(key = lambda x: data['costs'][x][1] - data['costs'][x][0], reverse = True)
		self.solution = []
		self.population = []

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
			for key in self.data['polynomial_gains'].keys():
				k = key.replace("(","").replace(")","").replace("'","").split(",")
				k = list(map(int,k))
				if all(elem in investments for elem in k):
					of += self.data['polynomial_gains'][key]
		else:
			of = -1
		return of

	def createPopulation(self):
		for k in range(100):
			parent = [i for i in range(0,len(self.contSolution)) if random.uniform(0,1) <= self.contSolution[i]]
			chromosome = ""
			for j in range(len(self.contSolution)):
				if j in parent:
					chromosome += '1'
				else:
					chromosome += '0'
			self.population.append(chromosome)
		
	def hashingPopulation(self):
		return [int(chromosome,2) for chromosome in self.population]

	def parentsSelection(self, counter):
		self.population = deepcopy(list(set(self.population)))
		self.population.sort(key = lambda x: self.fitnessScore(x), reverse = True)
		self.population = deepcopy(self.population[:int(100/(2**counter))])

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
		#SOLUTION 2: FLIPPING THE MOST PROBABLE INVESTMENT (not so useful)
		"""
		for i in range(0,len(self.population),5):
			with concurrent.futures.ProcessPoolExecutor(5) as executor:
				for chromosome in executor.map(self.mutation_procedure, self.population[i:i+5]):
					pass
		"""

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
		self.createPopulation()
		self.parentsSelection(counter)
		"""
		#FIRST METHOD: EPSILON
		epsilon = 0.1
		while self.diffOptimum() > epsilon :
			self.crossover()
			self.mutation()
			self.parentsSelection(1)
		return self.sequenceOpt[-1]
		"""
		#SECONDO METHOD: DECREASING POPULATION SIZE
		while len(self.population) != 1 :
			counter+=1
			self.crossover()
			self.mutation()
			self.parentsSelection(counter)
		return self.population[0],self.fitnessScore(self.population[0])
		#"""

##################################################################################################################

if __name__ =='__main__':

	#xls = ExcelFile('results_math.xlsx')
	#df = xls.parse(xls.sheet_names[0])

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
		print(name_file)
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

		#START OF THE GENETIC ALGORITHM
		g = GAHeuristic(sol, dict_data)
		solGA, objfun = g.run()
		timeStop = t.time()

		sol2 = []
		#sol with number of the items
		for i in range(0,len(solGA)):
			if solGA[i]=='1':
				sol2.append(i)

		name.append(name_file)
		oflist.append(objfun)
		time.append(round(timeStop-timeStart,3))
		sollist.append(str(sol2))
		print(str(objfun).replace('.',','))
		print(str(round(timeStop-timeStart,3)).replace('.',','))
"""
	df['Genetic'] = name
	df['O.F. GA'] = oflist
	df['C.T. GA'] = time
	df['Sol GA'] = sollist

	perc = []
	diftime = []
	for i in range(0,len(oflist)):
		p = round(100-(df['O.F. GA'][i]*100/float(df['O.F. model'][i])),4)
		perc.append(str(p).replace('.',','))
		t = round(float(df['C.T. model'][i])-float(df['C.T. GA'][i]),4)
		diftime.append(str(t).replace('.',','))

	df['% O.F difference GA'] = perc
	df['Time difference GA'] = diftime
	df.to_excel("results_final.xlsx") 
"""
		
		