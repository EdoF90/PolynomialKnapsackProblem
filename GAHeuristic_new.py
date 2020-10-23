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
import threading


#INITIALIZING THREAD
class initializingThread(threading.Thread):
	def __init__(self, contSolution, pos, data):
		threading.Thread.__init__(self)
		self.contSolution = contSolution
		self.pos = pos
		self.data = data

	def fitnessScore(self, chromosome):
		of = 0
		investments = [i for i in range(0,len(chromosome)) if chromosome[i]=='1']
		investments.sort(key = lambda x: self.data['costs'][x][1] - self.data['costs'][x][0], reverse = True)
		#TRY TO FIND WAYS TO AVOID SORTING
		
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

	def run(self):
		global population, terminating
		parent = [i for i in range(0,len(self.contSolution)) if random.uniform(0,1) <= self.contSolution[i]]
		chromosome = ""
		for j in range(len(self.contSolution)):
			if j in parent:
				chromosome +='1'
			else:
				chromosome +='0'

		population.append((chromosome,self.fitnessScore(chromosome)))
		terminating[self.pos] = True


#CROSSOVER THREAD
class CrossoverThread(threading.Thread):
	def __init__(self, chromosome1, chromosome2, pos):
		threading.Thread.__init__(self)
		self.chromosome1 = chromosome1
		self.chromosome2 = chromosome2
		self.pos = pos

	def run(self):
		global newpopulation, terminating
		crossoverPoint = random.randint(0, len(self.chromosome1))
		newpopulation.append(self.chromosome1[:crossoverPoint]+self.chromosome2[crossoverPoint:])
		newpopulation.append(self.chromosome2[:crossoverPoint]+self.chromosome1[crossoverPoint:])
		terminating[self.pos] = True


#MUTATION THREAD
class MutationThread(threading.Thread):
	def __init__(self, chromosome, pos, data):
		threading.Thread.__init__(self)
		self.chromosome = chromosome
		self.pos = pos
		self.data = data

	def fitnessScore(self, chromosome):
		of = 0
		investments = [i for i in range(0,len(chromosome)) if chromosome[i]=='1']
		investments.sort(key = lambda x: self.data['costs'][x][1] - self.data['costs'][x][0], reverse = True)
		#TRY TO FIND WAYS TO AVOID SORTING
		
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


	def run(self):
		global population, terminating
		#SOLUTON 1: TOTALLY RANDOM
		mutationPoint = random.randint(0, len(self.chromosome)-1)
		self.chromosome = list(self.chromosome)
		self.chromosome[mutationPoint] = str(int(not bool(int(self.chromosome[mutationPoint]))))
		self.chromosome = ''.join(self.chromosome)
		population.append((self.chromosome, self.fitnessScore(self.chromosome)))
		terminating[self.pos] = True

		#SOLUTION 2: FLIPPING THE MOST PROBABLE INVESTMENT



class GAHeuristic(object):
	def __init__(self, contSolution, data):
		self.contSolution = contSolution
		self.data = data
		self.items = list(range(data['n_items']))
		self.items.sort(key = lambda x: data['costs'][x][1] - data['costs'][x][0], reverse = True)


	def createPopulation(self):
		global terminating, population, newpopulation
		population = []
		terminating = []
		newpopulation = []

		threads = []
		for k in range(100):
			threads.append(initializingThread(self.contSolution, k, self.data))
			terminating.append(False)

		for t in threads:
			t.run()

		while sum(terminating) != len(terminating):
			continue
		
	'''
	def hashingPopulation(self):
		return [int(chromosome,2) for chromosome in self.population]
	'''

	def parentsSelection(self, counter):
		global population
		population = deepcopy(list(set(population)))
		population.sort(key = lambda x: x[1], reverse = True)
		population = deepcopy(population[:int(100/(2**counter))])


	def crossover(self):
		global newpopulation, population, terminating
		newpopulation = []
		terminating = []

		newpopulation = [elem[0] for elem in population]
		couples = list(itertools.combinations(newpopulation,2))

		threads = []
		for pos,parents in enumerate(couples):
			threads.append(CrossoverThread(parents[0], parents[1],pos))
			terminating.append(False)

		for t in threads:
			t.run()

		while sum(terminating) != len(terminating):
			continue


	def mutation(self):
		global newpopulation, terminating, population
		terminating = []
		threads = []
		
		for pos,chromosome in enumerate(newpopulation):
			threads.append(MutationThread(chromosome,pos, self.data))
			terminating.append(False)

		for t in threads:
			t.run()

		while sum(terminating) != len(terminating):
			continue
			

	def getOptimum(self):
		global population
		print("Best Obj.Func. : {}, Solution: {}".format(population[0][1], population[0][0]))
	
	def diffOptimum(self):
		global population
		opt = population[0]
		self.sequenceOpt.append(opt)
		if len(self.sequenceOpt) > 1:
			return self.sequenceOpt[-1][1] - self.sequenceOpt[-2][1]
		return 1

	def run(self):
		counter = 0
		global population
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
		while len(population) != 1:
			counter+=1
			self.crossover()
			self.mutation()
			self.parentsSelection(counter)
		return population[0]
		#"""
		
		

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

		#print("\nsolution: {}".format(sol))
		#print("objective function: {}".format(of))

		#init global var:
		population = []
		terminating = []
		newpopulation = []
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
		p = round(100-(df['O.F. GA'][i]*100/float(df['O.F. model'][i].replace(',','.'))),4)
		perc.append(str(p))
		t = round(float(df['C.T. model'][i])-float(df['C.T. GA'][i]),4)
		diftime.append(str(t))

	df['% O.F difference GA'] = perc
	df['Time difference GA'] = diftime
	df.to_excel("results_final.xlsx") 
"""
		
		