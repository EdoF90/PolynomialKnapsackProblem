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
	

class GAHeuristic(object):
	def __init__(self, contSolution, data, n_chromosomes, penalization, weight):
		self.contSolution = contSolution
		self.data = data
		self.items = list(range(data['n_items']))
		self.items.sort(key = lambda x: data['costs'][x][1] - data['costs'][x][0], reverse = True)
		self.solution = []
		self.population = []
		synWork=[key.replace("(","").replace(")","").replace("'","").split(",") for key in self.data['polynomial_gains'].keys()]
		self.synSet=[set(map(int,k)) for k in synWork]
		self.counterInf=0
		self.n_chromosomes=n_chromosomes
		self.penalization=penalization
		self.weight=weight


	def fitnessScore(self, chromosome):
		""" calculate the score of each possible solution
		Args: 
			chromosome: a possible solution
		Return: 
			of: value of the objective function of this chromosome
		"""
		of = 0
		investments = [i for i in range(0,len(chromosome)) if chromosome[i]=='1']	
		investments.sort(key = lambda x: self.data['costs'][x][1] - self.data['costs'][x][0], reverse = True)
		# CHECK FOR FEASIBILITY
		upperCosts = np.sum([self.data['costs'][x][1] for x in investments[:self.data['gamma']]])
		nominalCosts = np.sum([self.data['costs'][x][0] for x in investments[self.data['gamma']:]])
		# IF FEASIBLE, CALCULATE THE OBJECTIVE FUNCTION
		if upperCosts + nominalCosts <= self.data['budget']:
			of += np.sum([self.data['profits'][x] for x in investments])
			of -= upperCosts
			of -= nominalCosts
			investments=set(investments)
			for it in range(len(self.synSet)):
				syn=self.synSet[it]
				if syn.issubset(investments):
					of += self.data['polynomial_gains'][list(self.data['polynomial_gains'].keys())[it]]
		# IF INFEASIBLE, RETURN -1
		else:
			of = -1
			self.counterInf+=1
		return of


	def createPopulation(self):
		""" create the initial population  
		Args: 
			none
		Return: 
			none
		"""
		for k in range(self.n_chromosomes):
			chromosome=""
			count=0
			for i in range(0,len(self.contSolution)):
				if random.uniform(0,1) <= self.contSolution[i]-self.penalization*int(k/int(self.n_chromosomes*self.weight)):
					chromosome += '1'
					count+=1
				else:
					chromosome += '0'
			self.population.append(chromosome)
		
	def parentsSelection(self, counter):
		""" select the parents for the generation of the nexts chromosomes reducing the population size in an exponential way
		Args: 
			counter: variable that grows with each loop of the while in the run function
					 useful considering an ever smaller portion of the populaion until convergence
		Return: 
			none
		"""
		self.population = deepcopy(list(set(self.population)))
		self.counterInf=0
		self.population.sort(key = lambda x: self.fitnessScore(x), reverse = True)
		self.population = deepcopy(self.population[:int(self.n_chromosomes/(2**counter))])
		if counter==0 and len(self.population)==1:
			self.population += list(map(self.mapping,list(itertools.combinations(self.population[0],len(self.population[0])-1))))

	def crossover(self):
		""" combine the genetic information of two parents to generate new offspring
			the couples of parents are all the pobbile combination in the population 
		Args: 
			none
		Return: 
			none
		"""
		newpopulation = []
		couples = list(itertools.combinations(self.population,2))
		for chromosome1, chromosome2 in couples:
			crossoverPoint = random.randint(0, len(chromosome1))
			newpopulation.append(chromosome1[:crossoverPoint] + chromosome2[crossoverPoint:])
			newpopulation.append(chromosome2[:crossoverPoint] + chromosome1[crossoverPoint:])
		self.population = deepcopy(self.population + newpopulation)

	def mutation(self):
		""" alters chromosome's gene inverting their values from its initial state
			it maintain genetic diversity from one generation of a population to the next
		Args: 
			none
		Return: 
			none
		"""
		for chromosome in self.population:
			mutationPoint = random.randint(0, len(chromosome)-1)
			chromosome = list(chromosome)
			chromosome[mutationPoint] = str(int(not bool(int(chromosome[mutationPoint]))))
			chromosome = ''.join(chromosome)

	def run(self):
		counter = 0
		self.sequenceOpt = []
		self.createPopulation()
		self.parentsSelection(counter)
		while len(self.population) != 1 :
			counter+=1
			self.crossover()
			self.mutation()
			self.parentsSelection(counter)
		return self.population[0],self.fitnessScore(self.population[0])		