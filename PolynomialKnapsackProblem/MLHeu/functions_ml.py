import numpy as np
import random
import json
import math


def classifier_set(tuning=False):
	""" prepare the lists of classifiers and relative names we will evaluate 
	Args: 
		none
	Return: 
		classifiers: list of classifiers
		names: list of the name of the classifier
	"""
	if tuning==False:
		classifiers = [
			KNeighborsClassifier(50),
			SVC(kernel="linear", C=0.025, probability=True),
			SVC(gamma=1, C=1, probability=True),
			GaussianProcessClassifier(1.0 * RBF(1.0)),
			DecisionTreeClassifier(criterion= 'entropy', min_samples_leaf= 30, min_samples_split= 10, splitter= 'random'),
			RandomForestClassifier(n_estimators=50, min_samples_leaf=30, min_samples_split=2),
			MLPClassifier(early_stopping=True, hidden_layer_sizes=100,learning_rate_init=0.1),
			AdaBoostClassifier(n_estimators= 50),
			GaussianNB(),
			LogisticRegression()
			]
		names = ["KNN",
			 	"L SVM",
			 	"RBF SVM", 
			 	"GP",
				"DT",
				"RF",
				"NN", 
				"AB",
				"NB",
				"LR"
				]
	return classifiers, names


def countSynergies(item, polynomial_gains):
	""" Count how many positive and negative synergies each item has
	Args: 
		item : the considered item
		polynomial_gains: dictionary, keys are the set of items for a synergy and the value is the corresponding value
	Return: 
		positive_syn: sum of the positive synergy of the item
		negative_syn: sum of the negative synergy of the item
	"""
	positive_syn=0
	negative_syn=0
	for k_poly in polynomial_gains.keys():
		if item in k_poly[1:-1].split(', '):        
			if polynomial_gains[k_poly]>0:
				positive_syn+=1
			else:
				negative_syn+=1
		return (positive_syn, negative_syn)


def prepare_set(N_ITEMS, N_FEATURES, dict_data, sol_cont):
	""" Preparewhat we will pass to the ml 
	Args: 
		N_ITEMS : int, how many items are in this instance
		N_FEATURES: int, how many feature will be passed to the ml alorithm
		dict_data : dictionary with the configuration of the instance 
	Return: 
		X : matrix (N_ITEMS x N_FEATURES)
	"""
	X = np.zeros((N_ITEMS, N_FEATURES))
	for i in range(N_ITEMS):
			positive_syn, negative_syn = countSynergies(str(i), dict_data['polynomial_gains'])
			X[i, 0] = sol_cont[i]
			X[i, 1] = dict_data['profits'][i]
			X[i, 2] = dict_data['costs'][i][0]/dict_data['budget']
			X[i, 3] = dict_data['costs'][i][1]/dict_data['budget']
			X[i, 4] = positive_syn
			X[i, 5] = negative_syn
	return X

def fix_variables(N_ITEMS, y_mlP, FIXED_PERCENTAGE):
	""" find which item will be setted as constraint during the execution of the solver
	Args: 
		N_ITEMS : int, how many items are in this instance
		y_mlP: matrix (N_ITEMS x 2), result of the prediction of the ml algorithm
		FIXED_PERCENTAGE : percentage of the instance that will be ste
	Return: 
		y_ml : list which has -1 where the constrint will not be setted, 
						1 if we want to include the item in the solution
						0 if we do not want the item in the solution
	"""
	list_ymlProb_0 = list() 
	count_0 = 0
	list_ymlProb_1 = list()  
	count_1 = 0
	for it in range(N_ITEMS):
		if y_mlP[it][0]>0.5:
			count_0+=1
			list_ymlProb_0.append((y_mlP[it][0],it))
		else:
			count_1+=1
			list_ymlProb_1.append((y_mlP[it][1],it))
		
	list_ymlProb_0 = sorted(list_ymlProb_0, key=lambda kv: kv[0], reverse=False)
	list_ymlProb_1 = sorted(list_ymlProb_1, key=lambda kv: kv[0], reverse=False)

	y_ml=np.ones(N_ITEMS)*(-1)   
	while len(list_ymlProb_0)>FIXED_PERCENTAGE*count_0:
		ele=list_ymlProb_0.pop()
		y_ml[ele[1]]=0
	
	while len(list_ymlProb_1)>FIXED_PERCENTAGE*count_1:	
		ele=list_ymlProb_1.pop()
		y_ml[ele[1]]=1

	return y_ml


def create_instances(CONFIG_DIR, el):
	""" create an instance for the problem, take as input how many items the inctance will have
		save the instance in a file called with a speaking name about its composition
	Args: 
		CONFIG_DIR : folder where the instance will be salved
		el: how many element the instance will have
	Return: 
		config: dictionary, with the properties of the instance :
				n_items: int, given by input of the function 
				gamma: int, taken from an uniform distribution
				nominal cost: float, aken from an uniform distribution
				upper cost: float, nominal cost multiplied by 1+something
				profit: float, uniform with upper bound nominal cost multiplied by 0.8
				polynomial gains: synergies between the items
	"""
	random.seed(44)
	config = {}
	#N_ITEM
	config['n_items'] = int(el)
	#GAMMA
	config['gamma'] = int(random.uniform(0.2,0.6)*el)

	matrix_costs = np.zeros((config['n_items'],2), dtype = float)

	d = [0.3,0.6,0.9]
	for i in range(config['n_items']):
		matrix_costs[i,0] = random.uniform(1,50)
		matrix_costs[i,1] =	(1+random.choice(d)) * matrix_costs[i,0]

	array_profits = np.zeros((config['n_items']), dtype = float)

	for i in range(config['n_items']):
		array_profits[i] = random.uniform(0.8*np.max(matrix_costs[:,0]),100)	

	m = [2,3,4]
	config['budget'] = np.sum(matrix_costs[:,0])/random.choice(m)

	items = list(range(config['n_items']))
	polynomial_gains = {}

	n_it=0

	for i in range(2, config['n_items']):
		if config['n_items']>1000:
			for j in range(int(config['n_items']/2**((i-1)))):
				n_it+=1
				elem = str(tuple(np.random.choice(items, i, replace = False)))
				polynomial_gains[elem] = random.uniform(1, 100/i)
		elif config['n_items']<=1000 and config['n_items']>300:
			for j in range(int(config['n_items']/2**(math.sqrt(i-1)))):
				n_it+=1
				elem = str(tuple(np.random.choice(items, i, replace = False)))
				polynomial_gains[elem] = random.uniform(1, 100/i)
		else:		
			for j in range(int(config['n_items']/(i-1))):
				n_it+=1
				elem = str(tuple(np.random.choice(items, i, replace = False)))
				polynomial_gains[elem] = random.uniform(1, 100/i)
	array_profits = list(array_profits)
	matrix_costs = matrix_costs.reshape(config['n_items'],2)
	matrix_costs = matrix_costs.tolist()
	config['profits'] = array_profits
	config['costs'] = matrix_costs

	config['polynomial_gains'] = polynomial_gains	

	if config['n_items'] > 1000:
		namefile = CONFIG_DIR+"E_{}_{}_{}.json".format(config['n_items'],config['gamma'],round(config['budget'],3))
	elif config['n_items'] > 300 and config['n_items']<=1000:
		namefile = CONFIG_DIR+"S_{}_{}_{}.json".format(config['n_items'],config['gamma'],round(config['budget'],3))
	else:
		namefile = CONFIG_DIR+"L_{}_{}_{}.json".format(config['n_items'],config['gamma'],round(config['budget'],3))
	with open(namefile,'w') as f:
		json.dump(config, f, indent=4)
	return config