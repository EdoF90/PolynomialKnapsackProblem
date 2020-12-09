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
			if polynomial_gains[kp_oly]>0:
				positive_syn+=1
			else:
				negative_syn+=1
		return (positive_syn, negative_syn)




def prepare_set(N_ITEMS, N_FEATURES, dict_data):
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
			list_ymlP_0.append((y_mlP[it][0],it))
		else:
			count_1+=1
			list_ymlP_1.append((y_mlP[it][1],it))
	#print(f"\nThe number of zero-one elements are: {count_0}-{count_1}")
		
	list_ymlP_0 = sorted(list_ymlP_0, key=lambda kv: kv[0], reverse=False)
	list_ymlP_1 = sorted(list_ymlP_1, key=lambda kv: kv[0], reverse=False)


	y_ml=np.ones(N_ITEMS)*(-1)   
	while len(list_ymlP_0)>FIXED_PERCENTAGE*count_0:
		ele=list_ymlP_0.pop()
		y_ml[ele[1]]=0
	while len(list_ymlP_1)>FIXED_PERCENTAGE*count_1:
				
		ele=list_ymlP_1.pop()
		y_ml[ele[1]]=1

return y_ml