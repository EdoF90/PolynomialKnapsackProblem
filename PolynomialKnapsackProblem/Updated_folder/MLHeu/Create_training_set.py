import os
from Instance import  Instance
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack

N_INSTANCES = 3
N_FEATURES = 6
file_path = "./model_data/train.csv"

if os.path.isfile(file_path):
	file_output = open(
		file_path,
		"a"
	)
else:
	file_output = open(
		file_path,
		"w"
	)
	file_output.write("instance,relax,profit,nominalC,upperC,PosSinCount,NegSinCount,label,\n")

for n_instance in range(N_INSTANCES):
	inst = Instance()
	dict_data = inst.get_data()

	var_type = 'discrete'
	of, sol, comp_time = solve_polynomial_knapsack(dict_data, var_type)

	var_type = 'continuous'
	of, sol_cont, comp_time = solve_polynomial_knapsack(dict_data, var_type)

	# create Training
	for i, ele in enumerate(sol):
		sinP=0
		sinN=0
		for kpoly in dict_data['polynomial_gains'].keys():
			
			for indK in kpoly[1:-1].split(', '):
				if str(i)==indK:
					if dict_data['polynomial_gains'][kpoly]>0:
						sinP+=1
					else:
						sinN+=1
		file_output.write("{},{},{},{},{},{},{},{}\n".format(
			f"inst_{n_instance}", sol_cont[i], dict_data['profits'][0][i], dict_data['costs'][i][0]/dict_data['budget'], dict_data['costs'][i][1]/dict_data['budget'], sinP, sinN, sol[i]
		))
file_output.close()