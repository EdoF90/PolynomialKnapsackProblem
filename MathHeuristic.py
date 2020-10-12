from Instance import Instance
import logging
import numpy as np
import json
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack

	
if __name__ == '__main__':

	workbook = xlsxwriter.Workbook('results_math.xlsx')
    worksheet = workbook.add_worksheet()

    # Add a bold format to use to highlight cells.
    format_header = workbook.add_format(properties={'bold': True, 'font_color': 'white'})
    format_header.set_bg_color('navy')
    format_header.set_font_size(14)
    worksheet.set_column('A:A', 30)
    worksheet.set_column('B:B', 25)
    worksheet.set_column('C:C', 25)
    worksheet.set_column('D:D', 100)
    worksheet.write('A1', 'File name', format_header)
    worksheet.write('B1', 'Objective Function', format_header)
    worksheet.write('C1', 'Computational Time', format_header)
    worksheet.write('D1', 'Solution', format_header)

    # Start from the first cell below the headers.
    row = 1
    col = 0

	log_name = "logs/polynomial_knapsack.log"
	logging.basicConfig(
		filename=log_name,
		format='%(asctime)s %(levelname)s: %(message)s',
		level=logging.INFO, datefmt="%H:%M:%S",
		filemode='w'
	)
	
	list_of_files = os.listdir("config")

	for name_file in list_of_files:

		fp = open("config/"+name_file, 'r')
		sim_setting = json.load(fp)
		fp.close()

		inst = Instance(sim_setting)
		dict_data = inst.get_data()

		var_type = 'continuous'
		heuristic = False
		indexes = []
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

		#print("\nsolution: {}".format(sol))
		#print("objective function: {}".format(of))

		objfun=str(of).replace(".",",")

        worksheet.write(row, 0, name_file)
        worksheet.write(row, 1, objfun)
        worksheet.write(row, 2, comp_time)
        worksheet.write(row, 3, str(sol))
        row += 1

    workbook.close()		