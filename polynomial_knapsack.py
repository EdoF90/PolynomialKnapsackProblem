#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import numpy as np
import json
import csv
from Instance import Instance
from csv import writer
import os
import xlsxwriter
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack


if __name__ == '__main__':
    
    workbook = xlsxwriter.Workbook('results.xlsx')
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

        of, sol, comp_time = solve_polynomial_knapsack(dict_data)

        #print("\nsolution: {}".format(sol))
        #print("objective function: {}".format(of))
        
        """
        # printing results of a file
        file_output = open(
            "./results/exp_general_table.csv",
            "w"
        )
        file_output.write("method, of, sol, time\n")
        file_output.write("{}, {}, {}, {}\n".format(
            "heu", of_heu, sol_heu, comp_time_heu
        ))
        file_output.write("{}, {}, {}, {}\n".format(
            "exact", of_exact, sol_exact, comp_time_exact
        ))
        file_output.close()
        """

        objfun=str(of).replace(".",",")

        worksheet.write(row, 0, name_file)
        worksheet.write(row, 1, objfun)
        worksheet.write(row, 2, comp_time)
        worksheet.write(row, 3, str(sol))
        row += 1

    workbook.close()

