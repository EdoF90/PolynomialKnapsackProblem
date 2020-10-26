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
import pandas as pd


if __name__ == '__main__':
    
    #df = pd.DataFrame()

    log_name = "logs/polynomial_knapsack.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )
    
    list_of_files = os.listdir("config_enormi")

    name=[]
    oflist=[]
    time=[]
    sollist=[]

    for name_file in list_of_files:

        fp = open("config_enormi/"+name_file, 'r')
        sim_setting = json.load(fp)
        fp.close()

        inst = Instance(sim_setting)
        dict_data = inst.get_data()

        var_type = 'binary'
        heuristic = False
        indexes = []
        of, sol, comp_time = solve_polynomial_knapsack(dict_data, var_type, heuristic, indexes)
        of=round(of,3)
        objfun=str(of).replace(".",",")

        sol2=[]
        #sol with number of the items
        for i in range(0,len(sol)):
            if sol[i]==1:
                sol2.append(i)

        name.append(name_file)
        oflist.append(objfun)
        time.append(round(comp_time,3))
        sollist.append(str(sol2))
        print(str(objfun).replace('.',','))
        print(str(round(comp_time,3)).replace('.',','))
    """
    df['Name model']=name
    df['O.F. model']=oflist
    df['C.T. model']=time
    df['Sol model']=sollist

    df.to_excel("results_model.xlsx") 
    """

