# -*- coding: utf-8 -*-
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import time
import xlsxwriter
from openpyxl import load_workbook

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from instance import  Instance
from solve_polynomial_knapsack import solve_polynomial_knapsack


if __name__ == '__main__':
    log_name = "logs/polynomial_knapsack.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )


    file_path = "./model_data/ml_test.csv"
    file_model = './model_data/finalized_model.sav'
    N_INSTANCES = 3
    N_FEATURES = 6

     
    #generate_db = True
    #run_ml = False
     

    #generate_db = False
    #run_ml = True

      
    generate_db = False
    run_ml = False

     
    if generate_db:
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
        
    elif run_ml:
        
        df = pd.read_csv(file_path, header = 0)

        df = df._get_numeric_data()
        numeric_headers = list(df.columns.values)

        # remove the label tag
        numeric_headers.pop()

        X = df[numeric_headers]
        X= X.drop('label', axis=1)
        X = X.to_numpy()
        y = df['label']

        print(y)
        #test
        y=y.apply(lambda row: int(row)) 

        #
        y=y.to_numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)# ,random_state=109)
        clf = make_pipeline(StandardScaler(), LogisticRegression())
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(f"Result: {1 - sum(abs(y_pred-y_test))/len(y_test)}")

        #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        #print("Precision:",metrics.precision_score(y_test, y_pred))
        #print("Recall:",metrics.recall_score(y_test, y_pred))
        pickle.dump(clf, open(file_model, 'wb'))
        writer = pd.ExcelWriter('resultsTogether.xlsx', engine='xlsxwriter')
        writer.save()

    else:

        list_of_files = os.listdir("config")
        count=0
        for name_file in list_of_files:
            
                if not name_file.startswith('.'): #filter out strange .DS_store on MacOS
                    count+=1
                    print("\nDoing file {} ({} of {})".format(name_file,count,len(list_of_files)))

                    #inst = Instance()
                    fp = open("config/"+name_file, 'r')
                    sim_setting = json.load(fp)
                    fp.close()

                    inst = Instance(sim_setting)

                    dict_data = inst.get_data()
                    #namefile=dict_data['namefile'][7:-9]
                    namefile=name_file #for files writing 

                    start=time.time()
            
                    clf = pickle.load(open(file_model, 'rb'))
                    
                    #evaluate continuous solution
                    var_type = 'continuous'
                    of, sol_cont, comp_time = solve_polynomial_knapsack(dict_data, var_type)
                    #print('countinuous')
                    #samples for ML prediction
                    X = np.zeros((len(sol_cont), N_FEATURES))

                    for i, ele in enumerate(sol_cont):
                        sinP=0
                        sinN=0
                        for kpoly in dict_data['polynomial_gains'].keys():                
                                for indK in kpoly[1:-1].split(', '):
                                    if str(i)==indK:
                                        if dict_data['polynomial_gains'][kpoly]>0:
                                            sinP+=1
                                        else:
                                            sinN+=1
                        X[i, 0] = sol_cont[i]
                        #X[i, 1] = dict_data['profits'][0][i]
                        X[i, 1] = dict_data['profits'][i]
                        X[i, 2] = dict_data['costs'][i][0]/dict_data['budget']
                        X[i, 3] = dict_data['costs'][i][1]/dict_data['budget']
                        X[i, 4] = sinP
                        X[i, 5] = sinN

                    #predict probabilities of variables to be 0 or 1
                    y_mlP = clf.predict_proba(X)
                    #print('predict')
                    #put probabilities into lists
                    list_ymlP_0=list()
                    for i, ele in enumerate(y_mlP):
                        list_ymlP_0.append([ele[0], i])

                    list_ymlP_1=list()
                    for i, ele in enumerate(y_mlP):
                        list_ymlP_1.append([ele[1], i])
                    list_ymlP_0 = sorted(list_ymlP_0, key=lambda kv: kv[0], reverse=False)
                    list_ymlP_1 = sorted(list_ymlP_1, key=lambda kv: kv[0], reverse=False)

                    #list to pass to the solver to impose constraints on variables fixed to 0 or 1
                    y_ml=np.ones(dict_data['n_items'])*(-1)   
                    #print('fix')
                    #30% max probabs to be 1 fixed to 1
                    while len(list_ymlP_0)>0.7*dict_data['n_items']:
                        ele=list_ymlP_0.pop()
                        y_ml[ele[1]]=0

                    #30% max probabs to be 0 fixed to 0
                    while len(list_ymlP_1)>0.7*dict_data['n_items']:
                        ele=list_ymlP_1.pop()
                        y_ml[ele[1]]=1

                    #print(f"Ml sol: {y_ml}")
                    #print('try')
                    #second run (discrete with imposed contraints to reduce search space)
                    var_type = 'discrete'
                    of_ml, sol_ml, comp_time_ml = solve_polynomial_knapsack(dict_data, var_type, indexes=y_ml)#, time_limit=30*60)

                    end=time.time()-start
                    #print('Ml done')

                    #evaluate exact solution to compare with heuristic result
                    #of_exact, sol_exact, comp_time_exact = solve_polynomial_knapsack(dict_data, var_type, time_limit=30*60) 

                    #results
                    #print(f'Time exact sol {comp_time_exact}')
                    print(f'Time ml sol: {end}')
                    
                    #Gap=(of_exact - of_ml)*100/ of_ml
                    #Gap=str(Gap).replace(".",",")
                    #Diff_time=comp_time_exact-end
                    #Diff_time=str(Diff_time).replace(".",",")

                    #print('Exact of', of_exact)
                    print('ML of: ', of_ml)
                    #print("Gap:",  Gap)

                    df=pd.DataFrame(columns=["Instance_name", "Time_MLHeu", "Of_MLHeu"])
                    df.loc[0,:]=[namefile,end,of_ml]
                    #df = pd.DataFrame({'Instance name ' : namefile, 
                    #           'Diff time': Diff_time,
                    #           'Gap ' : [Gap],
                    #           'Time solver' : comp_time_exact,
                    #           'Time heu' : end,
                    #           'Of heu': of_ml,
                    #           '%time' : (1-end/comp_time_exact)*100})
                    
                    #append results to excel file
                    writer = pd.ExcelWriter('resultsMLHeu.xlsx', engine='openpyxl')
                    writer.book = load_workbook('resultsMLHeu.xlsx')
                    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
                    reader = pd.read_excel(r'resultsMLHeu.xlsx')
                    df.to_excel(writer,index=False,header=False,startrow=len(reader)+1)

                writer.close() 
