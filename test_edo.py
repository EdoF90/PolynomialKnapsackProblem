# -*- coding: utf-8 -*-
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from simulator.instance import InstanceEdo
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack


if __name__ == '__main__':
    log_name = "logs/polynomial_knapsack.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )

    fp = open("config/edo_instance.json", 'r')
    sim_setting = json.load(fp)
    fp.close()

    file_path = "./results/ml_test.csv"
    file_model = './results/finalized_model.sav'
    N_INSTANCES = 3
    N_FEATURES = 2

    '''
    generate_db = True
    run_ml = False
    '''
    '''
    generate_db = False
    run_ml = True
    '''
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
            file_output.write("instance,relax,profit,label\n")
        for n_instance in range(N_INSTANCES):
            inst = InstanceEdo(sim_setting)
            dict_data = inst.get_data()

            var_type = 'discrete'
            of, sol, comp_time = solve_polynomial_knapsack(dict_data, var_type)

            var_type = 'continuous'
            of, sol_cont, comp_time = solve_polynomial_knapsack(dict_data, var_type)

            print(of, sol, comp_time)
            print(dict_data['profits'])

            # create Training

            for i, ele in enumerate(sol):
                file_output.write("{},{},{},{}\n".format(
                    f"inst_{n_instance}", sol_cont[i], dict_data['profits'][0][i], ele
                ))
        file_output.close()
    elif run_ml:
        # create prediction (numero tra 0 e 1) e metto ad uno solo i maggiori
        
        df = pd.read_csv(file_path, header = 0)

        df = df._get_numeric_data()
        numeric_headers = list(df.columns.values)

        # remove the label tag
        numeric_headers.pop()

        X = df[numeric_headers].to_numpy()
        y = df['label'].to_numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)# ,random_state=109)
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # PROVA logistic regression
        print(f"Risult: {1 - sum(abs(y_pred-y_test))/len(y_test)}")

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("Precision:",metrics.precision_score(y_test, y_pred))
        print("Recall:",metrics.recall_score(y_test, y_pred))
        pickle.dump(clf, open(file_model, 'wb'))

    else:
        clf = pickle.load(open(file_model, 'rb'))

        inst = InstanceEdo(sim_setting)
        dict_data = inst.get_data()

        var_type = 'continuous'
        of, sol_cont, comp_time = solve_polynomial_knapsack(dict_data, var_type)
        print(f"sol_cont: {sol_cont}")

        X = np.zeros((len(sol_cont), N_FEATURES))
        for i, ele in enumerate(sol_cont):
            X[i, 0] = sol_cont[i]
            X[i, 1] = dict_data['profits'][0][i]
        y_ml = clf.predict(X)
        print(f"predicted sol: {y_ml}")


        var_type = 'discrete'
        of_exact, sol_exact, comp_time_exact = solve_polynomial_knapsack(dict_data, var_type)
        print(f"predicted sol: {sol_exact}")

        of_ml, sol_ml, comp_time_ml = solve_polynomial_knapsack(dict_data, var_type, fix_sol=y_ml)

        print("Accuracy:",metrics.accuracy_score(y_ml, sol_exact))
        print("Precision:",metrics.precision_score(y_ml, sol_exact))
        print("Recall:",metrics.recall_score(y_ml, sol_exact))
        if sum(abs(sol_ml-y_ml)) > 0:
            print("CODE ERROR")
        print("Gap:",  (of_exact - of_ml)/ of_ml)
