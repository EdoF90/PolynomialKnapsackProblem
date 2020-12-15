from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack
import json
import matplotlib.pyplot as plt
from functions_ml import classifier_set

""" 
Validation of the classifier over the train set

"""

classifiers, names = classifier_set()

df =  pd.read_csv('model_data/train.csv', header = 0)
df = df._get_numeric_data()
numeric_headers = list(df.columns.values)
numeric_headers.pop()
X = df[numeric_headers]
X= X.drop('label', axis=1)
X = X.to_numpy()
y = df['label']
y=y.apply(lambda row: int(row)) 
y=y.to_numpy()

print("START TRAINING OVER CLASSIFIER!\n")
accuracies = {}
mean_accuracies = []

for name, clf in zip(names, classifiers):
	clf = make_pipeline(StandardScaler(),clf)
	scores = cross_val_score(clf,X,y,cv=5)
	accuracies[name]=scores
	print(f"{name} trained")
	print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	mean_accuracies.append(accuracies[name].mean())

plt.figure()
plt.bar(names, mean_accuracies)
plt.grid()
plt.title("Total Accuracies")
plt.ylim(0.9)
plt.savefig('Accuracy.png')
plt.show()


