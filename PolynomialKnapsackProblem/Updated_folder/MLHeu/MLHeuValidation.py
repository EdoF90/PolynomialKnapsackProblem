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
from Instance import  Instance
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack
import json
import matplotlib.pyplot as plt


classifiers = [
	KNeighborsClassifier(50),
	#SVC(kernel="linear", C=0.025),
	SVC(gamma=1, C=1),
	#GaussianProcessClassifier(1.0 * RBF(1.0)),
	DecisionTreeClassifier(criterion= 'entropy', min_samples_leaf= 30, min_samples_split= 10, splitter= 'random'),
	RandomForestClassifier(n_estimators=100, min_samples_leaf=50, min_samples_split=2),
	MLPClassifier(early_stopping=True, hidden_layer_sizes=200,learning_rate_init=0.001),
	AdaBoostClassifier(n_estimators= 50),
	#GaussianNB(),
	LogisticRegression()]


names = ["Nearest Neighbors",
	 	#"Linear SVM",
	 	"RBF SVM", #"Gaussian Process",
		#"Decision Tree",
		"Random Forest",
		"Neural Net", 
		"AdaBoost",
		#"Naive Bayes",
		"LogisticRegression"]

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


