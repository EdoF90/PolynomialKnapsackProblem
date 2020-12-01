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
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from Instance import  Instance
from solver.solve_polynomial_knapsack import solve_polynomial_knapsack
import json
import matplotlib.pyplot as plt


classifiers = [
	KNeighborsClassifier(3),
	SVC(kernel="linear", C=0.025),
	SVC(gamma=2, C=1),
	#GaussianProcessClassifier(1.0 * RBF(1.0)),
	DecisionTreeClassifier(max_depth=5),
	RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	MLPClassifier(alpha=1, max_iter=1000),
	AdaBoostClassifier(),
	GaussianNB(),
	LogisticRegression()]


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", #"Gaussian Process",
		 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
		 "Naive Bayes","LogisticRegression"]

X =  pd.read_csv('model_data/train.csv', header = 0)
X = X._get_numeric_data()
numeric_headers = list(X.columns.values)
numeric_headers.pop()
y = X['label']
X = X[numeric_headers]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("START TRAINING OVER CLASSIFIER!")
for name, clf in zip(names, classifiers):
	clf = make_pipeline(StandardScaler(),clf)
	clf.fit(X_train,y_train)
	print(f"{name} trained")

accuracies = {}
accuracies_class1 = {}
accuracies_class0 = {}

for name in names:
	accuracies[name] = []
	accuracies_class0[name] = []
	accuracies_class1[name] = []

print("\tSTART PREDICTING")
for name,clf in zip(names, classifiers):
	print(f"\t\t{name}")
	scaler=StandardScaler().fit(X_test)
	X_test = scaler.transform(X_test)
	ypred = clf.predict(X_test)

	accuracies[name].append(
		np.sum([pred == true for pred, true in zip(ypred, y_test)])/len(y_test)
		)
	accuracies_class0[name].append(
		np.sum([pred == true for pred, true in zip(ypred, y_test) if pred == 0])/np.sum([1 for pred in ypred if pred == 0])
		)
	accuracies_class1[name].append(
		np.sum([pred == true for pred, true in zip(ypred, y_test) if pred == 1])/np.sum([1 for pred in ypred if pred == 1])
		)

#STATISTICS & PLOT RESULTS

mean_accuracies = []
mean_accuracies_class1 = []
mean_accuracies_class0 = []

for it in range(len(names)):
	name=names[it]
	mean_accuracies.append(np.mean(accuracies[name]))
	mean_accuracies_class1.append(np.mean(accuracies_class1[name]))
	mean_accuracies_class0.append(np.mean(accuracies_class0[name]))
	
fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(30,30))
ax1.bar(names, mean_accuracies)
ax1.grid()
ax1.set_title("Total Accuracies")
ax1.set_ylim(0.9)
ax2.bar(names, mean_accuracies_class0)
ax2.grid()
ax2.set_title("Accuracies on Class 0")
ax2.set_ylim(0.9)
ax3.bar(names, mean_accuracies_class1)
ax3.grid()
ax3.set_title("Accuracies on Class 1")
ax3.set_ylim(0.9)
plt.savefig('Accuracy.png')
plt.show()


