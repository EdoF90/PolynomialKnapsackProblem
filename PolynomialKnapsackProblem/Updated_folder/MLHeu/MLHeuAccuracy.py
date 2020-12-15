from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from solve_polynomial_knapsack import solve_polynomial_knapsack
import json
import matplotlib.pyplot as plt
from functions_ml import classifier_set

""" 
In order to select the classifier that better fit our problem
and in the meanwhile the quantity of item to fix in heuristic

The train set train.csv is already present

The output is a plot with four subplot beacause four different percentuage 
of the total number of items for each instance are tested.
In each subplot there are all the classifier we tested.
The best one is chosen with these plots.

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


#The train set is divided into train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("START TRAINING OVER CLASSIFIER")
for name, clf in zip(names, classifiers):
	pipe = make_pipeline(StandardScaler(),clf)
	pipe.fit(X_train,y_train)
	print(f"{name} trained")

percentuages= [1, 0.8, 0.9, 0.85]
accuracies = np.zeros((len(percentuages),len(names)))

print("\tSTART PREDICTING")
ind=0
for  name,clf in zip(names, classifiers):
	print(f"\t\t{name}")
	scaler=StandardScaler()
	scaler.fit(X_test)
	X_test=scaler.transform(X_test)

	ypred0 = []
	ypred1 = []
	probs= clf.predict_proba(X_test)

	for it in range(len(probs)):
		prob=probs[it]
		if prob[0]>0.5:
			#print(f"Since the prob of 0 was {prob[0]} I assigned {class_assigned}")
			ypred0.append((0,prob[0],int(y_test[it])))
		else:
			#print(f"Since the prob of 1 was {prob[1]} I assigned {class_assigned}")
			ypred1.append((1,prob[1],int(y_test[it])))
	
	ypred1.sort(key=lambda el: el[1], reverse=True)
	ypred0.sort(key=lambda el: el[1], reverse=True)

	for en, p in enumerate(percentuages):
		acc_1=np.sum([pred[0] == pred[2] for pred in ypred1[:int(len(ypred1)*0.8)]])/(len(ypred1)*p)
		acc_0=np.sum([pred[0] == pred[2] for pred in ypred0[:int(len(ypred0)*0.8)]])/(len(ypred0)*p)
		accuracies[en][ind] = (acc_1+acc_0)/2
	ind+=1

#plot the results :  accuracies of the classifiers and percentuage fixed 
fig ,axs= plt.subplots(2,2)

counter=0
for dx in range(2):
	for sx in range(2):
		print(dx,sx)
		ax=axs[dx,sx]
		p= percentuages[counter]*100

		ax.bar(names, [el for el in accuracies[counter]])
		print(p,counter)
		print(accuracies[counter])
		ax.grid()
		ax.tick_params(axis="x", labelsize=6, rotation=45)
		ax.tick_params(axis="y", labelsize=8)
		ax.set_title(f"Accuracies on the {p} %",fontsize=10)
		ax.set_ylim(0.9,0.98)
		counter+=1

#plt.savefig('Accuracy.png')
plt.show()


