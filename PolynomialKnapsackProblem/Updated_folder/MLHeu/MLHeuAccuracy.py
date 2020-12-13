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
from Instance import  Instance
from solve_polynomial_knapsack import solve_polynomial_knapsack
import json
import matplotlib.pyplot as plt

""" 
In order to select the classifier that better fit our problem
and in the meanwhile the quantity of item to fix in heuristic

The train set is already present

"""



classifiers = [
	KNeighborsClassifier(50),
	SVC(kernel="linear", C=0.025),
	SVC(gamma=1, C=1, probability=True),
	GaussianProcessClassifier(1.0 * RBF(1.0)),
	DecisionTreeClassifier(criterion= 'entropy', min_samples_leaf= 30, min_samples_split= 10, splitter= 'random'),
	RandomForestClassifier(n_estimators=50, min_samples_leaf=30, min_samples_split=2),
	MLPClassifier(early_stopping=True, hidden_layer_sizes=100,learning_rate_init=0.1),
	AdaBoostClassifier(n_estimators= 50),
	GaussianNB(),
	LogisticRegression()
	]


names = ["KNN",
	 	"Linear SVM",
	 	"RBF SVM", 
	 	"Gaussian Process",
		"DT",
		"RF",
		"NN", 
		"AB",
		"Naive Bayes",
		"LR"
		]


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

accuracies = {}
percentuages= [1, 0.8, 0.9, 0.85]

print("\tSTART PREDICTING")
for name,clf in zip(names, classifiers):
	print(f"\t\t{name}")
	scaler=StandardScaler()
	scaler.fit(X_test)
	X_test=scaler.transform(X_test)

	ypred0 = []
	ypred1 = []
	probs= clf.predict_proba(X_test)
	#classes= clf.predict(X_test)

	for it in range(len(probs)):
		prob=probs[it]
		#class_assigned=classes[it]
		if prob[0]>0.5:
			#print(f"Since the prob of 0 was {prob[0]} I assigned {class_assigned}")
			ypred0.append((0,prob[0],int(y_test[it])))
		else:
			#print(f"Since the prob of 1 was {prob[1]} I assigned {class_assigned}")
			ypred1.append((1,prob[1],int(y_test[it])))
	
	ypred1.sort(key=lambda el: el[1], reverse=True)
	ypred0.sort(key=lambda el: el[1], reverse=True)

	for p in percentuages:
		acc_1=np.sum([pred[0] == pred[2] for pred in ypred1[:int(len(ypred1)*0.8)]])/(len(ypred1)*p)
		acc_0=np.sum([pred[0] == pred[2] for pred in ypred0[:int(len(ypred0)*0.8)]])/(len(ypred0)*p)
		percentuage_str=str(p)
		accuracies[p][name]=(acc_1+acc_0)/2
		print(p,name)
		print(accuracies[p][name])



#plot the results :  accuracies of the classifiers and percentuage fixed 

fig, axs = plt.subplots(2, 2)

counter=0
for dx in range(2):
	for sx in range(2):
		p= int(percentuage[counter])*100
		print(accuracies[p])
		axs[dx,sx].bar(names, [el for el in accuracies[p].values()])
		axs[dx,sx].grid()
		axs[dx,sx].set_title("Accuracies on the {p} %",fontsize=10)
		axs[dx,sx].set_ylim(0.928,0.98)
		counter+=1

for ax in axs.flat:
	ax.tick_params(axis="x", labelsize=8)
	ax.tick_params(axis="y", labelsize=8)

#plt.savefig('Accuracy.png')
plt.show()


