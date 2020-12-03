from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
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
	KNeighborsClassifier(50),
	#SVC(kernel="linear", C=0.025),
	#SVC(gamma=1, C=1, probability=True),
	#GaussianProcessClassifier(1.0 * RBF(1.0)),
	#DecisionTreeClassifier(max_depth=1),
	RandomForestClassifier(n_estimators=100, min_samples_leaf=50, min_samples_split=2),
	MLPClassifier(early_stopping=True, hidden_layer_sizes=200,learning_rate_init=0.001),
	AdaBoostClassifier(n_estimators= 50),
	#GaussianNB(),
	LogisticRegression()
	]


names = ["Nearest Neighbors",
	 	#"Linear SVM",
	 	#"RBF SVM", #"Gaussian Process",
		#"Decision Tree",
		"Random Forest",
		"Neural Net", 
		"AdaBoost",
		#"Naive Bayes",
		"LogisticRegression"
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("START TRAINING OVER CLASSIFIER")
for name, clf in zip(names, classifiers):
	pipe = make_pipeline(StandardScaler(),clf)
	pipe.fit(X_train,y_train)
	print(f"{name} trained")

accuracies100 = {}

accuracies80 = {}

accuracies90 = {}

accuracies20 = {}

for name in names:
	accuracies100[name] = []
	accuracies80[name] = []
	accuracies90[name] = []
	accuracies20[name] = []

print("\tSTART PREDICTING")
for name,clf in zip(names, classifiers):
	print(f"\t\t{name}")
	scaler=StandardScaler()
	scaler.fit(X_test)
	X_test=scaler.transform(X_test)

	ypred0 = []
	ypred1 = []
	probs= clf.predict_proba(X_test)
	classes= clf.predict(X_test)

	for it in range(len(probs)):
		prob=probs[it]
		class_assigned=classes[it]
		if prob[0]>0.5:
			#print(f"Since the prob of 0 was {prob[0]} I assigned {class_assigned}")
			ypred0.append((0,prob[0],int(y_test[it])))
		else:
			#print(f"Since the prob of 1 was {prob[1]} I assigned {class_assigned}")
			ypred1.append((1,prob[1],int(y_test[it])))
	
	ypred1.sort(key=lambda el: el[1], reverse=True)
	ypred0.sort(key=lambda el: el[1], reverse=True)

	acc_1=np.sum([pred[0] == pred[2] for pred in ypred1])/len(ypred1)
	acc_0=np.sum([pred[0] == pred[2] for pred in ypred0])/len(ypred0)

	accuracies100[name].append(
		(acc_1+acc_0)/2
		)

	acc_1=np.sum([pred[0] == pred[2] for pred in ypred1[:int(len(ypred1)*0.8)]])/(len(ypred1)*0.8)
	acc_0=np.sum([pred[0] == pred[2] for pred in ypred0[:int(len(ypred0)*0.8)]])/(len(ypred0)*0.8)

	accuracies80[name].append(
		(acc_1+acc_0)/2
		)

	acc_1=np.sum([pred[0] == pred[2] for pred in ypred1[:int(len(ypred1)*0.9)]])/(len(ypred1)*0.9)
	acc_0=np.sum([pred[0] == pred[2] for pred in ypred0[:int(len(ypred0)*0.9)]])/(len(ypred0)*0.9)

	accuracies90[name].append(
		(acc_1+acc_0)/2
		)

	acc_1=np.sum([pred[0] == pred[2] for pred in ypred1[int(len(ypred1)*0.8):]])/(len(ypred1)*0.2)
	acc_0=np.sum([pred[0] == pred[2] for pred in ypred0[int(len(ypred0)*0.8):]])/(len(ypred0)*0.2)

	accuracies20[name].append(
		(acc_1+acc_0)/2
		)


#STATISTICS

mean_accuracies_100 = []
mean_accuracies_80 = []
mean_accuracies_90 = []
mean_accuracies_20 = []

for name in names:
	mean_accuracies_100.append(np.mean(accuracies100[name]))
	mean_accuracies_80.append(np.mean(accuracies80[name]))
	mean_accuracies_90.append(np.mean(accuracies90[name]))
	mean_accuracies_20.append(np.mean(accuracies20[name]))

#PLOT RESULTS
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.bar(names, mean_accuracies_100)
ax1.grid()
ax1.set_title("Accuracies on the 100 %")
ax2.bar(names, mean_accuracies_80)
ax2.grid()
ax2.set_title("Accuracies on the 80 %")

ax3.bar(names, mean_accuracies_90)
ax3.grid()
ax3.set_title("Accuracies on the 90 %")
"""
ax3.bar(names, mean_accuracies_20)
ax3.grid()
ax3.set_title("Accuracies on the last 20 %")
"""
#plt.savefig('Accuracy.png')
plt.show()


