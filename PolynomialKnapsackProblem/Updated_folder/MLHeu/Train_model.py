import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

file_path = "./model_data/train.csv"
file_model = './model_data/finalized_model_rTrees.sav'

df = pd.read_csv(file_path, header = 0)
df = df._get_numeric_data()
numeric_headers = list(df.columns.values)
# remove the label tag
numeric_headers.pop()
X = df[numeric_headers]
X= X.drop('label', axis=1)
X = X.to_numpy()
y = df['label']
y=y.apply(lambda row: int(row)) 
y=y.to_numpy()
clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=50, min_samples_leaf=30, min_samples_split=2))
clf.fit(X, y)
pickle.dump(clf, open(file_model, 'wb'))
