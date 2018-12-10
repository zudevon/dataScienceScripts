import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from matplotlib import pyplot as plt




# TEST AND TRAIN VARIABLES
X = df.iloc[:, 0:7].values
y = df.iloc[:, 7].values
# print X
# print y
y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

# print df.shape
# print df
print " STATS FOR 2ND DATASET "

""" RANDOM FOREST PREDICTION """
# X, y = make_classification(n_samples=100, n_features=5, n_informative=5, n_redundant=0, random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators=120, max_depth=5, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print metrics.accuracy_score(y_test, y_pred), " -- Accuracy with Random Forests\n"
print "-------------- Importances per column with RF"
print(clf.feature_importances_), "\n"

""" GRADIENT BOOST PREDICTION """
clf = GradientBoostingClassifier(random_state=0, n_estimators=120, learning_rate=0.2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print metrics.accuracy_score(y_test, y_pred), " -- Accuracy with Grad Boost\n"
print "-------------- Importances per column with GB"
print(clf.feature_importances_)
