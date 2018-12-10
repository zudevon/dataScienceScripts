import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from matplotlib import pyplot as plt

"""
AHA stats for high blood pressure across america
- For non-Hispanic whites, 33.4 percent of men and 30.7 percent of women. 
- For non-Hispanic blacks, 42.6 percent of men and 47.0 percent of women. 
- For Mexican Americans, 30.1 percent of men and 28.8 percent of women.
"""

NA = np.nan

def conv(someList):
    for x in range(0, len(someList)):

        if someList[x] == 'N' or someList[x] == 'No' or someList[x] == 'Female' or someList[x] == 'female' \
                or someList[x] == 'F':
            someList[x] = 0
        else:
             someList[x] = 1

    return someList


def bmi3Cat_conv(someList):
    for i in range(0, len(someList)):

        if someList[i] == 'Poor Health':
            someList[i] = 0

        if someList[i] == 'Intermediate Health':
            someList[i] = 1

        if someList[i] == 'Ideal Health':
            someList[i] = 2

        if someList[i] == NA:
            someList[i] = 0

    return someList


# PULL CSV'S
validation1 = pd.read_csv('/Users/a/myProgs/merckProject/csvData/validation1_int.csv', dtype={(437, 438): int})
validation2 = pd.read_csv('/Users/a/myProgs/merckProject/csvData/validation2_int.csv')
validation3 = pd.read_csv('/Users/a/myProgs/merckProject/csvData/validation3_int.csv')

HTN = validation1.iloc[:, 149].values  # Hypertension Visit #1
HTN2 = validation2.iloc[:, 118].values  # Hypertension Visit #2
HTN3 = validation3.iloc[:, 119].values  # Hypertension Visit #3
age = validation3.iloc[:, 40].values  # age of patient
alcohol = validation3.iloc[:, 47].values  # alcohol usage
PA = validation3.iloc[:, 378].values  # Physical activity
BMI = validation3.iloc[:, 376].values  # Body mass index
Diabetes = validation3.iloc[:, 133].values  # Diabetes
glucose_level = validation3.iloc[:, 125].values  # Glucose Number
nutrition = validation1.iloc[:, 463].values  # Nutrition status
idealHealthChol = validation3.iloc[:, 144].values  # ideal health catagory
famSize = validation3.iloc[:, 364].values  # Family Size
selfIncome = validation3.iloc[:, 363].values  # Head of household self Income
famIncome = validation3.iloc[:, 356].values  # Combined Family income
education = validation3.iloc[:, 372].values  # Education Level

" More variables for 2nd run "
hoodProbs = validation1.iloc[:, 494].values  # Neighborhood problems
hoodSocialProbs = validation1.iloc[:, 492].values  # Daily Discrimination
hoodViolence = validation1.iloc[:, 493].values  # Neighborhood Violence
favFoodStores = validation1.iloc[:, 497].values  # Miles to favorable food
PAFacilities = validation1.iloc[:, 496].values  # Miles from Physical facilty
landPerSqMi = validation1.iloc[:, 498].values  # land owned per suare mile
popDensityPerSqMi = validation1.iloc[:, 499].values  # Population per square mile from home

conv(HTN)
conv(HTN2)
conv(HTN3)
conv(alcohol)
conv(PA)
bmi3Cat_conv(BMI)
conv(Diabetes)
bmi3Cat_conv(nutrition)
# CONVERT EDUCATION

print len(HTN), " Population size"
print np.sum(HTN), " Amount of people that had Hypertension on visit #1"
print np.sum(HTN2), " Amount of people that had Hypertension on visit #2"
print np.sum(HTN3), " Amount of people that had Hypertension on visit #3"

print "543 patients gained HTN in the course of 8 - 10 years. \n\n\n"


# GET FEATURE MATRIX
df = pd.DataFrame([age, alcohol, PA, BMI, Diabetes, glucose_level, nutrition, idealHealthChol, famSize, selfIncome, famIncome])
df = pd.DataFrame.transpose(df)
df['11'] = HTN3
df = df.dropna()
# print df
# print df.shape

# TEST AND TRAIN VARIABLES
X = df.iloc[:, 0:11].values
y = df.iloc[:, 11].values
# print X
# print y
y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)


# ----------------- done with prepossessing

""" LOGISTIC REGRESSION """
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred), "-- Accuracy for Logistic Regression"



""" K NEIGHBORS METHOD """
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred), " -- Accuracy for KNeighbors\n"


""" DECISION TREE PREDICTION / CLASSIFIER """
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# print classifier.predict([[0, 0, 0, 0, 1000, 0, 0, 0, 0, 0]]), " With a blood pressure of 1000 !!!!"
print metrics.accuracy_score(y_test, y_pred), "  -- Accuracy for SKlearn Decision Tree"


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
print "-------------- Importances per column with RF"
print(clf.feature_importances_)

# GET FEATURE MATRIX
df = pd.DataFrame([hoodProbs, hoodSocialProbs, hoodViolence, favFoodStores, PAFacilities, landPerSqMi, popDensityPerSqMi])
df = pd.DataFrame.transpose(df)
df['7'] = HTN3
df = df.dropna()
# print df
# print df.shape

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
