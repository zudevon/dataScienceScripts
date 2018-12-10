
# TEST AND TRAIN VARIABLES
X = df.iloc[:, 0:10].values
y = df.iloc[:, 10].values
# print X
# print y
y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)


# print X_train.shape
# print X_test.shape
# print y_train.shape
# print y_test.shape

# ----------------- done with prepossessing

""" LOGISTIC REGRESSION """
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred), "-- Accuracy for Logistic Regression"


""" K NEIGHBORS METHOD """
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred), " -- Accuracy for KNeighbors"


""" DECISION TREE PREDICTION / CLASSIFIER """
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X, y)
print classifier.predict([[0, 0, 0, 0, 1000, 0, 0, 0, 0, 0]])
