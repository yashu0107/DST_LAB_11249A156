import pandas as pd
from sklearn.model selection import train test split
from sklearn.svm import SVC
from sklearn.metrics import accuracy score
from sklearn.datasets import load iris
# Load dataset
data = load_iris()
X = data.data
y = data.target
# Split dataset
X train, X test, y train, y test = train_test_split(X, y, test_size=0.2)
# Linear Kernel
model linear = SVC(kernel='linear')
model linear.fit(X train, y train)
pred linear = model linear.predict(X test)
acc linear = accuracy score(y test, pred linear)
print("Linear Kernel Accuracy:", acc linear)
# Polynomial Kernel
model poly = SVC(kernel='poly', degree=3)
model poly.fit(X train, y train)
pred poly = model poly.predict(X test)
acc_poly = accuracy score(y test, pred_poly)
print("Polynomial Kernel Accuracy:", acc poly)
# RBF Kernel
model rbf = SVC(kernel='rbf')
model rbf.fit(X train, y train)
pred_rbf = model_rbf.predict(X test)
acc_rbf = accuracy score(y test, pred rbf)
print("RBF Kernel Accuracy:", acc_rbf)
