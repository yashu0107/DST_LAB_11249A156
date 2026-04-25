from sklearn.ensemble import RandomForestclassifier, AdaBoostClassifier,StackingClassifier
from sklearn.svm import SVC
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
ada = AdaBoostClassifier(n_estimators=100).fit(X_train, y_train)
estimators = [('rf', RandomForestClassifier()), ('svc',
SVC (probability=True))]
stack = StackingClassifier(estimators=estimators,
final_estimator=LogisticRegression()).fit(X_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test,
rf.predict(X_test)):.2f}"(
print(f"AdaBoost Accuracy: {accuracy_score(y_test,
ada.predict(✗_test)):.2f}")
print(f"Stacking Accuracy: {accuracy_score(y_test,
stack.predict(X_test)):.2f}")
