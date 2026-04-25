from sklearn.tree import DecisionTreeClassifier
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
test_size=0.2, random_state=42)
dt_clf = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
print(f"Decision Tree Classifier Accuracy: {accuracy_score(y_test,
dt_clf.predict(X_test)):.2f}")
print("Feature Importances:", dt_clf.feature_importances_)
