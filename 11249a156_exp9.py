# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Step 2: Load dataset
data = pd.read_csv("mobile_price.csv")
# Step 3: Define features and target
X = data.drop("price_range", axis=1)
y = data["price_range"]
# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random state=42)
#Step 5: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Step 6: Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)
# Step 7: Train model
knn.fit(X_train, y_train)
# Step 8: Prediction
y_pred = knn.predict(X_test)
# Step 9: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
