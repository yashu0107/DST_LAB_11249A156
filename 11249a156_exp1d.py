# STEP 1: IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# STEP 2: CREATE DATASET
data = {
'Student': ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10'],
'Classes_Attended': [30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
'Internal_Marks': [35, 38, 42, 46, 50, 55, 60, 65, 70, 75]
}
df = pd.DataFrame(data)
print(df)
# STEP 3: FEATURES & TARGET
X = df[['Classes_Attended']]
y = df['Internal_Marks']
# STEP 4: TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=7
)
# STEP 5: TRAIN MODEL
model = LinearRegression()
model.fit(X_train, y_train)
# STEP 6: PREDICTION
predictions = model.predict(X_test)
# STEP 7: EVALUATION
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
print("Marks per Class:", model.coef_[0])
print("Base Marks:", model.intercept_)
# STEP 8: VISUALIZATION
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Classes Attended")
plt.ylabel("Internal Marks")
plt.title("Classes Attended vs Internal Marks")
plt.show()
