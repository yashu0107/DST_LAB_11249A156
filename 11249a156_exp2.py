import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# STEP 1: LOAD DATA
data = {
    'Country': ['India', 'USA', 'India', 'USA', 'UK', 'India'],
    'Age': [22, 25, np.nan, 30, 28, 35],
    'Salary': [40000, 60000, 50000, np.nan, 72000, 58000],
    'Purchased': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes']
}
df_combined = pd.DataFrame(data)

# STEP 2: INSPECT DATA (Handling Missing Values) --
# Logic: Fill missing Age/Salary with the Average (Mean) of that column
df_combined['Age'] = df_combined['Age'].fillna(df_combined['Age'].mean())
df_combined['Salary'] = df_combined['Salary'].fillna(df_combined['Salary'].mean())

#--- STEP 3: CONVERT TEXT TO NUMBERS (Encoding) ---
# We use 'One Hot Encoding' (get_dummies)
df_combined_encoded = pd.get_dummies(df_combined, columns=['Country'])

# For the Target column (Purchased), let's map Yes/No manually
df_combined_encoded['Purchased'] = df_combined_encoded['Purchased'].map({'Yes': 1, 'No': 0})

# -- STEP 4: SCALE FEATURES ---
# Separate Features (X) and Target (y)
X_combined = df_combined_encoded.drop('Purchased', axis=1)
y_combined = df_combined_encoded['Purchased']

# Scale
scaler_combined = StandardScaler()
X_combined_scaled = scaler_combined.fit_transform(X_combined)
print("--- FINAL PROCESSED DATA (All steps combined) ---")
print(pd.DataFrame(X_combined_scaled, columns=X_combined.columns).head())
