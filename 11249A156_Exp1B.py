import pandas as pd
# From a Dictionary
data = {
'Name': ['Alice', 'Bob', 'Charlie'],
'Age': [25, 30, 35],
'City': ['New York', 'Paris', 'London']
}
df = pd.DataFrame(data)
# Viewing data
print(df.head())      # First 5 rows
print(df.info())      # Summary of data types and non-nulls
print(df.describe())  # Statistical summary (mean, std, min, max)

#B. Selection and Filtering
# Selecting a column
ages = df['Age']
# Filtering rows
above_25 = df[df['Age'] > 25]
# iloc (integer-based) vs loc (label-based)
row_0 = df.iloc[0]          # First row (Alice's data)
specific_val = df.loc[0, 'City'] # 'New York'
print(ages)
print(above_25)
print(row_0)
print(specific_val)

#C. Data Cleaning (Handling Missing Values)
# Checking for nulls
print(df.isnull().sum())
# Dropping missing values
df_clean = df.dropna()
# Filling missing values
df_filled = df.fillna(0) # Fill NaNs with 0
# Or fill with mean:
# df['Age'] = df['Age'].fillna(df['Age'].mean())
