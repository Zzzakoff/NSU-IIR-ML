import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#1

data = np.zeros((40, 3), dtype=int)
data[:, 0] = np.random.randint(10, 101, size=40)
data[:, 1] = np.random.randint(10, 101, size=40)
data[:, 2] = np.random.randint(0, 2, size=40)

x = data[:, :2] #признаки
y = data[:, 2] #target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

print(f"x_train: {x_train.shape}")
print(f"x_test: {x_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

print("____________________________")

#2.1

df1 = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45, 50],
    'ID_System': [np.nan, 102, np.nan, 105, np.nan, 107],
    'Target': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes']
})

df1 = df1.dropna(subset=['ID_System'])
df1["Target"] = df1["Target"].replace("Yes" , 1).replace("No", 0)

print(df1)

print("____________________________")

#2.2

df2 = pd.DataFrame({
    "City": ["Moscow", "Moscow", "London", "Moscow", np.nan, "Moscow", "London"],
    "Age": [20, 25, 30, 35, 40, 45, 50],
    "Target": ["Basic", "Basic", "Silver", "Silver", "Gold", "Gold", "Gold"]
})

imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
df2[["City"]] = imputer.fit_transform(df2[["City"]])

encoder = OrdinalEncoder(categories=[["Basic", "Silver", "Gold"]])
df2["Target"] = encoder.fit_transform(df2[["Target"]])

print(df2)

print("____________________________")

#2.3

df3 = pd.DataFrame({
    'Pulse': [70, 72, 75, np.nan, 68, 71, 73, 74],
    'Temp': [36.6, 36.7, 36.8, 36.6, 36.9, 36.6, 36.7, 36.8],
    'Target': ['A', 'A', 'B', 'A', 'B', 'A', 'B', 'C']
})

imputer = SimpleImputer(strategy='median')
df3[['Pulse']] = imputer.fit_transform(df3[['Pulse']])

encoder = OrdinalEncoder(categories=[['A', 'B', 'C']])
df3['Target'] = encoder.fit_transform(df3[['Target']])

print(df3)

print("____________________________")

#2.4

df4 = pd.DataFrame({
    'Days_Since_Last_Incident': [10, 5, 20, np.nan, 15, 30],
    'Risk_Score': [0.1, 0.2, 0.1, 0.4, 0.2, 0.1],
    'Target': ['Safe', 'Safe', 'Warning', 'Safe', 'Safe', 'Warning']
})

imputer = SimpleImputer(strategy='median')
df4[['Days_Since_Last_Incident']] = imputer.fit_transform(df4[['Days_Since_Last_Incident']])

encoder = LabelEncoder()
df4['Target'] = encoder.fit_transform(df4['Target'])

print(df4)

print("____________________________")

#2.5

df5 = pd.DataFrame({
    'Bonus_Points': [100, 500, np.nan, 200, np.nan, 800],
    'Salary_K': [50, 100, 40, 120, 30, 150],
    'Target': ['Low', 'High', 'Low', 'High', 'Low', 'High']
})

imputer = SimpleImputer(strategy='median')
df5[['Bonus_Points']] = imputer.fit_transform(df5[['Bonus_Points']])

encoder = OrdinalEncoder(categories=[['Low', 'High']])
df5['Target'] = encoder.fit_transform(df5[['Target']])

print(df5)

print("____________________________")

#3.1

df6 = pd.DataFrame({
    'Completion_Pct': [10, 25, 45, 50, 75, 85, 95, 100],
    'Experience_Years': [1, 2, 3, 4, 5, 6, 7, 8],
    'Target': ['Low', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'High', 'High']
})

scaler = MinMaxScaler()
df6[['Completion_Pct', 'Experience_Years']] = scaler.fit_transform(df6[['Completion_Pct', 'Experience_Years']]
)

encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df6['Target'] = encoder.fit_transform(df6[['Target']])

print(df6)

print("____________________________")

#3.2

df7 = pd.DataFrame({
    'Income_K': [30, 35, 40, 45, 50, 42, 38, 1000],
    'Credit_Score': [600, 620, 640, 610, 650, 630, 615, 800],
    'Target': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
})

scaler = StandardScaler()
df7[['Income_K', 'Credit_Score']] = scaler.fit_transform(
    df7[['Income_K', 'Credit_Score']]
)

encoder = LabelEncoder()
df7['Target'] = encoder.fit_transform(df7['Target'])

print(df7)