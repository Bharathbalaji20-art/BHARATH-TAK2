import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('/content/task2 (1).csv')

print("=== Basic Info ===")
print(df.info())
print("\n=== Missing Values ===")
print(df.isnull().sum())

for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

cat_cols = df.select_dtypes(include='object').columns
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

num_cols = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

plt.figure(figsize=(14, 8))
for i, col in enumerate(num_cols):
    plt.subplot((len(num_cols) + 1) // 2, 2, i + 1)
    sns.boxplot(x=df[col], color='salmon')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("\n✅ Cleaned and Preprocessed Dataset Shape:", df.shape)
