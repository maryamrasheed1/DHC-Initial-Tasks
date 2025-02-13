Code:

# Import necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset

file_path = "C:/Users/ALI/Desktop/Word/titanic3.xlsx"  # Replace with the actual file path
df = pd.read_excel(file_path)

# Display the first few rows to confirm the data is loaded correctly

print("First 5 rows of the dataset:")
print(df.head())

# Dataset information

print("\nDataset Information:")
df.info()

# Summary statistics

print("\nSummary Statistics:")
print(df.describe())

# ------------------ Data Cleaning ------------------

# Check for missing values

print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values

df['age'].fillna(df['age'].median(), inplace=True)  # Replace missing 'age' with median
df['fare'].fillna(df['fare'].mean(), inplace=True)  # Replace missing 'fare' with mean
df.dropna(subset=['embarked'], inplace=True)  # Drop rows with missing 'embarked'

# Remove duplicates

df.drop_duplicates(inplace=True)

# Handle outliers for the 'fare' column using the IQR method

Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['fare'] >= lower_bound) & (df['fare'] <= upper_bound)]





# ------------------ Visualizations ------------------

# Bar chart for 'sex'

sns.countplot(x='sex', data=df, palette='pastel')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Bar chart for 'pclass'

sns.countplot(x='pclass', data=df, palette='muted')
plt.title('Passenger Class Distribution')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# Histogram for 'age'

df['age'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap for numeric features

numeric_columns = ['age', 'fare', 'sibsp', 'parch']
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Insights from visualizations

print("\nInsights:")
print("1. Gender distribution shows more males than females in the dataset.")
print("2. Passenger class distribution indicates most passengers were in 3rd class.")
print("3. Age distribution is slightly skewed, with a significant number of younger passengers.")
print("4. Correlation heatmap shows relationships between numeric features like age, fare, siblings/spouse, and parents/children.")

# Save the cleaned dataset for future use

cleaned_file_path = "C:/Users/ALI/Desktop/Word/titanic_cleaned.xlsx"
df.to_excel(cleaned_file_path, index=False)
print(f"\nCleaned dataset saved at: {cleaned_file_path}")

