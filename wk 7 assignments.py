# Data Analysis and Visualization with Pandas and Matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn style for better-looking plots
sns.set_style("whitegrid")

# ------------------------------
# Task 1: Load and Explore Dataset
# ------------------------------

# Using Iris dataset as an example
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Inspect the first few rows
print("First 5 rows of the dataset:")
print(df.head(), "\n")

# Check data types and missing values
print("Dataset info:")
print(df.info(), "\n")

print("Missing values per column:")
print(df.isnull().sum(), "\n")

# Handle missing values (if any)
df.fillna(method='ffill', inplace=True)  # Forward fill example

# ------------------------------
# Task 2: Basic Data Analysis
# ------------------------------

# Summary statistics of numerical columns
print("Summary statistics:")
print(df.describe(), "\n")

# Group by species and compute mean of numerical columns
grouped = df.groupby('species').mean()
print("Mean values per species:")
print(grouped, "\n")

# Example insights
print("Observation example: Versicolor has the largest average petal length.\n")

# ------------------------------
# Task 3: Data Visualization
# ------------------------------

# 1. Line Chart: Cumulative sum of sepal length
plt.figure(figsize=(8, 5))
df['sepal length (cm)'].cumsum().plot(kind='line', color='blue')
plt.title('Cumulative Sepal Length')
plt.xlabel('Sample Index')
plt.ylabel('Cumulative Length (cm)')
plt.show()

# 2. Bar Chart: Average petal length per species
plt.figure(figsize=(8, 5))
grouped['petal length (cm)'].plot(kind='bar', color='skyblue')
plt.title('Average Petal Length per Species')
plt.ylabel('Petal Length (cm)')
plt.xlabel('Species')
plt.show()

# 3. Histogram: Distribution of sepal width
plt.figure(figsize=(8, 5))
df['sepal width (cm)'].plot(kind='hist', bins=10, color='lightgreen')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.show()

# 4. Scatter Plot: Sepal length vs Petal length
plt.figure(figsize=(8, 5))
plt.scatter(
    df['sepal length (cm)'],
    df['petal length (cm)'],
    c=df['species'].cat.codes,  # Different colors per species
    cmap='viridis'
)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(iris.target_names)
plt.show()

# ------------------------------
# Optional: Error Handling Example (if reading CSV)
# ------------------------------
"""
try:
    df_csv = pd.read_csv('your_dataset.csv')
except FileNotFoundError:
    print("File not found. Please check the path.")
except pd.errors.ParserError:
    print("Error parsing the CSV file.")
"""
