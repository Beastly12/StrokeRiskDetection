import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set aesthetic style for plots
sns.set_theme(style="whitegrid")

# Load dataset
df = pd.read_csv('./data/healthcare-dataset-stroke-data.csv', encoding='ISO-8859-1',
                 na_values=['na', 'NA', 'Unknown', ''])

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())


print("\n" + "=" * 70)
print("MISSING VALUES ANALYSIS")
print("=" * 70)

missing_count = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_data = pd.DataFrame({
    'Missing_Count': missing_count,
    'Missing_Percentage': missing_percentage
})

missing_data = missing_data[missing_data['Missing_Count'] > 0]
missing_data = missing_data.sort_values('Missing_Percentage', ascending=False)

print("\nColumns with missing data:")
print(missing_data)

# Visualization: Missing Values
if not missing_data.empty:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing_data.index, y='Missing_Percentage', data=missing_data, hue=missing_data.index,
                palette='viridis', legend=False)
    plt.title('Percentage of Missing Values by Feature', fontsize=14)
    plt.ylabel('Percentage (%)')
    plt.xlabel('Features')
    plt.savefig('missing_values.png')
    print("-> Visualization saved: missing_values.png")


print("\n" + "=" * 70)
print("DUPLICATE RECORDS")
print("=" * 70)
duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")


print("\n" + "=" * 70)
print("OUTLIER DETECTION (IQR Method)")
print("=" * 70)

numerical_cols = ['age', 'avg_glucose_level', 'bmi']

# Visualization: Boxplots for Outliers
plt.figure(figsize=(15, 6))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(y=df[col], color='salmon', width=0.5)
    plt.title(f'Outliers in {col}', fontsize=12)

    # Calculate IQR for the printed report
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"\n{col}:")
    print(f"  Lower Bound: {lower_bound:.2f} | Upper Bound: {upper_bound:.2f}")
    print(f"  Outliers: {len(outliers)} ({(len(outliers) / len(df) * 100):.2f}%)")

plt.tight_layout()
plt.savefig('outlier_boxplots.png')
print("-> Visualization saved: outlier_boxplots.png")


# 1. Target Variable Distribution (Class Balance)
plt.figure(figsize=(7, 5))
sns.countplot(x='stroke', data=df, hue='stroke', palette='Set2', legend=False)
plt.title('Distribution of Stroke (Target Variable)', fontsize=14)
plt.savefig('stroke_distribution.png')

# 2. Distribution of Numerical Features (Histograms)
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df[col].dropna(), kde=True, color='teal')
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.savefig('numerical_histograms.png')

# 3. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14)
plt.savefig('correlation_heatmap.png')

print("\nAll visualizations have been generated and saved as PNG files.")