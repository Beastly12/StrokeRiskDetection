import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('./data/healthcare-dataset-stroke-data.csv', encoding='ISO-8859-1',
                     na_values=['na', 'NA', 'Unknown', ''])

print("="*70)
print("DATASET OVERVIEW")
print("="*70)
print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())


#checking for Missing values

print("\n" + "="*70)
print("MISSING VALUES ANALYSIS")
print("="*70)


missing_count = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

#  combining into a table
missing_data = pd.DataFrame({
    'Missing_Count': missing_count,
    'Missing_Percentage': missing_percentage
})

# showing only columns that actually have missing data
missing_data = missing_data[missing_data['Missing_Count'] > 0]
missing_data = missing_data.sort_values('Missing_Percentage', ascending=False)

print("\nColumns with missing data:")
print(missing_data)


#checking for duplicates
print("\n" + "="*70)
print("DUPLICATE RECORDS")
print("="*70)

duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")

if duplicate_count > 0:
    print("\nShowing first 5 duplicates:")
    print(df[df.duplicated(keep=False)].head())

print("\n" + "=" * 70)
print("OUTLIER DETECTION (IQR Method)")
print("=" * 70)

# Select ONLY the columns where outliers make sense
numerical_cols = ['age', 'avg_glucose_level', 'bmi']

for col in numerical_cols:
    # Calculate Q1, Q3, and IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    # Calculate percentage
    outlier_percentage = (len(outliers) / len(df)) * 100

    print(f"\n{col}:")
    print(f"  Lower Bound: {lower_bound:.2f}")
    print(f"  Upper Bound: {upper_bound:.2f}")
    print(f"  Outliers: {len(outliers)} ({outlier_percentage:.2f}%)")
    print(f"  Min value: {df[col].min():.2f}")
    print(f"  Max value: {df[col].max():.2f}")