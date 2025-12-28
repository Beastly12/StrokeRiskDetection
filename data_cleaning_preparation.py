import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("STROKE PREDICTION DATASET - DATA QUALITY & CLEANING")
print("=" * 70)



print("\n" + "=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

df = pd.read_csv('./data/healthcare-dataset-stroke-data.csv', encoding='ISO-8859-1',
                     na_values=['na', 'NA', 'Unknown', ''])

df_original = df.copy()  # Save original for comparison

print(f"\n✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn names: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

# Quick data types check
print("\nData types:")
print(df.dtypes)


# TEXT DATA STANDARDIZATION


print("\n" + "=" * 70)
print("STEP 1.5: TEXT DATA STANDARDIZATION")
print("=" * 70)

print("\nStandardizing categorical text for consistent presentation...")

# Get all categorical columns
categorical_cols = ['gender', 'ever_married', 'work_type',
                    'Residence_type', 'smoking_status']

# Store "before" values for documentation
text_changes = {}

for col in categorical_cols:
    print(f"\n{col}:")
    print(f"  Before: {df[col].unique()}")

    # Defensive cleaning: strip whitespace (even though none found)
    df[col] = df[col].str.strip()

    # Standardize to Title Case for professional presentation
    # BUT: Keep special cases as-is
    if col == 'work_type':
        # Keep "children" lowercase (it's not a title)
        # Capitalize others
        df[col] = df[col].replace({
            'children': 'Children',
            # Others already capitalized correctly
        })
    elif col == 'smoking_status':
        # Standardize smoking status to Title Case
        df[col] = df[col].replace({
            'formerly smoked': 'Formerly Smoked',
            'never smoked': 'Never Smoked',
            'smokes': 'Smokes',
            # 'Unknown' already correct
        })

    print(f"  After: {df[col].unique()}")

    # Track what changed
    text_changes[col] = {
        'before': df_original[col].unique().tolist(),
        'after': df[col].unique().tolist()
    }

print("\n✓ Text standardization complete")

# DATA QUALITY ASSESSMENT

print("\n" + "=" * 70)
print("STEP 2: DATA QUALITY ASSESSMENT")
print("=" * 70)

# 2.1 Check Missing Values
print("\n" + "-" * 70)
print("2.1 Missing Values Analysis")
print("-" * 70)

missing_count = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': missing_count,
    'Missing_Percentage': missing_percentage
})

missing_data = missing_data[missing_data['Missing_Count'] > 0]
missing_data = missing_data.sort_values('Missing_Percentage', ascending=False)

if len(missing_data) > 0:
    print("\nColumns with missing data:")
    print(missing_data.to_string(index=False))

    # VISUALIZATION: Missing Data
    plt.figure(figsize=(10, 6))
    sns.barplot(data=missing_data, x='Missing_Percentage', y='Column',
                palette='Reds_r', hue='Column', legend=False)
    plt.title('Missing Data Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Percentage Missing (%)', fontsize=12)
    plt.ylabel('Column', fontsize=12)
    plt.axvline(x=5, color='red', linestyle='--', linewidth=2, label='5% Threshold')
    plt.axvline(x=20, color='darkred', linestyle='--', linewidth=2, label='20% Threshold (Critical)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('01_missing_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n📊 Saved: 01_missing_data_analysis.png")
else:
    print("\n✓ No missing values found!")

# 2.2 Check Duplicates
print("\n" + "-" * 70)
print("2.2 Duplicate Records")
print("-" * 70)

duplicate_count = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicate_count}")

if duplicate_count > 0:
    print("⚠️  Found duplicates - may need to remove")
else:
    print("✓ No duplicates found")

# 2.3 Check Outliers
print("\n" + "-" * 70)
print("2.3 Outlier Detection (IQR Method)")
print("-" * 70)

numerical_cols = ['age', 'avg_glucose_level', 'bmi']
outlier_summary = []

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_pct = (len(outliers) / len(df)) * 100

    outlier_summary.append({
        'Column': col,
        'Outliers': len(outliers),
        'Percentage': outlier_pct,
        'Min': df[col].min(),
        'Max': df[col].max(),
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound
    })

    print(f"\n{col}:")
    print(f"  Outliers: {len(outliers)} ({outlier_pct:.2f}%)")
    print(f"  Range: {df[col].min():.2f} - {df[col].max():.2f}")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

# VISUALIZATION: Box Plots for Outliers
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for idx, col in enumerate(numerical_cols):
    bp = axes[idx].boxplot(df[col].dropna(), patch_artist=True,
                           boxprops=dict(facecolor='lightblue', linewidth=1.5),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           flierprops=dict(marker='o', markerfacecolor='red',
                                           markersize=5, alpha=0.5))

    axes[idx].set_title(f'{col}\nOutlier Detection', fontweight='bold')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(axis='y', alpha=0.3)

    # Add statistics box
    stats_text = (f"Min: {df[col].min():.1f}\n"
                  f"Q1: {df[col].quantile(0.25):.1f}\n"
                  f"Median: {df[col].median():.1f}\n"
                  f"Q3: {df[col].quantile(0.75):.1f}\n"
                  f"Max: {df[col].max():.1f}")

    axes[idx].text(0.98, 0.97, stats_text, transform=axes[idx].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   verticalalignment='top', horizontalalignment='right',
                   fontsize=9)

plt.suptitle('Outlier Analysis: Box Plots', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('02_outlier_analysis_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n📊 Saved: 02_outlier_analysis_boxplots.png")

# DATA CLEANING

print("\n" + "=" * 70)
print("STEP 3: DATA CLEANING")
print("=" * 70)

# 3.1 Handle Missing Smoking Status
print("\n" + "-" * 70)
print("3.1 Cleaning: Smoking Status")
print("-" * 70)

smoking_missing = df['smoking_status'].isnull().sum()
print(f"\nMissing smoking_status: {smoking_missing} ({smoking_missing / len(df) * 100:.2f}%)")
print("Decision: Create 'Unknown' category (30% missing exceeds imputation threshold)")

# VISUALIZATION: Before Cleaning
smoking_before = df['smoking_status'].value_counts(dropna=False)

# Apply cleaning
df['smoking_status'] = df['smoking_status'].fillna('Unknown')

# After cleaning
smoking_after = df['smoking_status'].value_counts()

print(f"✓ Created 'Unknown' category for {smoking_missing} missing values")

# VISUALIZATION: Before vs After
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# BEFORE
axes[0].bar(range(len(smoking_before)), smoking_before.values,
            color='coral', edgecolor='black', linewidth=1.5)
axes[0].set_xticks(range(len(smoking_before)))
axes[0].set_xticklabels(smoking_before.index, rotation=45, ha='right')
axes[0].set_title('BEFORE Cleaning\n(1,544 missing values shown as NaN)',
                  fontweight='bold', fontsize=12)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(smoking_before.values):
    axes[0].text(i, v + 30, f'{v}\n({v / len(df) * 100:.1f}%)',
                 ha='center', fontweight='bold', fontsize=9)

# AFTER
axes[1].bar(range(len(smoking_after)), smoking_after.values,
            color='lightgreen', edgecolor='black', linewidth=1.5)
axes[1].set_xticks(range(len(smoking_after)))
axes[1].set_xticklabels(smoking_after.index, rotation=45, ha='right')
axes[1].set_title('AFTER Cleaning\n("Unknown" category created)',
                  fontweight='bold', fontsize=12)
axes[1].set_ylabel('Count', fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

for i, v in enumerate(smoking_after.values):
    axes[1].text(i, v + 30, f'{v}\n({v / len(df) * 100:.1f}%)',
                 ha='center', fontweight='bold', fontsize=9)

plt.suptitle('Smoking Status: Data Cleaning Impact', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('03_smoking_status_cleaning.png', dpi=300, bbox_inches='tight')
plt.show()
print("📊 Saved: 03_smoking_status_cleaning.png")

# 3.2 Handle Missing BMI
print("\n" + "-" * 70)
print("3.2 Cleaning: BMI Imputation")
print("-" * 70)

bmi_missing = df['bmi'].isnull().sum()
print(f"\nMissing BMI: {bmi_missing} ({bmi_missing / len(df) * 100:.2f}%)")
print("Decision: Impute using age-gender group medians")

# Store original BMI for comparison
bmi_before = df['bmi'].copy()

# Create age groups
df['age_group_temp'] = pd.cut(df['age'],
                              bins=[0, 18, 35, 50, 65, 100],
                              labels=['0-18', '19-35', '36-50', '51-65', '65+'])

# Calculate medians by group
bmi_medians = df.groupby(['age_group_temp', 'gender'], observed=False)['bmi'].median()

print("\nMedian BMI by Age Group and Gender:")
print(bmi_medians)


# Imputation function
def impute_bmi(row):
    if pd.isnull(row['bmi']):
        try:
            return bmi_medians.loc[(row['age_group_temp'], row['gender'])]
        except KeyError:
            return df['bmi'].median()
    return row['bmi']


# Apply imputation
df['bmi'] = df.apply(impute_bmi, axis=1)
df = df.drop('age_group_temp', axis=1)

print(f"\n✓ Imputed {bmi_missing} BMI values using age-gender group medians")

# VISUALIZATION: BMI Distribution Before vs After
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# BEFORE
axes[0].hist(bmi_before.dropna(), bins=30, color='coral',
             edgecolor='black', alpha=0.7, linewidth=1)
axes[0].set_title(f'BEFORE Imputation\n({bmi_missing} missing values excluded)',
                  fontweight='bold')
axes[0].set_xlabel('BMI')
axes[0].set_ylabel('Frequency')
axes[0].grid(axis='y', alpha=0.3)
axes[0].axvline(bmi_before.median(), color='red', linestyle='--',
                linewidth=2, label=f'Median: {bmi_before.median():.1f}')
axes[0].legend()

# AFTER
axes[1].hist(df['bmi'], bins=30, color='lightgreen',
             edgecolor='black', alpha=0.7, linewidth=1)
axes[1].set_title('AFTER Imputation\n(All values present)', fontweight='bold')
axes[1].set_xlabel('BMI')
axes[1].set_ylabel('Frequency')
axes[1].grid(axis='y', alpha=0.3)
axes[1].axvline(df['bmi'].median(), color='red', linestyle='--',
                linewidth=2, label=f'Median: {df["bmi"].median():.1f}')
axes[1].legend()

plt.suptitle('BMI Distribution: Imputation Impact', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('04_bmi_imputation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("📊 Saved: 04_bmi_imputation_comparison.png")

# 3.3 Outlier Decision - Glucose & BMI
print("\n" + "-" * 70)
print("3.3 Outlier Handling Decision")
print("-" * 70)

print("\nGlucose Outliers:")
Q1_glucose = df['avg_glucose_level'].quantile(0.25)
Q3_glucose = df['avg_glucose_level'].quantile(0.75)
IQR_glucose = Q3_glucose - Q1_glucose
glucose_outliers = df[(df['avg_glucose_level'] < Q1_glucose - 1.5 * IQR_glucose) |
                      (df['avg_glucose_level'] > Q3_glucose + 1.5 * IQR_glucose)]

print(f"  Count: {len(glucose_outliers)} ({len(glucose_outliers) / len(df) * 100:.2f}%)")
print(f"  Range: {df['avg_glucose_level'].min():.2f} - {df['avg_glucose_level'].max():.2f} mg/dL")
print("  Decision: KEEP (clinically valid - diabetic patients can have 200-300+ mg/dL)")

print("\nBMI Outliers:")
Q1_bmi = df['bmi'].quantile(0.25)
Q3_bmi = df['bmi'].quantile(0.75)
IQR_bmi = Q3_bmi - Q1_bmi
bmi_outliers = df[(df['bmi'] < Q1_bmi - 1.5 * IQR_bmi) |
                  (df['bmi'] > Q3_bmi + 1.5 * IQR_bmi)]

print(f"  Count: {len(bmi_outliers)} ({len(bmi_outliers) / len(df) * 100:.2f}%)")
print(f"  Range: {df['bmi'].min():.2f} - {df['bmi'].max():.2f}")
print("  Decision: KEEP (medically valid - obesity range documented up to 100+)")

print("\n✓ All outliers retained as clinically valid extreme values")

# CLEANING SUMMARY

print("\n" + "=" * 70)
print("STEP 4: DATA CLEANING SUMMARY")
print("=" * 70)

print(f"\nOriginal dataset: {len(df_original)} rows × {df_original.shape[1]} columns")
print(f"Cleaned dataset: {len(df)} rows × {df.shape[1]} columns")
print(f"Rows removed: {len(df_original) - len(df)}")

print("\n✓ Cleaning actions performed:")
print(f"  1. Smoking status: Created 'Unknown' category ({smoking_missing} values)")
print(f"  2. BMI: Imputed using age-gender medians ({bmi_missing} values)")
print(f"  3. Outliers: Retained all as medically valid")

print("\n✓ Final data quality:")
missing_final = df.isnull().sum().sum()
print(f"  Missing values: {missing_final}")
print(f"  Duplicates: {df.duplicated().sum()}")
print(f"  Completeness: {((df.size - missing_final) / df.size * 100):.2f}%")

if missing_final == 0:
    print("\n✅ Dataset is now 100% complete and ready for analysis!")

# VISUALIZATION: Summary of Changes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Missing values comparison
ax1 = axes[0, 0]
comparison_data = pd.DataFrame({
    'Before': df_original.isnull().sum(),
    'After': df.isnull().sum()
})
comparison_data = comparison_data[comparison_data['Before'] > 0]
comparison_data.plot(kind='bar', ax=ax1, color=['coral', 'lightgreen'],
                     edgecolor='black', linewidth=1.5)
ax1.set_title('Missing Values: Before vs After', fontweight='bold')
ax1.set_ylabel('Count')
ax1.set_xticklabels(comparison_data.index, rotation=45, ha='right')
ax1.legend(['Before Cleaning', 'After Cleaning'])
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Data completeness
ax2 = axes[0, 1]
completeness_before = ((df_original.size - df_original.isnull().sum().sum()) / df_original.size * 100)
completeness_after = ((df.size - df.isnull().sum().sum()) / df.size * 100)
bars = ax2.bar(['Before', 'After'], [completeness_before, completeness_after],
               color=['coral', 'lightgreen'], edgecolor='black', linewidth=2)
ax2.set_title('Data Completeness', fontweight='bold')
ax2.set_ylabel('Percentage (%)')
ax2.set_ylim([95, 101])
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, [completeness_before, completeness_after]):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height,
             f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')

# Plot 3: Smoking status final distribution
ax3 = axes[1, 0]
smoking_final = df['smoking_status'].value_counts()
colors_smoking = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
ax3.pie(smoking_final, labels=smoking_final.index, autopct='%1.1f%%',
        colors=colors_smoking, startangle=90, textprops={'fontweight': 'bold'})
ax3.set_title('Final Smoking Status Distribution\n(with "Unknown" category)',
              fontweight='bold')

# Plot 4: BMI distribution final
ax4 = axes[1, 1]
ax4.hist(df['bmi'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
ax4.set_title('Final BMI Distribution\n(after imputation)', fontweight='bold')
ax4.set_xlabel('BMI')
ax4.set_ylabel('Frequency')
ax4.axvline(df['bmi'].median(), color='red', linestyle='--', linewidth=2,
            label=f'Median: {df["bmi"].median():.1f}')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('Data Cleaning Summary Dashboard', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('05_cleaning_summary_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n📊 Saved: 05_cleaning_summary_dashboard.png")

# Save cleaned data
df.to_csv('stroke_data_cleaned.csv', index=False)
print("\n💾 Cleaned dataset saved as: 'stroke_data_cleaned.csv'")

print("\n" + "=" * 70)
print("DATA QUALITY ASSESSMENT & CLEANING COMPLETE!")
print("=" * 70)
print("\nGenerated visualizations:")
print("  1. 01_missing_data_analysis.png")
print("  2. 02_outlier_analysis_boxplots.png")
print("  3. 03_smoking_status_cleaning.png")
print("  4. 04_bmi_imputation_comparison.png")
print("  5. 05_cleaning_summary_dashboard.png")
print("\nNext step: Data Exploration (Part A.3)")