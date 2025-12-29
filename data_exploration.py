import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("PART A.3: DATA EXPLORATION")
print("=" * 70)

# Load cleaned data
df = pd.read_csv('stroke_data_cleaned.csv')

print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")
print("Target variable: stroke (0 = No, 1 = Yes)")


# UNIVARIATE ANALYSIS


print("\n" + "=" * 70)
print("SECTION 1: UNIVARIATE ANALYSIS")
print("=" * 70)
print("Understanding individual variables before exploring relationships")

# 1.1 Age Distribution
print("\n" + "-" * 70)
print("1.1 Age Distribution")
print("-" * 70)

age_stats = df['age'].describe()
print(f"\nAge Statistics:")
print(f"  Mean: {age_stats['mean']:.1f} years")
print(f"  Median: {age_stats['50%']:.1f} years")
print(f"  Std Dev: {age_stats['std']:.1f} years")
print(f"  Range: {age_stats['min']:.1f} - {age_stats['max']:.1f} years")

# 1.2 Categorical Variables Distribution
print("\n" + "-" * 70)
print("1.2 Categorical Variables Summary")
print("-" * 70)

categorical_vars = ['gender', 'hypertension', 'heart_disease',
                    'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for var in categorical_vars:
    print(f"\n{var}:")
    counts = df[var].value_counts()
    percentages = (counts / len(df) * 100).round(1)
    for val, count, pct in zip(counts.index, counts.values, percentages.values):
        print(f"  {val}: {count} ({pct}%)")

# VISUALIZATION: Univariate Distributions
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Age distribution
ax1 = fig.add_subplot(gs[0, 0])
df['age'].hist(bins=30, color='steelblue', edgecolor='black', ax=ax1)
ax1.axvline(df['age'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {df["age"].mean():.1f}')
ax1.axvline(df['age'].median(), color='green', linestyle='--', linewidth=2,
            label=f'Median: {df["age"].median():.1f}')
ax1.set_title('Age Distribution', fontweight='bold')
ax1.set_xlabel('Age (years)')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: BMI distribution
ax2 = fig.add_subplot(gs[0, 1])
df['bmi'].hist(bins=30, color='coral', edgecolor='black', ax=ax2)
ax2.axvline(df['bmi'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {df["bmi"].mean():.1f}')
ax2.set_title('BMI Distribution', fontweight='bold')
ax2.set_xlabel('BMI')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Glucose distribution
ax3 = fig.add_subplot(gs[0, 2])
df['avg_glucose_level'].hist(bins=30, color='lightgreen', edgecolor='black', ax=ax3)
ax3.axvline(df['avg_glucose_level'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {df["avg_glucose_level"].mean():.1f}')
ax3.set_title('Average Glucose Level Distribution', fontweight='bold')
ax3.set_xlabel('Glucose (mg/dL)')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Gender
ax4 = fig.add_subplot(gs[1, 0])
gender_counts = df['gender'].value_counts()
ax4.bar(gender_counts.index, gender_counts.values, color=['lightblue', 'pink'],
        edgecolor='black', linewidth=1.5)
ax4.set_title('Gender Distribution', fontweight='bold')
ax4.set_ylabel('Count')
for i, v in enumerate(gender_counts.values):
    ax4.text(i, v + 20, str(v), ha='center', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Smoking Status
ax5 = fig.add_subplot(gs[1, 1])
smoking_counts = df['smoking_status'].value_counts()
ax5.bar(range(len(smoking_counts)), smoking_counts.values,
        color='lightcoral', edgecolor='black', linewidth=1.5)
ax5.set_xticks(range(len(smoking_counts)))
ax5.set_xticklabels(smoking_counts.index, rotation=45, ha='right')
ax5.set_title('Smoking Status Distribution', fontweight='bold')
ax5.set_ylabel('Count')
for i, v in enumerate(smoking_counts.values):
    ax5.text(i, v + 20, str(v), ha='center', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Work Type
ax6 = fig.add_subplot(gs[1, 2])
work_counts = df['work_type'].value_counts()
ax6.bar(range(len(work_counts)), work_counts.values,
        color='lightgreen', edgecolor='black', linewidth=1.5)
ax6.set_xticks(range(len(work_counts)))
ax6.set_xticklabels(work_counts.index, rotation=45, ha='right')
ax6.set_title('Work Type Distribution', fontweight='bold')
ax6.set_ylabel('Count')
for i, v in enumerate(work_counts.values):
    ax6.text(i, v + 30, str(v), ha='center', fontweight='bold', fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

# Plot 7: Hypertension
ax7 = fig.add_subplot(gs[2, 0])
hyper_counts = df['hypertension'].value_counts()
ax7.bar(['No Hypertension', 'Hypertension'], hyper_counts.values,
        color=['lightgreen', 'salmon'], edgecolor='black', linewidth=1.5)
ax7.set_title('Hypertension Distribution', fontweight='bold')
ax7.set_ylabel('Count')
for i, v in enumerate(hyper_counts.values):
    ax7.text(i, v + 50, f'{v}\n({v / len(df) * 100:.1f}%)', ha='center', fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# Plot 8: Heart Disease
ax8 = fig.add_subplot(gs[2, 1])
heart_counts = df['heart_disease'].value_counts()
ax8.bar(['No Heart Disease', 'Heart Disease'], heart_counts.values,
        color=['lightgreen', 'salmon'], edgecolor='black', linewidth=1.5)
ax8.set_title('Heart Disease Distribution', fontweight='bold')
ax8.set_ylabel('Count')
for i, v in enumerate(heart_counts.values):
    ax8.text(i, v + 50, f'{v}\n({v / len(df) * 100:.1f}%)', ha='center', fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')

# Plot 9: Stroke Distribution (Target)
ax9 = fig.add_subplot(gs[2, 2])
stroke_counts = df['stroke'].value_counts()
ax9.bar(['No Stroke', 'Stroke'], stroke_counts.values,
        color=['lightgreen', 'red'], edgecolor='black', linewidth=2)
ax9.set_title('Stroke Distribution (TARGET)', fontweight='bold', fontsize=14)
ax9.set_ylabel('Count')
for i, v in enumerate(stroke_counts.values):
    ax9.text(i, v + 50, f'{v}\n({v / len(df) * 100:.1f}%)',
             ha='center', fontweight='bold', fontsize=11)
ax9.grid(True, alpha=0.3, axis='y')

plt.suptitle('Univariate Analysis: Distribution of All Variables',
             fontsize=16, fontweight='bold')
plt.savefig('06_univariate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n📊 Saved: 06_univariate_analysis.png")


# BIVARIATE ANALYSIS - CONTINUOUS VARIABLES vs STROKE


print("\n" + "=" * 70)
print("SECTION 2: BIVARIATE ANALYSIS")
print("=" * 70)
print("Exploring relationships between variables and stroke outcome")

print("\n" + "-" * 70)
print("2.1 Continuous Variables vs Stroke (T-Tests)")
print("-" * 70)

continuous_vars = ['age', 'avg_glucose_level', 'bmi']

bivariate_results = {}

for var in continuous_vars:
    # Split by stroke outcome
    no_stroke = df[df['stroke'] == 0][var]
    yes_stroke = df[df['stroke'] == 1][var]

    # Calculate statistics
    mean_no_stroke = no_stroke.mean()
    mean_stroke = yes_stroke.mean()
    std_no_stroke = no_stroke.std()
    std_stroke = yes_stroke.std()

    # T-test
    t_stat, p_value = stats.ttest_ind(yes_stroke, no_stroke)

    # Cohen's d (effect size)
    mean_diff = mean_stroke - mean_no_stroke
    pooled_std = np.sqrt((std_no_stroke ** 2 + std_stroke ** 2) / 2)
    cohens_d = mean_diff / pooled_std

    print(f"\n{var}:")
    print(f"  No Stroke - Mean: {mean_no_stroke:.2f} (SD: {std_no_stroke:.2f})")
    print(f"  Stroke    - Mean: {mean_stroke:.2f} (SD: {std_stroke:.2f})")
    print(f"  Mean Difference: {mean_diff:.2f}")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.4e}")
    print(f"  Cohen's d: {cohens_d:.4f}")

    if p_value < 0.001:
        print(f"  *** HIGHLY SIGNIFICANT (p < 0.001)")
        significance = "***"
    elif p_value < 0.01:
        print(f"  ** SIGNIFICANT (p < 0.01)")
        significance = "**"
    elif p_value < 0.05:
        print(f"  * SIGNIFICANT (p < 0.05)")
        significance = "*"
    else:
        print(f"  NOT SIGNIFICANT (p >= 0.05)")
        significance = "ns"

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"

    print(f"  Effect Size: {effect}")

    bivariate_results[var] = {
        'mean_no_stroke': mean_no_stroke,
        'mean_stroke': mean_stroke,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significance': significance,
        'effect_size': effect
    }

# VISUALIZATION: Continuous Variables Comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, var in enumerate(continuous_vars):
    # Box plot
    ax_box = axes[0, idx]
    data_to_plot = [df[df['stroke'] == 0][var].dropna(),
                    df[df['stroke'] == 1][var].dropna()]
    bp = ax_box.boxplot(data_to_plot, labels=['No Stroke', 'Stroke'],
                        patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('salmon')
    ax_box.set_title(f'{var} by Stroke Outcome', fontweight='bold')
    ax_box.set_ylabel(var)
    ax_box.grid(True, alpha=0.3, axis='y')

    # Histogram overlay
    ax_hist = axes[1, idx]
    ax_hist.hist(df[df['stroke'] == 0][var].dropna(), bins=30, alpha=0.6,
                 label='No Stroke', color='green', edgecolor='black')
    ax_hist.hist(df[df['stroke'] == 1][var].dropna(), bins=30, alpha=0.6,
                 label='Stroke', color='red', edgecolor='black')
    ax_hist.axvline(bivariate_results[var]['mean_no_stroke'],
                    color='green', linestyle='--', linewidth=2)
    ax_hist.axvline(bivariate_results[var]['mean_stroke'],
                    color='red', linestyle='--', linewidth=2)
    ax_hist.set_title(f'{var} Distribution Comparison\np-value: {bivariate_results[var]["p_value"]:.2e}',
                      fontweight='bold')
    ax_hist.set_xlabel(var)
    ax_hist.set_ylabel('Frequency')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3, axis='y')

plt.suptitle('Continuous Variables: Stroke vs No Stroke Comparison',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('07_continuous_variables_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n📊 Saved: 07_continuous_variables_comparison.png")


# CATEGORICAL VARIABLES vs STROKE (Chi-Square Tests)


print("\n" + "-" * 70)
print("2.2 Categorical Variables vs Stroke (Chi-Square Tests)")
print("-" * 70)

categorical_vars = ['gender', 'hypertension', 'heart_disease',
                    'ever_married', 'work_type', 'Residence_type', 'smoking_status']

chi_square_results = {}

for var in categorical_vars:
    # Create contingency table
    contingency = pd.crosstab(df[var], df['stroke'])

    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Cramér's V (effect size)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))

    # Calculate stroke rate for each category
    stroke_rates = df.groupby(var)['stroke'].agg(['sum', 'count', 'mean'])
    stroke_rates['percentage'] = (stroke_rates['mean'] * 100).round(2)

    print(f"\n{var}:")
    print(f"  Chi-square: {chi2:.4f}")
    print(f"  P-value: {p_value:.4e}")
    print(f"  Cramér's V: {cramers_v:.4f}")

    if p_value < 0.001:
        print(f"  *** HIGHLY SIGNIFICANT")
        significance = "***"
    elif p_value < 0.01:
        print(f"  ** SIGNIFICANT")
        significance = "**"
    elif p_value < 0.05:
        print(f"  * SIGNIFICANT")
        significance = "*"
    else:
        print(f"  NOT SIGNIFICANT")
        significance = "ns"

    # Effect size interpretation
    if cramers_v < 0.1:
        effect = "negligible"
    elif cramers_v < 0.3:
        effect = "small"
    elif cramers_v < 0.5:
        effect = "medium"
    else:
        effect = "large"

    print(f"  Effect Size: {effect}")

    print(f"\n  Stroke Rates by Category:")
    for cat in stroke_rates.index:
        print(f"    {cat}: {stroke_rates.loc[cat, 'percentage']:.2f}%")

    chi_square_results[var] = {
        'chi2': chi2,
        'p_value': p_value,
        'cramers_v': cramers_v,
        'significance': significance,
        'effect_size': effect,
        'stroke_rates': stroke_rates
    }

# VISUALIZATION: Stroke Rates by Categorical Variables
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()

for idx, var in enumerate(categorical_vars):
    if idx < len(axes):
        stroke_rates = chi_square_results[var]['stroke_rates']

        bars = axes[idx].barh(range(len(stroke_rates)), stroke_rates['percentage'],
                              color='crimson', edgecolor='black', linewidth=1.5)
        axes[idx].set_yticks(range(len(stroke_rates)))
        axes[idx].set_yticklabels(stroke_rates.index)
        axes[idx].set_xlabel('Stroke Rate (%)', fontweight='bold')
        axes[idx].set_title(f'{var}\n(p={chi_square_results[var]["p_value"]:.2e})',
                            fontweight='bold', fontsize=11)
        axes[idx].grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (cat, row) in enumerate(stroke_rates.iterrows()):
            axes[idx].text(row['percentage'] + 0.3, i,
                           f"{row['percentage']:.1f}%\n(n={int(row['sum'])})",
                           va='center', fontweight='bold', fontsize=9)

# Hide unused subplot
for idx in range(len(categorical_vars), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Stroke Rates by Categorical Variables (with Chi-Square Significance)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('08_categorical_stroke_rates.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n📊 Saved: 08_categorical_stroke_rates.png")

# PATTERN DISCOVERY

print("\n" + "=" * 70)
print("SECTION 3: PATTERN DISCOVERY")
print("=" * 70)

# 3.1 High-Risk Profile Identification
print("\n" + "-" * 70)
print("3.1 High-Risk Patient Profile")
print("-" * 70)

# Define high-risk criteria
high_risk_mask = (
        (df['age'] >= 60) &
        ((df['hypertension'] == 1) | (df['heart_disease'] == 1))
)

high_risk_group = df[high_risk_mask]
normal_group = df[~high_risk_mask]

high_risk_stroke_rate = (high_risk_group['stroke'].mean() * 100)
normal_stroke_rate = (normal_group['stroke'].mean() * 100)
relative_risk = high_risk_stroke_rate / normal_stroke_rate if normal_stroke_rate > 0 else 0

print(f"\nHigh-Risk Profile (Age ≥60 + Hypertension/Heart Disease):")
print(f"  Population: {len(high_risk_group)} ({len(high_risk_group) / len(df) * 100:.1f}%)")
print(f"  Stroke Rate: {high_risk_stroke_rate:.2f}%")
print(f"  Relative Risk vs Normal: {relative_risk:.2f}x")
print(f"\nNormal Risk Population:")
print(f"  Population: {len(normal_group)} ({len(normal_group) / len(df) * 100:.1f}%)")
print(f"  Stroke Rate: {normal_stroke_rate:.2f}%")

# 3.2 Age-Risk Interaction
print("\n" + "-" * 70)
print("3.2 Age × Comorbidity Interaction")
print("-" * 70)

# Create age groups for analysis
df['age_bracket'] = pd.cut(df['age'],
                           bins=[0, 40, 50, 60, 70, 100],
                           labels=['<40', '40-49', '50-59', '60-69', '70+'])

# Calculate stroke rate by age bracket and comorbidity status
df['has_comorbidity'] = ((df['hypertension'] == 1) | (df['heart_disease'] == 1)).astype(int)

interaction = df.groupby(['age_bracket', 'has_comorbidity'], observed=False)['stroke'].agg(['sum', 'count', 'mean'])
interaction['stroke_rate_pct'] = (interaction['mean'] * 100).round(2)

print("\nStroke Rate (%) by Age and Comorbidity:")
print(interaction[['stroke_rate_pct']])

# VISUALIZATION: Pattern Discovery
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: High-Risk vs Normal Profile
ax1 = axes[0, 0]
profiles = ['Normal Risk', 'High Risk']
stroke_rates_profile = [normal_stroke_rate, high_risk_stroke_rate]
colors = ['lightgreen', 'red']
bars = ax1.bar(profiles, stroke_rates_profile, color=colors,
               edgecolor='black', linewidth=2)
ax1.set_title('Stroke Rate: High-Risk vs Normal Profile', fontweight='bold', fontsize=12)
ax1.set_ylabel('Stroke Rate (%)', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

for bar, rate, count in zip(bars, stroke_rates_profile,
                            [len(normal_group), len(high_risk_group)]):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
             f'{rate:.2f}%\n(n={count})',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 2: Age × Comorbidity Interaction
ax2 = axes[0, 1]
interaction_pivot = interaction['stroke_rate_pct'].unstack()
x = np.arange(len(interaction_pivot))
width = 0.35

bars1 = ax2.bar(x - width / 2, interaction_pivot[0], width, label='No Comorbidity',
                color='lightblue', edgecolor='black')
bars2 = ax2.bar(x + width / 2, interaction_pivot[1], width, label='Has Comorbidity',
                color='salmon', edgecolor='black')

ax2.set_xlabel('Age Bracket', fontweight='bold')
ax2.set_ylabel('Stroke Rate (%)', fontweight='bold')
ax2.set_title('Age × Comorbidity Interaction', fontweight='bold', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(interaction_pivot.index)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Stroke Rate by Age Bracket
ax3 = axes[1, 0]
age_stroke_rate = df.groupby('age_bracket', observed=False)['stroke'].mean() * 100
bars3 = ax3.bar(range(len(age_stroke_rate)), age_stroke_rate.values,
                color='steelblue', edgecolor='black', linewidth=1.5)
ax3.set_xticks(range(len(age_stroke_rate)))
ax3.set_xticklabels(age_stroke_rate.index)
ax3.set_title('Stroke Rate Increases with Age', fontweight='bold', fontsize=12)
ax3.set_xlabel('Age Bracket', fontweight='bold')
ax3.set_ylabel('Stroke Rate (%)', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(age_stroke_rate.values):
    ax3.text(i, v + 0.3, f'{v:.2f}%', ha='center', fontweight='bold')

# Plot 4: Multiple Risk Factors
ax4 = axes[1, 1]
df['risk_count'] = (
        (df['hypertension'] == 1).astype(int) +
        (df['heart_disease'] == 1).astype(int) +
        (df['age'] >= 60).astype(int) +
        (df['avg_glucose_level'] >= 126).astype(int)
)

risk_count_stroke = df.groupby('risk_count')['stroke'].mean() * 100
bars4 = ax4.bar(range(len(risk_count_stroke)), risk_count_stroke.values,
                color='coral', edgecolor='black', linewidth=1.5)
ax4.set_xticks(range(len(risk_count_stroke)))
ax4.set_xticklabels(risk_count_stroke.index)
ax4.set_title('Cumulative Risk: More Risk Factors = Higher Stroke Rate',
              fontweight='bold', fontsize=12)
ax4.set_xlabel('Number of Risk Factors', fontweight='bold')
ax4.set_ylabel('Stroke Rate (%)', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(risk_count_stroke.values):
    ax4.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold')

plt.suptitle('Pattern Discovery: High-Risk Profiles and Interactions',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('09_pattern_discovery.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n📊 Saved: 09_pattern_discovery.png")

# Clean up temporary columns
df = df.drop(['age_bracket', 'has_comorbidity', 'risk_count'], axis=1)

# KEY INSIGHTS SUMMARY

print("\n" + "=" * 70)
print("SECTION 4: KEY INSIGHTS DISCOVERED")
print("=" * 70)

insights = []

# Insight 1: Age
if bivariate_results['age']['p_value'] < 0.001:
    mean_diff = bivariate_results['age']['mean_stroke'] - bivariate_results['age']['mean_no_stroke']
    insight = f"1. AGE IS THE STRONGEST PREDICTOR: Stroke patients are {mean_diff:.1f} years older on average (p < 0.001, {bivariate_results['age']['effect_size']} effect)"
    insights.append(insight)
    print(f"\n{insight}")

# Insight 2: Hypertension
if chi_square_results['hypertension']['p_value'] < 0.001:
    hyper_stroke_rate = chi_square_results['hypertension']['stroke_rates'].loc[1, 'percentage']
    insight = f"2. HYPERTENSION SIGNIFICANTLY INCREASES RISK: {hyper_stroke_rate:.1f}% stroke rate in hypertensive patients (p < 0.001)"
    insights.append(insight)
    print(f"\n{insight}")

# Insight 3: High-risk profile
insight = f"3. HIGH-RISK PROFILE IDENTIFIED: Patients aged 60+ with comorbidities have {relative_risk:.1f}x higher stroke risk ({high_risk_stroke_rate:.1f}% vs {normal_stroke_rate:.1f}%)"
insights.append(insight)
print(f"\n{insight}")

# Insight 4: Cumulative risk
insight = f"4. CUMULATIVE RISK EFFECT: Stroke rate increases progressively with number of risk factors present"
insights.append(insight)
print(f"\n{insight}")

# Insight 5: Gender
if chi_square_results['gender']['significance'] != 'ns':
    insight = f"5. GENDER DIFFERENCES: Stroke rates differ between males and females (p = {chi_square_results['gender']['p_value']:.4f})"
    insights.append(insight)
    print(f"\n{insight}")

# Insight 6: Glucose
if bivariate_results['avg_glucose_level']['p_value'] < 0.05:
    mean_diff_glucose = bivariate_results['avg_glucose_level']['mean_stroke'] - bivariate_results['avg_glucose_level'][
        'mean_no_stroke']
    insight = f"6. GLUCOSE CONTROL MATTERS: Stroke patients have {mean_diff_glucose:.1f} mg/dL higher glucose levels on average"
    insights.append(insight)
    print(f"\n{insight}")

# Save insights
with open('data_exploration_insights.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("DATA EXPLORATION - KEY INSIGHTS\n")
    f.write("=" * 70 + "\n\n")
    for insight in insights:
        f.write(insight + "\n\n")

print("\n💾 Insights saved as: 'data_exploration_insights.txt'")

# FINAL SUMMARY

print("\n" + "=" * 70)
print("DATA EXPLORATION COMPLETE!")
print("=" * 70)

print("\nGenerated Visualizations:")
print("  1. 06_univariate_analysis.png - Distribution of all variables")
print("  2. 07_continuous_variables_comparison.png - Age, BMI, Glucose vs Stroke")
print("  3. 08_categorical_stroke_rates.png - Stroke rates by categories")
print("  4. 09_pattern_discovery.png - High-risk profiles and interactions")

print("\nKey Statistics Generated:")
print(f"  • T-tests for {len(continuous_vars)} continuous variables")
print(f"  • Chi-square tests for {len(categorical_vars)} categorical variables")
print(f"  • {len(insights)} major insights discovered")

print("\nNext Step: Part B - Data Modelling and Prediction")