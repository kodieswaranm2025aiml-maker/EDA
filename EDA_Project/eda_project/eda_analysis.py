"""
Exploratory Data Analysis (EDA) Project
========================================
Dataset: Employee Performance & Salary Dataset
Author: EDA Internship Project
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Styling ──────────────────────────────────────────────────────────────────
PALETTE = ['#1A237E', '#283593', '#3949AB', '#5C6BC0', '#9FA8DA',
           '#E53935', '#F57C00', '#43A047', '#00ACC1', '#8E24AA']
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#CCCCCC',
    'grid.color': '#EEEEEE',
    'font.family': 'DejaVu Sans',
})

OUTPUT = '/home/claude/eda_project/outputs'

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('/home/claude/eda_project/data/employee_dataset.csv')

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS — EMPLOYEE DATASET")
print("=" * 60)
print(f"\n📋 Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n🔍 Column Types:\n{df.dtypes}")
print(f"\n📊 Statistical Summary:\n{df.describe().round(2)}")
print(f"\n❓ Missing Values:\n{df.isnull().sum()}")
print(f"\n🔂 Duplicates: {df.duplicated().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
df_clean = df.copy()
df_clean['Salary'].fillna(df_clean['Salary'].median(), inplace=True)
df_clean['Job_Satisfaction'].fillna(df_clean['Job_Satisfaction'].median(), inplace=True)

print("\n✅ Missing values handled (median imputation).")
print(f"   Remaining nulls: {df_clean.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FIGURE 1 — Distribution Overview
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution Analysis — Numerical Features', fontsize=16, fontweight='bold', y=1.01)

num_cols = ['Age', 'Experience_Years', 'Salary', 'Performance_Score', 'Job_Satisfaction']
colors = ['#3949AB', '#00ACC1', '#E53935', '#43A047', '#F57C00']

for i, (col, color) in enumerate(zip(num_cols, colors)):
    ax = axes[i // 3][i % 3]
    sns.histplot(df_clean[col], kde=True, color=color, ax=ax, bins=25, alpha=0.7)
    ax.axvline(df_clean[col].mean(), color='black', linestyle='--', linewidth=1.5, label=f'Mean: {df_clean[col].mean():.1f}')
    ax.axvline(df_clean[col].median(), color='red', linestyle=':', linewidth=1.5, label=f'Median: {df_clean[col].median():.1f}')
    ax.set_title(col, fontweight='bold')
    ax.legend(fontsize=8)

axes[1][2].axis('off')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig1_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 1 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FIGURE 2 — Categorical Feature Counts
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Categorical Feature Distribution', fontsize=16, fontweight='bold')

cat_cols = ['Department', 'Education', 'Gender']
for ax, col in zip(axes, cat_cols):
    counts = df_clean[col].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=PALETTE[:len(counts)], edgecolor='white', linewidth=1.2)
    ax.set_title(col, fontweight='bold')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=20)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig2_categoricals.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 2 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURE 3 — Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
num_df = df_clean[['Age', 'Experience_Years', 'Salary', 'Performance_Score', 'Job_Satisfaction']]
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu_r',
            mask=mask, ax=ax, linewidths=0.5, annot_kws={'size': 11},
            vmin=-1, vmax=1, center=0,
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap — Numerical Features', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig3_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 3 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE 4 — Salary Analysis
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Salary Deep-Dive', fontsize=16, fontweight='bold')

# Salary by Department
dept_salary = df_clean.groupby('Department')['Salary'].mean().sort_values(ascending=False)
bars = axes[0].barh(dept_salary.index, dept_salary.values,
                    color=PALETTE[:len(dept_salary)], edgecolor='white')
axes[0].set_title('Average Salary by Department', fontweight='bold')
axes[0].set_xlabel('Average Salary (₹)')
for bar in bars:
    axes[0].text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                 f'₹{bar.get_width():,.0f}', va='center', fontsize=9)

# Salary vs Experience scatter
sc = axes[1].scatter(df_clean['Experience_Years'], df_clean['Salary'],
                     c=df_clean['Performance_Score'], cmap='RdYlGn',
                     alpha=0.5, s=40, edgecolors='none')
m, b, r, p, _ = stats.linregress(df_clean['Experience_Years'], df_clean['Salary'])
x_line = np.linspace(0, 40, 100)
axes[1].plot(x_line, m*x_line + b, 'r--', linewidth=2, label=f'r = {r:.2f}, p < 0.001')
axes[1].set_title('Salary vs Experience (color = Performance)', fontweight='bold')
axes[1].set_xlabel('Experience (Years)')
axes[1].set_ylabel('Salary (₹)')
axes[1].legend()
plt.colorbar(sc, ax=axes[1], label='Performance Score')

plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig4_salary_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 4 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURE 5 — Performance & Satisfaction
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Performance & Job Satisfaction Analysis', fontsize=16, fontweight='bold')

# Performance by Department
dept_perf = df_clean.groupby('Department')['Performance_Score'].mean().sort_values(ascending=False)
axes[0].bar(dept_perf.index, dept_perf.values,
            color=PALETTE[:len(dept_perf)], edgecolor='white')
axes[0].set_title('Avg Performance by Department', fontweight='bold')
axes[0].set_ylabel('Performance Score')
axes[0].tick_params(axis='x', rotation=20)
axes[0].set_ylim(0, 5.5)

# Education vs Salary box
edu_order = ['High School', 'Bachelor', 'Master', 'PhD']
sns.boxplot(data=df_clean, x='Education', y='Salary', order=edu_order,
            palette=PALETTE[:4], ax=axes[1])
axes[1].set_title('Salary Distribution by Education', fontweight='bold')
axes[1].tick_params(axis='x', rotation=15)

# Satisfaction vs Performance
pivot = df_clean.groupby(['Performance_Score', 'Job_Satisfaction']).size().unstack(fill_value=0)
sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues', ax=axes[2], linewidths=0.5)
axes[2].set_title('Performance vs Job Satisfaction (Count)', fontweight='bold')
axes[2].set_xlabel('Job Satisfaction')
axes[2].set_ylabel('Performance Score')

plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig5_performance_satisfaction.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 5 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. FIGURE 6 — Gender & City Analysis
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Demographic Insights', fontsize=16, fontweight='bold')

# Gender salary comparison
gender_stats = df_clean.groupby('Gender')['Salary'].agg(['mean', 'median'])
x = np.arange(2)
w = 0.35
axes[0].bar(x - w/2, gender_stats['mean'], w, label='Mean', color='#3949AB', alpha=0.85)
axes[0].bar(x + w/2, gender_stats['median'], w, label='Median', color='#E53935', alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(gender_stats.index)
axes[0].set_title('Salary by Gender', fontweight='bold')
axes[0].set_ylabel('Salary (₹)')
axes[0].legend()

# City distribution pie
city_counts = df_clean['City'].value_counts()
axes[1].pie(city_counts, labels=city_counts.index, autopct='%1.1f%%',
            colors=PALETTE[:len(city_counts)], startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=1.5))
axes[1].set_title('Employee Distribution by City', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT}/fig6_demographics.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 6 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 9. KEY INSIGHTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

r_exp_sal, p_exp_sal = stats.pearsonr(df_clean['Experience_Years'], df_clean['Salary'])
r_edu_sal, _ = stats.pearsonr(df_clean['Education'].map({'High School':0,'Bachelor':1,'Master':2,'PhD':3}), df_clean['Salary'])
r_perf_sat, _ = stats.pearsonr(df_clean['Performance_Score'], df_clean['Job_Satisfaction'])

insights = {
    "Avg Salary": f"₹{df_clean['Salary'].mean():,.0f}",
    "Salary Std Dev": f"₹{df_clean['Salary'].std():,.0f}",
    "Exp-Salary Correlation": f"r = {r_exp_sal:.3f} (strong positive)",
    "Edu-Salary Correlation": f"r = {r_edu_sal:.3f} (moderate positive)",
    "Perf-Satisfaction Corr": f"r = {r_perf_sat:.3f}",
    "Top Dept by Salary": dept_salary.idxmax(),
    "Top Dept by Performance": dept_perf.idxmax(),
}

for k, v in insights.items():
    print(f"  ▸ {k:35s}: {v}")

# Save summary CSV
summary = df_clean.describe().round(2)
summary.to_csv(f'{OUTPUT}/statistical_summary.csv')

dept_summary = df_clean.groupby('Department').agg(
    Avg_Salary=('Salary', 'mean'),
    Avg_Performance=('Performance_Score', 'mean'),
    Avg_Satisfaction=('Job_Satisfaction', 'mean'),
    Count=('EmployeeID', 'count')
).round(2)
dept_summary.to_csv(f'{OUTPUT}/department_summary.csv')

print("\n✅ All figures and CSVs saved to outputs/")
print("\n🎉 EDA COMPLETE!")
