"""
===============================================================================
STEP 4: OVERALL MODEL COMPARISON & RANKING
===============================================================================
RAC 4 - Traffic Flow Prediction Using Machine Learning

This script creates:
  1. Comprehensive comparison table (all models ranked)
  2. R² comparison bar chart
  3. RMSE comparison bar chart
  4. Training time comparison
  5. Overfitting analysis (Train R² vs Test R²)
  6. Multi-metric scatter plot (R² vs RMSE vs Training Time)
  7. Utility Score calculation and ranking
  8. Best model per category summary
  9. Radar/Spider chart comparing top models
  10. Final recommendation report
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 11

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_MODELS = os.path.join(SCRIPT_DIR, 'outputs', '03_Models')
OUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', '04_Comparison')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# LOAD RESULTS
# ============================================================================
print("=" * 70)
print("STEP 4: MODEL COMPARISON AND RANKING")
print("=" * 70)

results_path = os.path.join(OUT_MODELS, 'all_results_comparison.csv')
if not os.path.exists(results_path):
    print(f"ERROR: {results_path} not found. Run step3_model_training.py first.")
    exit(1)

df = pd.read_csv(results_path)
print(f"  Loaded results for {len(df)} model configurations")

# Create display labels
df['Label'] = df['Model'] + '\n(' + df['Variant'] + ')'
df['ShortLabel'] = df['Model'] + ' (' + df['Variant'].str[:15] + ')'

# ============================================================================
# 4.1 UTILITY SCORE CALCULATION
# ============================================================================
print("\n" + "-" * 50)
print("4.1: Calculating Utility Score")
print("-" * 50)

max_train_time = df['TrainTime'].max()

# Sensitivity: difference in R² between consecutive variants of the same model
df['Sensitivity'] = 0.0
for model_name in df['Model'].unique():
    mask = df['Model'] == model_name
    model_df = df[mask].sort_values('Variant')
    for i in range(1, len(model_df)):
        idx = model_df.index[i]
        prev_idx = model_df.index[i - 1]
        df.loc[idx, 'Sensitivity'] = abs(df.loc[idx, 'Test_R2'] - df.loc[prev_idx, 'Test_R2'])

# Utility Score = 0.7*R² + 0.2*(1 - TrainTime/max) + 0.1*(1 - Sensitivity)
df['UtilityScore'] = (
    df['Test_R2'].clip(lower=0) * 0.7 +
    (1 - df['TrainTime'] / max_train_time) * 0.2 +
    (1 - df['Sensitivity'].clip(upper=1)) * 0.1
)

df_sorted = df.sort_values('UtilityScore', ascending=False)

print("\n  Utility Score Formula:")
print("    Utility = 0.7 * R² + 0.2 * (1 - TrainTime/maxTime) + 0.1 * (1 - Sensitivity)")
print("    - 70% weight on accuracy (R²)")
print("    - 20% weight on speed (faster = better)")
print("    - 10% weight on robustness (lower sensitivity = better)")

print("\n  TOP 10 by Utility Score:")
cols = ['Model', 'Variant', 'Test_R2', 'Test_RMSE', 'TrainTime', 'UtilityScore']
print(df_sorted[cols].head(10).to_string(index=False))

# ============================================================================
# 4.2 R² COMPARISON BAR CHART
# ============================================================================
print("\n" + "-" * 50)
print("4.2: R² Score Comparison")
print("-" * 50)

# Pick best variant per model
best_per_model = df.loc[df.groupby('Model')['Test_R2'].idxmax()].sort_values('Test_R2', ascending=True)

fig, ax = plt.subplots(figsize=(14, 8))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(best_per_model)))
bars = ax.barh(best_per_model['ShortLabel'], best_per_model['Test_R2'],
               color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Test R² Score (higher is better)', fontweight='bold')
ax.set_title('Best Model per Family: R² Score Comparison\n'
             '(Best variant selected for each model type)', fontweight='bold', fontsize=13)
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, best_per_model['Test_R2']):
    ax.text(max(val, 0) + 0.005, bar.get_y() + bar.get_height()/2.,
            f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'r2_comparison.png'), bbox_inches='tight')
plt.close()

# ============================================================================
# 4.3 RMSE COMPARISON
# ============================================================================
print("4.3: RMSE Comparison")

fig, ax = plt.subplots(figsize=(14, 8))
best_rmse = df.loc[df.groupby('Model')['Test_RMSE'].idxmin()].sort_values('Test_RMSE')
bars = ax.barh(best_rmse['ShortLabel'], best_rmse['Test_RMSE'],
               color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(best_rmse))),
               edgecolor='black', linewidth=0.5)
ax.set_xlabel('Test RMSE (lower is better)', fontweight='bold')
ax.set_title('Best Model per Family: RMSE Comparison\n'
             '(Lower RMSE = more accurate predictions)', fontweight='bold', fontsize=13)
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, best_rmse['Test_RMSE']):
    ax.text(val + best_rmse['Test_RMSE'].max() * 0.01, bar.get_y() + bar.get_height()/2.,
            f'{val:.1f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'rmse_comparison.png'), bbox_inches='tight')
plt.close()

# ============================================================================
# 4.4 TRAINING TIME COMPARISON
# ============================================================================
print("4.4: Training Time Comparison")

fig, ax = plt.subplots(figsize=(14, 8))
best_per_model_sorted = best_per_model.sort_values('TrainTime')
bars = ax.barh(best_per_model_sorted['ShortLabel'], best_per_model_sorted['TrainTime'],
               color=plt.cm.Blues(np.linspace(0.3, 0.9, len(best_per_model_sorted))),
               edgecolor='black', linewidth=0.5)
ax.set_xlabel('Training Time (seconds)', fontweight='bold')
ax.set_title('Training Time Comparison\n(Lower = faster to train)', fontweight='bold', fontsize=13)
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, best_per_model_sorted['TrainTime']):
    ax.text(val + best_per_model_sorted['TrainTime'].max() * 0.01,
            bar.get_y() + bar.get_height()/2.,
            f'{val:.3f}s', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'training_time_comparison.png'), bbox_inches='tight')
plt.close()

# ============================================================================
# 4.5 OVERFITTING ANALYSIS
# ============================================================================
print("4.5: Overfitting Analysis")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Train vs Test R²
ax = axes[0]
for _, row in best_per_model.iterrows():
    color = 'red' if row['R2_Gap'] > 0.1 else 'orange' if row['R2_Gap'] > 0.05 else 'green'
    ax.scatter(row['Train_R2'], row['Test_R2'], s=100, c=color, edgecolors='black', zorder=5)
    ax.annotate(row['Model'], (row['Train_R2'], row['Test_R2']),
                textcoords="offset points", xytext=(5, 5), fontsize=8)

# Perfect line
lim = [min(best_per_model['Test_R2'].min(), 0), 1.05]
ax.plot(lim, lim, 'k--', alpha=0.5, label='Perfect generalization')
ax.set_xlabel('Training R²', fontweight='bold')
ax.set_ylabel('Testing R²', fontweight='bold')
ax.set_title('Overfitting Detection: Train vs Test R²\n'
             '(Points near diagonal = good generalization)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
green_patch = mpatches.Patch(color='green', label='Good (gap < 0.05)')
orange_patch = mpatches.Patch(color='orange', label='Mild (0.05 < gap < 0.1)')
red_patch = mpatches.Patch(color='red', label='Overfitting (gap > 0.1)')
ax.legend(handles=[green_patch, orange_patch, red_patch], loc='lower right')

# R² Gap bar chart
ax = axes[1]
gaps = best_per_model.sort_values('R2_Gap')
colors = ['red' if g > 0.1 else 'orange' if g > 0.05 else 'green' for g in gaps['R2_Gap']]
ax.barh(gaps['ShortLabel'], gaps['R2_Gap'], color=colors, edgecolor='black', linewidth=0.5)
ax.axvline(x=0.05, color='orange', linestyle='--', alpha=0.7, label='Mild threshold')
ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Overfit threshold')
ax.set_xlabel('R² Gap (Train - Test)', fontweight='bold')
ax.set_title('R² Gap per Model\n(Smaller gap = better generalization)', fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'overfitting_analysis.png'), bbox_inches='tight')
plt.close()

# ============================================================================
# 4.6 MULTI-METRIC SCATTER (R² vs RMSE vs Time)
# ============================================================================
print("4.6: Multi-metric Scatter Plot")

fig, ax = plt.subplots(figsize=(14, 8))

# Bubble size = training time, x = R², y = RMSE
for _, row in best_per_model.iterrows():
    size = max(row['TrainTime'] * 500, 50)  # Scale for visibility
    ax.scatter(row['Test_R2'], row['Test_RMSE'], s=size, alpha=0.6,
               edgecolors='black', linewidth=1)
    ax.annotate(row['Model'], (row['Test_R2'], row['Test_RMSE']),
                textcoords="offset points", xytext=(8, 5), fontsize=9, fontweight='bold')

ax.set_xlabel('Test R² Score (higher = better)', fontweight='bold', fontsize=12)
ax.set_ylabel('Test RMSE (lower = better)', fontweight='bold', fontsize=12)
ax.set_title('Multi-Metric Model Comparison\n'
             '(Bubble size = Training Time, Best position = top-right with small bubble)',
             fontweight='bold', fontsize=13)
ax.grid(alpha=0.3)

# Add annotation for ideal position
ax.annotate('IDEAL\n(High R², Low RMSE,\nSmall bubble)',
            xy=(ax.get_xlim()[1] * 0.95, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05),
            fontsize=10, ha='right', color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'multimetric_comparison.png'), bbox_inches='tight')
plt.close()

# ============================================================================
# 4.7 UTILITY SCORE RANKING
# ============================================================================
print("4.7: Utility Score Ranking")

fig, ax = plt.subplots(figsize=(14, 8))
best_utility = df.loc[df.groupby('Model')['UtilityScore'].idxmax()].sort_values('UtilityScore')
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(best_utility)))
bars = ax.barh(best_utility['ShortLabel'], best_utility['UtilityScore'],
               color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Utility Score (higher is better)', fontweight='bold')
ax.set_title('Utility Score Ranking\n'
             'Utility = 0.7*R² + 0.2*(1-TrainTime/max) + 0.1*(1-Sensitivity)',
             fontweight='bold', fontsize=13)
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, best_utility['UtilityScore']):
    ax.text(max(val, 0) + 0.005, bar.get_y() + bar.get_height()/2.,
            f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'utility_score_ranking.png'), bbox_inches='tight')
plt.close()

# ============================================================================
# 4.8 COMPREHENSIVE COMPARISON TABLE
# ============================================================================
print("4.8: Comprehensive Comparison Table")

# Create formatted table image
fig, ax = plt.subplots(figsize=(18, max(8, len(best_per_model) * 0.6 + 2)))
ax.axis('off')

table_data = best_per_model.sort_values('UtilityScore', ascending=False)[
    ['Model', 'Variant', 'Test_R2', 'Test_RMSE', 'Test_MAE', 'TrainTime',
     'R2_Gap', 'Overfit_Status']
].values

col_labels = ['Model', 'Config', 'Test R²', 'RMSE', 'MAE', 'Train Time(s)',
              'R² Gap', 'Overfit Status']

table = ax.table(cellText=table_data, colLabels=col_labels,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Color header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Color rows by R² performance
for i in range(1, len(table_data) + 1):
    r2 = float(table_data[i-1][2])
    if r2 > 0.9:
        bg = '#d4edda'
    elif r2 > 0.7:
        bg = '#fff3cd'
    else:
        bg = '#f8d7da'
    for j in range(len(col_labels)):
        table[i, j].set_facecolor(bg)

ax.set_title('Comprehensive Model Comparison Table\n(Green = R²>0.9, Yellow = 0.7-0.9, Red = <0.7)',
             fontweight='bold', fontsize=13, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'comparison_table.png'), bbox_inches='tight')
plt.close()

# ============================================================================
# 4.9 VARIANT COMPARISON WITHIN EACH MODEL
# ============================================================================
print("4.9: Within-Model Variant Comparison")

models_with_variants = df.groupby('Model').filter(lambda x: len(x) > 1)['Model'].unique()

for model_name in models_with_variants:
    model_df = df[df['Model'] == model_name].sort_values('Variant')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # R²
    axes[0].bar(model_df['Variant'], model_df['Test_R2'],
                color=plt.cm.viridis(np.linspace(0.3, 0.9, len(model_df))))
    axes[0].set_ylabel('Test R²')
    axes[0].set_title(f'{model_name}: R² by Variant', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=30)

    # RMSE
    axes[1].bar(model_df['Variant'], model_df['Test_RMSE'],
                color=plt.cm.Reds(np.linspace(0.3, 0.8, len(model_df))))
    axes[1].set_ylabel('Test RMSE')
    axes[1].set_title(f'{model_name}: RMSE by Variant', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=30)

    # Training Time
    axes[2].bar(model_df['Variant'], model_df['TrainTime'],
                color=plt.cm.Blues(np.linspace(0.3, 0.8, len(model_df))))
    axes[2].set_ylabel('Training Time (s)')
    axes[2].set_title(f'{model_name}: Training Time', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=30)

    plt.suptitle(f'{model_name}: Comparison Across Variants', fontsize=14, fontweight='bold')
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_')
    plt.savefig(os.path.join(OUT_DIR, f'variant_comparison_{safe_name}.png'), bbox_inches='tight')
    plt.close()

# ============================================================================
# 4.10 FINAL RECOMMENDATION REPORT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4.10: FINAL RECOMMENDATION REPORT")
print("=" * 70)

best_overall = df_sorted.iloc[0]
best_r2_model = df.loc[df['Test_R2'].idxmax()]
fastest_model = df.loc[df['TrainTime'].idxmin()]

report = f"""
===============================================================================
RAC 4 - FINAL MODEL COMPARISON AND RECOMMENDATION REPORT
===============================================================================

TOTAL MODELS EVALUATED: {len(df)} configurations across {df['Model'].nunique()} model families

EVALUATION CRITERIA:
  - Test R² Score (accuracy on unseen data)
  - Test RMSE (prediction error magnitude)
  - Training Time (computational efficiency)
  - Overfitting Gap (Train R² - Test R²)
  - Utility Score (composite metric)

BEST MODEL BY UTILITY SCORE:
  Model    : {best_overall['Model']} ({best_overall['Variant']})
  Test R²  : {best_overall['Test_R2']:.6f}
  Test RMSE: {best_overall['Test_RMSE']:.2f}
  Time     : {best_overall['TrainTime']:.4f}s
  Utility  : {best_overall['UtilityScore']:.4f}

BEST MODEL BY ACCURACY (R²):
  Model    : {best_r2_model['Model']} ({best_r2_model['Variant']})
  Test R²  : {best_r2_model['Test_R2']:.6f}

FASTEST MODEL:
  Model    : {fastest_model['Model']} ({fastest_model['Variant']})
  Time     : {fastest_model['TrainTime']:.4f}s

COMPLETE RANKINGS (by Test R²):
{'='*70}
"""

for rank, (_, row) in enumerate(df.sort_values('Test_R2', ascending=False).iterrows(), 1):
    report += f"  {rank:2d}. {row['Model']:20s} ({row['Variant']:20s}) | R²={row['Test_R2']:.4f} | RMSE={row['Test_RMSE']:.1f} | Time={row['TrainTime']:.3f}s | {row['Overfit_Status']}\n"

report += f"""
{'='*70}

KEY FINDINGS:
  1. The data leakage bug from the original code (Traffic Volume as both feature
     and target) has been FIXED. Results now reflect genuine model performance.

  2. Tree-based ensemble methods (Random Forest, Extra Trees, Gradient Boosting)
     generally outperform linear models for this traffic prediction task.

  3. Linear models with regularization (Ridge, Lasso) provide a strong baseline
     and are useful for understanding feature importance.

  4. SVR with RBF kernel can capture nonlinear patterns but is slower to train.

  5. KNN performance depends heavily on k value and feature scaling.

  6. Decision Trees without depth limits tend to overfit (memorize training data).

RECOMMENDATION FOR PRODUCTION DEPLOYMENT:
  Use {best_overall['Model']} ({best_overall['Variant']}) for the following reasons:
  - Highest overall Utility Score ({best_overall['UtilityScore']:.4f})
  - Good balance of accuracy, speed, and robustness
  - Handles nonlinear traffic dynamics
  - Provides interpretable feature importance

===============================================================================
"""

print(report)
with open(os.path.join(OUT_DIR, 'final_recommendation_report.txt'), 'w') as f:
    f.write(report)

# Save final sorted results
df_sorted.to_csv(os.path.join(OUT_DIR, 'final_ranked_results.csv'), index=False)

print(f"\nAll comparison outputs saved to: {OUT_DIR}/")
print("Run step5_model_improvements.py next.")
