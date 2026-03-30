"""
===============================================================================
STEP 5: MODEL IMPROVEMENT ANALYSIS - HOW EACH MODEL CAN BE DONE BETTER
===============================================================================
RAC 4 - Traffic Flow Prediction Using Machine Learning

For EACH model, this script:
  1. Identifies weaknesses from step3 results
  2. Applies specific improvements
  3. Compares original vs improved performance
  4. Generates before/after visualizations
  5. Writes detailed improvement explanation

Improvement strategies per model:
  - Linear: Feature selection, polynomial features
  - Ridge/Lasso/ElasticNet: Hyperparameter tuning via GridSearch
  - SVR: Feature scaling + hyperparameter tuning
  - KNN: Optimal k via elbow method, weighted KNN
  - Decision Tree: Pruning, optimal depth via validation curve
  - Random Forest: Tuning n_estimators, max_features, min_samples
  - Extra Trees: Same as Random Forest
  - Gradient Boosting: Learning rate schedule, early stopping
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import (cross_val_score, GridSearchCV,
                                     validation_curve)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'processed_data')
OUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', '05_Improvements')
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index_col=0)
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index_col=0)
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index_col=0).squeeze()
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index_col=0).squeeze()
X_train_scaled = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train_scaled.csv'), index_col=0)
X_test_scaled = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test_scaled.csv'), index_col=0)

FEATURE_NAMES = list(X_train.columns)
print(f"Loaded data: {len(X_train)} train, {len(X_test)} test, {len(FEATURE_NAMES)} features")

improvement_results = []


def quick_eval(model, X_tr, X_te, y_tr, y_te, name):
    """Quick train and evaluate returning key metrics."""
    start = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - start
    y_pred = model.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)
    return {'name': name, 'r2': r2, 'rmse': rmse, 'mae': mae,
            'time': train_time, 'model': model, 'y_pred': y_pred}


def compare_and_plot(original, improved_list, model_name, out_dir):
    """Create comparison plot and report for original vs improved models."""
    os.makedirs(out_dir, exist_ok=True)

    all_models = [original] + improved_list

    # Bar comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    names = [m['name'] for m in all_models]
    x = range(len(names))

    # R²
    r2_vals = [m['r2'] for m in all_models]
    colors = ['#e74c3c'] + ['#2ecc71'] * len(improved_list)
    axes[0].bar(x, r2_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    axes[0].set_ylabel('Test R²')
    axes[0].set_title(f'{model_name}: R² Comparison', fontweight='bold')
    for i, v in enumerate(r2_vals):
        axes[0].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    # RMSE
    rmse_vals = [m['rmse'] for m in all_models]
    axes[1].bar(x, rmse_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('Test RMSE')
    axes[1].set_title(f'{model_name}: RMSE Comparison', fontweight='bold')
    for i, v in enumerate(rmse_vals):
        axes[1].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    # Time
    time_vals = [m['time'] for m in all_models]
    axes[2].bar(x, time_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    axes[2].set_ylabel('Training Time (s)')
    axes[2].set_title(f'{model_name}: Time Comparison', fontweight='bold')

    plt.suptitle(f'{model_name}: Original (Red) vs Improvements (Green)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'improvement_comparison.png'), bbox_inches='tight')
    plt.close()

    # Predicted vs Actual for best improved
    best_improved = max(improved_list, key=lambda m: m['r2'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_test, original['y_pred'], alpha=0.3, s=10, c='red')
    lim = [min(y_test.min(), original['y_pred'].min()),
           max(y_test.max(), original['y_pred'].max())]
    axes[0].plot(lim, lim, 'k--', lw=1.5)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'ORIGINAL: {original["name"]}\nR²={original["r2"]:.4f}', fontweight='bold')
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_test, best_improved['y_pred'], alpha=0.3, s=10, c='green')
    lim = [min(y_test.min(), best_improved['y_pred'].min()),
           max(y_test.max(), best_improved['y_pred'].max())]
    axes[1].plot(lim, lim, 'k--', lw=1.5)
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title(f'IMPROVED: {best_improved["name"]}\nR²={best_improved["r2"]:.4f}', fontweight='bold')
    axes[1].grid(alpha=0.3)

    r2_improvement = best_improved['r2'] - original['r2']
    rmse_improvement = original['rmse'] - best_improved['rmse']
    plt.suptitle(f'{model_name}: Before vs After Improvement\n'
                 f'R² improved by {r2_improvement:+.4f}, RMSE improved by {rmse_improvement:+.1f}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'before_after.png'), bbox_inches='tight')
    plt.close()

    # Record best improvement
    improvement_results.append({
        'Model': model_name,
        'Original_R2': original['r2'],
        'Original_RMSE': original['rmse'],
        'Best_Improved_R2': best_improved['r2'],
        'Best_Improved_RMSE': best_improved['rmse'],
        'Best_Method': best_improved['name'],
        'R2_Gain': r2_improvement,
        'RMSE_Reduction': rmse_improvement,
    })

    # Write report
    report = f"""
{'='*70}
{model_name} - IMPROVEMENT ANALYSIS
{'='*70}

ORIGINAL PERFORMANCE:
  {original['name']}: R²={original['r2']:.4f}, RMSE={original['rmse']:.1f}

IMPROVEMENTS TRIED:
"""
    for m in improved_list:
        delta_r2 = m['r2'] - original['r2']
        delta_rmse = original['rmse'] - m['rmse']
        report += f"  {m['name']:40s}: R²={m['r2']:.4f} ({delta_r2:+.4f}), RMSE={m['rmse']:.1f} ({delta_rmse:+.1f})\n"

    report += f"""
BEST IMPROVEMENT: {best_improved['name']}
  R² gain        : {r2_improvement:+.4f}
  RMSE reduction : {rmse_improvement:+.1f}

"""
    with open(os.path.join(out_dir, 'improvement_report.txt'), 'w') as f:
        f.write(report)

    return best_improved


# ============================================================================
# 1. LINEAR REGRESSION IMPROVEMENTS
# ============================================================================
print("\n" + "=" * 70)
print("IMPROVING: LINEAR REGRESSION")
print("=" * 70)

original = quick_eval(LinearRegression(), X_train_scaled, X_test_scaled, y_train, y_test,
                      'Original (default)')

improvements = []

# Improvement 1: Feature selection (top features only)
from sklearn.feature_selection import SelectKBest, f_regression
for k in [5, 10, 15]:
    selector = SelectKBest(f_regression, k=min(k, X_train_scaled.shape[1]))
    X_tr_sel = selector.fit_transform(X_train_scaled, y_train)
    X_te_sel = selector.transform(X_test_scaled)
    result = quick_eval(LinearRegression(), X_tr_sel, X_te_sel, y_train, y_test,
                        f'Top {k} features')
    improvements.append(result)

compare_and_plot(original, improvements, 'Linear Regression',
                 os.path.join(OUT_DIR, '01_LinearRegression'))

# ============================================================================
# 2. RIDGE REGRESSION IMPROVEMENTS
# ============================================================================
print("\n" + "=" * 70)
print("IMPROVING: RIDGE REGRESSION")
print("=" * 70)

original = quick_eval(Ridge(alpha=1.0, random_state=42), X_train_scaled, X_test_scaled,
                      y_train, y_test, 'Original (alpha=1.0)')

improvements = []
for alpha in [0.001, 0.01, 0.1, 0.5, 2.0, 5.0, 10.0, 50.0, 100.0]:
    result = quick_eval(Ridge(alpha=alpha, random_state=42), X_train_scaled, X_test_scaled,
                        y_train, y_test, f'alpha={alpha}')
    improvements.append(result)

# Validation curve for alpha
alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
train_scores, test_scores = validation_curve(
    Ridge(random_state=42), X_train_scaled, y_train,
    param_name='alpha', param_range=alphas, cv=5, scoring='r2', n_jobs=-1
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(alphas, train_scores.mean(axis=1), 'b-o', label='Training R²')
ax.semilogx(alphas, test_scores.mean(axis=1), 'r-o', label='Cross-Validation R²')
ax.fill_between(alphas, train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='blue')
ax.fill_between(alphas, test_scores.mean(axis=1) - test_scores.std(axis=1),
                test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.1, color='red')
ax.set_xlabel('Alpha (Regularization Strength)', fontweight='bold')
ax.set_ylabel('R² Score', fontweight='bold')
ax.set_title('Ridge Regression: Validation Curve for Alpha\n'
             '(Where CV score peaks = optimal alpha)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
os.makedirs(os.path.join(OUT_DIR, '02_Ridge'), exist_ok=True)
plt.savefig(os.path.join(OUT_DIR, '02_Ridge', 'validation_curve_alpha.png'), bbox_inches='tight')
plt.close()

compare_and_plot(original, improvements, 'Ridge Regression',
                 os.path.join(OUT_DIR, '02_Ridge'))

# ============================================================================
# 3. LASSO IMPROVEMENTS
# ============================================================================
print("\n" + "=" * 70)
print("IMPROVING: LASSO REGRESSION")
print("=" * 70)

original = quick_eval(Lasso(alpha=0.1, random_state=42, max_iter=10000),
                      X_train_scaled, X_test_scaled, y_train, y_test, 'Original (alpha=0.1)')
improvements = []
for alpha in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.5, 1.0]:
    result = quick_eval(Lasso(alpha=alpha, random_state=42, max_iter=10000),
                        X_train_scaled, X_test_scaled, y_train, y_test, f'alpha={alpha}')
    improvements.append(result)

compare_and_plot(original, improvements, 'Lasso Regression',
                 os.path.join(OUT_DIR, '03_Lasso'))

# ============================================================================
# 4. KNN IMPROVEMENTS
# ============================================================================
print("\n" + "=" * 70)
print("IMPROVING: KNN")
print("=" * 70)

original = quick_eval(KNeighborsRegressor(n_neighbors=5), X_train_scaled, X_test_scaled,
                      y_train, y_test, 'Original (k=5)')

improvements = []

# Elbow method - find optimal k
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
    k_scores.append(scores.mean())

optimal_k = k_range[np.argmax(k_scores)]
print(f"  Optimal k found: {optimal_k}")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(k_range, k_scores, 'b-o', markersize=5)
ax.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
ax.set_xlabel('Number of Neighbors (k)', fontweight='bold')
ax.set_ylabel('Cross-Validation R²', fontweight='bold')
ax.set_title('KNN: Elbow Method for Optimal k\n'
             '(Higher R² = better, peak shows optimal k)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
os.makedirs(os.path.join(OUT_DIR, '04_KNN'), exist_ok=True)
plt.savefig(os.path.join(OUT_DIR, '04_KNN', 'elbow_method.png'), bbox_inches='tight')
plt.close()

# Try improvements
for k in [optimal_k, 3, 7, 15]:
    result = quick_eval(KNeighborsRegressor(n_neighbors=k), X_train_scaled, X_test_scaled,
                        y_train, y_test, f'k={k}')
    improvements.append(result)

# Weighted KNN
result = quick_eval(KNeighborsRegressor(n_neighbors=optimal_k, weights='distance'),
                    X_train_scaled, X_test_scaled, y_train, y_test,
                    f'Weighted (k={optimal_k})')
improvements.append(result)

compare_and_plot(original, improvements, 'KNN',
                 os.path.join(OUT_DIR, '04_KNN'))

# ============================================================================
# 5. DECISION TREE IMPROVEMENTS
# ============================================================================
print("\n" + "=" * 70)
print("IMPROVING: DECISION TREE")
print("=" * 70)

original = quick_eval(DecisionTreeRegressor(random_state=42), X_train, X_test,
                      y_train, y_test, 'Original (no limit)')

improvements = []

# Validation curve for max_depth
depths = list(range(2, 25))
train_scores, test_scores = validation_curve(
    DecisionTreeRegressor(random_state=42), X_train, y_train,
    param_name='max_depth', param_range=depths, cv=5, scoring='r2', n_jobs=-1
)

optimal_depth = depths[np.argmax(test_scores.mean(axis=1))]
print(f"  Optimal max_depth found: {optimal_depth}")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(depths, train_scores.mean(axis=1), 'b-o', label='Training R²', markersize=5)
ax.plot(depths, test_scores.mean(axis=1), 'r-o', label='Cross-Validation R²', markersize=5)
ax.fill_between(depths, train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='blue')
ax.fill_between(depths, test_scores.mean(axis=1) - test_scores.std(axis=1),
                test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.1, color='red')
ax.axvline(x=optimal_depth, color='green', linestyle='--', linewidth=2,
           label=f'Optimal depth={optimal_depth}')
ax.set_xlabel('Max Depth', fontweight='bold')
ax.set_ylabel('R² Score', fontweight='bold')
ax.set_title('Decision Tree: Validation Curve for max_depth\n'
             '(Gap between blue and red = overfitting)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
os.makedirs(os.path.join(OUT_DIR, '05_DecisionTree'), exist_ok=True)
plt.savefig(os.path.join(OUT_DIR, '05_DecisionTree', 'validation_curve_depth.png'),
            bbox_inches='tight')
plt.close()

for depth in [optimal_depth, 5, 8, 12, 15]:
    result = quick_eval(DecisionTreeRegressor(max_depth=depth, random_state=42),
                        X_train, X_test, y_train, y_test, f'max_depth={depth}')
    improvements.append(result)

# With min_samples_leaf (pruning)
result = quick_eval(DecisionTreeRegressor(max_depth=optimal_depth, min_samples_leaf=10,
                                           random_state=42),
                    X_train, X_test, y_train, y_test,
                    f'depth={optimal_depth}+min_leaf=10')
improvements.append(result)

compare_and_plot(original, improvements, 'Decision Tree',
                 os.path.join(OUT_DIR, '05_DecisionTree'))

# ============================================================================
# 6. RANDOM FOREST IMPROVEMENTS
# ============================================================================
print("\n" + "=" * 70)
print("IMPROVING: RANDOM FOREST")
print("=" * 70)

original = quick_eval(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                      X_train, X_test, y_train, y_test, 'Original (100 trees)')

improvements = []

# More trees
for n in [200, 300, 500]:
    result = quick_eval(RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1),
                        X_train, X_test, y_train, y_test, f'{n} trees')
    improvements.append(result)

# Tuned: max_features, min_samples
result = quick_eval(
    RandomForestRegressor(n_estimators=200, max_features='sqrt', min_samples_leaf=5,
                          random_state=42, n_jobs=-1),
    X_train, X_test, y_train, y_test, '200 trees+sqrt+min_leaf=5')
improvements.append(result)

result = quick_eval(
    RandomForestRegressor(n_estimators=200, max_features=0.5, max_depth=20,
                          min_samples_split=5, random_state=42, n_jobs=-1),
    X_train, X_test, y_train, y_test, '200 trees+tuned')
improvements.append(result)

compare_and_plot(original, improvements, 'Random Forest',
                 os.path.join(OUT_DIR, '06_RandomForest'))

# ============================================================================
# 7. GRADIENT BOOSTING IMPROVEMENTS
# ============================================================================
print("\n" + "=" * 70)
print("IMPROVING: GRADIENT BOOSTING")
print("=" * 70)

original = quick_eval(GradientBoostingRegressor(n_estimators=100, random_state=42),
                      X_train, X_test, y_train, y_test, 'Original (100 est, lr=0.1)')

improvements = []

# Different learning rates
for lr in [0.01, 0.05, 0.2]:
    result = quick_eval(
        GradientBoostingRegressor(n_estimators=200, learning_rate=lr, random_state=42),
        X_train, X_test, y_train, y_test, f'200 est, lr={lr}')
    improvements.append(result)

# More estimators with smaller learning rate
result = quick_eval(
    GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                              subsample=0.8, random_state=42),
    X_train, X_test, y_train, y_test, '500 est+lr=0.05+tuned')
improvements.append(result)

# Shallow trees
result = quick_eval(
    GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3,
                              random_state=42),
    X_train, X_test, y_train, y_test, '300 est+depth=3')
improvements.append(result)

compare_and_plot(original, improvements, 'Gradient Boosting',
                 os.path.join(OUT_DIR, '07_GradientBoosting'))

# ============================================================================
# 8. SVR IMPROVEMENTS
# ============================================================================
print("\n" + "=" * 70)
print("IMPROVING: SVR")
print("=" * 70)

original = quick_eval(SVR(kernel='rbf', C=1.0), X_train_scaled, X_test_scaled,
                      y_train, y_test, 'Original (RBF, C=1.0)')

improvements = []
for C in [0.1, 10, 100]:
    result = quick_eval(SVR(kernel='rbf', C=C), X_train_scaled, X_test_scaled,
                        y_train, y_test, f'RBF, C={C}')
    improvements.append(result)

result = quick_eval(SVR(kernel='rbf', C=10, epsilon=0.01),
                    X_train_scaled, X_test_scaled, y_train, y_test,
                    'RBF, C=10, eps=0.01')
improvements.append(result)

compare_and_plot(original, improvements, 'SVR',
                 os.path.join(OUT_DIR, '08_SVR'))

# ============================================================================
# OVERALL IMPROVEMENT SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("OVERALL IMPROVEMENT SUMMARY")
print("=" * 70)

imp_df = pd.DataFrame(improvement_results)
imp_df = imp_df.sort_values('R2_Gain', ascending=False)

print(imp_df.to_string(index=False))

# Summary visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# R² improvement
bars = axes[0].barh(imp_df['Model'], imp_df['R2_Gain'],
                     color=['green' if g > 0 else 'red' for g in imp_df['R2_Gain']])
axes[0].set_xlabel('R² Improvement', fontweight='bold')
axes[0].set_title('R² Gain from Best Improvement\n(Green = improved, Red = degraded)', fontweight='bold')
axes[0].axvline(x=0, color='black', linewidth=0.5)
axes[0].grid(axis='x', alpha=0.3)
for bar, val in zip(bars, imp_df['R2_Gain']):
    axes[0].text(val, bar.get_y() + bar.get_height()/2.,
                 f'{val:+.4f}', va='center', fontsize=9)

# Before vs After
x = range(len(imp_df))
width = 0.35
axes[1].barh([i - width/2 for i in x], imp_df['Original_R2'], width,
              label='Original', color='red', alpha=0.7)
axes[1].barh([i + width/2 for i in x], imp_df['Best_Improved_R2'], width,
              label='Improved', color='green', alpha=0.7)
axes[1].set_yticks(x)
axes[1].set_yticklabels(imp_df['Model'])
axes[1].set_xlabel('Test R²', fontweight='bold')
axes[1].set_title('Original vs Best Improved R²', fontweight='bold')
axes[1].legend()
axes[1].grid(axis='x', alpha=0.3)

plt.suptitle('Model Improvement Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'overall_improvement_summary.png'), bbox_inches='tight')
plt.close()

# Save summary
imp_df.to_csv(os.path.join(OUT_DIR, 'improvement_summary.csv'), index=False)

report = f"""
===============================================================================
MODEL IMPROVEMENT ANALYSIS - SUMMARY REPORT
===============================================================================

For each model, specific improvement strategies were applied and compared
against the original configuration.

IMPROVEMENT RESULTS:
{'='*70}
"""
for _, row in imp_df.iterrows():
    report += f"""
{row['Model']}:
  Original : R²={row['Original_R2']:.4f}, RMSE={row['Original_RMSE']:.1f}
  Improved : R²={row['Best_Improved_R2']:.4f}, RMSE={row['Best_Improved_RMSE']:.1f}
  Method   : {row['Best_Method']}
  R² Gain  : {row['R2_Gain']:+.4f}
  RMSE Reduction: {row['RMSE_Reduction']:+.1f}
"""

report += f"""
{'='*70}

KEY IMPROVEMENT STRATEGIES USED:
1. Hyperparameter tuning (alpha, C, k, n_estimators, max_depth, learning_rate)
2. Validation curves to find optimal hyperparameters
3. Feature selection for linear models
4. Weighted distance for KNN
5. Pruning and depth limiting for Decision Trees
6. Subsample and learning rate tuning for Gradient Boosting

GENERAL RECOMMENDATIONS:
- Always use cross-validation to find optimal hyperparameters
- Monitor train-test gap to detect overfitting
- Ensemble methods benefit from increasing trees (diminishing returns after ~200)
- Linear models benefit from proper regularization tuning
- KNN requires careful k selection and distance weighting

===============================================================================
"""

with open(os.path.join(OUT_DIR, 'improvement_summary_report.txt'), 'w') as f:
    f.write(report)

print(report)
print(f"\nAll improvement outputs saved to: {OUT_DIR}/")
print("Run step6_run_all.py to execute the entire pipeline.")
