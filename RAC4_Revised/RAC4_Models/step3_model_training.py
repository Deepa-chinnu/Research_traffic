"""
===============================================================================
STEP 3: COMPREHENSIVE MODEL TRAINING & EVALUATION
===============================================================================
RAC 4 - Traffic Flow Prediction Using Machine Learning

This script trains and evaluates ALL models INDIVIDUALLY with:
  - Separate output folder for each model
  - Detailed performance metrics (Train + Test)
  - Predicted vs Actual scatter plots
  - Residual analysis
  - Overfitting detection
  - For tree-based: Full tree visualizations and feature importance
  - For each model: Written text report explaining results
  - Cross-validation for robust evaluation

Models: LinearRegression, Ridge, Lasso, ElasticNet, SVR, KNN,
        DecisionTree, RandomForest, ExtraTrees, GradientBoosting
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
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor)
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error, r2_score)
from sklearn.model_selection import cross_val_score

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'processed_data')
OUT_BASE = os.path.join(SCRIPT_DIR, 'outputs', '03_Models')
os.makedirs(OUT_BASE, exist_ok=True)

# ============================================================================
# LOAD PROCESSED DATA
# ============================================================================
print("=" * 70)
print("LOADING PROCESSED DATA")
print("=" * 70)

X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index_col=0)
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index_col=0)
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index_col=0).squeeze()
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index_col=0).squeeze()

# Also load scaled versions for models that need scaling
X_train_scaled = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train_scaled.csv'), index_col=0)
X_test_scaled = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test_scaled.csv'), index_col=0)

print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples : {len(X_test)}")
print(f"  Features        : {X_train.shape[1]}")

FEATURE_NAMES = list(X_train.columns)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

all_results = []


def evaluate_model(model, X_tr, X_te, y_tr, y_te, model_name, variant_name,
                   out_dir, use_cv=True, explanation=""):
    """Train, evaluate, and generate comprehensive outputs for a single model."""

    os.makedirs(out_dir, exist_ok=True)

    # Train
    start = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - start

    # Predict
    y_train_pred = model.predict(X_tr)
    y_test_pred = model.predict(X_te)

    # Metrics - Training
    train_r2 = r2_score(y_tr, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_tr, y_train_pred))
    train_mae = mean_absolute_error(y_tr, y_train_pred)

    # Metrics - Testing
    test_r2 = r2_score(y_te, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_te, y_test_pred))
    test_mae = mean_absolute_error(y_te, y_test_pred)
    try:
        test_mape = mean_absolute_percentage_error(y_te, y_test_pred)
    except Exception:
        test_mape = np.nan

    # Overfitting check
    r2_gap = train_r2 - test_r2
    if r2_gap > 0.1:
        overfit_status = "OVERFITTING (train R² >> test R²)"
    elif r2_gap > 0.05:
        overfit_status = "SLIGHT OVERFITTING"
    elif r2_gap < -0.05:
        overfit_status = "UNDERFITTING (test > train, unusual)"
    else:
        overfit_status = "GOOD GENERALIZATION"

    # Cross-validation (on unscaled data for tree models, scaled for linear)
    cv_r2 = None
    if use_cv:
        try:
            cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring='r2', n_jobs=-1)
            cv_r2 = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception:
            cv_r2 = None
            cv_std = None

    # ---- VISUALIZATIONS ----

    # 1. Predicted vs Actual (Train + Test side by side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(y_tr, y_train_pred, alpha=0.3, s=10, color='blue')
    min_val = min(y_tr.min(), y_train_pred.min())
    max_val = max(y_tr.max(), y_train_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Traffic Volume')
    axes[0].set_ylabel('Predicted Traffic Volume')
    axes[0].set_title(f'TRAINING Data\nR²={train_r2:.4f}, RMSE={train_rmse:.1f}', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_te, y_test_pred, alpha=0.3, s=10, color='red')
    min_val = min(y_te.min(), y_test_pred.min())
    max_val = max(y_te.max(), y_test_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Traffic Volume')
    axes[1].set_ylabel('Predicted Traffic Volume')
    axes[1].set_title(f'TESTING Data (Unseen)\nR²={test_r2:.4f}, RMSE={test_rmse:.1f}', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f'{model_name} ({variant_name}): Predicted vs Actual',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'predicted_vs_actual.png'), bbox_inches='tight')
    plt.close()

    # 2. Residual Analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    residuals = y_te.values - y_test_pred
    axes[0].scatter(y_test_pred, residuals, alpha=0.3, s=10, color='steelblue')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals (Actual - Predicted)')
    axes[0].set_title('Residual Plot\n(Should be random around 0)', fontweight='bold')
    axes[0].grid(alpha=0.3)

    axes[1].hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Residual Distribution\n(Mean={np.mean(residuals):.1f}, Std={np.std(residuals):.1f})',
                      fontweight='bold')

    # Residuals over index (ordered)
    axes[2].plot(range(len(residuals)), sorted(residuals), color='steelblue', linewidth=0.8)
    axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Sorted Sample Index')
    axes[2].set_ylabel('Residual')
    axes[2].set_title('Sorted Residuals\n(Ideal: symmetric around 0)', fontweight='bold')
    axes[2].grid(alpha=0.3)

    plt.suptitle(f'{model_name} ({variant_name}): Residual Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'residual_analysis.png'), bbox_inches='tight')
    plt.close()

    # 3. Overfitting comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ['R² Score', 'RMSE', 'MAE']
    train_vals = [train_r2, train_rmse, train_mae]
    test_vals = [test_r2, test_rmse, test_mae]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, train_vals, width, label='Training', color='blue', alpha=0.7)
    ax.bar(x + width/2, test_vals, width, label='Testing', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title(f'{model_name}: Training vs Testing Metrics\nStatus: {overfit_status}',
                 fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for i, (tv, tsv) in enumerate(zip(train_vals, test_vals)):
        ax.text(i - width/2, tv, f'{tv:.4f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, tsv, f'{tsv:.4f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'train_vs_test_metrics.png'), bbox_inches='tight')
    plt.close()

    # ---- TEXT REPORT ----
    report = f"""
{'='*70}
{model_name.upper()} - {variant_name}
DETAILED ANALYSIS REPORT
{'='*70}

MODEL DESCRIPTION:
{explanation}

TRAINING PERFORMANCE:
  R² Score : {train_r2:.6f}
  RMSE     : {train_rmse:.2f}
  MAE      : {train_mae:.2f}

TESTING PERFORMANCE (Unseen Data):
  R² Score : {test_r2:.6f}
  RMSE     : {test_rmse:.2f}
  MAE      : {test_mae:.2f}
  MAPE     : {test_mape:.4f}

OVERFITTING ANALYSIS:
  R² Gap (Train - Test): {r2_gap:+.6f}
  Status: {overfit_status}
  {'The model memorizes training data but fails on new data.' if r2_gap > 0.1 else 'The model generalizes well to unseen data.' if r2_gap < 0.05 else 'Mild overfitting - acceptable for most applications.'}

"""
    if cv_r2 is not None:
        report += f"""5-FOLD CROSS-VALIDATION:
  Mean R²  : {cv_r2:.6f}
  Std R²   : {cv_std:.6f}
  This confirms {'consistent' if cv_std < 0.05 else 'variable'} performance across data splits.

"""
    report += f"""TRAINING TIME: {train_time:.4f} seconds

FILES GENERATED:
  - predicted_vs_actual.png  : Scatter plot of predictions (train + test)
  - residual_analysis.png    : Residual plots and histogram
  - train_vs_test_metrics.png: Side-by-side metric comparison
{'='*70}
"""
    with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
        f.write(report)

    print(f"  {model_name} ({variant_name}): Test R²={test_r2:.4f}, RMSE={test_rmse:.1f}, "
          f"Time={train_time:.3f}s [{overfit_status}]")

    result = {
        'Model': model_name, 'Variant': variant_name,
        'Train_R2': train_r2, 'Train_RMSE': train_rmse, 'Train_MAE': train_mae,
        'Test_R2': test_r2, 'Test_RMSE': test_rmse, 'Test_MAE': test_mae,
        'Test_MAPE': test_mape, 'R2_Gap': r2_gap, 'Overfit_Status': overfit_status,
        'CV_R2': cv_r2, 'TrainTime': train_time,
    }
    all_results.append(result)

    return model, y_train_pred, y_test_pred, result


def plot_feature_importance(model, feature_names, model_name, out_dir):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(12, max(6, len(feature_names) * 0.35)))
    y_pos = range(len(importances))
    ax.barh(y_pos, importances[indices], color='forestgreen', edgecolor='black', linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel('Feature Importance', fontweight='bold')
    ax.set_title(f'{model_name}: Feature Importance\n'
                 f'(Top 3 features explain {sum(importances[indices[:3]]):.1%} of predictions)',
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (idx, imp) in enumerate(zip(indices, importances[indices])):
        ax.text(imp + 0.001, i, f'{imp:.4f}', va='center', fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_importance.png'), bbox_inches='tight')
    plt.close()


# ============================================================================
# MODEL 1: LINEAR REGRESSION
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 1: LINEAR REGRESSION")
print("=" * 70)

out_dir = os.path.join(OUT_BASE, '01_LinearRegression')
model_lr, _, _, _ = evaluate_model(
    LinearRegression(), X_train_scaled, X_test_scaled, y_train, y_test,
    'Linear Regression', 'default',
    out_dir, use_cv=True,
    explanation="""Linear Regression fits a straight line (hyperplane) to the data.
  Formula: y = b0 + b1*x1 + b2*x2 + ... + bn*xn
  It finds coefficients that minimize the sum of squared errors.
  - No regularization (can overfit with many features)
  - Assumes linear relationship between features and target
  - Very fast training, highly interpretable
  - Uses SCALED features (StandardScaler) for stable coefficients"""
)

# Show coefficients
coef_df = pd.DataFrame({
    'Feature': FEATURE_NAMES,
    'Coefficient': model_lr.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(12, max(6, len(FEATURE_NAMES) * 0.35)))
colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
ax.barh(range(len(coef_df)), coef_df['Coefficient'], color=colors, alpha=0.7)
ax.set_yticks(range(len(coef_df)))
ax.set_yticklabels(coef_df['Feature'], fontsize=9)
ax.set_xlabel('Coefficient Value')
ax.set_title('Linear Regression Coefficients\n(Green = positive effect, Red = negative effect)',
             fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.5)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'coefficients.png'), bbox_inches='tight')
plt.close()

# ============================================================================
# MODEL 2: RIDGE REGRESSION (multiple alphas)
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 2: RIDGE REGRESSION")
print("=" * 70)

for alpha in [0.1, 1.0, 10.0]:
    out_dir = os.path.join(OUT_BASE, '02_Ridge', f'alpha_{alpha}')
    evaluate_model(
        Ridge(alpha=alpha, random_state=42),
        X_train_scaled, X_test_scaled, y_train, y_test,
        'Ridge Regression', f'alpha={alpha}',
        out_dir, use_cv=True,
        explanation=f"""Ridge Regression adds L2 regularization to Linear Regression.
  Formula: y = b0 + b1*x1 + ... + bn*xn + lambda * sum(bi^2)
  alpha={alpha} controls regularization strength:
    - Higher alpha = more regularization = simpler model (less overfitting)
    - Lower alpha = less regularization = closer to basic Linear Regression
  - Prevents overfitting by penalizing large coefficients
  - Handles multicollinearity (correlated features) well
  - Does NOT perform feature selection (all features kept)"""
    )

# ============================================================================
# MODEL 3: LASSO REGRESSION
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 3: LASSO REGRESSION")
print("=" * 70)

for alpha in [0.001, 0.01, 0.1, 1.0]:
    out_dir = os.path.join(OUT_BASE, '03_Lasso', f'alpha_{alpha}')
    model_lasso, _, _, _ = evaluate_model(
        Lasso(alpha=alpha, random_state=42, max_iter=10000),
        X_train_scaled, X_test_scaled, y_train, y_test,
        'Lasso Regression', f'alpha={alpha}',
        out_dir, use_cv=True,
        explanation=f"""Lasso Regression adds L1 regularization.
  Formula: y = b0 + b1*x1 + ... + bn*xn + lambda * sum(|bi|)
  alpha={alpha}:
    - L1 penalty can set some coefficients to EXACTLY ZERO
    - This means Lasso performs automatic FEATURE SELECTION
    - Features with zero coefficient are considered unimportant
  - Useful for identifying which features matter most"""
    )
    # Show which features Lasso eliminated
    n_zero = (model_lasso.coef_ == 0).sum()
    if n_zero > 0:
        zero_feats = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if model_lasso.coef_[i] == 0]
        with open(os.path.join(out_dir, 'report.txt'), 'a') as f:
            f.write(f"\nFEATURE SELECTION by Lasso (alpha={alpha}):\n")
            f.write(f"  Features ELIMINATED (coefficient = 0): {n_zero}\n")
            for feat in zero_feats:
                f.write(f"    - {feat}\n")
            f.write(f"  Features KEPT: {len(FEATURE_NAMES) - n_zero}\n")

# ============================================================================
# MODEL 4: ELASTICNET
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 4: ELASTICNET")
print("=" * 70)

for alpha in [0.001, 0.01, 0.1]:
    out_dir = os.path.join(OUT_BASE, '04_ElasticNet', f'alpha_{alpha}')
    evaluate_model(
        ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42, max_iter=10000),
        X_train_scaled, X_test_scaled, y_train, y_test,
        'ElasticNet', f'alpha={alpha}, l1_ratio=0.5',
        out_dir, use_cv=True,
        explanation=f"""ElasticNet combines both L1 and L2 regularization.
  Formula: y = b0 + ... + lambda1*sum(|bi|) + lambda2*sum(bi^2)
  alpha={alpha}, l1_ratio=0.5 (50% L1 + 50% L2):
    - Gets benefits of BOTH Ridge (stability) and Lasso (feature selection)
    - l1_ratio controls mix: 0=Ridge, 1=Lasso, 0.5=equal mix
  - Best when features are correlated AND you want some feature selection"""
    )

# ============================================================================
# MODEL 5: SVR (Support Vector Regression)
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 5: SUPPORT VECTOR REGRESSION (SVR)")
print("=" * 70)

for kernel in ['rbf', 'linear']:
    out_dir = os.path.join(OUT_BASE, '05_SVR', f'kernel_{kernel}')
    evaluate_model(
        SVR(kernel=kernel, C=1.0),
        X_train_scaled, X_test_scaled, y_train, y_test,
        'SVR', f'kernel={kernel}, C=1.0',
        out_dir, use_cv=True,
        explanation=f"""Support Vector Regression with {kernel} kernel.
  - SVR finds a function that deviates from target by at most epsilon
  - kernel='{kernel}': {'RBF (Radial Basis Function) maps data to higher dimensions to find nonlinear patterns' if kernel == 'rbf' else 'Linear kernel finds a hyperplane, similar to linear regression with regularization'}
  - C=1.0 controls trade-off between fitting training data and smoothness
  - Uses SCALED features (SVR is sensitive to feature magnitudes)
  - Memory efficient: only uses support vectors for prediction"""
    )

# ============================================================================
# MODEL 6: KNN (K-Nearest Neighbors)
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 6: K-NEAREST NEIGHBORS (KNN)")
print("=" * 70)

for k in [3, 5, 11]:
    out_dir = os.path.join(OUT_BASE, '06_KNN', f'k_{k}')
    evaluate_model(
        KNeighborsRegressor(n_neighbors=k),
        X_train_scaled, X_test_scaled, y_train, y_test,
        'KNN', f'k={k}',
        out_dir, use_cv=True,
        explanation=f"""K-Nearest Neighbors Regression with k={k}.
  - Predicts traffic volume as the AVERAGE of {k} most similar records
  - 'Similar' = closest in feature space (Euclidean distance)
  - k={k}: {'Small k = more sensitive to local patterns, risk of overfitting' if k <= 5 else 'Larger k = smoother predictions, more robust but may miss local patterns'}
  - No actual 'training' - just stores the data (lazy learning)
  - Prediction time scales with dataset size
  - Uses SCALED features (distance-based)"""
    )

# ============================================================================
# MODEL 7: DECISION TREE
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 7: DECISION TREE")
print("=" * 70)

for depth in [6, 12, None]:
    depth_label = str(depth) if depth is not None else 'Unlimited'
    out_dir = os.path.join(OUT_BASE, '07_DecisionTree', f'max_depth_{depth_label}')

    model_dt, _, _, _ = evaluate_model(
        DecisionTreeRegressor(max_depth=depth, random_state=42),
        X_train, X_test, y_train, y_test,  # Trees don't need scaling
        'Decision Tree', f'max_depth={depth_label}',
        out_dir, use_cv=True,
        explanation=f"""Decision Tree Regressor with max_depth={depth_label}.
  - Creates binary splits based on feature values
  - At each node, picks the feature and threshold that best reduces error
  - max_depth={depth_label}: {'Limits tree to {0} levels, preventing overfitting'.format(depth) if depth is not None else 'No depth limit - tree grows until leaves are pure (HIGH OVERFITTING RISK)'}
  - Does NOT need feature scaling
  - Highly interpretable: you can visualize the entire tree structure
  - Can capture nonlinear patterns and feature interactions"""
    )

    # Feature importance
    plot_feature_importance(model_dt, FEATURE_NAMES, f'Decision Tree (depth={depth_label})', out_dir)

    # TREE VISUALIZATION (the committee asked for this!)
    fig, ax = plt.subplots(figsize=(24, 14))
    max_display_depth = min(depth if depth is not None else 6, 6)
    plot_tree(model_dt, feature_names=FEATURE_NAMES, filled=True, rounded=True,
              fontsize=7, ax=ax, max_depth=max_display_depth,
              impurity=True, proportion=True)
    ax.set_title(f'Decision Tree Structure (max_depth={depth_label})\n'
                 f'Showing top {max_display_depth} levels | '
                 f'Total nodes: {model_dt.tree_.node_count} | '
                 f'Total leaves: {model_dt.tree_.n_leaves}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tree_structure.png'), bbox_inches='tight', dpi=150)
    plt.close()

    # Text representation of tree rules
    tree_rules = export_text(model_dt, feature_names=FEATURE_NAMES, max_depth=5)
    with open(os.path.join(out_dir, 'tree_rules.txt'), 'w') as f:
        f.write(f"Decision Tree Rules (max_depth={depth_label})\n")
        f.write(f"Total nodes: {model_dt.tree_.node_count}\n")
        f.write(f"Total leaves: {model_dt.tree_.n_leaves}\n")
        f.write(f"Max actual depth: {model_dt.get_depth()}\n\n")
        f.write("HOW TO READ: At each node, if condition is TRUE go LEFT, else go RIGHT\n\n")
        f.write(tree_rules)

    print(f"    -> Tree nodes: {model_dt.tree_.node_count}, Leaves: {model_dt.tree_.n_leaves}")

# ============================================================================
# MODEL 8: RANDOM FOREST
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 8: RANDOM FOREST")
print("=" * 70)

for n_trees in [50, 100, 200]:
    out_dir = os.path.join(OUT_BASE, '08_RandomForest', f'{n_trees}_trees')

    model_rf, _, _, _ = evaluate_model(
        RandomForestRegressor(n_estimators=n_trees, random_state=42, n_jobs=-1),
        X_train, X_test, y_train, y_test,
        'Random Forest', f'{n_trees} trees',
        out_dir, use_cv=True,
        explanation=f"""Random Forest with {n_trees} decision trees.
  - Creates {n_trees} different decision trees using random subsets of data
  - Each tree sees a random sample of features at each split (decorrelation)
  - Final prediction = AVERAGE of all {n_trees} tree predictions
  - This reduces overfitting compared to a single Decision Tree
  - Does NOT need feature scaling
  - Key advantages: robust to outliers, handles missing data, fast training"""
    )

    # Feature importance
    plot_feature_importance(model_rf, FEATURE_NAMES, f'Random Forest ({n_trees} trees)', out_dir)

    # Visualize 3 sample trees from the forest
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    for i in range(3):
        plot_tree(model_rf.estimators_[i], feature_names=FEATURE_NAMES,
                  filled=True, rounded=True, fontsize=5, ax=axes[i], max_depth=3)
        axes[i].set_title(f'Tree {i+1} of {n_trees}\n(Showing top 3 levels)',
                          fontsize=11, fontweight='bold')
    plt.suptitle(f'Random Forest: 3 Sample Trees from {n_trees}-Tree Ensemble\n'
                 f'Each tree is different due to random sampling (bagging)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sample_trees.png'), bbox_inches='tight', dpi=150)
    plt.close()

    # Single full tree visualization
    fig, ax = plt.subplots(figsize=(24, 14))
    plot_tree(model_rf.estimators_[0], feature_names=FEATURE_NAMES,
              filled=True, rounded=True, fontsize=6, ax=ax, max_depth=5)
    ax.set_title(f'Random Forest: Detailed View of Tree #1 (of {n_trees})\n'
                 f'Top 5 levels shown', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'single_tree_detail.png'), bbox_inches='tight', dpi=150)
    plt.close()

# ============================================================================
# MODEL 9: EXTRA TREES
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 9: EXTRA TREES (Extremely Randomized Trees)")
print("=" * 70)

for n_trees in [50, 100, 200]:
    out_dir = os.path.join(OUT_BASE, '09_ExtraTrees', f'{n_trees}_trees')

    model_et, _, _, _ = evaluate_model(
        ExtraTreesRegressor(n_estimators=n_trees, random_state=42, n_jobs=-1),
        X_train, X_test, y_train, y_test,
        'Extra Trees', f'{n_trees} trees',
        out_dir, use_cv=True,
        explanation=f"""Extra Trees (Extremely Randomized Trees) with {n_trees} trees.
  - Similar to Random Forest but with MORE randomization
  - Key difference: Random Forest finds the BEST split, Extra Trees uses RANDOM splits
  - This makes Extra Trees FASTER to train (no optimization at each split)
  - Can be more robust to noise due to extra randomization
  - Does NOT need feature scaling"""
    )
    plot_feature_importance(model_et, FEATURE_NAMES, f'Extra Trees ({n_trees} trees)', out_dir)

# ============================================================================
# MODEL 10: GRADIENT BOOSTING
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 10: GRADIENT BOOSTING")
print("=" * 70)

for n_est in [100, 200]:
    out_dir = os.path.join(OUT_BASE, '10_GradientBoosting', f'{n_est}_estimators')

    model_gb, _, _, _ = evaluate_model(
        GradientBoostingRegressor(n_estimators=n_est, random_state=42,
                                  max_depth=5, learning_rate=0.1),
        X_train, X_test, y_train, y_test,
        'Gradient Boosting', f'{n_est} estimators, lr=0.1',
        out_dir, use_cv=True,
        explanation=f"""Gradient Boosting with {n_est} sequential estimators.
  - Unlike Random Forest which builds trees IN PARALLEL and averages them,
    Gradient Boosting builds trees ONE BY ONE, each correcting the previous errors
  - Step 1: Build tree 1, calculate errors
  - Step 2: Build tree 2 to predict those errors
  - Step N: Each new tree fixes remaining errors
  - learning_rate=0.1: How much each tree contributes (smaller = more trees needed but better)
  - max_depth=5: Each individual tree is shallow (weak learner)
  - Often achieves highest accuracy but slower training"""
    )
    plot_feature_importance(model_gb, FEATURE_NAMES, f'Gradient Boosting ({n_est} est.)', out_dir)

    # Show learning curve (how error decreases with more trees)
    if hasattr(model_gb, 'train_score_'):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, n_est + 1), model_gb.train_score_, 'b-', label='Training Loss')
        ax.set_xlabel('Number of Boosting Iterations')
        ax.set_ylabel('Loss (Deviance)')
        ax.set_title(f'Gradient Boosting Learning Curve ({n_est} iterations)\n'
                     f'Shows how error decreases as more trees are added', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'learning_curve.png'), bbox_inches='tight')
        plt.close()

# ============================================================================
# SAVE OVERALL RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("SAVING OVERALL COMPARISON RESULTS")
print("=" * 70)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('Test_R2', ascending=False)
results_df.to_csv(os.path.join(OUT_BASE, 'all_results_comparison.csv'), index=False)
results_df.to_excel(os.path.join(OUT_BASE, 'all_results_comparison.xlsx'), index=False)

print("\n  TOP 10 MODELS BY TEST R²:")
print(results_df[['Model', 'Variant', 'Test_R2', 'Test_RMSE', 'TrainTime', 'Overfit_Status']].head(10).to_string(index=False))

print(f"\n  Results saved to: {OUT_BASE}/all_results_comparison.csv")
print(f"  Individual model outputs in: {OUT_BASE}/01_LinearRegression/ through 10_GradientBoosting/")
print("\nRun step4_model_comparison.py next.")
