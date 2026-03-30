"""
===============================================================================
STEP 6: MASTER RUNNER - EXECUTE ENTIRE PIPELINE
===============================================================================
RAC 4 - Traffic Flow Prediction Using Machine Learning

This script runs all 5 steps in sequence:
  Step 1: Data Preprocessing (EDA, cleaning, feature engineering, train-test split)
  Step 2: Temporal Analysis (monthly trends, day-of-week, seasonal, weather impact)
  Step 3: Model Training (10 models, individual outputs with all visualizations)
  Step 4: Model Comparison (overall ranking, utility score, recommendations)
  Step 5: Model Improvements (how each model can be improved)

Total expected output: ~100+ plots, reports, and data files

Usage:
  python step6_run_all.py

Or run each step individually:
  python step1_data_preprocessing.py
  python step2_temporal_analysis.py
  python step3_model_training.py
  python step4_model_comparison.py
  python step5_model_improvements.py
===============================================================================
"""

import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

steps = [
    ('step1_data_preprocessing.py', 'Data Preprocessing & EDA'),
    ('step2_temporal_analysis.py', 'Temporal Pattern Analysis'),
    ('step3_model_training.py', 'Model Training & Individual Evaluation'),
    ('step4_model_comparison.py', 'Model Comparison & Ranking'),
    ('step5_model_improvements.py', 'Model Improvement Analysis'),
]

print("=" * 70)
print("RAC 4 REVISED - COMPLETE PIPELINE EXECUTION")
print("=" * 70)
print(f"\nThis will run {len(steps)} steps. Total time: approximately 5-15 minutes.")
print("All outputs will be saved to: RAC4_Revised/outputs/\n")

overall_start = time.time()

for i, (script, description) in enumerate(steps, 1):
    print(f"\n{'#' * 70}")
    print(f"# STEP {i}/{len(steps)}: {description}")
    print(f"# Script: {script}")
    print(f"{'#' * 70}\n")

    script_path = os.path.join(SCRIPT_DIR, script)
    if not os.path.exists(script_path):
        print(f"  ERROR: {script} not found! Skipping.")
        continue

    step_start = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=SCRIPT_DIR,
        capture_output=False
    )

    step_time = time.time() - step_start

    if result.returncode == 0:
        print(f"\n  Step {i} completed successfully in {step_time:.1f} seconds.")
    else:
        print(f"\n  Step {i} FAILED (return code: {result.returncode}).")
        print(f"  You can run it individually: python {script}")

overall_time = time.time() - overall_start

print(f"\n{'=' * 70}")
print(f"PIPELINE COMPLETE")
print(f"{'=' * 70}")
print(f"Total execution time: {overall_time:.1f} seconds ({overall_time/60:.1f} minutes)")
print(f"\nOutput folder structure:")
print(f"""
  RAC4_Revised/outputs/
  |-- 01_Preprocessing/        <- EDA plots, distributions, correlations
  |-- 02_Temporal_Analysis/    <- Monthly trends, day-of-week, weather, heatmaps
  |-- 03_Models/               <- Individual model folders
  |   |-- 01_LinearRegression/ <- Predictions, residuals, coefficients
  |   |-- 02_Ridge/            <- Per-alpha results + validation curves
  |   |-- 03_Lasso/            <- Per-alpha results + feature selection
  |   |-- 04_ElasticNet/       <- Per-alpha results
  |   |-- 05_SVR/              <- Per-kernel results
  |   |-- 06_KNN/              <- Per-k results
  |   |-- 07_DecisionTree/     <- Tree structures, rules, feature importance
  |   |-- 08_RandomForest/     <- Sample trees, feature importance, comparisons
  |   |-- 09_ExtraTrees/       <- Feature importance, comparisons
  |   |-- 10_GradientBoosting/ <- Learning curves, feature importance
  |   +-- all_results_comparison.csv
  |-- 04_Comparison/           <- Overall rankings, charts, recommendation
  |-- 05_Improvements/         <- Per-model improvement analysis
  +-- processed_data/          <- Cleaned datasets for reproducibility
""")
print("Each model folder contains:")
print("  - predicted_vs_actual.png  (scatter: actual vs predicted)")
print("  - residual_analysis.png    (residual plots)")
print("  - train_vs_test_metrics.png (overfitting check)")
print("  - report.txt               (detailed text explanation)")
print("  - feature_importance.png   (for tree-based models)")
print("  - tree_structure.png       (for Decision Tree)")
print("  - sample_trees.png         (for Random Forest)")
print("  - learning_curve.png       (for Gradient Boosting)")
