# RAC 4 Revised - Traffic Flow Prediction Using Machine Learning

## What Changed from Original RAC 4

### Critical Bug Fix
The original code had **Traffic Volume** listed as BOTH a feature and the target variable. This caused data leakage where models could "cheat" by using the answer to predict the answer (resulting in R²=1.0). This has been **FIXED** - Traffic Volume is now ONLY the target.

### New Features Added (based on committee feedback)
1. **Complete EDA with preprocessing explanations** - every step documented
2. **Temporal analysis with fluctuation explanations** - why traffic goes up/down each month
3. **Individual model folders** - each model has its own visualizations and report
4. **Decision Tree structure visualizations** - full tree diagrams and rules
5. **Random Forest sample tree visualizations** - see individual trees in the ensemble
6. **Overfitting detection** - train vs test comparison for every model
7. **Cross-validation** - 5-fold CV for robust evaluation
8. **Model improvement analysis** - how each model can be made better
9. **Utility Score** - composite metric balancing accuracy, speed, and robustness

## How to Run

### Option 1: Run everything at once
```bash
cd RAC4_Revised
python step6_run_all.py
```

### Option 2: Run step by step
```bash
python step1_data_preprocessing.py    # EDA + preprocessing
python step2_temporal_analysis.py     # Temporal patterns
python step3_model_training.py        # Train all models
python step4_model_comparison.py      # Compare and rank
python step5_model_improvements.py    # Improvement analysis
```

### Requirements
```
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

## Output Structure
```
RAC4_Revised/outputs/
├── 01_Preprocessing/          # EDA, distributions, correlations
├── 02_Temporal_Analysis/      # Monthly trends, seasonal patterns
├── 03_Models/                 # Individual model results
│   ├── 01_LinearRegression/
│   ├── 02_Ridge/
│   ├── 03_Lasso/
│   ├── 04_ElasticNet/
│   ├── 05_SVR/
│   ├── 06_KNN/
│   ├── 07_DecisionTree/       # Includes tree_structure.png, tree_rules.txt
│   ├── 08_RandomForest/       # Includes sample_trees.png
│   ├── 09_ExtraTrees/
│   ├── 10_GradientBoosting/   # Includes learning_curve.png
│   └── all_results_comparison.csv
├── 04_Comparison/             # Rankings, utility scores, recommendation
├── 05_Improvements/           # Before/after for each model
└── processed_data/            # Cleaned data for reproducibility
```

## Models Evaluated (10 families, 25+ configurations)

| Category | Models | Key Hyperparameters |
|----------|--------|-------------------|
| Linear | Linear Regression, Ridge, Lasso, ElasticNet | alpha |
| SVM | SVR (RBF, Linear) | C, kernel |
| Instance-Based | KNN | k (neighbors) |
| Tree-Based | Decision Tree | max_depth |
| Ensemble | Random Forest, Extra Trees, Gradient Boosting | n_estimators, learning_rate |

## Dataset
- **Source**: Bangalore Traffic Dataset
- **Period**: 2022-01-01 to 2024-08-09
- **Records**: 8,936
- **Features**: 10 numeric + 7 temporal + 3 encoded + weather dummies
- **Target**: Traffic Volume (vehicles/day)
