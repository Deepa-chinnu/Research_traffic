"""
===============================================================================
STEP 1: DATA LOADING, EXPLORATION & PREPROCESSING
===============================================================================
RAC 4 - Traffic Flow Prediction Using Machine Learning
Dataset: Bangalore Traffic Dataset (2022-01-01 to 2024-08-09)

This script demonstrates EVERY preprocessing step with explanations:
  1. Loading raw data
  2. Exploratory Data Analysis (EDA)
  3. Missing value handling
  4. Feature engineering (temporal features from Date)
  5. Encoding categorical variables
  6. Correlation analysis
  7. Feature scaling
  8. Train-Test split
  9. Saving cleaned data for model training

IMPORTANT FIX: The original code had 'Traffic Volume' in both features AND
target, causing data leakage (R²=1.0). This version correctly separates them.
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 11

# ============================================================================
# PATHS
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'RAC4', 'Banglore_traffic_Dataset.csv')
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(SCRIPT_DIR, 'Banglore_traffic_Dataset.csv')
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found. Place 'Banglore_traffic_Dataset.csv' in RAC4/ or RAC4_Revised/"
    )

OUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', '01_Preprocessing')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# 1. LOAD RAW DATA
# ============================================================================
print("=" * 70)
print("STEP 1.1: LOADING RAW DATASET")
print("=" * 70)

df_raw = pd.read_csv(DATA_PATH)

print(f"  Dataset path : {DATA_PATH}")
print(f"  Total records: {len(df_raw)}")
print(f"  Total columns: {len(df_raw.columns)}")
print(f"\n  Column Names:")
for i, col in enumerate(df_raw.columns, 1):
    print(f"    {i:2d}. {col}")

print(f"\n  First 5 rows:")
print(df_raw.head().to_string(index=False))

# ============================================================================
# 2. DATA TYPES & BASIC STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.2: DATA TYPES AND BASIC STATISTICS")
print("=" * 70)

print("\n  Data Types:")
print(df_raw.dtypes.to_string())

print(f"\n  Numeric Column Statistics:")
print(df_raw.describe().round(2).to_string())

# ============================================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.3: MISSING VALUE ANALYSIS")
print("=" * 70)

missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df.to_string())

# Visualize missing values
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['green' if v == 0 else 'red' for v in missing.values]
ax.barh(range(len(missing)), missing.values, color=colors)
ax.set_yticks(range(len(missing)))
ax.set_yticklabels(missing.index, fontsize=9)
ax.set_xlabel('Number of Missing Values')
ax.set_title('Missing Values per Column\n(Green = No Missing, Red = Has Missing)', fontweight='bold')
for i, v in enumerate(missing.values):
    ax.text(v + 0.5, i, str(v), va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'missing_values.png'), bbox_inches='tight')
plt.close()
print(f"\n  Saved: {OUT_DIR}/missing_values.png")

# ============================================================================
# 4. IDENTIFY FEATURE AND TARGET COLUMNS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.4: IDENTIFYING FEATURES AND TARGET")
print("=" * 70)

TARGET = 'Traffic Volume'

# Numeric features (EXCLUDING the target variable - THIS FIXES DATA LEAKAGE)
NUMERIC_FEATURES = [
    'Average Speed', 'Travel Time Index', 'Congestion Level',
    'Road Capacity Utilization', 'Incident Reports',
]

CATEGORICAL_FEATURES = ['Area Name', 'Road/Intersection Name',
                        'Weather Conditions', 'Roadwork and Construction Activity']
DATE_COLUMN = 'Date'

print(f"  Target Variable: {TARGET}")
print(f"\n  Numeric Features ({len(NUMERIC_FEATURES)}):")
for f in NUMERIC_FEATURES:
    print(f"    - {f}")
print(f"\n  Categorical Features ({len(CATEGORICAL_FEATURES)}):")
for f in CATEGORICAL_FEATURES:
    print(f"    - {f}")
print(f"\n  Date Column: {DATE_COLUMN}")

print(f"\n  ** IMPORTANT: 'Traffic Volume' is ONLY the target variable.")
print(f"  ** It is NOT included in the features to prevent data leakage.")
print(f"  ** (Original code had this bug causing R²=1.0)")

# ============================================================================
# 5. CLEAN NUMERIC DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.5: CLEANING NUMERIC DATA")
print("=" * 70)

df = df_raw.copy()
initial_rows = len(df)

# Convert numeric columns
all_numeric = NUMERIC_FEATURES + [TARGET]
for col in all_numeric:
    before = len(df)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[col])
    after = len(df)
    if before - after > 0:
        print(f"  {col}: Removed {before - after} invalid rows")

print(f"\n  Rows before cleaning: {initial_rows}")
print(f"  Rows after cleaning : {len(df)}")
print(f"  Rows removed        : {initial_rows - len(df)}")

# ============================================================================
# 6. FEATURE ENGINEERING - TEMPORAL FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.6: FEATURE ENGINEERING - TEMPORAL FEATURES FROM DATE")
print("=" * 70)

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek          # 0=Monday, 6=Sunday
df['DayName'] = df['Date'].dt.day_name()
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)  # 1 if Sat/Sun
df['Quarter'] = df['Date'].dt.quarter

TEMPORAL_FEATURES = ['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear',
                     'IsWeekend', 'Quarter']

print("  Created temporal features:")
for f in TEMPORAL_FEATURES:
    print(f"    - {f}: unique values = {df[f].nunique()}")

print(f"\n  WHY temporal features matter:")
print(f"    - Traffic patterns change by day of week (weekday vs weekend)")
print(f"    - Seasonal patterns exist (monsoon, festivals, holidays)")
print(f"    - Year-over-year trends capture infrastructure changes")

# ============================================================================
# 7. ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.7: ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

# Weather Conditions
print(f"\n  Weather Conditions distribution:")
print(df['Weather Conditions'].value_counts().to_string())

weather_dummies = pd.get_dummies(df['Weather Conditions'], prefix='Weather', drop_first=True)
print(f"  -> Created {len(weather_dummies.columns)} dummy columns (dropped first to avoid multicollinearity)")

# Roadwork
print(f"\n  Roadwork distribution:")
print(df['Roadwork and Construction Activity'].value_counts().to_string())
df['Roadwork_Binary'] = (df['Roadwork and Construction Activity'] == 'Yes').astype(int)

# Area Name - Label encode
print(f"\n  Area Names ({df['Area Name'].nunique()} unique):")
print(df['Area Name'].value_counts().to_string())
area_mapping = {name: i for i, name in enumerate(df['Area Name'].unique())}
df['Area_Encoded'] = df['Area Name'].map(area_mapping)

# Road Name - Label encode
print(f"\n  Roads ({df['Road/Intersection Name'].nunique()} unique):")
road_mapping = {name: i for i, name in enumerate(df['Road/Intersection Name'].unique())}
df['Road_Encoded'] = df['Road/Intersection Name'].map(road_mapping)

ENCODED_FEATURES = ['Area_Encoded', 'Road_Encoded', 'Roadwork_Binary']

# ============================================================================
# 8. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.8: CORRELATION ANALYSIS")
print("=" * 70)

corr_cols = NUMERIC_FEATURES + [TARGET]
corr_matrix = df[corr_cols].corr()

print(f"\n  Correlation with '{TARGET}':")
target_corr = corr_matrix[TARGET].drop(TARGET).sort_values(ascending=False)
for feat, corr_val in target_corr.items():
    strength = "Strong" if abs(corr_val) > 0.7 else "Moderate" if abs(corr_val) > 0.4 else "Weak"
    direction = "Positive" if corr_val > 0 else "Negative"
    print(f"    {feat:35s}: {corr_val:+.4f} ({strength} {direction})")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center='light', as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={'size': 8})
ax.set_title('Feature Correlation Heatmap\n(Lower triangle shown to avoid redundancy)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'correlation_heatmap.png'), bbox_inches='tight')
plt.close()
print(f"\n  Saved: correlation_heatmap.png")

# ============================================================================
# 9. TARGET VARIABLE DISTRIBUTION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.9: TARGET VARIABLE ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram
axes[0].hist(df[TARGET], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(df[TARGET].mean(), color='red', linestyle='--', label=f'Mean: {df[TARGET].mean():.0f}')
axes[0].axvline(df[TARGET].median(), color='green', linestyle='--', label=f'Median: {df[TARGET].median():.0f}')
axes[0].set_xlabel('Traffic Volume')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Traffic Volume', fontweight='bold')
axes[0].legend()

# Box plot
axes[1].boxplot(df[TARGET], vert=True)
axes[1].set_ylabel('Traffic Volume')
axes[1].set_title('Box Plot - Outlier Detection', fontweight='bold')
q1 = df[TARGET].quantile(0.25)
q3 = df[TARGET].quantile(0.75)
iqr = q3 - q1
axes[1].text(1.15, q3, f'Q3: {q3:.0f}', va='center')
axes[1].text(1.15, q1, f'Q1: {q1:.0f}', va='center')
axes[1].text(1.15, df[TARGET].median(), f'Median: {df[TARGET].median():.0f}', va='center')

# QQ Plot equivalent - sorted values
sorted_vals = np.sort(df[TARGET].values)
theoretical = np.linspace(0, 1, len(sorted_vals))
axes[2].plot(theoretical, sorted_vals, 'b.', markersize=1)
axes[2].set_xlabel('Percentile')
axes[2].set_ylabel('Traffic Volume')
axes[2].set_title('Cumulative Distribution of Traffic Volume', fontweight='bold')

plt.suptitle('Target Variable: Traffic Volume', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'target_distribution.png'), bbox_inches='tight')
plt.close()

stats = df[TARGET].describe()
print(f"  Mean   : {stats['mean']:.2f}")
print(f"  Std    : {stats['std']:.2f}")
print(f"  Min    : {stats['min']:.2f}")
print(f"  Max    : {stats['max']:.2f}")
print(f"  Median : {stats['50%']:.2f}")
print(f"  Saved: target_distribution.png")

# ============================================================================
# 10. FEATURE DISTRIBUTIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.10: FEATURE DISTRIBUTIONS")
print("=" * 70)

n_feats = len(NUMERIC_FEATURES)
n_cols = 3
n_rows = (n_feats + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
axes = axes.flatten()

for i, feat in enumerate(NUMERIC_FEATURES):
    axes[i].hist(df[feat], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(feat, fontweight='bold', fontsize=10)
    axes[i].set_ylabel('Count')
    mean_val = df[feat].mean()
    axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7)
    axes[i].text(0.95, 0.95, f'Mean: {mean_val:.1f}\nStd: {df[feat].std():.1f}',
                 transform=axes[i].transAxes, ha='right', va='top', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Hide unused axes
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of All Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'feature_distributions.png'), bbox_inches='tight')
plt.close()
print(f"  Saved: feature_distributions.png")

# ============================================================================
# 11. PAIRWISE SCATTER PLOTS (Top correlated features)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.11: PAIRWISE SCATTER PLOTS (Top Features vs Target)")
print("=" * 70)

top_features = target_corr.abs().sort_values(ascending=False).head(4).index.tolist()
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, feat in enumerate(top_features):
    axes[i].scatter(df[feat], df[TARGET], alpha=0.2, s=5, color='steelblue')
    axes[i].set_xlabel(feat, fontsize=9)
    axes[i].set_ylabel('Traffic Volume' if i == 0 else '')
    corr_val = df[feat].corr(df[TARGET])
    axes[i].set_title(f'{feat}\nvs Traffic Volume (r={corr_val:.3f})', fontsize=10, fontweight='bold')

plt.suptitle('Top 4 Correlated Features vs Traffic Volume', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'scatter_top_features.png'), bbox_inches='tight')
plt.close()
print(f"  Saved: scatter_top_features.png")

# ============================================================================
# 12. PREPARE FINAL FEATURE MATRIX
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.12: PREPARING FINAL FEATURE MATRIX")
print("=" * 70)

# Combine all weather dummies into main dataframe
df = pd.concat([df, weather_dummies], axis=1)

FINAL_FEATURES = (NUMERIC_FEATURES + TEMPORAL_FEATURES +
                  ENCODED_FEATURES + list(weather_dummies.columns))

X = df[FINAL_FEATURES].copy()
y = df[TARGET].copy()

print(f"  Total features: {len(FINAL_FEATURES)}")
print(f"  Feature list:")
for i, f in enumerate(FINAL_FEATURES, 1):
    print(f"    {i:2d}. {f}")
print(f"\n  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# ============================================================================
# 13. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.13: TRAIN-TEST SPLIT (70% Train, 30% Test)")
print("=" * 70)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Testing set : {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"  Random state: 42 (for reproducibility)")

# Visualize the split
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(y_train, bins=40, alpha=0.7, color='blue', label=f'Train ({len(y_train)})')
axes[0].hist(y_test, bins=40, alpha=0.7, color='red', label=f'Test ({len(y_test)})')
axes[0].set_xlabel('Traffic Volume')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Train vs Test - Target Distribution', fontweight='bold')
axes[0].legend()

# Show split is representative
axes[1].bar(['Train Mean', 'Test Mean'], [y_train.mean(), y_test.mean()], color=['blue', 'red'], alpha=0.7)
axes[1].set_ylabel('Traffic Volume')
axes[1].set_title('Mean Traffic Volume: Train vs Test\n(Should be similar for good split)', fontweight='bold')
for i, v in enumerate([y_train.mean(), y_test.mean()]):
    axes[1].text(i, v + 200, f'{v:.0f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'train_test_split.png'), bbox_inches='tight')
plt.close()
print(f"  Saved: train_test_split.png")

# ============================================================================
# 14. FEATURE SCALING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.14: FEATURE SCALING (StandardScaler)")
print("=" * 70)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

print("  Scaler fitted on TRAINING data only (prevents data leakage)")
print("  Test data transformed using training scaler parameters")
print(f"\n  Before scaling - X_train sample stats:")
print(f"    Mean of first feature: {X_train.iloc[:, 0].mean():.2f}")
print(f"    Std of first feature : {X_train.iloc[:, 0].std():.2f}")
print(f"\n  After scaling - X_train_scaled sample stats:")
print(f"    Mean of first feature: {X_train_scaled.iloc[:, 0].mean():.4f} (should be ~0)")
print(f"    Std of first feature : {X_train_scaled.iloc[:, 0].std():.4f} (should be ~1)")

print(f"\n  WHY scale features?")
print(f"    - Linear Regression, Ridge, Lasso, ElasticNet: Need scaling for regularization")
print(f"    - SVR: Sensitive to feature magnitudes")
print(f"    - KNN: Distance-based, requires similar scales")
print(f"    - Decision Tree, Random Forest, Gradient Boosting: Do NOT need scaling")
print(f"    - LSTM: Benefits from normalized inputs")

# ============================================================================
# 15. SAVE PROCESSED DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.15: SAVING PROCESSED DATA")
print("=" * 70)

PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'processed_data')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Save everything needed by model scripts
df.to_csv(os.path.join(PROCESSED_DIR, 'cleaned_full_dataset.csv'), index=False)
X_train.to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index=True)
X_test.to_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index=True)
y_train.to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=True, header=True)
y_test.to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=True, header=True)
X_train_scaled.to_csv(os.path.join(PROCESSED_DIR, 'X_train_scaled.csv'), index=True)
X_test_scaled.to_csv(os.path.join(PROCESSED_DIR, 'X_test_scaled.csv'), index=True)

# Save feature list and metadata
import json
metadata = {
    'target': TARGET,
    'numeric_features': NUMERIC_FEATURES,
    'temporal_features': TEMPORAL_FEATURES,
    'encoded_features': ENCODED_FEATURES,
    'weather_dummies': list(weather_dummies.columns),
    'all_features': FINAL_FEATURES,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'total_size': len(df),
    'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
    'area_mapping': area_mapping,
    'road_mapping': road_mapping,
}
with open(os.path.join(PROCESSED_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2, default=str)

print(f"  Saved all processed data to: {PROCESSED_DIR}/")
print(f"    - cleaned_full_dataset.csv")
print(f"    - X_train.csv, X_test.csv")
print(f"    - y_train.csv, y_test.csv")
print(f"    - X_train_scaled.csv, X_test_scaled.csv")
print(f"    - metadata.json")

# ============================================================================
# 16. PREPROCESSING SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE - SUMMARY")
print("=" * 70)

summary = f"""
PREPROCESSING SUMMARY REPORT
=============================

Dataset: Bangalore Traffic Dataset
Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}
Total Records: {len(df)}

FEATURES USED ({len(FINAL_FEATURES)} total):
  - Numeric features: {len(NUMERIC_FEATURES)} (from original dataset)
  - Temporal features: {len(TEMPORAL_FEATURES)} (engineered from Date column)
  - Encoded features: {len(ENCODED_FEATURES)} (from categorical columns)
  - Weather dummies: {len(weather_dummies.columns)} (one-hot encoded)

TARGET: {TARGET}

TRAIN-TEST SPLIT:
  - Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)
  - Testing:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)
  - Random State: 42

KEY PREPROCESSING STEPS:
  1. Removed rows with null/invalid numeric values
  2. Parsed Date column -> Year, Month, Day, DayOfWeek, WeekOfYear, IsWeekend, Quarter
  3. One-hot encoded Weather Conditions (dropped first category)
  4. Binary encoded Roadwork (Yes=1, No=0)
  5. Label encoded Area Name and Road/Intersection Name
  6. Applied StandardScaler (fitted on training data only)

DATA LEAKAGE FIX:
  - Original code included 'Traffic Volume' as BOTH a feature and the target
  - This caused artificially perfect results (R²=1.0)
  - FIXED: 'Traffic Volume' is now ONLY the target variable
"""

print(summary)
with open(os.path.join(OUT_DIR, 'preprocessing_summary.txt'), 'w') as f:
    f.write(summary)

print(f"\nAll outputs saved to: {OUT_DIR}/")
print("Run step2_temporal_analysis.py next.")
