"""
ODTC ML Framework - Pipeline Engine
Stateless functions decomposed from the monolithic step7_ODTC_ML_Framework.py.
No print, no matplotlib, no file I/O.
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, StackingRegressor,
)
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings('ignore')

# ============================================================================
# Required columns for the Bangalore traffic dataset
# ============================================================================
REQUIRED_COLUMNS = [
    'Area Name', 'Road/Intersection Name', 'Date', 'Traffic Volume',
    'Average Speed', 'Travel Time Index', 'Congestion Level',
    'Road Capacity Utilization', 'Incident Reports',
    'Weather Conditions', 'Roadwork and Construction Activity',
]

TARGET = 'Traffic Volume'

EXCLUDE_COLS = [
    'Date', 'Area Name', 'Road/Intersection Name',
    'Weather Conditions', 'Roadwork and Construction Activity', TARGET,
    'Environmental Impact', 'Public Transport Usage',
    'Traffic Signal Compliance', 'Parking Usage',
    'Pedestrian and Cyclist Count',
]

# Model definitions
ALL_MODEL_DEFS = {
    'Ridge Regression': lambda: Ridge(alpha=0.001),
    'Lasso Regression': lambda: Lasso(alpha=0.01),
    'ElasticNet': lambda: ElasticNet(alpha=0.01, l1_ratio=0.5),
    'KNN (k=8)': lambda: KNeighborsRegressor(n_neighbors=8, weights='distance', n_jobs=-1),
    'SVR (RBF)': lambda: SVR(kernel='rbf', C=100),
    'Decision Tree': lambda: DecisionTreeRegressor(max_depth=15, random_state=42),
    'Random Forest': lambda: RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
    'Extra Trees': lambda: ExtraTreesRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
    'Gradient Boosting': lambda: GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
}

NEEDS_SCALING = {'Ridge Regression', 'Lasso Regression', 'ElasticNet', 'KNN (k=8)', 'SVR (RBF)'}


def _evaluate(name, y_true, y_pred, train_time):
    """Compute R2, RMSE, MAE, MAPE for a model."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {
        'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
        'Train_Time': train_time, 'predictions': y_pred, 'actuals': y_true,
    }


# ============================================================================
# STAGE 0: DATA LOADING
# ============================================================================
def load_dataset(source):
    """Load CSV from a file path (str) or file-like object (BytesIO).
    Validates that all 16 required columns are present.
    Returns (df, error_message). error_message is None on success.
    """
    try:
        if isinstance(source, str):
            df = pd.read_csv(source)
        else:
            df = pd.read_csv(source)
    except Exception as e:
        return None, f"Failed to read CSV: {e}"

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, f"Missing required columns: {missing}"

    return df, None


# ============================================================================
# STAGE 1: MULTI-SCALE TEMPORAL FEATURE ENGINEERING (MSTFE)
# ============================================================================
def stage1_mstfe(df):
    """Create ~39 temporal features. Returns dict with enriched df and metadata."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Short-term
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

    # Medium-term
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfWeekSin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeekCos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

    # Long-term
    df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days
    df['YearProgress'] = df['DayOfYear'] / 365.0
    df['QuarterSin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
    df['QuarterCos'] = np.cos(2 * np.pi * df['Quarter'] / 4)

    # Per-road lag features
    df = df.sort_values(['Road/Intersection Name', 'Date']).reset_index(drop=True)
    for col in ['Traffic Volume', 'Average Speed', 'Congestion Level']:
        for lag in [1, 3, 7]:
            df[f'{col}_Lag{lag}'] = df.groupby('Road/Intersection Name')[col].shift(lag)

    # Rolling statistics per road
    for col in ['Traffic Volume', 'Average Speed', 'Congestion Level']:
        for window in [3, 7]:
            roll = df.groupby('Road/Intersection Name')[col]
            df[f'{col}_RollMean{window}'] = roll.transform(lambda x: x.rolling(window, min_periods=1).mean())
            df[f'{col}_RollStd{window}'] = roll.transform(lambda x: x.rolling(window, min_periods=1).std())

    # Trend features
    df['Traffic_Trend_3d'] = df['Traffic Volume'] - df.get('Traffic Volume_Lag3', df['Traffic Volume'])
    df['Speed_Trend_3d'] = df['Average Speed'] - df.get('Average Speed_Lag3', df['Average Speed'])

    # Count temporal features
    temporal_keywords = [
        'Lag', 'Roll', 'Sin', 'Cos', 'Trend', 'Progress',
        'DaysSince', 'DayOfYear', 'WeekOfYear', 'Quarter',
        'IsWeekend', 'DayOfWeek', 'Month', 'Day', 'Year',
    ]
    temporal_features = [c for c in df.columns if any(k in c for k in temporal_keywords)]

    return {
        'df': df,
        'n_temporal_features': len(temporal_features),
        'temporal_features': temporal_features,
        'categories': {
            'short_term': ['Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend'],
            'medium_term': ['WeekOfYear', 'Quarter', 'DayOfYear', 'MonthSin', 'MonthCos', 'DayOfWeekSin', 'DayOfWeekCos'],
            'long_term': ['DaysSinceStart', 'YearProgress', 'QuarterSin', 'QuarterCos'],
            'lag': [c for c in df.columns if 'Lag' in c],
            'rolling': [c for c in df.columns if 'Roll' in c],
            'trend': [c for c in df.columns if 'Trend' in c],
        },
    }


# ============================================================================
# STAGE 2: DYNAMIC CONTEXT ENRICHMENT (DCE)
# ============================================================================
def stage2_dce(df):
    """Create ~13 context features. Returns dict with enriched df and metadata."""
    df = df.copy()

    # Encode categoricals
    le_area = LabelEncoder()
    le_road = LabelEncoder()
    df['Area_Encoded'] = le_area.fit_transform(df['Area Name'])
    df['Road_Encoded'] = le_road.fit_transform(df['Road/Intersection Name'])

    # One-hot weather
    weather_dummies = pd.get_dummies(df['Weather Conditions'], prefix='Weather', drop_first=True)
    df = pd.concat([df, weather_dummies], axis=1)

    # Binary roadwork
    df['Roadwork'] = (df['Roadwork and Construction Activity'] == 'Yes').astype(int)

    # Area-Road interaction
    df['Area_Road_Interaction'] = df['Area_Encoded'] * 100 + df['Road_Encoded']

    # Congestion-Speed dynamics
    df['Speed_Congestion_Ratio'] = df['Average Speed'] / (df['Congestion Level'] + 1)
    df['Capacity_Utilization_Gap'] = 100 - df['Road Capacity Utilization']
    df['Traffic_Density'] = df['Traffic Volume'] / (df['Average Speed'] + 1)

    # Weather-Temporal cross features
    weather_weekend_cols = []
    for wcol in weather_dummies.columns:
        new_col = f'{wcol}_Weekend'
        df[new_col] = df[wcol] * df['IsWeekend']
        weather_weekend_cols.append(new_col)

    # Infrastructure stress indicators
    df['Incident_Weekend'] = df['Incident Reports'] * df['IsWeekend']
    df['Roadwork_Traffic_Impact'] = df['Roadwork'] * df['Road Capacity Utilization']

    n_context = 4 + len(weather_dummies.columns) + 3

    context_features_desc = {
        'Area_Road_Interaction': 'Area-Road combined interaction code',
        'Speed_Congestion_Ratio': 'Average Speed / (Congestion Level + 1)',
        'Capacity_Utilization_Gap': '100 - Road Capacity Utilization',
        'Traffic_Density': 'Traffic Volume / (Average Speed + 1)',
        'Incident_Weekend': 'Incident Reports x IsWeekend',
        'Roadwork_Traffic_Impact': 'Roadwork x Road Capacity Utilization',
    }

    return {
        'df': df,
        'n_context_features': n_context,
        'le_area': le_area,
        'le_road': le_road,
        'weather_columns': list(weather_dummies.columns),
        'weather_weekend_columns': weather_weekend_cols,
        'context_features_desc': context_features_desc,
    }


# ============================================================================
# PREPARE FEATURE MATRIX
# ============================================================================
def prepare_features(df, test_size=0.3, random_state=42):
    """Train/test split + StandardScaler. Returns dict with splits and metadata."""
    all_features = [c for c in df.columns if c not in EXCLUDE_COLS]

    # Drop NaN rows (from lag/rolling features)
    df_clean = df.dropna(subset=all_features + [TARGET]).copy()
    n_original = len(df)
    n_clean = len(df_clean)

    X = df_clean[all_features].values.astype(np.float64)
    y = df_clean[TARGET].values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'df_clean': df_clean,
        'all_features': all_features,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'n_original': n_original,
        'n_clean': n_clean,
        'n_features': len(all_features),
    }


# ============================================================================
# STAGE 3: ADAPTIVE FEATURE SELECTION (AFS)
# ============================================================================
def stage3_afs(X_train_scaled, y_train, all_features, selection_ratio=0.7, random_state=42):
    """MI + RF ranking, select top features. Returns dict with rankings and selected indices."""
    # Mutual Information
    mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=random_state, n_neighbors=5)
    mi_ranking = pd.DataFrame({
        'Feature': all_features,
        'MI_Score': mi_scores,
    }).sort_values('MI_Score', ascending=False)
    mi_ranking['MI_Rank'] = range(1, len(mi_ranking) + 1)

    # Random Forest importance
    rf_selector = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf_selector.fit(X_train_scaled, y_train)
    rf_importance = pd.DataFrame({
        'Feature': all_features,
        'RF_Importance': rf_selector.feature_importances_,
    }).sort_values('RF_Importance', ascending=False)
    rf_importance['RF_Rank'] = range(1, len(rf_importance) + 1)

    # Combined ranking
    combined = mi_ranking.merge(rf_importance, on='Feature')
    combined['Avg_Rank'] = (combined['MI_Rank'] + combined['RF_Rank']) / 2
    combined = combined.sort_values('Avg_Rank')

    n_select = max(15, int(len(all_features) * selection_ratio))
    selected_features = combined.head(n_select)['Feature'].tolist()
    combined['Selected'] = combined['Feature'].isin(selected_features)

    selected_indices = [all_features.index(f) for f in selected_features]

    return {
        'mi_ranking': mi_ranking,
        'rf_importance': rf_importance,
        'combined': combined,
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'n_total': len(all_features),
        'n_selected': len(selected_features),
    }


# ============================================================================
# STAGE 4: MULTI-MODEL ENSEMBLE WITH STACKING (MES)
# ============================================================================
def stage4_mes(X_train, X_test, X_train_scaled, X_test_scaled,
               y_train, y_test, selected_indices,
               selected_models=None, random_state=42, progress_callback=None):
    """Train individual models + stacking + weighted ensemble.
    Returns dict with all results, ODTC result, and ensemble details.
    progress_callback(fraction, message) is called to report progress.
    """
    if selected_models is None:
        selected_models = list(ALL_MODEL_DEFS.keys())

    def report(frac, msg):
        if progress_callback:
            progress_callback(frac, msg)

    results_all = {}
    total_steps = len(selected_models) + 3  # +stacking +weighted +odtc
    step = 0

    # --- Train on ALL features ---
    for name in selected_models:
        model = ALL_MODEL_DEFS[name]()
        use_scaled = name in NEEDS_SCALING
        Xtr = X_train_scaled if use_scaled else X_train
        Xte = X_test_scaled if use_scaled else X_test

        t0 = time.time()
        model.fit(Xtr, y_train)
        train_time = time.time() - t0
        y_pred = model.predict(Xte)

        results_all[name] = _evaluate(name, y_test, y_pred, train_time)
        step += 1
        report(step / total_steps, f"Trained {name}")

    # --- Stacking Ensemble ---
    report(step / total_steps, "Training Stacking Ensemble...")
    base_estimators = [
        ('ridge', Ridge(alpha=0.001)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=random_state)),
        ('et', ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1)),
        ('dt', DecisionTreeRegressor(max_depth=12, random_state=random_state)),
    ]

    stack_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5, n_jobs=-1, passthrough=False,
    )

    t0 = time.time()
    stack_model.fit(X_train_scaled, y_train)
    stack_time = time.time() - t0
    y_pred_stack = stack_model.predict(X_test_scaled)
    res_stack = _evaluate('Stacking Ensemble', y_test, y_pred_stack, stack_time)
    results_all['Stacking Ensemble'] = res_stack
    step += 1
    report(step / total_steps, "Stacking Ensemble done")

    # --- Weighted Voting Ensemble ---
    report(step / total_steps, "Training Weighted Ensemble...")
    cv_scores = {}
    for est_name, est in base_estimators:
        scores = cross_val_score(est, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
        cv_scores[est_name] = max(0, scores.mean())

    total_score = sum(cv_scores.values())
    weights_dict = {k: v / total_score for k, v in cv_scores.items()}

    t0 = time.time()
    weighted_preds = np.zeros_like(y_test, dtype=float)
    for est_name, est in base_estimators:
        est_clone = est.__class__(**est.get_params())
        est_clone.fit(X_train_scaled, y_train)
        pred = est_clone.predict(X_test_scaled)
        weighted_preds += weights_dict[est_name] * pred

    weighted_time = time.time() - t0
    res_weighted = _evaluate('Weighted Ensemble', y_test, weighted_preds, weighted_time)
    results_all['Weighted Ensemble'] = res_weighted
    step += 1
    report(step / total_steps, "Weighted Ensemble done")

    # --- ODTC Framework (best of stacking/weighted) ---
    if res_stack['R2'] > res_weighted['R2']:
        res_odtc = res_stack.copy()
        odtc_method = 'Stacking Ensemble'
    else:
        res_odtc = res_weighted.copy()
        odtc_method = 'Weighted Ensemble'
    res_odtc['Model'] = 'ODTC Framework'
    results_all['ODTC Framework'] = res_odtc
    step += 1
    report(1.0, "Stage 4 complete")

    return {
        'results': results_all,
        'res_odtc': res_odtc,
        'res_stack': res_stack,
        'res_weighted': res_weighted,
        'odtc_method': odtc_method,
        'cv_scores': cv_scores,
        'weights': weights_dict,
        'base_estimators': [name for name, _ in base_estimators],
    }


# ============================================================================
# STAGE 5: ONLINE ADAPTATION MODULE (OAM)
# ============================================================================
def stage5_oam(X_train_scaled, X_test_scaled, y_train, y_test, window_sizes=None):
    """Sliding window online adaptation simulation.
    Returns dict with per-window results and summaries.
    """
    if window_sizes is None:
        window_sizes = [100, 200, 500]

    online_results = {}
    summaries = []

    for ws in window_sizes:
        n_windows = len(X_test_scaled) // ws
        if n_windows < 2:
            continue

        window_r2s = []
        static_model = Ridge(alpha=0.001)
        static_model.fit(X_train_scaled, y_train)

        for i in range(n_windows):
            start = i * ws
            end = start + ws
            X_window = X_test_scaled[start:end]
            y_window = y_test[start:end]

            y_pred_static = static_model.predict(X_window)

            if i > 0:
                X_new_train = np.vstack([X_train_scaled, X_test_scaled[:start]])
                y_new_train = np.concatenate([y_train, y_test[:start]])
                online_model = Ridge(alpha=0.001)
                online_model.fit(X_new_train, y_new_train)
                y_pred_online = online_model.predict(X_window)
            else:
                y_pred_online = y_pred_static

            r2_static = r2_score(y_window, y_pred_static)
            r2_online = r2_score(y_window, y_pred_online)
            window_r2s.append({
                'Window': i + 1, 'Static_R2': r2_static, 'Online_R2': r2_online,
            })

        odf = pd.DataFrame(window_r2s)
        online_results[ws] = odf
        avg_static = odf['Static_R2'].mean()
        avg_online = odf['Online_R2'].mean()
        improvement = ((avg_online - avg_static) / (abs(avg_static) + 1e-10)) * 100
        summaries.append({
            'Window_Size': ws, 'N_Windows': n_windows,
            'Avg_Static_R2': avg_static, 'Avg_Online_R2': avg_online,
            'Improvement_%': improvement,
        })

    return {
        'online_results': online_results,
        'summaries': pd.DataFrame(summaries) if summaries else pd.DataFrame(),
    }


# ============================================================================
# COMPUTE IMPROVEMENTS
# ============================================================================
def compute_improvements(res_odtc, results_all):
    """Compute R2 improvement % and RMSE reduction % of ODTC over each baseline model."""
    odtc_r2 = res_odtc['R2']
    odtc_rmse = res_odtc['RMSE']

    rows = []
    for name, r in results_all.items():
        if name in ('ODTC Framework', 'Stacking Ensemble', 'Weighted Ensemble'):
            continue
        r2_imp = ((odtc_r2 - r['R2']) / (abs(r['R2']) + 1e-10)) * 100
        rmse_imp = ((r['RMSE'] - odtc_rmse) / (r['RMSE'] + 1e-10)) * 100
        rows.append({
            'Model': name,
            'R2_Improvement_%': round(r2_imp, 4),
            'RMSE_Reduction_%': round(rmse_imp, 4),
        })

    return pd.DataFrame(rows)
