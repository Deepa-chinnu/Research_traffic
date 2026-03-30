# ODTC ML Framework - Dashboard User Guide

## Overview

The ODTC (Online Dynamic Temporal Context) Dashboard is an interactive Streamlit interface for the traffic flow prediction pipeline. It wraps the 5-stage ML framework into a point-and-click tool with real-time progress tracking and interactive Plotly charts.

## How to Run

```bash
pip install -r requirements_ui.txt
streamlit run app.py --server.headless true
```

Open **http://localhost:8501** in your browser.

---

## Sidebar Controls

### Data Source
- **Use default Bangalore dataset**: Loads `RAC4/Banglore_traffic_Dataset.csv` (8936 records, 16 columns, 8 areas, 16 roads)
- **Upload CSV**: Supply your own dataset (must contain all 16 required columns)

### Pipeline Parameters

| Parameter | Range | Default | Meaning |
|---|---|---|---|
| **Test split ratio** | 0.1 - 0.5 | 0.3 | Fraction of data held out for testing. 0.3 means 70% train, 30% test |
| **Feature selection threshold** | 0.3 - 1.0 | 0.7 | Fraction of features to keep after ranking. 0.7 keeps top 70%, drops bottom 30%. Lower = fewer but stronger features |
| **Random state** | 0 - 9999 | 42 | Seed for reproducibility. Same value = same results every run |

### Model Selection
Choose which of the 9 base models to train:
- **Ridge Regression** - Linear model with L2 regularization
- **Lasso Regression** - Linear model with L1 regularization (can zero out features)
- **ElasticNet** - Combines L1 and L2 regularization
- **KNN (k=8)** - Predicts based on 8 nearest neighbors
- **SVR (RBF)** - Support Vector Regression (slow on large datasets)
- **Decision Tree** - Single tree-based model
- **Random Forest** - Ensemble of 200 decision trees
- **Extra Trees** - Ensemble of 200 randomized trees
- **Gradient Boosting** - Sequential boosting with 300 estimators

Minimum 2 models required for the ensemble to work.

### Online Adaptation
- **Window sizes**: Controls the sliding window sizes for Stage 5 simulation (default: 100, 200, 500)

### Run Button
Click **"Run Full Pipeline"** to execute all 5 stages. A progress bar shows real-time status.

---

## Dashboard Tabs

### Tab 1: Dashboard
The main results overview:
- **4 metric cards**: R², RMSE, MAE, MAPE for the ODTC Framework
- **Ranked results table**: All models sorted by R² score
- **R² bar chart**: Visual ranking of all models
- **Time vs Accuracy scatter**: Shows training time trade-offs
- **Improvement chart**: How much ODTC improves over each baseline model

### Tab 2: Data Explorer
Explore the raw dataset:
- Dataset preview (first 100 rows)
- Descriptive statistics (mean, std, min, max, quartiles)
- Column distribution histograms (select any numeric column)
- Area breakdown pie chart
- Road breakdown bar chart

### Tab 3: Stage 1 - MSTFE (Multi-Scale Temporal Feature Engineering)
Shows the ~39 temporal features created:
- **Short-term**: Year, Month, Day, DayOfWeek, IsWeekend
- **Medium-term**: WeekOfYear, Quarter, cyclical sin/cos encodings
- **Long-term**: DaysSinceStart, YearProgress, seasonal indicators
- **Lag features**: 1, 3, 7-day lags for Traffic Volume, Speed, Congestion
- **Rolling stats**: 3-day and 7-day rolling mean and standard deviation
- **Trend features**: 3-day traffic and speed trends

Includes an interactive time-series chart showing traffic volume with rolling means per road.

### Tab 4: Stage 2 - DCE (Dynamic Context Enrichment)
Shows the ~13 context features:
- Area-Road interaction codes
- Speed-Congestion dynamics (ratio, gap, density)
- Weather-temporal cross features (weather × weekend)
- Infrastructure stress indicators

Includes Speed vs Congestion scatter plot by area and weather impact bar chart.

### Tab 5: Stage 3 - AFS (Adaptive Feature Selection)
Shows how features were ranked and selected:
- **Mutual Information** bar chart (top 20 features)
- **Random Forest Importance** bar chart (top 20 features)
- **Combined ranking table** with MI rank, RF rank, average rank, and selection status

### Tab 6: Stage 4 - MES (Multi-Model Ensemble with Stacking)
The core model training results:
- Complete results table for all models
- 2×2 metric comparison (R², RMSE, MAE, MAPE)
- **Predicted vs Actual scatter** with model selector dropdown
- ODTC improvement chart over baselines
- Ensemble details (stacking base estimators, weighted ensemble weights)
- Residual distribution histogram

### Tab 7: Stage 5 - OAM (Online Adaptation Module)
Simulates online learning:
- Per-window summary table (static vs online R², improvement %)
- **Static vs Online R² line chart** with filled area showing improvement
- Summary narrative with best improvement statistics

---

## Typical Workflow

1. Open the app and keep default settings for first run
2. Click **"Run Full Pipeline"** (takes 1-3 minutes depending on models selected)
3. Check the **Dashboard** tab for overall results
4. Browse **Stage 1-5** tabs to understand each pipeline step
5. Experiment: change test split, feature threshold, or deselect slow models (SVR)
6. Click **"Run Full Pipeline"** again to compare results

## Notes

- Results persist across tab switches (stored in session state)
- Deselecting SVR speeds up the pipeline significantly
- The ODTC Framework result is the best of Stacking or Weighted Ensemble
- All charts are interactive: hover for values, zoom, pan, and download as PNG
