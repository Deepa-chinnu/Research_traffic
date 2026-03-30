# ODTC ML Framework - Novel Contribution

## Online Dynamic Temporal Context (ODTC) Machine Learning Framework
**A Novel Framework for Urban Traffic Flow Prediction**

## Overview
The ODTC Framework is the novel contribution of this PhD research. It is a systematic
5-stage ML pipeline that transforms raw traffic data into highly accurate predictions
through feature engineering, context enrichment, adaptive selection, and ensemble stacking.

## 5-Stage Pipeline

| Stage | Abbreviation | Full Name | What It Does |
|-------|-------------|-----------|--------------|
| 1 | MSTFE | Multi-Scale Temporal Feature Engineering | Creates 39 temporal features (lag, rolling, trend, cyclical) |
| 2 | DCE | Dynamic Context Enrichment | Creates ~13 context features (interactions, dynamics, cross) |
| 3 | AFS | Adaptive Feature Selection | Selects top 47 from 68 features using MI + RF ranking |
| 4 | MES | Multi-Model Ensemble with Stacking | Level-0 base learners + Level-1 meta-learner |
| 5 | OAM | Online Adaptation Module | Sliding window retraining for real-world deployment |

## Results
- **ODTC Framework R² = 1.000000** (RMSE = 0.01, MAPE = 0.00%)
- Best ensemble method: Stacking (Ridge + RF + GB + ExtraTrees + DT with Ridge meta-learner)

## Files

### Scripts
| File | Description |
|------|-------------|
| `step7_ODTC_ML_Framework.py` | Main framework script (all 5 stages) |
| `generate_flow_diagrams.py` | Generates architecture and comparison diagrams |

### Output Files (in `outputs/`)
| File | Description |
|------|-------------|
| `01_feature_selection.png` | MI and RF feature importance rankings |
| `02_framework_architecture.png` | 5-stage framework architecture diagram |
| `03_R2_ranking.png` | All models ranked by R2 score |
| `04_multi_metric_comparison.png` | R2, RMSE, MAE, MAPE bar charts |
| `05_improvement_over_baselines.png` | ODTC improvement % over each model |
| `06_predicted_vs_actual.png` | Scatter plots (Ridge vs GB vs ODTC) |
| `07_online_adaptation.png` | Static vs online adapted model |
| `08_time_vs_performance.png` | Training time vs accuracy tradeoff |
| `09_results_table.png` | Complete results summary table |
| `framework_report.txt` | Comprehensive text report with committee Q&A |
| `comparison_results.csv` | Raw CSV results |

### Flow Diagrams (in `outputs/`)
| File | Description |
|------|-------------|
| `FULL_RESEARCH_ARCHITECTURE.png` | Complete PhD research flow (all 4 phases) |
| `ODTC_FRAMEWORK_DETAILED_PIPELINE.png` | Detailed 5-stage pipeline diagram |
| `MODEL_COMPARISON_BEFORE_VS_AFTER.png` | Individual models vs ODTC framework |
| `LITERATURE_VS_ODTC_COMPARISON.png` | ODTC vs literature review papers |

## How to Run
```bash
python step7_ODTC_ML_Framework.py     # Run the framework
python generate_flow_diagrams.py       # Generate flow diagrams
```

## Inspired By (Literature Review)
- Bartlett et al. - Deep RNN framework with STATS metric
- Li et al. (2022) - Spatio-Temporal Graph Neural Network
- Zhou et al. (2024) - Knowledge Representation STGNN
- Zou et al. (2024) - CNN-LSTM hybrid with detector clustering
