"""
ODTC ML Framework - Interactive Streamlit Dashboard
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from odtc_engine import (
    load_dataset, stage1_mstfe, stage2_dce, prepare_features,
    stage3_afs, stage4_mes, stage5_oam, compute_improvements,
    ALL_MODEL_DEFS, TARGET,
)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="ODTC ML Framework",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'RAC4', 'Banglore_traffic_Dataset.csv')

MODEL_COLORS = {
    'ODTC Framework': '#e74c3c',
    'Stacking Ensemble': '#f39c12',
    'Weighted Ensemble': '#f1c40f',
    'Ridge Regression': '#2ecc71',
    'Lasso Regression': '#1abc9c',
    'ElasticNet': '#16a085',
    'KNN (k=8)': '#3498db',
    'SVR (RBF)': '#e67e22',
    'Decision Tree': '#95a5a6',
    'Random Forest': '#27ae60',
    'Extra Trees': '#2980b9',
    'Gradient Boosting': '#8e44ad',
}


def get_color(name):
    return MODEL_COLORS.get(name, '#34495e')


# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("ODTC Framework")
st.sidebar.markdown("---")

# Data source
st.sidebar.subheader("Data Source")
use_default = st.sidebar.checkbox("Use default Bangalore dataset", value=True)
uploaded_file = None
if not use_default:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

# Pipeline parameters
st.sidebar.subheader("Pipeline Parameters")
test_size = st.sidebar.slider("Test split ratio", 0.1, 0.5, 0.3, 0.05)
selection_ratio = st.sidebar.slider("Feature selection threshold", 0.3, 1.0, 0.7, 0.05)
random_state = st.sidebar.number_input("Random state", 0, 9999, 42)

# Model selection
st.sidebar.subheader("Model Selection")
all_model_names = list(ALL_MODEL_DEFS.keys())
selected_models = st.sidebar.multiselect(
    "Select models to train",
    all_model_names,
    default=all_model_names,
)

if len(selected_models) < 2:
    st.sidebar.warning("Select at least 2 models for ensemble.")

# Online adaptation
st.sidebar.subheader("Online Adaptation")
window_options = [50, 100, 200, 500, 1000]
window_sizes = st.sidebar.multiselect(
    "Window sizes",
    window_options,
    default=[100, 200, 500],
)

# SVR warning
if 'SVR (RBF)' in selected_models:
    st.sidebar.info("SVR can be slow on large datasets.")

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Run Full Pipeline", type="primary", disabled=len(selected_models) < 2)


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================
def run_pipeline():
    """Execute the full ODTC pipeline and store results in session_state."""
    progress = st.progress(0, text="Loading dataset...")

    # --- Load ---
    if use_default:
        source = DEFAULT_DATA_PATH
    elif uploaded_file is not None:
        source = uploaded_file
    else:
        st.error("No dataset provided. Upload a CSV or enable the default dataset.")
        return

    df_raw, err = load_dataset(source)
    if err:
        st.error(err)
        return
    st.session_state['df_raw'] = df_raw
    progress.progress(5, text="Dataset loaded. Running Stage 1: MSTFE...")

    # --- Stage 1 ---
    s1 = stage1_mstfe(df_raw)
    st.session_state['stage1'] = s1
    progress.progress(15, text="Stage 1 complete. Running Stage 2: DCE...")

    # --- Stage 2 ---
    s2 = stage2_dce(s1['df'])
    st.session_state['stage2'] = s2
    progress.progress(25, text="Stage 2 complete. Preparing features...")

    # --- Prepare ---
    prep = prepare_features(s2['df'], test_size=test_size, random_state=random_state)
    st.session_state['prep'] = prep
    progress.progress(35, text="Features ready. Running Stage 3: AFS...")

    # --- Stage 3 ---
    s3 = stage3_afs(
        prep['X_train_scaled'], prep['y_train'],
        prep['all_features'], selection_ratio=selection_ratio,
        random_state=random_state,
    )
    st.session_state['stage3'] = s3
    progress.progress(40, text="Stage 3 complete. Running Stage 4: MES...")

    # --- Stage 4 ---
    def progress_cb(frac, msg):
        pct = int(45 + frac * 40)
        progress.progress(min(pct, 85), text=f"Stage 4: {msg}")

    s4 = stage4_mes(
        prep['X_train'], prep['X_test'],
        prep['X_train_scaled'], prep['X_test_scaled'],
        prep['y_train'], prep['y_test'],
        s3['selected_indices'],
        selected_models=selected_models,
        random_state=random_state,
        progress_callback=progress_cb,
    )
    st.session_state['stage4'] = s4
    progress.progress(85, text="Stage 4 complete. Running Stage 5: OAM...")

    # --- Stage 5 ---
    s5 = stage5_oam(
        prep['X_train_scaled'], prep['X_test_scaled'],
        prep['y_train'], prep['y_test'],
        window_sizes=window_sizes if window_sizes else [100, 200, 500],
    )
    st.session_state['stage5'] = s5
    progress.progress(95, text="Stage 5 complete. Computing improvements...")

    # --- Improvements ---
    imp_df = compute_improvements(s4['res_odtc'], s4['results'])
    st.session_state['improvements'] = imp_df

    # --- Ranked results ---
    ranked = sorted(s4['results'].items(), key=lambda x: x[1]['R2'], reverse=True)
    st.session_state['ranked'] = ranked

    st.session_state['pipeline_complete'] = True
    progress.progress(100, text="Pipeline complete!")


if run_btn:
    run_pipeline()

# ============================================================================
# MAIN AREA - TABS
# ============================================================================
st.title("Online Dynamic Temporal Context (ODTC) ML Framework")

tabs = st.tabs([
    "Dashboard", "Data Explorer",
    "Stage 1: MSTFE", "Stage 2: DCE", "Stage 3: AFS",
    "Stage 4: MES", "Stage 5: OAM",
])

pipeline_done = st.session_state.get('pipeline_complete', False)

# ============================================================================
# TAB 0: DASHBOARD
# ============================================================================
with tabs[0]:
    if not pipeline_done:
        st.info("Configure parameters in the sidebar and click **Run Full Pipeline** to begin.")
    else:
        s4 = st.session_state['stage4']
        res_odtc = s4['res_odtc']
        ranked = st.session_state['ranked']
        imp_df = st.session_state['improvements']

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R\u00b2 Score", f"{res_odtc['R2']:.6f}")
        c2.metric("RMSE", f"{res_odtc['RMSE']:.2f}")
        c3.metric("MAE", f"{res_odtc['MAE']:.2f}")
        c4.metric("MAPE", f"{res_odtc['MAPE']:.4f}%")

        st.markdown(f"**ODTC Best Method:** {s4['odtc_method']}")

        # Ranked results table
        st.subheader("Model Rankings")
        table_rows = []
        for i, (name, r) in enumerate(ranked, 1):
            table_rows.append({
                'Rank': i, 'Model': name,
                'R\u00b2': round(r['R2'], 6), 'RMSE': round(r['RMSE'], 2),
                'MAE': round(r['MAE'], 2), 'MAPE (%)': round(r['MAPE'], 4),
                'Time (s)': round(r['Train_Time'], 2),
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        col_left, col_right = st.columns(2)

        # R2 bar chart
        with col_left:
            names = [n for n, _ in ranked]
            r2_vals = [r['R2'] for _, r in ranked]
            colors = [get_color(n) for n in names]
            fig = go.Figure(go.Bar(
                y=names, x=r2_vals, orientation='h',
                marker_color=colors, text=[f"{v:.6f}" for v in r2_vals],
                textposition='outside',
            ))
            fig.update_layout(
                title="Model Ranking by R\u00b2",
                xaxis_title="R\u00b2 Score", yaxis=dict(autorange="reversed"),
                height=400, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Time vs accuracy bubble chart
        with col_right:
            fig = go.Figure()
            for name, r in ranked:
                fig.add_trace(go.Scatter(
                    x=[r['Train_Time']], y=[r['R2']],
                    mode='markers+text', text=[name], textposition='top center',
                    marker=dict(size=18, color=get_color(name), line=dict(width=1, color='black')),
                    name=name, showlegend=False,
                ))
            fig.update_layout(
                title="Training Time vs R\u00b2",
                xaxis_title="Training Time (s)", yaxis_title="R\u00b2 Score",
                height=400, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Improvement table
        st.subheader("ODTC Improvement Over Baselines")
        if not imp_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=imp_df['Model'], y=imp_df['R2_Improvement_%'],
                name='R\u00b2 Improvement %', marker_color='#2ecc71',
            ))
            fig.add_trace(go.Bar(
                x=imp_df['Model'], y=imp_df['RMSE_Reduction_%'],
                name='RMSE Reduction %', marker_color='#3498db',
            ))
            fig.update_layout(
                barmode='group', height=400,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(imp_df, use_container_width=True, hide_index=True)


# ============================================================================
# TAB 1: DATA EXPLORER
# ============================================================================
with tabs[1]:
    if not pipeline_done:
        st.info("Run the pipeline first.")
    else:
        df_raw = st.session_state['df_raw']

        st.subheader("Dataset Preview")
        st.dataframe(df_raw.head(100), use_container_width=True, height=300)

        st.subheader("Descriptive Statistics")
        st.dataframe(df_raw.describe(), use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Column Distributions")
            numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            hist_col = st.selectbox("Select column", numeric_cols, key="hist_col")
            fig = px.histogram(df_raw, x=hist_col, nbins=40, title=f"Distribution of {hist_col}",
                               color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Area Breakdown")
            area_counts = df_raw['Area Name'].value_counts()
            fig = px.pie(values=area_counts.values, names=area_counts.index,
                         title="Records by Area")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Road Breakdown")
        road_counts = df_raw['Road/Intersection Name'].value_counts()
        fig = px.bar(x=road_counts.index, y=road_counts.values,
                     title="Records by Road/Intersection",
                     labels={'x': 'Road', 'y': 'Count'},
                     color_discrete_sequence=['#27ae60'])
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 2: STAGE 1 - MSTFE
# ============================================================================
with tabs[2]:
    if not pipeline_done:
        st.info("Run the pipeline first.")
    else:
        s1 = st.session_state['stage1']
        st.subheader(f"Stage 1: Multi-Scale Temporal Feature Engineering")
        st.markdown(f"**Total temporal features created:** {s1['n_temporal_features']}")

        # Category breakdown
        cats = s1['categories']
        cat_data = []
        for cat_name, features in cats.items():
            cat_data.append({'Category': cat_name, 'Count': len(features), 'Features': ', '.join(features[:5]) + ('...' if len(features) > 5 else '')})
        st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)

        # Sample data with new columns
        st.subheader("Sample Data with Temporal Features")
        sample_cols = ['Date', 'Road/Intersection Name', 'Traffic Volume'] + s1['temporal_features'][:10]
        available_cols = [c for c in sample_cols if c in s1['df'].columns]
        st.dataframe(s1['df'][available_cols].head(20), use_container_width=True, height=300)

        # Traffic time-series with rolling means
        st.subheader("Traffic Volume with Rolling Means")
        df_s1 = s1['df'].copy()
        roads = df_s1['Road/Intersection Name'].unique()
        selected_road = st.selectbox("Select road", roads, key="s1_road")
        road_df = df_s1[df_s1['Road/Intersection Name'] == selected_road].sort_values('Date')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=road_df['Date'], y=road_df['Traffic Volume'],
                                 mode='lines', name='Traffic Volume', line=dict(color='#3498db', width=1)))
        if 'Traffic Volume_RollMean3' in road_df.columns:
            fig.add_trace(go.Scatter(x=road_df['Date'], y=road_df['Traffic Volume_RollMean3'],
                                     mode='lines', name='3-day Rolling Mean', line=dict(color='#e74c3c', width=2)))
        if 'Traffic Volume_RollMean7' in road_df.columns:
            fig.add_trace(go.Scatter(x=road_df['Date'], y=road_df['Traffic Volume_RollMean7'],
                                     mode='lines', name='7-day Rolling Mean', line=dict(color='#2ecc71', width=2)))
        fig.update_layout(title=f"Traffic Volume - {selected_road}", height=400)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 3: STAGE 2 - DCE
# ============================================================================
with tabs[3]:
    if not pipeline_done:
        st.info("Run the pipeline first.")
    else:
        s2 = st.session_state['stage2']
        st.subheader("Stage 2: Dynamic Context Enrichment")
        st.markdown(f"**Total context features created:** ~{s2['n_context_features']}")

        # Feature descriptions
        st.subheader("Context Feature Descriptions")
        desc_rows = [{'Feature': k, 'Description': v} for k, v in s2['context_features_desc'].items()]
        st.dataframe(pd.DataFrame(desc_rows), use_container_width=True, hide_index=True)

        col_l, col_r = st.columns(2)

        df_s2 = s2['df']

        # Speed-Congestion scatter by area
        with col_l:
            st.subheader("Speed vs Congestion by Area")
            sample = df_s2.sample(min(2000, len(df_s2)), random_state=42)
            fig = px.scatter(sample, x='Congestion Level', y='Average Speed',
                             color='Area Name', opacity=0.5,
                             title="Speed-Congestion Scatter by Area")
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        # Weather impact
        with col_r:
            st.subheader("Weather Impact on Traffic")
            weather_avg = df_s2.groupby('Weather Conditions')[TARGET].mean().reset_index()
            fig = px.bar(weather_avg, x='Weather Conditions', y=TARGET,
                         title="Avg Traffic Volume by Weather",
                         color_discrete_sequence=['#e67e22'])
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 4: STAGE 3 - AFS
# ============================================================================
with tabs[4]:
    if not pipeline_done:
        st.info("Run the pipeline first.")
    else:
        s3 = st.session_state['stage3']
        st.subheader("Stage 3: Adaptive Feature Selection")
        st.markdown(f"**Selected {s3['n_selected']} features from {s3['n_total']} total**")

        col_l, col_r = st.columns(2)

        # MI scores top 20
        with col_l:
            top20_mi = s3['mi_ranking'].head(20)
            fig = go.Figure(go.Bar(
                y=top20_mi['Feature'], x=top20_mi['MI_Score'],
                orientation='h', marker_color='#3498db',
                text=[f"{v:.4f}" for v in top20_mi['MI_Score']],
                textposition='outside',
            ))
            fig.update_layout(
                title="Top 20 - Mutual Information",
                yaxis=dict(autorange="reversed"), height=600,
                margin=dict(l=10, r=80, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # RF importance top 20
        with col_r:
            top20_rf = s3['rf_importance'].head(20)
            fig = go.Figure(go.Bar(
                y=top20_rf['Feature'], x=top20_rf['RF_Importance'],
                orientation='h', marker_color='#27ae60',
                text=[f"{v:.4f}" for v in top20_rf['RF_Importance']],
                textposition='outside',
            ))
            fig.update_layout(
                title="Top 20 - Random Forest Importance",
                yaxis=dict(autorange="reversed"), height=600,
                margin=dict(l=10, r=80, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Combined ranking table
        st.subheader("Combined Feature Ranking")
        display_df = s3['combined'][['Feature', 'MI_Score', 'MI_Rank', 'RF_Importance', 'RF_Rank', 'Avg_Rank', 'Selected']].copy()
        display_df = display_df.sort_values('Avg_Rank')
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)


# ============================================================================
# TAB 5: STAGE 4 - MES
# ============================================================================
with tabs[5]:
    if not pipeline_done:
        st.info("Run the pipeline first.")
    else:
        s4 = st.session_state['stage4']
        ranked = st.session_state['ranked']
        results = s4['results']
        imp_df = st.session_state['improvements']

        st.subheader("Stage 4: Multi-Model Ensemble with Stacking")
        st.markdown(f"**ODTC Best Method:** {s4['odtc_method']}")

        # Model results table
        st.subheader("All Model Results")
        rows = []
        for i, (name, r) in enumerate(ranked, 1):
            rows.append({
                'Rank': i, 'Model': name,
                'R\u00b2': round(r['R2'], 6), 'RMSE': round(r['RMSE'], 2),
                'MAE': round(r['MAE'], 2), 'MAPE (%)': round(r['MAPE'], 4),
                'Time (s)': round(r['Train_Time'], 2),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # 2x2 metric comparison
        st.subheader("Multi-Metric Comparison")
        key_models = [n for n, _ in ranked if n not in ('Stacking Ensemble', 'Weighted Ensemble')]
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=('R\u00b2 Score', 'RMSE (Lower is Better)',
                                            'MAE (Lower is Better)', 'MAPE % (Lower is Better)'))

        names_k = [n for n in key_models if n in results]
        colors_k = [get_color(n) for n in names_k]

        for metric, row, col in [('R2', 1, 1), ('RMSE', 1, 2), ('MAE', 2, 1), ('MAPE', 2, 2)]:
            vals = [results[n][metric] for n in names_k]
            fig.add_trace(go.Bar(x=names_k, y=vals, marker_color=colors_k, showlegend=False), row=row, col=col)

        fig.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Predicted vs actual scatter
        st.subheader("Predicted vs Actual")
        model_sel = st.selectbox("Select model", [n for n, _ in ranked], key="pva_model")
        r = results[model_sel]
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=r['actuals'], y=r['predictions'],
            mode='markers', marker=dict(size=4, color=get_color(model_sel), opacity=0.4),
            name='Predictions',
        ))
        max_val = max(r['actuals'].max(), r['predictions'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', line=dict(dash='dash', color='black'),
            name='Perfect',
        ))
        fig.update_layout(
            title=f"{model_sel} - R\u00b2={r['R2']:.6f}",
            xaxis_title="Actual", yaxis_title="Predicted",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Improvement chart
        st.subheader("ODTC Improvement Over Baselines")
        if not imp_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=imp_df['Model'], y=imp_df['R2_Improvement_%'],
                                 name='R\u00b2 Improvement %', marker_color='#2ecc71'))
            fig.add_trace(go.Bar(x=imp_df['Model'], y=imp_df['RMSE_Reduction_%'],
                                 name='RMSE Reduction %', marker_color='#3498db'))
            fig.update_layout(barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Ensemble details
        st.subheader("Ensemble Details")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("**Stacking Ensemble Base Estimators:**")
            st.write(', '.join(s4['base_estimators']))
            st.markdown(f"**Stacking R\u00b2:** {s4['res_stack']['R2']:.6f}")
        with col_e2:
            st.markdown("**Weighted Ensemble Weights:**")
            w_df = pd.DataFrame([
                {'Estimator': k, 'Weight': round(v, 4)} for k, v in s4['weights'].items()
            ])
            st.dataframe(w_df, use_container_width=True, hide_index=True)
            st.markdown(f"**Weighted R\u00b2:** {s4['res_weighted']['R2']:.6f}")

        # Residual histogram
        st.subheader("Residual Distribution")
        residuals = r['actuals'] - r['predictions']
        fig = px.histogram(x=residuals, nbins=50, title=f"Residuals - {model_sel}",
                           labels={'x': 'Residual'}, color_discrete_sequence=['#8e44ad'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 6: STAGE 5 - OAM
# ============================================================================
with tabs[6]:
    if not pipeline_done:
        st.info("Run the pipeline first.")
    else:
        s5 = st.session_state['stage5']
        st.subheader("Stage 5: Online Adaptation Module")

        if s5['summaries'].empty:
            st.warning("No windows could be created with the selected sizes. Try smaller window sizes.")
        else:
            # Summary table
            st.subheader("Per-Window Summary")
            st.dataframe(s5['summaries'], use_container_width=True, hide_index=True)

            # Pick best window for chart
            online_results = s5['online_results']
            available_ws = list(online_results.keys())
            chart_ws = st.selectbox("Window size to visualize", available_ws, key="oam_ws")

            odf = online_results[chart_ws]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=odf['Window'], y=odf['Static_R2'],
                mode='lines+markers', name='Static Model',
                line=dict(color='#e74c3c', width=2),
            ))
            fig.add_trace(go.Scatter(
                x=odf['Window'], y=odf['Online_R2'],
                mode='lines+markers', name='Online Adapted',
                line=dict(color='#2ecc71', width=2),
                fill='tonexty', fillcolor='rgba(46,204,113,0.15)',
            ))
            fig.update_layout(
                title=f"Static vs Online Adapted R\u00b2 (Window={chart_ws})",
                xaxis_title="Window Number", yaxis_title="R\u00b2 Score",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary narrative
            best_row = s5['summaries'].loc[s5['summaries']['Improvement_%'].idxmax()]
            st.markdown(f"""
**Summary:** The online adaptation module demonstrates that incrementally retraining
with new data can improve prediction accuracy. The best improvement was
**{best_row['Improvement_%']:.2f}%** with window size **{int(best_row['Window_Size'])}**,
achieving an average online R\u00b2 of **{best_row['Avg_Online_R2']:.6f}** vs
static R\u00b2 of **{best_row['Avg_Static_R2']:.6f}**.
""")
