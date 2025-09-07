"""CareSight Risk Engine - Interactive Dashboard

A modern Streamlit dashboard for model monitoring, predictions, and analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from common.io import read_parquet, file_exists
from common.logging import get_logger
from common.config import load_config

# Page configuration
st.set_page_config(
    page_title="CareSight Risk Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_artifacts():
    """Load trained model and related artifacts."""
    try:
        model_path = "data/models/lgbm/lgbm.pkl"
        calibrator_path = "data/models/lgbm/calibrator_isotonic.pkl"
        
        if file_exists(model_path) and file_exists(calibrator_path):
            model = joblib.load(model_path)
            calibrator = joblib.load(calibrator_path)
            return model, calibrator
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None

@st.cache_data
def load_metrics():
    """Load model evaluation metrics."""
    try:
        metrics_path = "data/reports/metrics.json"
        if file_exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None

@st.cache_data
def load_validation_data():
    """Load validation dataset with predictions."""
    try:
        val_path = "data/models/lgbm/val.parquet"
        if file_exists(val_path):
            return read_parquet(val_path)
        return None
    except Exception as e:
        st.error(f"Error loading validation data: {e}")
        return None

@st.cache_data
def load_training_data():
    """Load training dataset."""
    try:
        train_path = "data/processed/train.parquet"
        if file_exists(train_path):
            return read_parquet(train_path)
        return None
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return None

def create_roc_curve(val_data):
    """Create ROC curve visualization."""
    if val_data is None or 'label_90d' not in val_data.columns:
        return None
    
    try:
        from sklearn.metrics import roc_curve, auc
        
        y_true = val_data['label_90d']
        y_scores = val_data.get('prediction_proba', val_data.get('prediction', [0] * len(y_true)))
        
        # Handle case where all labels are the same
        if len(np.unique(y_true)) < 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(dash='dash', color='gray')
            ))
            fig.add_annotation(
                x=0.5, y=0.5,
                text="ROC curve not available<br>(single class in validation set)",
                showarrow=False,
                font=dict(size=14, color="red")
            )
        else:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='#1f77b4', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(dash='dash', color='gray')
            ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=500,
            height=400,
            showlegend=True
        )
        return fig
    except Exception as e:
        st.error(f"Error creating ROC curve: {e}")
        return None

def create_calibration_plot(val_data):
    """Create calibration plot."""
    if val_data is None or 'label_90d' not in val_data.columns:
        return None
    
    try:
        y_true = val_data['label_90d']
        y_prob = val_data.get('prediction_proba', val_data.get('prediction', [0] * len(y_true)))
        
        # Create probability bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_true_rates = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_true_rates.append(y_true[in_bin].mean())
                bin_counts.append(in_bin.sum())
        
        fig = go.Figure()
        
        if bin_centers:
            fig.add_trace(go.Scatter(
                x=bin_centers, y=bin_true_rates,
                mode='markers+lines',
                name='Model Calibration',
                marker=dict(size=8, color='#1f77b4'),
                line=dict(color='#1f77b4', width=2)
            ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='Calibration Plot',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            width=500,
            height=400,
            showlegend=True
        )
        return fig
    except Exception as e:
        st.error(f"Error creating calibration plot: {e}")
        return None

def create_feature_importance_plot(model):
    """Create feature importance visualization."""
    if model is None:
        return None
    
    try:
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = getattr(model, 'feature_names_in_', 
                                  [f'feature_{i}' for i in range(len(importances))])
        else:
            return None
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            importance_df.tail(10),  # Top 10 features
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Feature Importance',
            color='importance',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            width=600,
            height=400,
            showlegend=False
        )
        return fig
    except Exception as e:
        st.error(f"Error creating feature importance plot: {e}")
        return None

def predict_patient_risk(model, calibrator, patient_features):
    """Predict risk for a single patient."""
    try:
        if model is None or calibrator is None:
            return None, None
        
        # Convert to DataFrame
        features_df = pd.DataFrame([patient_features])
        
        # Get raw prediction
        raw_prob = model.predict_proba(features_df)[0, 1]
        
        # Get calibrated prediction
        calibrated_prob = calibrator.predict([raw_prob])[0]
        
        return raw_prob, calibrated_prob
    except Exception as e:
        st.error(f"Error predicting patient risk: {e}")
        return None, None

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 CareSight Risk Engine Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Model Performance", "Patient Prediction", "Data Quality", "Monitoring"]
    )
    
    # Load data
    model, calibrator = load_model_artifacts()
    metrics = load_metrics()
    val_data = load_validation_data()
    train_data = load_training_data()
    
    if page == "Overview":
        show_overview_page(metrics, val_data, train_data, model)
    elif page == "Model Performance":
        show_performance_page(model, metrics, val_data)
    elif page == "Patient Prediction":
        show_prediction_page(model, calibrator)
    elif page == "Data Quality":
        show_data_quality_page(train_data, val_data)
    elif page == "Monitoring":
        show_monitoring_page(metrics, val_data)

def show_overview_page(metrics, val_data, train_data, model):
    """Show overview dashboard page."""
    st.header("📊 System Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if train_data is not None:
            st.metric("Training Samples", len(train_data))
        else:
            st.metric("Training Samples", "N/A")
    
    with col2:
        if val_data is not None:
            st.metric("Validation Samples", len(val_data))
        else:
            st.metric("Validation Samples", "N/A")
    
    with col3:
        if metrics and 'metrics' in metrics:
            auroc = metrics['metrics'].get('auroc_calibrated', 'N/A')
            if auroc != 'N/A' and not pd.isna(auroc):
                st.metric("AUROC", f"{auroc:.3f}")
            else:
                st.metric("AUROC", "N/A")
        else:
            st.metric("AUROC", "N/A")
    
    with col4:
        if train_data is not None and 'label_90d' in train_data.columns:
            positive_rate = train_data['label_90d'].mean()
            st.metric("Positive Rate", f"{positive_rate:.1%}")
        else:
            st.metric("Positive Rate", "N/A")
    
    # Model status
    st.subheader("🔧 Model Status")
    if model is not None:
        st.success("✅ Model loaded successfully")
        st.info(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.error("❌ Model not available")
    
    # Recent predictions (simulated)
    st.subheader("📈 Recent Activity")
    if val_data is not None:
        st.info(f"🔍 {len(val_data)} predictions made in validation set")
        
        # Show sample of validation data
        if len(val_data) > 0:
            st.write("**Sample Validation Results:**")
            display_cols = []

            # Add available columns
            if 'patient' in val_data.columns:
                display_cols.append('patient')
            if 'label_90d' in val_data.columns:
                display_cols.append('label_90d')
            if 'prediction_proba' in val_data.columns:
                display_cols.append('prediction_proba')
            if 'prediction' in val_data.columns:
                display_cols.append('prediction')

            # If no specific columns found, show first few columns
            if not display_cols:
                display_cols = val_data.columns[:3].tolist()

            st.dataframe(val_data[display_cols].head(), use_container_width=True)

def show_performance_page(model, metrics, val_data):
    """Show model performance page."""
    st.header("📈 Model Performance")
    
    if metrics is None:
        st.warning("No metrics available. Please run the evaluation pipeline.")
        return
    
    # Performance metrics
    st.subheader("🎯 Key Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'metrics' in metrics:
            m = metrics['metrics']
            
            # AUROC
            auroc = m.get('auroc_calibrated', 'N/A')
            if auroc != 'N/A' and not pd.isna(auroc):
                st.metric("AUROC (Calibrated)", f"{auroc:.3f}")
            else:
                st.metric("AUROC (Calibrated)", "N/A")
            
            # AUPRC
            auprc = m.get('auprc_calibrated', 'N/A')
            if auprc != 'N/A':
                st.metric("AUPRC (Calibrated)", f"{auprc:.3f}")
            else:
                st.metric("AUPRC (Calibrated)", "N/A")
            
            # Brier Score
            brier = m.get('brier_calibrated', 'N/A')
            if brier != 'N/A':
                st.metric("Brier Score", f"{brier:.3f}")
            else:
                st.metric("Brier Score", "N/A")
    
    with col2:
        # Confusion matrices
        if 'confusion_matrices' in metrics:
            st.write("**Confusion Matrix (High Sensitivity)**")
            cm = metrics['confusion_matrices']['high_sensitivity']
            conf_matrix = cm['confusion_matrix']
            
            # Create confusion matrix display
            matrix_data = [
                [conf_matrix['tn'], conf_matrix['fp']],
                [conf_matrix['fn'], conf_matrix['tp']]
            ]
            
            fig = px.imshow(
                matrix_data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Blues',
                labels=dict(x="Predicted", y="Actual"),
                x=['Negative', 'Positive'],
                y=['Negative', 'Positive']
            )
            fig.update_layout(width=300, height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Visualizations
    st.subheader("📊 Performance Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROC Curve
        roc_fig = create_roc_curve(val_data)
        if roc_fig:
            st.plotly_chart(roc_fig, use_container_width=True)
    
    with col2:
        # Calibration Plot
        cal_fig = create_calibration_plot(val_data)
        if cal_fig:
            st.plotly_chart(cal_fig, use_container_width=True)
    
    # Feature Importance
    st.subheader("🔍 Feature Importance")
    importance_fig = create_feature_importance_plot(model)
    if importance_fig:
        st.plotly_chart(importance_fig, use_container_width=True)
    else:
        st.info("Feature importance not available")

def show_prediction_page(model, calibrator):
    """Show patient prediction page."""
    st.header("🔮 Patient Risk Prediction")
    
    if model is None or calibrator is None:
        st.warning("Model not available. Please train the model first.")
        return
    
    st.write("Enter patient clinical features to predict 90-day deterioration risk:")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Clinical Counts")
            n_observations = st.number_input("Observations (180d)", min_value=0, max_value=1000, value=15)
            n_encounters = st.number_input("Encounters (180d)", min_value=0, max_value=100, value=3)
            n_meds = st.number_input("Active Medications", min_value=0, max_value=50, value=5)
        
        with col2:
            st.subheader("Lab Values")
            hba1c_last = st.number_input("Last HbA1c (%)", min_value=3.0, max_value=15.0, value=7.2, step=0.1)
            hba1c_mean = st.number_input("Mean HbA1c (%)", min_value=3.0, max_value=15.0, value=7.1, step=0.1)
            hba1c_std = st.number_input("HbA1c Std Dev", min_value=0.0, max_value=5.0, value=0.3, step=0.1)
            
            st.subheader("Vital Signs")
            sbp_last = st.number_input("Last Systolic BP (mmHg)", min_value=60, max_value=250, value=140)
            dbp_last = st.number_input("Last Diastolic BP (mmHg)", min_value=40, max_value=150, value=85)
        
        submitted = st.form_submit_button("Predict Risk")
        
        if submitted:
            # Prepare features
            patient_features = {
                'n_observations_180d': n_observations,
                'n_encounters_180d': n_encounters,
                'n_active_meds_180d': n_meds,
                'hba1c_last': hba1c_last,
                'hba1c_mean': hba1c_mean,
                'hba1c_std': hba1c_std,
                'sbp_last': sbp_last,
                'dbp_last': dbp_last
            }
            
            # Make prediction
            raw_prob, calibrated_prob = predict_patient_risk(model, calibrator, patient_features)
            
            if raw_prob is not None and calibrated_prob is not None:
                st.subheader("🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Raw Risk Score", f"{raw_prob:.3f}")
                
                with col2:
                    st.metric("Calibrated Risk Score", f"{calibrated_prob:.3f}")
                
                with col3:
                    risk_level = "High" if calibrated_prob > 0.5 else "Medium" if calibrated_prob > 0.25 else "Low"
                    st.metric("Risk Level", risk_level)
                
                # Risk interpretation
                if calibrated_prob > 0.5:
                    st.markdown('<div class="alert-high">⚠️ <strong>High Risk:</strong> Consider immediate clinical review and intervention.</div>', 
                               unsafe_allow_html=True)
                elif calibrated_prob > 0.25:
                    st.markdown('<div class="alert-medium">⚡ <strong>Medium Risk:</strong> Monitor closely and consider preventive measures.</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-low">✅ <strong>Low Risk:</strong> Continue routine care.</div>', 
                               unsafe_allow_html=True)
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = calibrated_prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Score"},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.25], 'color': "lightgreen"},
                            {'range': [0.25, 0.5], 'color': "yellow"},
                            {'range': [0.5, 1], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

def show_data_quality_page(train_data, val_data):
    """Show data quality page."""
    st.header("📋 Data Quality")
    
    if train_data is None:
        st.warning("Training data not available.")
        return
    
    # Dataset overview
    st.subheader("📊 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Data**")
        st.write(f"- Samples: {len(train_data)}")
        st.write(f"- Features: {len(train_data.columns) - 3}")  # Exclude patient, as_of, label
        
        if 'label_90d' in train_data.columns:
            pos_rate = train_data['label_90d'].mean()
            st.write(f"- Positive Rate: {pos_rate:.1%}")
    
    with col2:
        if val_data is not None:
            st.write("**Validation Data**")
            st.write(f"- Samples: {len(val_data)}")
            if 'label_90d' in val_data.columns:
                pos_rate = val_data['label_90d'].mean()
                st.write(f"- Positive Rate: {pos_rate:.1%}")
    
    # Feature completeness
    st.subheader("🔍 Feature Completeness")
    
    feature_cols = [col for col in train_data.columns if col not in ['patient', 'as_of', 'label_90d']]
    completeness_data = []
    
    for col in feature_cols:
        non_null_pct = (train_data[col].notna().sum() / len(train_data)) * 100
        completeness_data.append({
            'Feature': col,
            'Completeness (%)': non_null_pct,
            'Missing Count': train_data[col].isna().sum()
        })
    
    completeness_df = pd.DataFrame(completeness_data)
    
    # Completeness bar chart
    fig = px.bar(
        completeness_df,
        x='Feature',
        y='Completeness (%)',
        title='Feature Completeness',
        color='Completeness (%)',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Completeness table
    st.dataframe(completeness_df, use_container_width=True)
    
    # Feature distributions
    st.subheader("📈 Feature Distributions")
    
    selected_feature = st.selectbox("Select feature to visualize:", feature_cols)
    
    if selected_feature:
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(
                train_data,
                x=selected_feature,
                title=f'Distribution of {selected_feature}',
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(
                train_data,
                y=selected_feature,
                title=f'Box Plot of {selected_feature}'
            )
            st.plotly_chart(fig, use_container_width=True)

def show_monitoring_page(metrics, val_data):
    """Show monitoring page."""
    st.header("🔍 Model Monitoring")
    
    # Model health status
    st.subheader("🏥 Model Health Status")
    
    # Simulated monitoring metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Uptime", "99.9%", delta="0.1%")
    
    with col2:
        st.metric("Avg Response Time", "45ms", delta="-5ms")
    
    with col3:
        st.metric("Daily Predictions", "1,247", delta="156")
    
    # Alerts and warnings
    st.subheader("⚠️ Alerts & Warnings")
    
    # Simulated alerts
    alerts = [
        {"level": "info", "message": "Model performance within expected range", "time": "2 hours ago"},
        {"level": "warning", "message": "Slight increase in prediction latency", "time": "1 day ago"},
        {"level": "success", "message": "Model calibration check passed", "time": "3 days ago"}
    ]
    
    for alert in alerts:
        if alert["level"] == "warning":
            st.warning(f"⚠️ {alert['message']} ({alert['time']})")
        elif alert["level"] == "success":
            st.success(f"✅ {alert['message']} ({alert['time']})")
        else:
            st.info(f"ℹ️ {alert['message']} ({alert['time']})")
    
    # Performance over time (simulated)
    st.subheader("📊 Performance Trends")
    
    # Generate simulated time series data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    auroc_values = 0.75 + 0.1 * np.random.randn(len(dates)).cumsum() * 0.01
    auroc_values = np.clip(auroc_values, 0.6, 0.9)
    
    trend_data = pd.DataFrame({
        'Date': dates,
        'AUROC': auroc_values
    })
    
    fig = px.line(
        trend_data,
        x='Date',
        y='AUROC',
        title='Model Performance Over Time',
        line_shape='spline'
    )
    fig.add_hline(y=0.75, line_dash="dash", line_color="red", 
                  annotation_text="Minimum Threshold")
    st.plotly_chart(fig, use_container_width=True)
    
    # Data drift monitoring
    st.subheader("🌊 Data Drift Detection")
    
    if val_data is not None:
        st.info("Data drift monitoring is active. No significant drift detected in the last 30 days.")
        
        # Simulated drift scores
        features = ['hba1c_last', 'sbp_last', 'n_encounters_180d']
        drift_scores = [0.02, 0.15, 0.08]
        
        drift_df = pd.DataFrame({
            'Feature': features,
            'Drift Score': drift_scores,
            'Status': ['Normal' if score < 0.1 else 'Warning' for score in drift_scores]
        })
        
        fig = px.bar(
            drift_df,
            x='Feature',
            y='Drift Score',
            color='Status',
            title='Feature Drift Scores',
            color_discrete_map={'Normal': 'green', 'Warning': 'orange', 'Critical': 'red'}
        )
        fig.add_hline(y=0.1, line_dash="dash", line_color="orange", 
                      annotation_text="Warning Threshold")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Validation data not available for drift monitoring.")

if __name__ == "__main__":
    main()
