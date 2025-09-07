"""SHAP-based model explainability for the CareSight Risk Engine.

This module provides global and local explanations for model predictions using SHAP
(SHapley Additive exPlanations) values.
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from common.io import read_parquet, write_parquet, ensure_dir, file_exists
from common.logging import get_logger

# Import SHAP with error handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


class SHAPExplainer:
    """SHAP-based explainer for LightGBM models."""
    
    def __init__(self, model_path: str, background_data_path: str = None):
        """Initialize SHAP explainer.
        
        Args:
            model_path: Path to trained model
            background_data_path: Path to background data for SHAP explainer
        """
        self.logger = get_logger("shap_explainer")
        self.model = None
        self.explainer = None
        self.background_data = None
        self.feature_names = None
        
        # Load model
        self._load_model(model_path)
        
        # Load background data if provided
        if background_data_path and file_exists(background_data_path):
            self._load_background_data(background_data_path)
    
    def _load_model(self, model_path: str):
        """Load the trained model."""
        try:
            if not file_exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = joblib.load(model_path)
            
            # Get feature names
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
            else:
                # Default feature names for our model
                self.feature_names = [
                    'n_observations_180d', 'n_encounters_180d', 'n_active_meds_180d',
                    'hba1c_last', 'hba1c_mean', 'hba1c_std', 'sbp_last', 'dbp_last'
                ]
            
            self.logger.info(f"Model loaded successfully from {model_path}")
            self.logger.info(f"Feature names: {self.feature_names}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _load_background_data(self, background_data_path: str):
        """Load background data for SHAP explainer."""
        try:
            self.background_data = read_parquet(background_data_path)
            
            # Select only feature columns
            feature_cols = [col for col in self.background_data.columns 
                          if col in self.feature_names]
            self.background_data = self.background_data[feature_cols]
            
            self.logger.info(f"Background data loaded: {self.background_data.shape}")
            
        except Exception as e:
            self.logger.error(f"Error loading background data: {e}")
            raise
    
    def initialize_explainer(self, explainer_type: str = "tree", max_evals: int = 100):
        """Initialize SHAP explainer.
        
        Args:
            explainer_type: Type of SHAP explainer ('tree', 'kernel', 'linear')
            max_evals: Maximum evaluations for kernel explainer
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        try:
            if explainer_type == "tree":
                # Tree explainer for tree-based models (LightGBM, XGBoost, etc.)
                self.explainer = shap.TreeExplainer(self.model)
                self.logger.info("Tree explainer initialized")
                
            elif explainer_type == "kernel":
                # Kernel explainer (model-agnostic but slower)
                if self.background_data is None:
                    raise ValueError("Background data required for kernel explainer")
                
                # Use a sample of background data for efficiency
                background_sample = self.background_data.sample(
                    n=min(100, len(self.background_data)), 
                    random_state=42
                )
                
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    background_sample,
                    max_evals=max_evals
                )
                self.logger.info("Kernel explainer initialized")
                
            elif explainer_type == "linear":
                # Linear explainer for linear models
                self.explainer = shap.LinearExplainer(
                    self.model, 
                    self.background_data if self.background_data is not None else np.zeros((1, len(self.feature_names)))
                )
                self.logger.info("Linear explainer initialized")
                
            else:
                raise ValueError(f"Unknown explainer type: {explainer_type}")
                
        except Exception as e:
            self.logger.error(f"Error initializing explainer: {e}")
            raise
    
    def compute_shap_values(self, data: pd.DataFrame, check_additivity: bool = False) -> np.ndarray:
        """Compute SHAP values for given data.
        
        Args:
            data: Input data for explanation
            check_additivity: Whether to check SHAP additivity
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            self.initialize_explainer()
        
        try:
            # Ensure data has correct feature columns
            feature_data = data[self.feature_names].copy()
            
            # Compute SHAP values
            self.logger.info(f"Computing SHAP values for {len(feature_data)} samples...")
            start_time = time.time()
            
            if isinstance(self.explainer, shap.TreeExplainer):
                shap_values = self.explainer.shap_values(feature_data)
                
                # For binary classification, take positive class SHAP values
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
                    
            elif isinstance(self.explainer, shap.KernelExplainer):
                shap_values = self.explainer.shap_values(feature_data)
                
                # For binary classification, take positive class SHAP values
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
                    
            else:
                shap_values = self.explainer.shap_values(feature_data)
            
            computation_time = time.time() - start_time
            self.logger.info(f"SHAP computation completed in {computation_time:.2f} seconds")
            
            # Check additivity if requested
            if check_additivity and hasattr(self.explainer, 'expected_value'):
                self._check_additivity(feature_data, shap_values)
            
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Error computing SHAP values: {e}")
            raise
    
    def _check_additivity(self, data: pd.DataFrame, shap_values: np.ndarray):
        """Check SHAP additivity property."""
        try:
            predictions = self.model.predict_proba(data)[:, 1]
            expected_value = self.explainer.expected_value
            
            if isinstance(expected_value, list):
                expected_value = expected_value[1]  # Positive class
            
            shap_sum = expected_value + shap_values.sum(axis=1)
            
            max_error = np.max(np.abs(predictions - shap_sum))
            self.logger.info(f"SHAP additivity check - Max error: {max_error:.6f}")
            
            if max_error > 1e-3:
                self.logger.warning(f"SHAP additivity violated - Max error: {max_error:.6f}")
            else:
                self.logger.info("SHAP additivity check passed")
                
        except Exception as e:
            self.logger.warning(f"Could not check SHAP additivity: {e}")
    
    def global_feature_importance(self, data: pd.DataFrame, top_k: int = None) -> pd.DataFrame:
        """Compute global feature importance using mean absolute SHAP values.
        
        Args:
            data: Data to compute importance on
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        shap_values = self.compute_shap_values(data)
        
        # Compute mean absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        if top_k is not None:
            importance_df = importance_df.head(top_k)
        
        self.logger.info(f"Global feature importance computed for {len(importance_df)} features")
        
        return importance_df
    
    def local_explanation(self, patient_data: pd.DataFrame, patient_id: str = None) -> Dict[str, Any]:
        """Generate local explanation for a single patient.
        
        Args:
            patient_data: Single patient data (1 row)
            patient_id: Patient identifier
            
        Returns:
            Dictionary with local explanation
        """
        if len(patient_data) != 1:
            raise ValueError("Patient data must contain exactly one row")
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(patient_data)
        
        # Get prediction
        prediction_proba = self.model.predict_proba(patient_data[self.feature_names])[0, 1]
        
        # Create explanation dictionary
        explanation = {
            'patient_id': patient_id or 'unknown',
            'prediction_probability': float(prediction_proba),
            'base_value': float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) 
                               else self.explainer.expected_value),
            'feature_contributions': {},
            'feature_values': {}
        }
        
        # Add feature contributions and values
        for i, feature in enumerate(self.feature_names):
            explanation['feature_contributions'][feature] = float(shap_values[0, i])
            explanation['feature_values'][feature] = float(patient_data[feature].iloc[0])
        
        # Sort features by absolute contribution
        sorted_features = sorted(
            explanation['feature_contributions'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        explanation['top_contributing_features'] = sorted_features[:5]
        
        self.logger.info(f"Local explanation generated for patient {patient_id}")
        
        return explanation
    
    def generate_explanation_report(self, patient_data: pd.DataFrame, patient_id: str, 
                                  output_dir: str = "data/reports/explanations") -> str:
        """Generate a comprehensive explanation report for a patient.
        
        Args:
            patient_data: Single patient data
            patient_id: Patient identifier
            output_dir: Directory to save report
            
        Returns:
            Path to generated report
        """
        ensure_dir(output_dir)
        
        # Generate local explanation
        explanation = self.local_explanation(patient_data, patient_id)
        
        # Create report content
        report_lines = [
            f"# Risk Prediction Explanation Report",
            f"**Patient ID:** {patient_id}",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Prediction Summary",
            f"- **Risk Score:** {explanation['prediction_probability']:.3f}",
            f"- **Risk Level:** {'High' if explanation['prediction_probability'] > 0.5 else 'Medium' if explanation['prediction_probability'] > 0.25 else 'Low'}",
            f"- **Base Risk:** {explanation['base_value']:.3f}",
            "",
            "## Feature Contributions",
            "The following features contributed most to this prediction:",
            ""
        ]
        
        # Add top contributing features
        for i, (feature, contribution) in enumerate(explanation['top_contributing_features'], 1):
            value = explanation['feature_values'][feature]
            direction = "increases" if contribution > 0 else "decreases"
            
            report_lines.extend([
                f"### {i}. {feature}",
                f"- **Value:** {value:.2f}",
                f"- **Contribution:** {contribution:+.3f}",
                f"- **Effect:** {direction} risk",
                ""
            ])
        
        # Add interpretation
        report_lines.extend([
            "## Interpretation",
            "This prediction is based on the patient's clinical features over the past 180 days.",
            "Positive contributions increase the risk of deterioration, while negative contributions decrease it.",
            "",
            "**Note:** This explanation is for informational purposes and should be interpreted by qualified healthcare professionals.",
        ])
        
        # Save report
        report_path = os.path.join(output_dir, f"explanation_{patient_id}_{int(time.time())}.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Explanation report saved to {report_path}")
        
        return report_path
    
    def create_summary_plot(self, data: pd.DataFrame, output_path: str = None, 
                          max_display: int = 10) -> str:
        """Create SHAP summary plot.
        
        Args:
            data: Data for plotting
            output_path: Path to save plot
            max_display: Maximum features to display
            
        Returns:
            Path to saved plot
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available for plotting")
        
        try:
            # Compute SHAP values
            shap_values = self.compute_shap_values(data)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, 
                data[self.feature_names], 
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            
            plt.title("SHAP Feature Importance Summary")
            plt.tight_layout()
            
            # Save plot
            if output_path is None:
                output_path = "data/reports/shap_summary_plot.png"
            
            ensure_dir(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"SHAP summary plot saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating summary plot: {e}")
            raise
    
    def create_waterfall_plot(self, patient_data: pd.DataFrame, patient_id: str,
                            output_path: str = None) -> str:
        """Create SHAP waterfall plot for a single patient.
        
        Args:
            patient_data: Single patient data
            patient_id: Patient identifier
            output_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available for plotting")
        
        try:
            # Compute SHAP values
            shap_values = self.compute_shap_values(patient_data)
            
            # Create waterfall plot
            plt.figure(figsize=(10, 8))
            
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=expected_value,
                    data=patient_data[self.feature_names].iloc[0].values,
                    feature_names=self.feature_names
                ),
                show=False
            )
            
            plt.title(f"SHAP Explanation for Patient {patient_id}")
            plt.tight_layout()
            
            # Save plot
            if output_path is None:
                output_path = f"data/reports/shap_waterfall_{patient_id}.png"
            
            ensure_dir(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"SHAP waterfall plot saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating waterfall plot: {e}")
            raise


def explain_high_risk_predictions(model_path: str, validation_data_path: str, 
                                threshold: float = 0.5, output_dir: str = "data/reports/explanations"):
    """Generate explanations for high-risk predictions.
    
    Args:
        model_path: Path to trained model
        validation_data_path: Path to validation data
        threshold: Risk threshold for high-risk classification
        output_dir: Directory to save explanations
    """
    logger = get_logger("explain_high_risk")
    
    try:
        # Load validation data
        val_data = read_parquet(validation_data_path)
        
        # Filter high-risk predictions
        if 'prediction_proba' in val_data.columns:
            high_risk_data = val_data[val_data['prediction_proba'] >= threshold]
        else:
            logger.warning("No prediction probabilities found in validation data")
            return
        
        if len(high_risk_data) == 0:
            logger.info("No high-risk predictions found")
            return
        
        logger.info(f"Found {len(high_risk_data)} high-risk predictions")
        
        # Initialize explainer
        explainer = SHAPExplainer(model_path, validation_data_path)
        explainer.initialize_explainer()
        
        # Generate explanations for each high-risk patient
        for idx, row in high_risk_data.iterrows():
            patient_id = row.get('patient', f'patient_{idx}')
            patient_data = pd.DataFrame([row])
            
            try:
                # Generate explanation report
                report_path = explainer.generate_explanation_report(
                    patient_data, patient_id, output_dir
                )
                
                # Create waterfall plot
                plot_path = explainer.create_waterfall_plot(
                    patient_data, patient_id,
                    os.path.join(output_dir, f"waterfall_{patient_id}.png")
                )
                
                logger.info(f"Explanation generated for patient {patient_id}")
                
            except Exception as e:
                logger.error(f"Error generating explanation for patient {patient_id}: {e}")
        
        # Create global summary plot
        summary_plot_path = explainer.create_summary_plot(
            high_risk_data,
            os.path.join(output_dir, "high_risk_summary.png")
        )
        
        logger.info(f"High-risk explanations completed. Reports saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in high-risk explanation generation: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    model_path = "data/models/lgbm/lgbm.pkl"
    validation_path = "data/models/lgbm/val.parquet"
    
    if file_exists(model_path) and file_exists(validation_path):
        explain_high_risk_predictions(model_path, validation_path)
    else:
        print("Model or validation data not found. Please run the training pipeline first.")
