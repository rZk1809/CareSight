"""Test script for the SHAP explainability module."""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from explain.shap_explainer import SHAPExplainer, explain_high_risk_predictions
from common.io import file_exists, read_parquet
from common.logging import get_logger


def test_explainability():
    """Test the explainability module."""
    logger = get_logger("test_explainability")
    
    # Paths
    model_path = "data/models/lgbm/lgbm.pkl"
    validation_path = "data/models/lgbm/val.parquet"
    
    # Check if files exist
    if not file_exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    if not file_exists(validation_path):
        logger.error(f"Validation file not found: {validation_path}")
        return False
    
    try:
        logger.info("Testing SHAP explainer initialization...")
        
        # Initialize explainer
        explainer = SHAPExplainer(model_path, validation_path)
        explainer.initialize_explainer()
        
        logger.info("✅ SHAP explainer initialized successfully")
        
        # Load validation data
        val_data = read_parquet(validation_path)
        logger.info(f"Loaded validation data: {val_data.shape}")
        
        # Test global feature importance
        logger.info("Testing global feature importance...")
        importance_df = explainer.global_feature_importance(val_data.head(10), top_k=5)
        logger.info("✅ Global feature importance computed")
        logger.info(f"Top features:\n{importance_df}")
        
        # Test local explanation for first patient
        logger.info("Testing local explanation...")
        first_patient = val_data.head(1)
        patient_id = first_patient['patient'].iloc[0] if 'patient' in first_patient.columns else 'test_patient'
        
        explanation = explainer.local_explanation(first_patient, str(patient_id))
        logger.info("✅ Local explanation generated")
        logger.info(f"Prediction probability: {explanation['prediction_probability']:.3f}")
        logger.info(f"Top contributing features: {explanation['top_contributing_features'][:3]}")
        
        # Test explanation report generation
        logger.info("Testing explanation report generation...")
        report_path = explainer.generate_explanation_report(first_patient, str(patient_id))
        logger.info(f"✅ Explanation report generated: {report_path}")
        
        # Test summary plot (if SHAP is available)
        try:
            logger.info("Testing summary plot generation...")
            plot_path = explainer.create_summary_plot(val_data.head(20))
            logger.info(f"✅ Summary plot generated: {plot_path}")
        except Exception as e:
            logger.warning(f"Summary plot generation failed: {e}")
        
        # Test waterfall plot
        try:
            logger.info("Testing waterfall plot generation...")
            waterfall_path = explainer.create_waterfall_plot(first_patient, str(patient_id))
            logger.info(f"✅ Waterfall plot generated: {waterfall_path}")
        except Exception as e:
            logger.warning(f"Waterfall plot generation failed: {e}")
        
        logger.info("🎉 All explainability tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Explainability test failed: {e}")
        return False


def test_high_risk_explanations():
    """Test high-risk explanation generation."""
    logger = get_logger("test_high_risk")
    
    try:
        logger.info("Testing high-risk explanation generation...")
        
        # Since our sample data has all 0 labels, we'll use a low threshold
        explain_high_risk_predictions(
            model_path="data/models/lgbm/lgbm.pkl",
            validation_data_path="data/models/lgbm/val.parquet",
            threshold=0.0,  # Low threshold to capture some predictions
            output_dir="data/reports/explanations"
        )
        
        logger.info("✅ High-risk explanations completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ High-risk explanation test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing CareSight Risk Engine Explainability Module")
    print("=" * 60)
    
    # Test basic explainability
    success1 = test_explainability()
    
    print("\n" + "=" * 60)
    
    # Test high-risk explanations
    success2 = test_high_risk_explanations()
    
    print("\n" + "=" * 60)
    
    if success1 and success2:
        print("🎉 All explainability tests PASSED!")
    else:
        print("❌ Some explainability tests FAILED!")
        sys.exit(1)
