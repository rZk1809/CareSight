"""Test script to validate the Streamlit dashboard before running."""

import sys
import os
import importlib.util
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from common.io import file_exists
from common.logging import get_logger


def test_dashboard_imports():
    """Test that all required modules can be imported."""
    from common.logging import get_logger
    logger = get_logger("test_dashboard_imports")
    
    try:
        # Test core imports
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        import joblib
        import json
        
        logger.info("✅ All core modules imported successfully")
        
        # Test custom imports
        from common.io import read_parquet, file_exists
        from common.logging import get_logger
        from common.config import load_config
        
        logger.info("✅ All custom modules imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during imports: {e}")
        return False


def test_dashboard_syntax():
    """Test that the dashboard file has valid Python syntax."""
    logger = get_logger("test_dashboard_syntax")
    
    try:
        dashboard_path = Path(__file__).parent.parent / "src" / "dashboards" / "streamlit_app.py"
        
        if not dashboard_path.exists():
            logger.error(f"❌ Dashboard file not found: {dashboard_path}")
            return False
        
        # Try to compile the file
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        compile(source_code, str(dashboard_path), 'exec')
        logger.info("✅ Dashboard syntax is valid")
        
        return True
        
    except SyntaxError as e:
        logger.error(f"❌ Syntax error in dashboard: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking dashboard syntax: {e}")
        return False


def test_required_files():
    """Test that required data files exist."""
    logger = get_logger("test_required_files")
    
    required_files = [
        "data/models/lgbm/lgbm.pkl",
        "data/models/lgbm/calibrator_isotonic.pkl",
        "data/reports/metrics.json"
    ]
    
    optional_files = [
        "data/models/lgbm/val.parquet",
        "data/processed/train.parquet"
    ]
    
    missing_required = []
    missing_optional = []
    
    for file_path in required_files:
        if not file_exists(file_path):
            missing_required.append(file_path)
        else:
            logger.info(f"✅ Found required file: {file_path}")
    
    for file_path in optional_files:
        if not file_exists(file_path):
            missing_optional.append(file_path)
        else:
            logger.info(f"✅ Found optional file: {file_path}")
    
    if missing_required:
        logger.error(f"❌ Missing required files: {missing_required}")
        logger.info("💡 Run 'make pipeline' to generate required files")
        return False
    
    if missing_optional:
        logger.warning(f"⚠️ Missing optional files: {missing_optional}")
        logger.info("💡 Some dashboard features may not work without these files")
    
    logger.info("✅ All required files are available")
    return True


def test_dashboard_functions():
    """Test that dashboard functions can be loaded."""
    logger = get_logger("test_dashboard_functions")
    
    try:
        # Import the dashboard module
        dashboard_path = Path(__file__).parent.parent / "src" / "dashboards" / "streamlit_app.py"
        
        spec = importlib.util.spec_from_file_location("streamlit_app", dashboard_path)
        dashboard_module = importlib.util.module_from_spec(spec)
        
        # Don't execute the module (which would run Streamlit), just load it
        # spec.loader.exec_module(dashboard_module)
        
        # Instead, just check that the file can be read and has the expected functions
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        expected_functions = [
            "def main(",
            "def show_overview_page(",
            "def show_performance_page(",
            "def show_prediction_page(",
            "def load_model_artifacts(",
            "def load_metrics("
        ]
        
        missing_functions = []
        for func in expected_functions:
            if func not in content:
                missing_functions.append(func)
        
        if missing_functions:
            logger.error(f"❌ Missing functions: {missing_functions}")
            return False
        
        logger.info("✅ All expected functions found in dashboard")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing dashboard functions: {e}")
        return False


def main():
    """Run all dashboard tests."""
    logger = get_logger("test_dashboard")
    
    print("Testing CareSight Risk Engine Dashboard")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_dashboard_imports),
        ("Syntax Test", test_dashboard_syntax),
        ("Required Files Test", test_required_files),
        ("Function Test", test_dashboard_functions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED! Dashboard is ready to run.")
        print("\n💡 To start the dashboard, run:")
        print("   python scripts/run_dashboard.py")
        print("   OR")
        print("   streamlit run src/dashboards/streamlit_app.py")
        return 0
    else:
        print("❌ Some tests FAILED! Please fix the issues before running the dashboard.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
