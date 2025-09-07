"""Test script for API endpoints (for CI/CD)."""

import sys
import time
import requests
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from common.logging import get_logger


def wait_for_api(base_url: str, max_wait: int = 60):
    """Wait for API to be ready."""
    logger = get_logger("wait_for_api")
    
    for i in range(max_wait):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("API is ready")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
    
    logger.error("API did not become ready in time")
    return False


def test_health_endpoint(base_url: str):
    """Test health endpoint."""
    logger = get_logger("test_health")
    
    try:
        response = requests.get(f"{base_url}/health")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Health check passed: {data['status']}")
            return True
        else:
            logger.error(f"Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return False


def test_model_info_endpoint(base_url: str, token: str):
    """Test model info endpoint."""
    logger = get_logger("test_model_info")
    
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{base_url}/model/info", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Model info retrieved: {data['model_name']}")
            return True
        else:
            logger.error(f"Model info failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return False


def test_prediction_endpoint(base_url: str, token: str):
    """Test prediction endpoint."""
    logger = get_logger("test_prediction")
    
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Sample prediction request
        payload = {
            "patient_id": "test_patient_123",
            "features": {
                "n_observations_180d": 15,
                "n_encounters_180d": 3,
                "n_active_meds_180d": 5,
                "hba1c_last": 7.2,
                "hba1c_mean": 7.1,
                "hba1c_std": 0.5,
                "sbp_last": 140,
                "dbp_last": 85
            }
        }
        
        response = requests.post(f"{base_url}/predict", headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Prediction successful: risk_score={data['risk_score']:.3f}")
            return True
        else:
            logger.error(f"Prediction failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return False


def test_batch_prediction_endpoint(base_url: str, token: str):
    """Test batch prediction endpoint."""
    logger = get_logger("test_batch_prediction")
    
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Sample batch prediction request
        payload = {
            "patients": [
                {
                    "patient_id": "batch_patient_1",
                    "features": {
                        "n_observations_180d": 12,
                        "n_encounters_180d": 2,
                        "n_active_meds_180d": 4,
                        "hba1c_last": 6.8,
                        "hba1c_mean": 6.9,
                        "hba1c_std": 0.3,
                        "sbp_last": 130,
                        "dbp_last": 80
                    }
                },
                {
                    "patient_id": "batch_patient_2",
                    "features": {
                        "n_observations_180d": 20,
                        "n_encounters_180d": 5,
                        "n_active_meds_180d": 8,
                        "hba1c_last": 8.1,
                        "hba1c_mean": 8.0,
                        "hba1c_std": 0.8,
                        "sbp_last": 160,
                        "dbp_last": 95
                    }
                }
            ]
        }
        
        response = requests.post(f"{base_url}/predict/batch", headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Batch prediction successful: {data['processed_count']} patients processed")
            return True
        else:
            logger.error(f"Batch prediction failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return False


def test_authentication(base_url: str):
    """Test authentication requirements."""
    logger = get_logger("test_auth")
    
    try:
        # Test without token
        response = requests.get(f"{base_url}/model/info")
        
        if response.status_code == 401:
            logger.info("Authentication properly required")
            return True
        else:
            logger.error(f"Authentication not enforced: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Authentication test error: {e}")
        return False


def main():
    """Main API testing workflow."""
    logger = get_logger("test_api")
    
    # Configuration
    base_url = "http://localhost:8000"
    token = "demo-token-12345"  # Demo token from FastAPI app
    
    try:
        logger.info("Starting API endpoint tests...")
        
        # Wait for API to be ready
        if not wait_for_api(base_url):
            logger.error("API not ready - aborting tests")
            return 1
        
        # Run tests
        tests = [
            ("Health Endpoint", lambda: test_health_endpoint(base_url)),
            ("Authentication", lambda: test_authentication(base_url)),
            ("Model Info", lambda: test_model_info_endpoint(base_url, token)),
            ("Single Prediction", lambda: test_prediction_endpoint(base_url, token)),
            ("Batch Prediction", lambda: test_batch_prediction_endpoint(base_url, token))
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            
            try:
                if test_func():
                    logger.info(f"✅ {test_name} PASSED")
                    passed += 1
                else:
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} ERROR: {e}")
        
        # Summary
        logger.info(f"API tests completed: {passed}/{total} passed")
        
        if passed == total:
            logger.info("🎉 All API tests PASSED!")
            return 0
        else:
            logger.error("❌ Some API tests FAILED!")
            return 1
            
    except Exception as e:
        logger.error(f"API testing failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
