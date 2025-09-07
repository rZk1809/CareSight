"""CareSight Risk Engine - FastAPI Application

Production-ready REST API for model serving with authentication, validation, and monitoring.
"""

import os
import sys
import time
import uuid
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from fastapi import FastAPI, HTTPException, Depends, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from common.io import file_exists
from common.logging import get_logger
from common.config import load_config
from serving.api_models import (
    PredictionRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse,
    ModelInfo, HealthStatus, ErrorResponse,
    ValidationErrorResponse, RiskLevel
)

# Initialize logger
logger = get_logger("fastapi_app")

# Security
security = HTTPBearer()

# Global model storage
class ModelManager:
    """Manages model loading and caching."""
    
    def __init__(self):
        self.model = None
        self.calibrator = None
        self.model_info = None
        self.last_loaded = None
        
    def load_models(self):
        """Load model artifacts."""
        try:
            model_path = "data/models/lgbm/lgbm.pkl"
            calibrator_path = "data/models/lgbm/calibrator_isotonic.pkl"
            
            if file_exists(model_path) and file_exists(calibrator_path):
                self.model = joblib.load(model_path)
                self.calibrator = joblib.load(calibrator_path)
                self.last_loaded = datetime.now()
                
                # Load model info
                self._load_model_info()
                
                logger.info("Models loaded successfully")
                return True
            else:
                logger.error("Model files not found")
                return False
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _load_model_info(self):
        """Load model metadata."""
        try:
            # Load metrics if available
            metrics_path = "data/reports/metrics.json"
            performance_metrics = {}
            
            if file_exists(metrics_path):
                import json
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    if 'metrics' in metrics_data:
                        performance_metrics = metrics_data['metrics']
            
            # Get feature names
            feature_names = []
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = list(self.model.feature_names_in_)
            else:
                # Default feature names
                feature_names = [
                    'n_observations_180d', 'n_encounters_180d', 'n_active_meds_180d',
                    'hba1c_last', 'hba1c_mean', 'hba1c_std', 'sbp_last', 'dbp_last'
                ]
            
            self.model_info = {
                "model_name": "CareSight Risk Engine",
                "model_version": "v1.0.0",
                "model_type": "LightGBM",
                "training_date": self.last_loaded.strftime('%Y-%m-%d') if self.last_loaded else "Unknown",
                "features": feature_names,
                "performance_metrics": performance_metrics,
                "calibration_method": "Isotonic Regression",
                "thresholds": {
                    "low_risk": 0.25,
                    "high_risk": 0.5
                }
            }
        except Exception as e:
            logger.error(f"Error loading model info: {e}")
            self.model_info = None
    
    def is_loaded(self):
        """Check if models are loaded."""
        return self.model is not None and self.calibrator is not None
    
    def predict(self, features_df: pd.DataFrame):
        """Make predictions."""
        if not self.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        try:
            # Get raw predictions
            raw_probs = self.model.predict_proba(features_df)[:, 1]
            
            # Get calibrated predictions
            calibrated_probs = self.calibrator.predict(raw_probs)
            
            return raw_probs, calibrated_probs
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )

# Initialize model manager
model_manager = ModelManager()

# Create FastAPI app
app = FastAPI(
    title="CareSight Risk Engine API",
    description="Production-ready API for healthcare risk prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API token (simplified for demo)."""
    # In production, implement proper JWT validation
    token = credentials.credentials
    
    # Simple token validation (replace with proper authentication)
    if token != "demo-token-12345":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Log request
    logger.info(f"Request {request_id}: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Request {request_id} completed in {process_time:.3f}s with status {response.status_code}")
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting CareSight Risk Engine API...")
    
    # Load models
    if not model_manager.load_models():
        logger.warning("Failed to load models on startup")
    else:
        logger.info("Models loaded successfully on startup")

# Health check endpoint
@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthStatus(
        status="healthy" if model_manager.is_loaded() else "degraded",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        model_loaded=model_manager.is_loaded(),
        dependencies={
            "model": "loaded" if model_manager.model is not None else "not_loaded",
            "calibrator": "loaded" if model_manager.calibrator is not None else "not_loaded"
        }
    )

# Readiness check endpoint
@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check endpoint."""
    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready - models not loaded"
        )
    
    return {"status": "ready", "timestamp": datetime.now().isoformat()}

# Model info endpoint
@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info(token: str = Depends(verify_token)):
    """Get model information and metadata."""
    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    if model_manager.model_info is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model info not available"
        )
    
    return ModelInfo(**model_manager.model_info)

def determine_risk_level(risk_score: float) -> RiskLevel:
    """Determine risk level based on score."""
    if risk_score >= 0.5:
        return RiskLevel.HIGH
    elif risk_score >= 0.25:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW

def calculate_confidence(raw_score: float, calibrated_score: float) -> float:
    """Calculate confidence score (simplified)."""
    # Simple confidence based on distance from decision boundary
    distance_from_boundary = abs(calibrated_score - 0.5)
    confidence = min(0.5 + distance_from_boundary, 1.0)
    return confidence

def prepare_features_dataframe(features: dict) -> pd.DataFrame:
    """Convert features dict to DataFrame."""
    # Ensure all required features are present
    required_features = [
        'n_observations_180d', 'n_encounters_180d', 'n_active_meds_180d',
        'hba1c_last', 'hba1c_mean', 'hba1c_std', 'sbp_last', 'dbp_last'
    ]
    
    feature_dict = {}
    for feature in required_features:
        feature_dict[feature] = features.get(feature, np.nan)
    
    return pd.DataFrame([feature_dict])

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(
    request: PredictionRequest,
    token: str = Depends(verify_token)
):
    """Predict risk for a single patient."""
    try:
        # Prepare features
        features_dict = request.features.dict()
        features_df = prepare_features_dataframe(features_dict)
        
        # Make prediction
        raw_scores, calibrated_scores = model_manager.predict(features_df)
        
        raw_score = float(raw_scores[0])
        calibrated_score = float(calibrated_scores[0])
        
        # Determine risk level
        risk_level = determine_risk_level(calibrated_score)
        
        # Calculate confidence
        confidence = calculate_confidence(raw_score, calibrated_score)
        
        return PredictionResponse(
            patient_id=request.patient_id,
            prediction_date=datetime.now().isoformat(),
            risk_score=calibrated_score,
            risk_level=risk_level,
            confidence=confidence,
            model_version=model_manager.model_info["model_version"] if model_manager.model_info else "unknown",
            raw_score=raw_score
        )
        
    except Exception as e:
        logger.error(f"Prediction error for patient {request.patient_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    token: str = Depends(verify_token)
):
    """Predict risk for multiple patients."""
    start_time = time.time()
    batch_id = str(uuid.uuid4())
    
    predictions = []
    failed_count = 0
    
    try:
        for patient_request in request.patients:
            try:
                # Prepare features
                features_dict = patient_request.features.dict()
                features_df = prepare_features_dataframe(features_dict)
                
                # Make prediction
                raw_scores, calibrated_scores = model_manager.predict(features_df)
                
                raw_score = float(raw_scores[0])
                calibrated_score = float(calibrated_scores[0])
                
                # Determine risk level
                risk_level = determine_risk_level(calibrated_score)
                
                # Calculate confidence
                confidence = calculate_confidence(raw_score, calibrated_score)
                
                prediction = PredictionResponse(
                    patient_id=patient_request.patient_id,
                    prediction_date=datetime.now().isoformat(),
                    risk_score=calibrated_score,
                    risk_level=risk_level,
                    confidence=confidence,
                    model_version=model_manager.model_info["model_version"] if model_manager.model_info else "unknown",
                    raw_score=raw_score
                )
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Prediction error for patient {patient_request.patient_id}: {e}")
                failed_count += 1
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            processed_count=len(predictions),
            failed_count=failed_count,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

# Model reload endpoint (for admin use)
@app.post("/model/reload", tags=["Admin"])
async def reload_model(token: str = Depends(verify_token)):
    """Reload model artifacts."""
    try:
        success = model_manager.load_models()
        if success:
            return {"status": "success", "message": "Model reloaded successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reload model"
            )
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {str(e)}"
        )

# Custom exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="ValidationError",
            message=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
