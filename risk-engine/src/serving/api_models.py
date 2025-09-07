"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level categories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PatientFeatures(BaseModel):
    """Patient clinical features for risk prediction."""
    
    # Clinical counts (180-day window)
    n_observations_180d: int = Field(
        ..., 
        ge=0, 
        le=1000, 
        description="Number of observations in the last 180 days"
    )
    n_encounters_180d: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Number of encounters in the last 180 days"
    )
    n_active_meds_180d: int = Field(
        ..., 
        ge=0, 
        le=50, 
        description="Number of active medications"
    )
    
    # HbA1c measurements
    hba1c_last: Optional[float] = Field(
        None, 
        ge=3.0, 
        le=15.0, 
        description="Last HbA1c value (%)"
    )
    hba1c_mean: Optional[float] = Field(
        None, 
        ge=3.0, 
        le=15.0, 
        description="Mean HbA1c value (%)"
    )
    hba1c_std: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=5.0, 
        description="Standard deviation of HbA1c values"
    )
    
    # Blood pressure measurements
    sbp_last: Optional[float] = Field(
        None, 
        ge=60, 
        le=250, 
        description="Last systolic blood pressure (mmHg)"
    )
    dbp_last: Optional[float] = Field(
        None, 
        ge=40, 
        le=150, 
        description="Last diastolic blood pressure (mmHg)"
    )
    
    @validator('hba1c_mean')
    def validate_hba1c_mean(cls, v, values):
        """Validate HbA1c mean is reasonable given last value."""
        if v is not None and 'hba1c_last' in values and values['hba1c_last'] is not None:
            # Mean should be within reasonable range of last value
            if abs(v - values['hba1c_last']) > 3.0:
                raise ValueError("HbA1c mean seems inconsistent with last value")
        return v
    
    @validator('dbp_last')
    def validate_blood_pressure(cls, v, values):
        """Validate diastolic BP is reasonable given systolic BP."""
        if v is not None and 'sbp_last' in values and values['sbp_last'] is not None:
            if v >= values['sbp_last']:
                raise ValueError("Diastolic BP cannot be higher than systolic BP")
        return v


class PredictionRequest(BaseModel):
    """Single patient prediction request."""
    
    patient_id: str = Field(
        ..., 
        min_length=1, 
        max_length=100, 
        description="Unique patient identifier"
    )
    as_of_date: Optional[str] = Field(
        None, 
        description="As-of date for prediction (YYYY-MM-DD format)"
    )
    features: PatientFeatures = Field(
        ..., 
        description="Patient clinical features"
    )
    
    @validator('as_of_date')
    def validate_as_of_date(cls, v):
        """Validate as-of date format."""
        if v is not None:
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError("as_of_date must be in YYYY-MM-DD format")
        return v


class BatchPredictionRequest(BaseModel):
    """Batch prediction request for multiple patients."""
    
    patients: List[PredictionRequest] = Field(
        ..., 
        min_items=1, 
        max_items=1000, 
        description="List of patients for prediction"
    )
    
    @validator('patients')
    def validate_unique_patient_ids(cls, v):
        """Ensure patient IDs are unique within the batch."""
        patient_ids = [p.patient_id for p in v]
        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError("Patient IDs must be unique within a batch")
        return v


class PredictionResponse(BaseModel):
    """Single patient prediction response."""
    
    patient_id: str = Field(..., description="Patient identifier")
    prediction_date: str = Field(..., description="Date when prediction was made")
    risk_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Calibrated risk score (0-1)"
    )
    risk_level: RiskLevel = Field(..., description="Risk level category")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Model confidence score"
    )
    model_version: str = Field(..., description="Model version used for prediction")
    
    # Optional detailed information
    raw_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Raw model score before calibration"
    )
    feature_contributions: Optional[Dict[str, float]] = Field(
        None, 
        description="Feature importance for this prediction"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    
    predictions: List[PredictionResponse] = Field(
        ..., 
        description="List of individual predictions"
    )
    batch_id: str = Field(..., description="Unique batch identifier")
    processed_count: int = Field(..., description="Number of successfully processed patients")
    failed_count: int = Field(..., description="Number of failed predictions")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class ModelInfo(BaseModel):
    """Model information and metadata."""
    
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type (e.g., LightGBM)")
    training_date: str = Field(..., description="Date when model was trained")
    features: List[str] = Field(..., description="List of required features")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    calibration_method: str = Field(..., description="Calibration method used")
    
    # Thresholds
    thresholds: Dict[str, float] = Field(
        ..., 
        description="Risk level thresholds"
    )


class HealthStatus(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    dependencies: Dict[str, str] = Field(..., description="Dependency status")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


class ValidationError(BaseModel):
    """Validation error details."""
    
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    invalid_value: Any = Field(..., description="The invalid value provided")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field details."""
    
    validation_errors: List[ValidationError] = Field(
        ..., 
        description="List of validation errors"
    )
