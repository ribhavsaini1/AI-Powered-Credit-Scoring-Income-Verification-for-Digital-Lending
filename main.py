"""
FastAPI application for AI-Powered Loan Eligibility & Risk Scoring System.
"""

import sys
from datetime import datetime
from typing import Dict, List, Any
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.schemas import (
    BorrowerFeatures,
    RiskPredictionResponse,
    ModelPerformanceResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthCheckResponse
)
from app.model_service import LoanRiskModelService
from app.config import settings, get_models_dir

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)

# Global model service instance
model_service = None

async def load_model():
    """Load the trained model and associated components."""
    global model_service
    
    try:
        models_dir = get_models_dir()
        model_service = LoanRiskModelService(models_dir)
        model_service.load_model()
        logger.info("Model service initialized successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup."""
    await load_model()
    yield
    # Cleanup code would go here

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Machine learning API for predicting loan default risk and making lending decisions",
    version=settings.app_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

def get_model_service():
    """Dependency to ensure model service is loaded."""
    if model_service is None or not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server status."
        )
    return model_service

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint providing API information."""
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    try:
        model_loaded = model_service is not None and model_service.is_loaded
        
        return HealthCheckResponse(
            status="healthy" if model_loaded else "unhealthy",
            version=settings.app_version,
            model_loaded=model_loaded,
            dependencies={
                "model_service": "loaded" if model_loaded else "not_loaded",
                "models_directory": "accessible" if get_models_dir().exists() else "not_found"
            }
        )
    except Exception as e:
        return HealthCheckResponse(
            status="unhealthy",
            version=settings.app_version,
            model_loaded=False,
            dependencies={"error": str(e)}
        )

@app.post("/predict", response_model=RiskPredictionResponse)
async def predict_risk(
    borrower: BorrowerFeatures,
    service: LoanRiskModelService = Depends(get_model_service)
) -> RiskPredictionResponse:
    """
    Predict loan default risk for a single borrower.
    
    Args:
        borrower: Borrower features and loan details
        
    Returns:
        Risk prediction with score, category, and recommendation
    """
    try:
        # Convert Pydantic model to dict
        borrower_data = borrower.model_dump()
        
        # Make prediction using model service
        risk_score, factors = service.predict_risk(borrower_data)
        
        # Get model info
        model_info = service.get_model_info()
        
        return RiskPredictionResponse(
            risk_score=risk_score,
            risk_category="",  # Will be set by validator
            recommendation="",  # Will be set by validator
            factors=factors,
            model_version=model_info["model_info"]["version"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    service: LoanRiskModelService = Depends(get_model_service)
) -> BatchPredictionResponse:
    """
    Predict loan default risk for multiple borrowers.
    
    Args:
        request: Batch prediction request with list of borrowers
        
    Returns:
        Batch prediction response with individual predictions and summary
    """
    try:
        start_time = time.time()
        predictions = []
        
        # Get model info once
        model_info = service.get_model_info()
        
        for borrower in request.borrowers:
            try:
                # Convert Pydantic model to dict
                borrower_data = borrower.model_dump()
                
                # Make prediction using model service
                risk_score, factors = service.predict_risk(borrower_data)
                
                # Create prediction response
                prediction = RiskPredictionResponse(
                    risk_score=risk_score,
                    risk_category="",  # Will be set by validator
                    recommendation="",  # Will be set by validator
                    factors=factors if request.include_details else {},
                    model_version=model_info["model_info"]["version"]
                )
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error predicting for borrower: {str(e)}")
                # Add failed prediction with error info
                prediction = RiskPredictionResponse(
                    risk_score=0.5,  # Default uncertain score
                    risk_category="UNKNOWN",
                    recommendation="REVIEW",
                    factors={"error": str(e)},
                    model_version=model_info["model_info"]["version"]
                )
                predictions.append(prediction)
        
        # Calculate summary statistics
        processing_time = time.time() - start_time
        
        approve_count = sum(1 for p in predictions if p.recommendation == "APPROVE")
        review_count = sum(1 for p in predictions if p.recommendation == "REVIEW")
        decline_count = sum(1 for p in predictions if p.recommendation == "DECLINE")
        average_risk_score = sum(p.risk_score for p in predictions) / len(predictions)
        
        summary = {
            "total_processed": len(predictions),
            "approve_count": approve_count,
            "review_count": review_count,
            "decline_count": decline_count,
            "average_risk_score": round(average_risk_score, 4)
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary,
            processing_time=round(processing_time, 3)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model/performance", response_model=ModelPerformanceResponse)
async def get_model_performance(
    service: LoanRiskModelService = Depends(get_model_service)
) -> ModelPerformanceResponse:
    """
    Get model performance metrics and feature importance.
    
    Returns:
        Model performance information including metrics and feature importance
    """
    try:
        # Get model info and feature importance
        model_info = service.get_model_info()
        feature_importance = service.get_feature_importance()
        
        return ModelPerformanceResponse(
            model_info={
                "model_type": model_info["model_info"]["type"],
                "version": model_info["model_info"]["version"],
                "training_date": model_info["model_info"]["training_date"]
            },
            performance_metrics={
                "auc_pr": model_info["performance_metrics"]["test_auc_pr"],
                "auc_roc": model_info["performance_metrics"]["test_auc_roc"],
                "cv_auc_pr_mean": model_info["performance_metrics"]["cv_auc_pr_mean"],
                "cv_auc_roc_mean": model_info["performance_metrics"]["cv_auc_roc_mean"]
            },
            feature_importance=feature_importance,
            data_info=model_info["dataset_info"]
        )
        
    except Exception as e:
        logger.error(f"Model performance error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model performance: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with custom error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with custom error response."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={"error_type": exc.__class__.__name__}
        ).model_dump()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.host, 
        port=settings.port,
        workers=settings.workers
    )
