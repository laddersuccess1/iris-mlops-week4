"""
FastAPI application for IRIS classification model
Serves predictions via REST API endpoints
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
from typing import List
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(
    title="IRIS Classification API",
    description="API for predicting IRIS flower species using ML model",
    version="1.0.0"
)

# Load the trained model
MODEL_PATH = os.getenv('MODEL_PATH', 'models/iris_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Iris class names
IRIS_CLASSES = ['setosa', 'versicolor', 'virginica']

# Pydantic models for request/response validation
class IrisFeatures(BaseModel):
    """Input features for IRIS prediction"""
    sepal_length: float = Field(..., description="Sepal length in cm", ge=0, le=10)
    sepal_width: float = Field(..., description="Sepal width in cm", ge=0, le=10)
    petal_length: float = Field(..., description="Petal length in cm", ge=0, le=10)
    petal_width: float = Field(..., description="Petal width in cm", ge=0, le=10)
    
    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str = Field(..., description="Predicted IRIS species")
    prediction_id: int = Field(..., description="Numeric class ID (0-2)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: dict = Field(..., description="Probabilities for each class")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "IRIS Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Kubernetes probes"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """
    Predict IRIS species based on flower measurements
    
    Args:
        features: IrisFeatures object with sepal and petal measurements
        
    Returns:
        PredictionResponse with prediction and confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Make prediction
        prediction_id = int(model.predict(input_data)[0])
        prediction_class = IRIS_CLASSES[prediction_id]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(probabilities[prediction_id])
        
        # Format probabilities dict
        prob_dict = {
            class_name: float(prob) 
            for class_name, prob in zip(IRIS_CLASSES, probabilities)
        }
        
        return {
            "prediction": prediction_class,
            "prediction_id": prediction_id,
            "confidence": confidence,
            "probabilities": prob_dict
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": str(type(model).__name__),
        "n_features": 4,
        "n_classes": 3,
        "classes": IRIS_CLASSES
    }

if __name__ == "__main__":
    # Run the application
    port = int(os.getenv('PORT', 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)


