"""
FastAPI endpoint for BioBERT predictions
======================================

This module provides a REST API for making predictions with the trained BioBERT model.
It's designed to be used by the Next.js dashboard for real-time classification.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os
import logging

# Add src to path
sys.path.append('src')
from biobert_classifier import BioBERTMedicalClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BioBERT Medical Literature Classifier API",
    description="API for classifying medical literature using BioBERT",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier = None
model_loaded = False

# Request/Response models
class PredictionRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.5

class PredictionResponse(BaseModel):
    text: str
    predicted_labels: List[str]
    probabilities: Dict[str, float]
    confidence: float
    max_probability: float

class BatchPredictionRequest(BaseModel):
    texts: List[str]
    threshold: Optional[float] = 0.5

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str

@app.on_event("startup")
async def load_model():
    """
    Load the trained BioBERT model on startup
    """
    global classifier, model_loaded
    
    try:
        model_path = "models/biobert_medical"
        
        if os.path.exists(model_path):
            logger.info("Loading BioBERT model...")
            classifier = BioBERTMedicalClassifier()
            classifier.load_model(model_path)
            model_loaded = True
            logger.info("BioBERT model loaded successfully!")
        else:
            logger.warning(f"Model not found at {model_path}. API will run without model.")
            model_loaded = False
            
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model_loaded = False

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        message="BioBERT API is running" if model_loaded else "Model not loaded"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """
    Make a prediction for a single text
    """
    if not model_loaded or classifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure the model is trained and available."
        )
    
    try:
        predictions = classifier.predict([request.text], threshold=request.threshold)
        prediction = predictions[0]
        
        return PredictionResponse(
            text=prediction['text'],
            predicted_labels=prediction['predicted_labels'],
            probabilities=prediction['probabilities'],
            confidence=prediction['confidence'],
            max_probability=prediction['max_probability']
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions for multiple texts
    """
    if not model_loaded or classifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure the model is trained and available."
        )
    
    try:
        predictions = classifier.predict(request.texts, threshold=request.threshold)
        
        return {
            "predictions": [
                PredictionResponse(
                    text=pred['text'],
                    predicted_labels=pred['predicted_labels'],
                    probabilities=pred['probabilities'],
                    confidence=pred['confidence'],
                    max_probability=pred['max_probability']
                ) for pred in predictions
            ]
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model
    """
    if not model_loaded or classifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded"
        )
    
    try:
        return {
            "model_name": classifier.model_name,
            "max_length": classifier.max_length,
            "device": classifier.device,
            "num_labels": len(classifier.label_encoder.classes_),
            "label_classes": classifier.label_encoder.classes_.tolist()
        }
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/model/feature-importance")
async def get_feature_importance(text: str):
    """
    Get feature importance (attention weights) for a given text
    """
    if not model_loaded or classifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded"
        )
    
    try:
        importance = classifier.get_feature_importance(text)
        return importance
        
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
