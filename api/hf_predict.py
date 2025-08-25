from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import sys
import logging

# Add src to path
sys.path.append('src')
from huggingface_classifier import get_classifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BioBERT Medical Literature Classifier API",
    description="API for classifying medical literature using BioBERT with Hugging Face integration",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    title: str
    abstract: str
    max_length: Optional[int] = 512

class BatchPredictionRequest(BaseModel):
    texts: List[str]
    max_length: Optional[int] = 512

class PredictionResponse(BaseModel):
    text: str
    predictions: List[Dict]
    top_prediction: Dict
    active_labels: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str

# Global variables
model_loaded = False
classifier = None

def get_hf_token(authorization: str = Header(None)):
    """Extract Hugging Face token from Authorization header"""
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]  # Remove "Bearer " prefix
    return os.getenv("HUGGING_FACE_TOKEN")

@app.on_event("startup")
async def load_model():
    """Load the BioBERT model on startup"""
    global classifier, model_loaded
    
    try:
        # Try to get token from environment
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        model_name = os.getenv("HF_MODEL_NAME", "pubmedbert-medical-classifier")
        
        logger.info("Initializing BioBERT classifier...")
        classifier = get_classifier(hf_token=hf_token, model_name=model_name)
        
        # Try loading from local first, then from Hub
        local_model_path = os.getenv("LOCAL_MODEL_PATH", "./pubmedbert_model_final")
        
        if os.path.exists(local_model_path):
            logger.info(f"Loading model from local path: {local_model_path}")
            success = classifier.load_model(local_model_path, local_files_only=True)
        else:
            logger.info(f"Loading model from Hugging Face Hub: {model_name}")
            success = classifier.load_model(model_name, local_files_only=False)
        
        if success:
            model_loaded = True
            logger.info("BioBERT model loaded successfully!")
        else:
            logger.error("Failed to load model")
            
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model_loaded = False

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        message="BioBERT API is running" if model_loaded else "Model not loaded"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest, hf_token: str = Depends(get_hf_token)):
    """Make a prediction for a single article"""
    if not model_loaded or classifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure the model is available."
        )
    
    try:
        # Combine title and abstract
        text = f"{request.title} [SEP] {request.abstract}"
        
        predictions = classifier.predict([text], max_length=request.max_length)
        prediction = predictions[0]
        
        return PredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest, hf_token: str = Depends(get_hf_token)):
    """Make predictions for multiple texts"""
    if not model_loaded or classifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded"
        )
    
    try:
        predictions = classifier.predict(request.texts, max_length=request.max_length)
        
        return {
            "predictions": [PredictionResponse(**pred) for pred in predictions]
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if not model_loaded or classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": classifier.model_name,
        "device": str(classifier.device),
        "num_classes": len(classifier.classes) if classifier.classes else 0,
        "classes": classifier.classes,
        "thresholds": classifier.thresholds
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
