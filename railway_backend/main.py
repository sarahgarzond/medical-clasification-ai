from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import requests
import zipfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Literature Classifier API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your V0 domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels
CLASS_LABELS = ["cardiovascular", "hepatorenal", "neurological", "oncological"]

class PredictionRequest(BaseModel):
    title: str
    abstract: str

class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: dict
    confidence: float

def download_model_from_github():
    """Download model from GitHub Releases"""
    try:
        # GitHub release URL - replace with your actual release URL
        model_url = os.getenv("MODEL_DOWNLOAD_URL", 
                             "https://github.com/sarahgarzond/medical-clasification-ai/releases/download/v1.0/pubmedbert_model_final.zip")
        
        model_dir = Path("./model")
        model_dir.mkdir(exist_ok=True)
        
        if not (model_dir / "config.json").exists():
            logger.info("Downloading model from GitHub...")
            
            # Download the model zip file
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            zip_path = model_dir / "model.zip"
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            
            # Remove zip file
            zip_path.unlink()
            
            logger.info("Model downloaded and extracted successfully")
        else:
            logger.info("Model already exists, skipping download")
            
        return model_dir
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise HTTPException(status_code=500, f"Failed to download model: {e}")

def load_model():
    """Load the BioBERT model and tokenizer"""
    global model, tokenizer
    
    try:
        model_dir = download_model_from_github()
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True
        )
        
        logger.info("Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True,
            num_labels=len(CLASS_LABELS)
        )
        
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    return {"message": "Medical Literature Classifier API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(device)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction on medical literature"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Combine title and abstract
        text = f"{request.title} [SEP] {request.abstract}"
        
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        # Format response
        predicted_class = CLASS_LABELS[predicted_class_idx]
        prob_dict = {
            label: float(probabilities[0][i].item()) 
            for i, label in enumerate(CLASS_LABELS)
        }
        
        return PredictionResponse(
            predicted_class=predicted_class,
            probabilities=prob_dict,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
