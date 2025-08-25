import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login, HfApi
import logging

logger = logging.getLogger(__name__)

class HuggingFaceClassifier:
    """
    BioBERT Medical Literature Classifier with Hugging Face Hub integration
    """
    
    def __init__(self, model_name="pubmedbert-medical-classifier", hf_token=None):
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.classes = None
        self.thresholds = None
        
        # Login to Hugging Face if token provided
        if hf_token:
            login(token=hf_token)
            logger.info("Logged in to Hugging Face Hub")
    
    def load_model(self, model_path_or_name=None, local_files_only=False):
        """
        Load model from local path or Hugging Face Hub
        """
        if model_path_or_name is None:
            model_path_or_name = self.model_name
            
        try:
            logger.info(f"Loading model from: {model_path_or_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path_or_name,
                local_files_only=local_files_only,
                use_auth_token=self.hf_token
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path_or_name,
                local_files_only=local_files_only,
                use_auth_token=self.hf_token
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            # Load thresholds and classes
            self._load_thresholds(model_path_or_name, local_files_only)
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _load_thresholds(self, model_path, local_files_only=False):
        """Load optimal thresholds for multi-label classification"""
        try:
            if os.path.exists(model_path):
                # Local path
                thresholds_path = os.path.join(model_path, "best_thresholds.json")
                if os.path.exists(thresholds_path):
                    with open(thresholds_path, 'r') as f:
                        threshold_data = json.load(f)
                    self.classes = threshold_data.get("classes", [])
                    self.thresholds = threshold_data.get("thresholds", [])
                else:
                    logger.warning("best_thresholds.json not found, using default thresholds")
                    self._set_default_thresholds()
            else:
                # Hugging Face Hub - try to download thresholds file
                try:
                    from huggingface_hub import hf_hub_download
                    thresholds_file = hf_hub_download(
                        repo_id=model_path,
                        filename="best_thresholds.json",
                        use_auth_token=self.hf_token,
                        local_files_only=local_files_only
                    )
                    with open(thresholds_file, 'r') as f:
                        threshold_data = json.load(f)
                    self.classes = threshold_data.get("classes", [])
                    self.thresholds = threshold_data.get("thresholds", [])
                except:
                    logger.warning("Could not load thresholds from Hub, using defaults")
                    self._set_default_thresholds()
                    
        except Exception as e:
            logger.error(f"Error loading thresholds: {e}")
            self._set_default_thresholds()
    
    def _set_default_thresholds(self):
        """Set default classes and thresholds"""
        self.classes = ["neurological", "cardiovascular", "oncological", "hepatorenal"]
        self.thresholds = [0.5] * len(self.classes)
    
    def predict(self, texts, max_length=512, batch_size=16):
        """
        Predict medical domains for given texts
        
        Args:
            texts: List of strings (title + abstract)
            max_length: Maximum token length
            batch_size: Batch size for processing
            
        Returns:
            List of predictions with probabilities and labels
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        all_predictions = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.sigmoid(outputs.logits).cpu().numpy()
            
            # Convert to predictions
            for j, probs in enumerate(probabilities):
                predictions = []
                for k, (class_name, prob, threshold) in enumerate(zip(self.classes, probs, self.thresholds)):
                    predictions.append({
                        "class": class_name,
                        "probability": float(prob),
                        "predicted": int(prob > threshold)
                    })
                
                # Add summary info
                top_class_idx = np.argmax(probs)
                result = {
                    "text": batch_texts[j][:100] + "..." if len(batch_texts[j]) > 100 else batch_texts[j],
                    "predictions": predictions,
                    "top_prediction": {
                        "class": self.classes[top_class_idx],
                        "probability": float(probs[top_class_idx])
                    },
                    "active_labels": [p["class"] for p in predictions if p["predicted"] == 1]
                }
                
                all_predictions.append(result)
        
        return all_predictions
    
    def upload_to_hub(self, repo_name, local_model_path, commit_message="Upload BioBERT medical classifier"):
        """
        Upload trained model to Hugging Face Hub
        """
        if not self.hf_token:
            raise ValueError("Hugging Face token required for uploading")
        
        try:
            api = HfApi()
            
            # Create repository
            api.create_repo(repo_id=repo_name, token=self.hf_token, exist_ok=True)
            
            # Upload model files
            api.upload_folder(
                folder_path=local_model_path,
                repo_id=repo_name,
                token=self.hf_token,
                commit_message=commit_message
            )
            
            logger.info(f"Model uploaded successfully to {repo_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            return False

# Global classifier instance for API
classifier_instance = None

def get_classifier(hf_token=None, model_name=None):
    """Get or create classifier instance"""
    global classifier_instance
    
    if classifier_instance is None:
        classifier_instance = HuggingFaceClassifier(
            model_name=model_name or "pubmedbert-medical-classifier",
            hf_token=hf_token
        )
    
    return classifier_instance
