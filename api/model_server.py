from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os

app = Flask(__name__)
CORS(app)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

def find_model_path():
    """Find the correct model path automatically"""
    possible_paths = [
        "models/biobert_medical",
        "models/pubmedbert_model_final", 
        "pubmedbert_model_final",
        "../pubmedbert_model_final",
        "../../pubmedbert_model_final"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            config_exists = (os.path.exists(os.path.join(path, "config.json")) or 
                           os.path.exists(os.path.join(path, "config")))
            model_exists = (os.path.exists(os.path.join(path, "model.safetensors")) or 
                          os.path.exists(os.path.join(path, "pytorch_model.bin")))
            
            if config_exists and model_exists:
                print(f"[v0] Found model at: {path}")
                return path
            else:
                print(f"[v0] Path {path} exists but missing essential files")
    
    return None

def verify_model_files(model_path):
    """Verify that all necessary model files exist"""
    files_to_check = [
        ("config.json", "config"),
        ("model.safetensors", "pytorch_model.bin"),
        ("tokenizer.json", "tokenizer"),
        ("vocab.txt", "vocab"),
        ("tokenizer_config.json", "tokenizer_config"),
        ("special_tokens_map.json", "special_tokens_map")
    ]
    
    print(f"[v0] Checking files in {model_path}:")
    print(f"[v0] All files in directory:")
    if os.path.exists(model_path):
        for file in os.listdir(model_path):
            print(f"[v0]   - {file}")
    
    for primary, alternative in files_to_check:
        primary_path = os.path.join(model_path, primary)
        alt_path = os.path.join(model_path, alternative)
        
        if os.path.exists(primary_path):
            print(f"[v0] ✓ Found {primary}")
        elif os.path.exists(alt_path):
            print(f"[v0] ✓ Found {alternative}")
        else:
            print(f"[v0] ✗ Missing {primary} or {alternative}")

MODEL_PATH = find_model_path()

if MODEL_PATH is None:
    print("[v0] Error: Could not find model files in any expected location")
    print("[v0] Please ensure your trained model is in one of these locations:")
    print("[v0] - models/biobert_medical/")
    print("[v0] - models/pubmedbert_model_final/")
    print("[v0] - pubmedbert_model_final/")
    exit(1)

verify_model_files(MODEL_PATH)
print(f"[v0] Loading model from {MODEL_PATH}")

try:
    print("[v0] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, 
        local_files_only=True,
        use_fast=False,
        trust_remote_code=False
    )
    
    print("[v0] Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=False
    )
    
    print("[v0] Model loaded successfully in offline mode")
    print(f"[v0] Model has {model.config.num_labels} output classes")
    
except Exception as e:
    print(f"[v0] Error loading model: {e}")
    print(f"[v0] Current working directory: {os.getcwd()}")
    print(f"[v0] Model path: {MODEL_PATH}")
    exit(1)

LABEL_MAPPING = {
    0: "neurological",
    1: "cardiovascular", 
    2: "oncological",
    3: "hepatorenal",
    4: "respiratory",
    5: "endocrine",
    6: "immunological"
}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": "BioBERT Medical Classifier"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        
        if not title and not abstract:
            return jsonify({"error": "Title or abstract required"}), 400
        
        # Combinar título y abstract
        text = f"{title} [SEP] {abstract}" if title and abstract else title or abstract
        
        # Tokenizar
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # Predicción
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convertir a resultados legibles
        results = []
        for i, prob in enumerate(probabilities[0]):
            label = LABEL_MAPPING.get(i, f"Class_{i}")
            confidence_score = float(prob)
            
            results.append({
                "class": label,
                "probability": confidence_score,
                "confidence": "High" if confidence_score > 0.7 else "Medium" if confidence_score > 0.4 else "Low"
            })
        
        # Ordenar por probabilidad
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            "predictions": results,
            "top_prediction": results[0],
            "input_text": text[:100] + "..." if len(text) > 100 else text
        })
        
    except Exception as e:
        print(f"[v0] Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("[v0] Starting BioBERT API server on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
