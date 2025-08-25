# 🚀 Guía Completa de Despliegue en Nube

## Arquitectura de Despliegue

\`\`\`
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │   ML Model      │
│   (Vercel)      │───▶│   (Railway/      │───▶│   (Hugging Face │
│   Next.js + V0  │    │    Render)       │    │    Hub)         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
\`\`\`

## 📋 Paso a Paso Completo

### **Paso 1: Preparar el Modelo para Hugging Face Hub**

\`\`\`bash
# 1. Instalar dependencias
pip install huggingface_hub transformers

# 2. Crear estructura del modelo
mkdir biobert-medical-classifier
cd biobert-medical-classifier

# 3. Copiar archivos del modelo entrenado
cp ../pubmedbert_model_final/* .

# 4. Crear archivo de configuración del modelo
\`\`\`

\`\`\`python
# model_config.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configuración del modelo
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
NUM_LABELS = 4
LABEL_MAPPING = {
    0: "cardiovascular",
    1: "hepatorenal", 
    2: "neurological",
    3: "oncological"
}

# Función para cargar el modelo
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        "./", 
        num_labels=NUM_LABELS,
        local_files_only=True
    )
    return tokenizer, model
\`\`\`

### **Paso 2: Subir Modelo a Hugging Face Hub**

\`\`\`python
# upload_model.py
from huggingface_hub import HfApi, create_repo
import os

# Tu token de Hugging Face
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"  # Reemplazar con tu token
REPO_NAME = "tu-usuario/biobert-medical-classifier"

# Crear repositorio
api = HfApi()
create_repo(
    repo_id=REPO_NAME,
    token=HF_TOKEN,
    private=False,  # Cambiar a True si quieres privado
    repo_type="model"
)

# Subir archivos
api.upload_folder(
    folder_path="./",
    repo_id=REPO_NAME,
    token=HF_TOKEN
)

print(f"Modelo subido exitosamente a: https://huggingface.co/{REPO_NAME}")
\`\`\`

### **Paso 3: Crear API Backend**

\`\`\`python
# app.py - FastAPI Backend
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pydantic import BaseModel
import os

app = FastAPI(title="BioBERT Medical Classifier API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración
MODEL_REPO = "tu-usuario/biobert-medical-classifier"  # Tu repo de HF
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# Cargar modelo al iniciar
tokenizer = None
model = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_REPO,
            use_auth_token=HF_TOKEN
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_REPO,
            use_auth_token=HF_TOKEN
        )
        print("Modelo cargado exitosamente")
    except Exception as e:
        print(f"Error cargando modelo: {e}")

class PredictionRequest(BaseModel):
    title: str
    abstract: str
    max_length: int = 512

class PredictionResponse(BaseModel):
    predictions: list
    top_prediction: dict
    active_labels: list

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not tokenizer or not model:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Combinar título y abstract
        text = f"{request.title} [SEP] {request.abstract}"
        
        # Tokenizar
        inputs = tokenizer(
            text,
            max_length=request.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Predicción
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = probabilities.cpu().numpy()[0]
        
        # Mapear resultados
        labels = ["cardiovascular", "hepatorenal", "neurological", "oncological"]
        results = []
        
        for i, prob in enumerate(predictions):
            results.append({
                "class": labels[i],
                "probability": float(prob),
                "predicted": 1 if prob > 0.5 else 0
            })
        
        # Encontrar predicción principal
        top_idx = predictions.argmax()
        top_prediction = {
            "class": labels[top_idx],
            "probability": float(predictions[top_idx])
        }
        
        # Etiquetas activas (probabilidad > 0.5)
        active_labels = [labels[i] for i, prob in enumerate(predictions) if prob > 0.5]
        
        return PredictionResponse(
            predictions=results,
            top_prediction=top_prediction,
            active_labels=active_labels
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
\`\`\`

### **Paso 4: Desplegar Backend en Railway**

\`\`\`bash
# 1. Crear requirements.txt
echo "fastapi==0.104.1
uvicorn==0.24.0
transformers==4.35.0
torch==2.1.0
huggingface_hub==0.17.3
python-multipart==0.0.6" > requirements.txt

# 2. Crear Procfile para Railway
echo "web: uvicorn app:app --host 0.0.0.0 --port \$PORT" > Procfile

# 3. Subir a GitHub
git init
git add .
git commit -m "Initial commit"
git push origin main

# 4. Conectar con Railway
# - Ir a railway.app
# - Conectar repositorio GitHub
# - Agregar variable de entorno: HUGGING_FACE_TOKEN
# - Desplegar automáticamente
\`\`\`

### **Paso 5: Actualizar Frontend para API en Nube**

\`\`\`tsx
// app/api/predict/route.ts
import { type NextRequest, NextResponse } from "next/server"

const API_URL = process.env.BACKEND_API_URL || "https://tu-app.railway.app"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
    
  } catch (error) {
    console.error("Prediction API error:", error)
    return NextResponse.json(
      { error: "Prediction service unavailable" }, 
      { status: 500 }
    )
  }
}
\`\`\`

### **Paso 6: Desplegar Frontend en Vercel**

\`\`\`bash
# 1. Configurar variables de entorno en Vercel
BACKEND_API_URL=https://tu-app.railway.app

# 2. Desplegar desde V0
# - Usar el botón "Publish" en V0
# - O conectar repositorio GitHub con Vercel

# 3. Verificar despliegue
curl https://tu-app.vercel.app/api/predict -X POST \
  -H "Content-Type: application/json" \
  -d '{"title":"Test","abstract":"Test abstract"}'
\`\`\`

## 🔧 **Configuración de Variables de Entorno**

### Railway (Backend):
\`\`\`
HUGGING_FACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
\`\`\`

### Vercel (Frontend):
\`\`\`
BACKEND_API_URL=https://tu-app.railway.app
NEXT_PUBLIC_API_URL=https://tu-app.railway.app
\`\`\`

## 📊 **Monitoreo y Logs**

\`\`\`python
# Agregar logging al backend
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(request: PredictionRequest):
    logger.info(f"Predicción solicitada: {request.title[:50]}...")
    # ... resto del código
    logger.info(f"Predicción completada: {top_prediction['class']}")
\`\`\`

## 🚀 **URLs Finales**

- **Frontend**: `https://tu-proyecto.vercel.app`
- **Backend API**: `https://tu-app.railway.app`
- **Modelo HF**: `https://huggingface.co/tu-usuario/biobert-medical-classifier`
- **Documentación API**: `https://tu-app.railway.app/docs`

## ✅ **Verificación Final**

1. ✅ Modelo subido a Hugging Face Hub
2. ✅ Backend desplegado en Railway
3. ✅ Frontend desplegado en Vercel/V0
4. ✅ API funcionando correctamente
5. ✅ Dashboard mostrando resultados reales

¡Tu solución BioBERT estará completamente desplegada en la nube!
