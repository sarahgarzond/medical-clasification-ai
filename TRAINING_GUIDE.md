# Guía Completa: Entrenamiento BioBERT Local

## ¿Por qué es una Propuesta Híbrida?

**Híbrida porque combina:**
1. **Entrenamiento local** (Python/GPU) → Genera modelo real
2. **Visualización V0** (Next.js) → Dashboard interactivo
3. **Datos estáticos** → Resultados del entrenamiento se cargan en V0

## Paso a Paso: Entrenamiento Local

### 1. Preparar Entorno Local
\`\`\`bash
# Crear directorio
mkdir medical-classifier-local
cd medical-classifier-local

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn
\`\`\`

### 2. Descargar Código Python
\`\`\`bash
# Copiar estos archivos desde V0:
# - src/biobert_classifier.py
# - scripts/train_biobert.py  
# - scripts/download_and_prepare_data.py
# - data/challenge_data-18-ago.csv
\`\`\`

### 3. Ejecutar Entrenamiento (2-4 horas)
\`\`\`bash
# Preparar datos
python scripts/download_and_prepare_data.py

# Entrenar modelo
python scripts/train_biobert.py

# Esto genera:
# - models/biobert_medical_classifier.pkl
# - results/model_results.json
# - results/confusion_matrix.png
# - results/training_log.txt
\`\`\`

### 4. Subir Resultados a V0

**Opción A: Copiar JSON manualmente**
1. Abrir `results/model_results.json`
2. Copiar contenido
3. Pegar en V0 como archivo estático

**Opción B: Actualizar código V0**
1. Reemplazar datos fallback en `lib/model-data.ts`
2. Usar métricas reales del entrenamiento

### 5. Verificar Resultados
- Dashboard V0 mostrará métricas reales
- Matriz de confusión actualizada
- Distribución de clases real
- Features importantes del modelo BioBERT

## Archivos Generados
\`\`\`
results/
├── model_results.json      # Métricas para V0
├── confusion_matrix.png    # Visualización
├── training_log.txt        # Log completo
└── feature_importance.csv  # Análisis detallado

models/
└── biobert_medical_classifier.pkl  # Modelo entrenado
\`\`\`

## Tiempo Estimado
- **Preparación**: 30 minutos
- **Entrenamiento**: 2-4 horas (depende de GPU)
- **Integración V0**: 15 minutos
- **Total**: ~3-5 horas
