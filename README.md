# Clasificador de Literatura Médica

Sistema de Inteligencia Artificial para la clasificación automática de artículos médicos en dominios específicos utilizando únicamente el título y abstract.

## 🎯 Objetivo

Implementar un sistema capaz de asignar artículos médicos a uno o varios dominios médicos (neurológico, hepatorenal, oncológico, cardiovascular) utilizando técnicas de machine learning y procesamiento de lenguaje natural.

## 🏗️ Arquitectura de la Solución

### Enfoque Híbrido
- **Machine Learning Tradicional**: Random Forest con TF-IDF vectorization
- **Procesamiento de NLP**: Preprocesamiento avanzado de texto médico
- **Pipeline Optimizado**: Sklearn pipeline para reproducibilidad

### Componentes Principales
1. **MedicalLiteratureClassifier**: Clasificador principal con TF-IDF + Random Forest
2. **DataProcessor**: Análisis y visualización de resultados
3. **Scripts de Entrenamiento**: Automatización del proceso completo

## 📊 Características del Modelo

- **Features**: TF-IDF con n-gramas (1,2), max 5000 características
- **Algoritmo**: Random Forest (200 árboles, profundidad 20)
- **Preprocesamiento**: Limpieza de texto, combinación título+abstract
- **Métricas**: F1-score ponderado, Accuracy, Matriz de confusión

## 🚀 Instalación y Uso

### Requisitos
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Preparar Datos de Muestra
\`\`\`bash
python scripts/create_sample_data.py
\`\`\`

### Entrenar Modelo
\`\`\`bash
python main.py --train data/train.csv --output results/
\`\`\`

### Hacer Predicciones
\`\`\`bash
python main.py --predict data/test.csv --model models/medical_classifier.pkl --output results/
\`\`\`

## 📁 Estructura del Proyecto

\`\`\`
medical-literature-classifier/
├── src/
│   ├── medical_classifier.py    # Clasificador principal
│   └── data_processor.py        # Procesamiento y visualización
├── scripts/
│   └── create_sample_data.py    # Generación de datos de muestra
├── data/                        # Datasets
├── models/                      # Modelos entrenados
├── results/                     # Resultados y visualizaciones
├── main.py                      # Script principal
├── requirements.txt             # Dependencias
└── README.md                    # Documentación
\`\`\`

## 📈 Métricas y Evaluación

El sistema genera automáticamente:

- **F1-Score ponderado** (métrica principal)
- **Accuracy**
- **Matriz de confusión**
- **Reporte de clasificación por clase**
- **Distribución de clases**
- **Características más importantes**

## 🔧 Formato de Datos

### Entrada (CSV)
\`\`\`csv
title,abstract,group
"Neurological disorders..","This study examines...","neurological"
\`\`\`

### Salida (CSV)
\`\`\`csv
title,abstract,group,group_predicted,prediction_confidence
"Neurological disorders..","This study examines...","neurological","neurological",0.95
\`\`\`

## 📊 Visualizaciones Incluidas

1. **Matriz de Confusión**: Análisis de errores de clasificación
2. **Distribución de Clases**: Balance del dataset
3. **Métricas por Clase**: Precision, Recall, F1-Score
4. **Características Importantes**: Top features del modelo

## 🎯 Justificación del Enfoque

### ¿Por qué Random Forest + TF-IDF?

1. **Interpretabilidad**: Permite identificar términos médicos importantes
2. **Robustez**: Maneja bien el desbalance de clases
3. **Eficiencia**: Entrenamiento rápido y predicciones en tiempo real
4. **Escalabilidad**: Fácil de actualizar con nuevos datos

### Preprocesamiento Especializado

- **Combinación título+abstract**: El título recibe doble peso
- **N-gramas**: Captura términos médicos compuestos
- **Filtrado de características**: Elimina ruido y mejora generalización

## 🔍 Evaluación del Modelo

### Métricas Principales
- **F1-Score Ponderado**: Métrica principal del desafío
- **Accuracy**: Precisión general
- **F1-Score Macro**: Rendimiento balanceado por clase

### Validación
- Split 80/20 para entrenamiento/prueba
- Estratificación por clase principal
- Validación cruzada en desarrollo

## 🚀 Uso en Producción

### Cargar Modelo Entrenado
\`\`\`python
from src.medical_classifier import MedicalLiteratureClassifier

classifier = MedicalLiteratureClassifier()
classifier.load_model('models/medical_classifier.pkl')

# Predecir nuevos artículos
predictions, confidence = classifier.predict(new_data)
\`\`\`

### API de Predicción
El modelo puede integrarse fácilmente en APIs REST o aplicaciones web.

## 📝 Documentación del Proceso

### Experimentos Realizados
1. **Baseline**: TF-IDF + Logistic Regression
2. **Optimización**: Random Forest con hiperparámetros ajustados
3. **Feature Engineering**: Combinación de título y abstract
4. **Validación**: Múltiples métricas y visualizaciones

### Decisiones de Diseño
- **Manejo de múltiples etiquetas**: Se toma la primera etiqueta como principal
- **Preprocesamiento**: Balance entre limpieza y preservación de términos médicos
- **Hiperparámetros**: Optimizados para el dominio médico

## 🎯 Resultados Esperados

Con el dataset de muestra:
- **F1-Score**: >0.85
- **Accuracy**: >0.80
- **Cobertura**: Todos los dominios médicos principales

## 🔄 Mejoras Futuras

1. **Modelos de Lenguaje**: Integración con BERT médico
2. **Ensemble**: Combinación de múltiples algoritmos
3. **Active Learning**: Mejora continua con feedback
4. **Multilingüe**: Soporte para múltiples idiomas

## 👥 Contribución

Este proyecto fue desarrollado como solución al desafío de clasificación de literatura médica, implementando mejores prácticas de ML y documentación completa para reproducibilidad.

## 📄 Licencia

Proyecto académico - Uso educativo y de investigación.
