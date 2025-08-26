# Medical Classification AI – Clasificación Inteligente de Literatura Médica

> **Innovación en la clasificación de artículos biomédicos mediante un enfoque híbrido (ML + LLM), optimizado para entornos con y sin GPU.**

---

## 1. Análisis Exploratorio y Comprensión del Problema

El reto consiste en **clasificar artículos médicos en múltiples dominios** (neurológico, hepatorenal, oncológico y cardiovascular) usando únicamente el **título y el abstract**.

### Desafíos identificados:
- Problema **multi-etiqueta**, ya que un artículo puede pertenecer a varias categorías simultáneamente.
- **Datos desbalanceados**, con clases que tienen muchos más ejemplos que otras.
- Lenguaje técnico y terminología médica compleja.

### Análisis inicial:
- Exploración de la **distribución de clases**, longitud promedio de abstracts y correlación entre etiquetas.
- **Visualizaciones y estadísticas** para identificar patrones y retos.
- Identificación de la necesidad de un **modelo contextualizado en biomedicina**.

> **Evidencias**: Gráficas y tablas en `/visualizations`.

## 2. Preparación y Preprocesamiento

### Pipeline:
- **Limpieza de texto**: 
  - Conversión a minúsculas.
  - Eliminación de caracteres especiales.
  - Stopwords médicas.
- **Tokenización y representación**:
  - **PubMedBERT tokenizer** para embeddings contextuales.
  - **TF-IDF** para el modelo alternativo.
- **Partición de datos**:
  - 80% entrenamiento / 20% validación.
  - Estratificación multi-etiqueta para mantener la proporción de clases.

### Justificación:
- **PubMedBERT**: Preentrenado en literatura biomédica → mejor comprensión contextual.
- **RandomForest + TF-IDF**: Ligero y ejecutable en CPU → ideal para Railway y demos rápidas.


## 3. Selección y Diseño de la Solución

Nuestro **enfoque híbrido innovador** combina:

1. **Modelo Principal**  
   - `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`.
   - Ajustado (fine-tuning) para clasificación multi-etiqueta.
   - Optimización de umbrales por clase.

2. **Modelo Alternativo**  
   - **Random Forest + TF-IDF**.
   - Permite predicciones rápidas en CPU y entornos sin GPU.

**¿Por qué es innovador?**
- Combina **Deep Learning de última generación** y **ML tradicional interpretable**.
- Diseño modular y escalable, adaptable a distintos entornos y datasets.

## 4. Validación, Métricas y Análisis de Errores (20 pts)

### Métrica Principal
- **Weighted F1-score**, ideal para datos desbalanceados.

### Resultados en validación (PubMedBERT):
Weighted F1: 0.96
Macro F1: 0.95

### Reporte de clasificación:
neurological: P=0.92, R=0.90, F1=0.91
hepatorenal: P=0.97, R=0.96, F1=0.96
cardiovascular: P=0.97, R=0.96, F1=0.97
oncological: P=0.99, R=0.96, F1=0.98


### Análisis de errores:
- La mayoría de falsos negativos aparecen en abstracts muy cortos.
- Confusión entre **neurológica** y **oncología** por términos compartidos.

> **Matriz de confusión y gráficas en** `/visualizations/confusion_matrix.png`.

---

## 5. Presentación y Reporte
Incluye:
- **Notebook de entrenamiento en Colab** con hiperparámetros y configuración.
- **Capturas y prompts utilizados en V0** para generar visualizaciones interactivas.
- **Dashboard en Railway** mostrando métricas y gráficas.

Ejemplos de visualizaciones generadas en V0:
- Distribución de etiquetas en el dataset.
- Matriz de confusión interactiva.

> **Ver carpeta `/visualizations` para capturas y prompts documentados.**

---

## 6. Organización del Repositorio
/data
/models
/notebooks
/visualizations
main.py
requirements.txt
README.md

## 7. Conclusión e Innovación

- **Innovador enfoque híbrido**: combina lo mejor de Deep Learning y ML clásico.
- **Adaptabilidad**: funciona en entornos con GPU y en servidores ligeros.
- **Presentación visual clara**: integración con V0 para dashboards y análisis interactivos.

Este sistema no solo clasifica artículos médicos con alta precisión, sino que también sienta bases para **aplicaciones reales en entornos biomédicos**.

---

## 8. Instalación y Ejecución

Requisitos: Node.js (con pnpm), Python 3.10+, y dependencias listadas en `requirements.txt`.

### Backend / API
```bash
pnpm install
pnpm build
pnpm start

