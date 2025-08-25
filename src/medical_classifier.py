"""
Medical Literature Classifier
Sistema de clasificación de literatura médica usando título y abstract
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

class MedicalLiteratureClassifier:
    """
    Clasificador de literatura médica que utiliza TF-IDF y Random Forest
    para predecir dominios médicos basado en título y abstract
    """
    
    def __init__(self):
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.feature_names = None
        
    def preprocess_text(self, text):
        """
        Preprocesa el texto combinando título y abstract
        """
        if pd.isna(text):
            return ""
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Remover caracteres especiales pero mantener espacios
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remover espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def prepare_features(self, df):
        """
        Combina título y abstract para crear features de texto
        """
        # Combinar título y abstract
        combined_text = []
        for idx, row in df.iterrows():
            title = str(row.get('title', '')) if pd.notna(row.get('title', '')) else ''
            abstract = str(row.get('abstract', '')) if pd.notna(row.get('abstract', '')) else ''
            
            # Dar más peso al título repitiéndolo
            combined = f"{title} {title} {abstract}"
            combined_text.append(self.preprocess_text(combined))
        
        return combined_text
    
    def train(self, df):
        """
        Entrena el modelo con el dataset proporcionado
        """
        print("Preparando datos de entrenamiento...")
        
        # Preparar features
        X_text = self.prepare_features(df)
        
        # Preparar labels - manejar múltiples grupos
        y_labels = []
        for group in df['group']:
            if pd.isna(group):
                y_labels.append('unknown')
            else:
                # Tomar el primer grupo si hay múltiples
                groups = str(group).split('|')
                y_labels.append(groups[0].strip())
        
        # Codificar labels
        y_encoded = self.label_encoder.fit_transform(y_labels)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Datos de entrenamiento: {len(X_train)} muestras")
        print(f"Datos de prueba: {len(X_test)} muestras")
        print(f"Clases únicas: {len(self.label_encoder.classes_)}")
        
        # Crear pipeline
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        # Entrenar modelo
        print("Entrenando modelo...")
        self.pipeline.fit(X_train, y_train)
        
        # Evaluar en conjunto de prueba
        y_pred = self.pipeline.predict(X_test)
        
        # Calcular métricas
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"\nMétricas en conjunto de prueba:")
        print(f"F1-Score (weighted): {f1_weighted:.4f}")
        print(f"F1-Score (macro): {f1_macro:.4f}")
        
        # Reporte detallado
        target_names = self.label_encoder.classes_
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Guardar nombres de features para importancia
        self.feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
        
        return {
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        }
    
    def predict(self, df):
        """
        Predice las clases para nuevos datos
        """
        if self.pipeline is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        
        X_text = self.prepare_features(df)
        y_pred_encoded = self.pipeline.predict(X_text)
        y_pred_proba = self.pipeline.predict_proba(X_text)
        
        # Decodificar predicciones
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Obtener probabilidades máximas
        max_probabilities = np.max(y_pred_proba, axis=1)
        
        return y_pred, max_probabilities
    
    def get_feature_importance(self, top_n=20):
        """
        Obtiene las características más importantes del modelo
        """
        if self.pipeline is None:
            raise ValueError("El modelo no ha sido entrenado.")
        
        # Obtener importancias del Random Forest
        importances = self.pipeline.named_steps['classifier'].feature_importances_
        
        # Crear DataFrame con features e importancias
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return feature_importance_df
    
    def save_model(self, filepath):
        """
        Guarda el modelo entrenado
        """
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath):
        """
        Carga un modelo previamente entrenado
        """
        model_data = joblib.load(filepath)
        self.pipeline = model_data['pipeline']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        print(f"Modelo cargado desde: {filepath}")

def load_and_prepare_data(filepath):
    """
    Carga y prepara los datos desde un archivo CSV
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Datos cargados: {len(df)} registros")
        print(f"Columnas: {list(df.columns)}")
        
        # Verificar columnas requeridas
        required_cols = ['title', 'abstract', 'group']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")
        
        # Mostrar distribución de clases
        print("\nDistribución de grupos:")
        group_counts = df['group'].value_counts()
        print(group_counts)
        
        return df
    
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None

if __name__ == "__main__":
    # Ejemplo de uso
    print("=== Clasificador de Literatura Médica ===")
    
    # Crear datos de ejemplo para demostración
    sample_data = {
        'title': [
            'Neurological disorders in elderly patients',
            'Hepatorenal syndrome treatment approaches',
            'Oncological markers in breast cancer',
            'Cardiovascular risk factors analysis'
        ],
        'abstract': [
            'This study examines neurological conditions affecting elderly populations...',
            'Hepatorenal syndrome represents a complex condition requiring specialized treatment...',
            'Oncological biomarkers play a crucial role in breast cancer diagnosis...',
            'Cardiovascular disease remains a leading cause of mortality worldwide...'
        ],
        'group': ['neurological', 'hepatorenal', 'oncological', 'cardiovascular']
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    # Crear y entrenar clasificador
    classifier = MedicalLiteratureClassifier()
    
    # Entrenar con datos de ejemplo
    results = classifier.train(df_sample)
    
    # Hacer predicciones
    predictions, probabilities = classifier.predict(df_sample)
    
    print("\nPredicciones:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Muestra {i+1}: {pred} (confianza: {prob:.3f})")
