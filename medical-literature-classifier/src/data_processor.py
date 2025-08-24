"""
Procesador de datos para el clasificador de literatura médica
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

class DataProcessor:
    """
    Clase para procesar y analizar datos del clasificador médico
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_predictions(self, y_true, y_pred, class_names=None):
        """
        Evalúa las predicciones y genera métricas completas
        """
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        # Reporte de clasificación
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        self.results = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'confusion_matrix': cm,
            'classification_report': report,
            'class_names': class_names if class_names else list(set(y_true))
        }
        
        return self.results
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Genera gráfico de matriz de confusión
        """
        if 'confusion_matrix' not in self.results:
            raise ValueError("Primero ejecuta evaluate_predictions()")
        
        plt.figure(figsize=(10, 8))
        cm = self.results['confusion_matrix']
        class_names = self.results['class_names']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_class_distribution(self, y_true, save_path=None):
        """
        Genera gráfico de distribución de clases
        """
        plt.figure(figsize=(12, 6))
        
        class_counts = Counter(y_true)
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.bar(classes, counts, color='skyblue', alpha=0.7)
        plt.title('Distribución de Clases en el Dataset')
        plt.xlabel('Grupos Médicos')
        plt.ylabel('Número de Artículos')
        plt.xticks(rotation=45)
        
        # Añadir valores en las barras
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(count), ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics_comparison(self, save_path=None):
        """
        Genera gráfico comparativo de métricas por clase
        """
        if 'classification_report' not in self.results:
            raise ValueError("Primero ejecuta evaluate_predictions()")
        
        report = self.results['classification_report']
        
        # Extraer métricas por clase
        classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        precision = [report[c]['precision'] for c in classes]
        recall = [report[c]['recall'] for c in classes]
        f1 = [report[c]['f1-score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.figure(figsize=(14, 8))
        plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Grupos Médicos')
        plt.ylabel('Score')
        plt.title('Métricas de Clasificación por Grupo')
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, output_path=None):
        """
        Genera reporte completo de resultados
        """
        if not self.results:
            raise ValueError("No hay resultados para reportar")
        
        report_text = f"""
=== REPORTE DE CLASIFICACIÓN DE LITERATURA MÉDICA ===

MÉTRICAS GENERALES:
- Accuracy: {self.results['accuracy']:.4f}
- F1-Score (Weighted): {self.results['f1_weighted']:.4f}
- F1-Score (Macro): {self.results['f1_macro']:.4f}
- F1-Score (Micro): {self.results['f1_micro']:.4f}

MÉTRICAS POR CLASE:
"""
        
        report = self.results['classification_report']
        for class_name in self.results['class_names']:
            if class_name in report:
                metrics = report[class_name]
                report_text += f"""
{class_name.upper()}:
  - Precision: {metrics['precision']:.4f}
  - Recall: {metrics['recall']:.4f}
  - F1-Score: {metrics['f1-score']:.4f}
  - Support: {metrics['support']}
"""
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        print(report_text)
        return report_text
    
    def save_results_json(self, filepath):
        """
        Guarda resultados en formato JSON
        """
        # Convertir numpy arrays a listas para JSON
        results_json = self.results.copy()
        results_json['confusion_matrix'] = results_json['confusion_matrix'].tolist()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        
        print(f"Resultados guardados en: {filepath}")

def process_csv_file(filepath, classifier):
    """
    Procesa un archivo CSV y genera predicciones
    """
    try:
        # Cargar datos
        df = pd.read_csv(filepath)
        print(f"Procesando archivo: {filepath}")
        print(f"Registros encontrados: {len(df)}")
        
        # Verificar columnas requeridas
        required_cols = ['title', 'abstract']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")
        
        # Hacer predicciones
        predictions, probabilities = classifier.predict(df)
        
        # Añadir predicciones al DataFrame
        df['group_predicted'] = predictions
        df['prediction_confidence'] = probabilities
        
        # Si existe columna 'group', calcular métricas
        if 'group' in df.columns:
            processor = DataProcessor()
            results = processor.evaluate_predictions(
                df['group'].fillna('unknown'), 
                df['group_predicted']
            )
            
            print("\nMétricas de evaluación:")
            print(f"F1-Score (Weighted): {results['f1_weighted']:.4f}")
            print(f"Accuracy: {results['accuracy']:.4f}")
            
            return df, processor
        
        return df, None
        
    except Exception as e:
        print(f"Error procesando archivo: {e}")
        return None, None
