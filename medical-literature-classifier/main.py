"""
Script principal para entrenar y evaluar el clasificador de literatura médica
"""

import pandas as pd
import numpy as np
import argparse
import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, multilabel_confusion_matrix

def load_model(model_path="./pubmedbert_model_final"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # thresholds y clases
    with open(f"{model_path}/best_thresholds.json") as f:
        th_data = json.load(f)
    classes = th_data["classes"]
    thresholds = th_data["thresholds"]

    def predict_batch(texts, max_len=512):
        enc = tokenizer(texts, return_tensors="pt", truncation=True,
                        padding=True, max_length=max_len).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()

        preds = (probs > thresholds).astype(int)
        return probs, preds

    return predict_batch, classes

def main():
    parser = argparse.ArgumentParser(description='Clasificador de Literatura Médica con PubMedBERT')
    parser.add_argument('--predict', type=str, help='Archivo CSV para predicción')
    parser.add_argument('--output', type=str, default='results/', help='Directorio de salida')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # cargar modelo
    predict_fn, classes = load_model("./pubmedbert_model_final")

    if args.predict:
        print("=== MODO PREDICCIÓN ===")
        df = pd.read_csv(args.predict)

        if not all(c in df.columns for c in ["title", "abstract"]):
            print("El CSV debe contener columnas 'title' y 'abstract'")
            return

        texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()

        probs, preds = predict_fn(texts)

        # agregar columnas de probabilidades
        for i, cls in enumerate(classes):
            df[f"prob_{cls}"] = probs[:, i]
            df[f"pred_{cls}"] = preds[:, i]

        # guardar resultados
        output_file = os.path.join(args.output, "predictions.csv")
        df.to_csv(output_file, index=False)
        print(f"✅ Predicciones guardadas en: {output_file}")

        # si hay ground truth en el CSV, calcular métricas
        if "labels" in df.columns:  # suponiendo que labels es lista separada por comas
            y_true = df["labels"].apply(lambda x: [1 if c in x.split(",") else 0 for c in classes])
            y_true = np.array(list(y_true))
            report = classification_report(y_true, preds, target_names=classes, zero_division=0, output_dict=True)

            # guardar métricas
            with open(os.path.join(args.output, "results.json"), "w") as f:
                json.dump({
                    "classification_report": report,
                    "confusion_matrix": multilabel_confusion_matrix(y_true, preds).tolist(),
                    "classes": classes
                }, f, indent=2)

            print("📊 Métricas guardadas en results.json")

    else:
        print("Debes especificar --predict")
        print("Ejemplo: python main.py --predict data/test.csv")


if __name__ == "__main__":
    main()
