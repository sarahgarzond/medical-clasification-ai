import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Ruta al modelo local
MODEL_PATH = r"C:\Users\user\Downloads\medical-literature-classifier\pubmedbert_model_final_2\pubmedbert_model_final"

# 2. Cargar modelo y tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# 3. Lista de etiquetas
labels = ["neurological", "hepatorenal", "cardiovascular", "Doncological"]

# 4. Ejemplo de artículo a testear
title = "Adrenoleukodystrophy: survey of 303 cases: biochemistry, diagnosis, and therapy."
abstract = "Adrenoleukodystrophy ( ALD ) is a genetically determined disorder associated with progressive central demyelination and adrenal cortical insufficiency . All affected persons show increased levels of saturated unbranched very-long-chain fatty acids , particularly hexacosanoate ( C26  0 ) , because of impaired capacity to degrade these acids . This degradation normally takes place in a subcellular organelle called the peroxisome , and ALD , together with Zellwegers cerebrohepatorenal syndrome , is now considered to belong to the newly formed category of peroxisomal disorders . Biochemical assays permit prenatal diagnosis , as well as identification of most heterozygotes . We have identified 303 patients with ALD in 217 kindreds . These patients show a wide phenotypic variation . Sixty percent of patients had childhood ALD and 17 % adrenomyeloneuropathy , both of which are X-linked , with the gene mapped to Xq28 . Neonatal ALD , a distinct entity with autosomal recessive inheritance and points of resemblance to Zellwegers syndrome , accounted for 7 % of the cases . Although excess C26  0 in the brain of patients with ALD is partially of dietary origin , dietary C26  0 restriction did not produce clear benefit . Bone marrow transplant lowered the plasma C26  0 level but failed to arrest neurological progression."
text = title + " " + abstract

# 5. Tokenizar entrada
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# 6. Pasar por el modelo
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits).squeeze().tolist()  # multi-label → sigmoid

# 7. Convertir probabilidades a 0/1 según umbral
threshold = 0.5
predictions = [1 if p >= threshold else 0 for p in probs]

# 8. Mostrar resultados
print("\n--- Predicciones del modelo ---")
for label, prob, pred in zip(labels, probs, predictions):
    print(f"{label:15s} | Prob: {prob:.4f} | Pred: {pred}")
