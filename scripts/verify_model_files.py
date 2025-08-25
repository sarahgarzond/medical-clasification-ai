"""
Script para verificar que todos los archivos del modelo estén presentes
"""
import os
import json

def verify_model_files(model_path="models/biobert_medical"):
    """
    Verifica que todos los archivos necesarios del modelo estén presentes
    """
    required_files = [
        "config.json",
        "tokenizer_config.json", 
        "vocab.txt",
        "special_tokens_map.json"
    ]
    
    # El modelo puede estar en formato safetensors o pytorch
    model_files = ["model.safetensors", "pytorch_model.bin"]
    
    print(f"Verificando archivos en: {model_path}")
    print("=" * 50)
    
    missing_files = []
    
    # Verificar archivos requeridos
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file} - Encontrado")
        else:
            print(f"❌ {file} - FALTANTE")
            missing_files.append(file)
    
    # Verificar archivo del modelo
    model_found = False
    for model_file in model_files:
        file_path = os.path.join(model_path, model_file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ {model_file} - Encontrado ({size_mb:.1f} MB)")
            model_found = True
            break
    
    if not model_found:
        print(f"❌ Archivo del modelo - FALTANTE")
        print(f"   Buscando: {' o '.join(model_files)}")
        missing_files.extend(model_files)
    
    print("=" * 50)
    
    if missing_files:
        print("❌ FALTAN ARCHIVOS:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nSolución:")
        print("1. Verifica que el entrenamiento se completó correctamente")
        print("2. Asegúrate de que los archivos se guardaron en la ruta correcta")
        print("3. Si usaste Hugging Face, verifica que se guardó con save_pretrained()")
        return False
    else:
        print("✅ TODOS LOS ARCHIVOS ESTÁN PRESENTES")
        print("El modelo está listo para usar en modo offline")
        return True

if __name__ == "__main__":
    verify_model_files()
