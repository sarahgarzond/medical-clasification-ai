import os
import shutil

def setup_model_directory():
    """
    Script to help user organize their trained model files
    """
    print("=== BioBERT Model Setup Helper ===")
    print()
    
    # Find where the user's model files are
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Look for common model file patterns
    model_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if any(pattern in file.lower() for pattern in ['model.safetensors', 'config.json', 'tokenizer', 'vocab']):
                model_files.append(os.path.join(root, file))
    
    if model_files:
        print("\nFound potential model files:")
        for file in model_files:
            print(f"  - {file}")
        
        # Suggest creating proper directory structure
        target_dir = "models/biobert_medical"
        print(f"\nRecommendation: Move all model files to '{target_dir}/'")
        print("This will ensure the API can find them automatically.")
        
        response = input("\nWould you like me to create the directory structure? (y/n): ")
        if response.lower() == 'y':
            os.makedirs(target_dir, exist_ok=True)
            print(f"Created directory: {target_dir}")
            print("Please manually copy your model files to this directory.")
    else:
        print("No model files found in current directory.")
        print("Please ensure your trained BioBERT model files are in the project directory.")

if __name__ == "__main__":
    setup_model_directory()
