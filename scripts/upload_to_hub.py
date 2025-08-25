#!/usr/bin/env python3
"""
Upload trained BioBERT model to Hugging Face Hub
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append('src')
from huggingface_classifier import HuggingFaceClassifier

def main():
    parser = argparse.ArgumentParser(description='Upload BioBERT model to Hugging Face Hub')
    parser.add_argument('--model_path', required=True, help='Path to local model directory')
    parser.add_argument('--repo_name', required=True, help='Hugging Face repository name (username/model-name)')
    parser.add_argument('--token', help='Hugging Face token (or set HUGGING_FACE_TOKEN env var)')
    parser.add_argument('--private', action='store_true', help='Create private repository')
    
    args = parser.parse_args()
    
    # Get token from args or environment
    hf_token = args.token or os.getenv('HUGGING_FACE_TOKEN')
    if not hf_token:
        print("Error: Hugging Face token required. Use --token or set HUGGING_FACE_TOKEN environment variable.")
        sys.exit(1)
    
    # Verify model path exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist.")
        sys.exit(1)
    
    # Check for required files
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
    missing_files = []
    
    for file in required_files:
        if not (model_path / file).exists():
            # Check for alternative names
            if file == 'pytorch_model.bin' and (model_path / 'model.safetensors').exists():
                continue
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: Missing files: {missing_files}")
        print("Continuing anyway...")
    
    # Initialize classifier and upload
    try:
        print(f"Initializing Hugging Face classifier...")
        classifier = HuggingFaceClassifier(hf_token=hf_token)
        
        print(f"Uploading model from {model_path} to {args.repo_name}...")
        success = classifier.upload_to_hub(
            repo_name=args.repo_name,
            local_model_path=str(model_path),
            commit_message="Upload BioBERT medical literature classifier"
        )
        
        if success:
            print(f"✅ Model uploaded successfully!")
            print(f"🔗 Model URL: https://huggingface.co/{args.repo_name}")
            print(f"📝 To use in code:")
            print(f"   classifier.load_model('{args.repo_name}')")
        else:
            print("❌ Upload failed. Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
