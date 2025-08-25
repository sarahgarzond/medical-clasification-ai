"""
Training script for BioBERT Medical Classifier
============================================

This script handles the complete training pipeline:
1. Data loading and preprocessing
2. Model training with BioBERT
3. Evaluation and metrics calculation
4. Results saving for dashboard visualization

Usage:
    python scripts/train_biobert.py --data_path data/medical_data.csv --output_dir models/biobert_medical
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from sklearn.model_selection import train_test_split
import logging

# Add src to path
sys.path.append('src')
from biobert_classifier import BioBERTMedicalClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess the medical literature dataset
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading data from {data_path}")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(data_path, encoding=encoding)
            logger.info(f"Successfully loaded data with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError(f"Could not load data from {data_path} with any encoding")
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Identify text and label columns
    text_columns = []
    label_column = None
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['title', 'abstract', 'text']):
            text_columns.append(col)
        elif any(keyword in col_lower for keyword in ['domain', 'class', 'label', 'category']):
            label_column = col
    
    if not text_columns:
        # Try to find any text-like columns
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].str.len().mean() > 20:
                text_columns.append(col)
    
    if not label_column:
        # Try to find label column
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.5:
                label_column = col
                break
    
    logger.info(f"Identified text columns: {text_columns}")
    logger.info(f"Identified label column: {label_column}")
    
    # Combine text columns
    if len(text_columns) == 1:
        df['combined_text'] = df[text_columns[0]].fillna('')
    elif len(text_columns) >= 2:
        # Assume first is title, second is abstract
        df['combined_text'] = (df[text_columns[0]].fillna('') + ' ' + 
                              df[text_columns[1]].fillna(''))
    else:
        raise ValueError("No suitable text columns found")
    
    # Process labels
    if label_column:
        df['domains'] = df[label_column].fillna('')
        
        # Handle different label formats
        if df['domains'].str.contains('|').any():
            # Multi-label format with pipe separator
            df['domains'] = df['domains'].apply(lambda x: x.split('|') if x else [])
        elif df['domains'].str.contains(',').any():
            # Multi-label format with comma separator
            df['domains'] = df['domains'].apply(lambda x: x.split(',') if x else [])
        else:
            # Single label format
            df['domains'] = df['domains'].apply(lambda x: [x] if x else [])
        
        # Clean labels
        df['domains'] = df['domains'].apply(
            lambda labels: [label.strip().lower() for label in labels if label.strip()]
        )
    else:
        raise ValueError("No suitable label column found")
    
    # Remove empty samples
    df = df[df['combined_text'].str.len() > 10]
    df = df[df['domains'].apply(len) > 0]
    
    logger.info(f"After preprocessing: {len(df)} samples")
    
    # Show label distribution
    all_labels = []
    for labels in df['domains']:
        all_labels.extend(labels)
    
    label_counts = pd.Series(all_labels).value_counts()
    logger.info(f"Label distribution:\n{label_counts}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Train BioBERT Medical Classifier')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the medical literature CSV file')
    parser.add_argument('--output_dir', type=str, default='models/biobert_medical',
                       help='Directory to save the trained model')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Proportion of training data to use for validation')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/results', exist_ok=True)
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(args.data_path)
        
        # Split data
        train_df, test_df = train_test_split(
            df, test_size=args.test_size, random_state=42, stratify=None
        )
        
        if args.val_size > 0:
            train_df, val_df = train_test_split(
                train_df, test_size=args.val_size, random_state=42, stratify=None
            )
        else:
            val_df = None
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df) if val_df is not None else 0}, Test: {len(test_df)}")
        
        # Initialize classifier
        classifier = BioBERTMedicalClassifier(max_length=args.max_length)
        
        # Train model
        logger.info("Starting training...")
        train_result = classifier.train(
            train_df=train_df,
            val_df=val_df,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        eval_results = classifier.evaluate(test_df)
        
        # Save results
        results_path = f'{args.output_dir}/results/evaluation_results.json'
        classifier.save_results(results_path)
        
        # Generate sample predictions for dashboard
        sample_texts = test_df['combined_text'].head(5).tolist()
        predictions = classifier.predict(sample_texts)
        
        # Save sample predictions
        import json
        with open(f'{args.output_dir}/results/sample_predictions.json', 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Model saved to: {args.output_dir}")
        print(f"F1-Score (weighted): {eval_results['f1_weighted']:.4f}")
        print(f"F1-Score (macro): {eval_results['f1_macro']:.4f}")
        print(f"F1-Score (micro): {eval_results['f1_micro']:.4f}")
        print(f"Number of test samples: {eval_results['num_samples']}")
        
        # Show per-class performance
        print("\nPer-class Performance:")
        for class_name, metrics in eval_results['classification_report'].items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                print(f"  {class_name}: F1={metrics['f1-score']:.3f}, "
                      f"Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}")
        
        print(f"\nResults saved to: {results_path}")
        print("You can now use the trained model for predictions!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
