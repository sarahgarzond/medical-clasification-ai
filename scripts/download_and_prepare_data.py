"""
Data Download and Preparation Script
==================================

This script downloads the real medical literature dataset and prepares it for BioBERT training.
It handles the CSV from the provided URL and converts it to the required format.
"""

import pandas as pd
import requests
import os
import logging
from typing import List, Dict
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_dataset(url: str, output_path: str) -> bool:
    """
    Download the dataset from the provided URL
    
    Args:
        url: URL to download the CSV file
        output_path: Local path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading dataset from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Dataset downloaded successfully to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        return False

def analyze_dataset_structure(df: pd.DataFrame) -> Dict:
    """
    Analyze the structure of the downloaded dataset
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with dataset statistics
    """
    logger.info("Analyzing dataset structure...")
    
    analysis = {
        'total_samples': len(df),
        'columns': df.columns.tolist(),
        'column_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'sample_data': df.head(3).to_dict('records')
    }
    
    # Identify potential text and label columns
    text_columns = []
    label_columns = []
    
    for col in df.columns:
        col_lower = col.lower()
        sample_values = df[col].dropna().head(10).astype(str)
        avg_length = sample_values.str.len().mean()
        
        if any(keyword in col_lower for keyword in ['title', 'abstract', 'text', 'content']):
            text_columns.append(col)
        elif any(keyword in col_lower for keyword in ['domain', 'class', 'label', 'category', 'type']):
            label_columns.append(col)
        elif avg_length > 50:  # Likely text column
            text_columns.append(col)
        elif df[col].nunique() < len(df) * 0.3 and avg_length < 30:  # Likely categorical
            label_columns.append(col)
    
    analysis['potential_text_columns'] = text_columns
    analysis['potential_label_columns'] = label_columns
    
    # Analyze labels if found
    if label_columns:
        for col in label_columns:
            unique_values = df[col].value_counts()
            analysis[f'{col}_distribution'] = unique_values.head(10).to_dict()
    
    return analysis

def prepare_medical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the medical dataset for BioBERT training
    
    Args:
        df: Raw DataFrame from CSV
        
    Returns:
        Processed DataFrame ready for training
    """
    logger.info("Preparing medical data for training...")
    
    # Create a copy to avoid modifying original
    processed_df = df.copy()
    
    # Find text columns (title, abstract, etc.)
    text_cols = []
    for col in processed_df.columns:
        if any(keyword in col.lower() for keyword in ['title', 'abstract', 'text', 'content']):
            text_cols.append(col)
    
    # If no obvious text columns, look for long text fields
    if not text_cols:
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object':
                avg_length = processed_df[col].astype(str).str.len().mean()
                if avg_length > 30:
                    text_cols.append(col)
    
    logger.info(f"Identified text columns: {text_cols}")
    
    # Combine text columns
    if len(text_cols) >= 2:
        # Assume first is title, second is abstract
        processed_df['title'] = processed_df[text_cols[0]].fillna('')
        processed_df['abstract'] = processed_df[text_cols[1]].fillna('')
        processed_df['combined_text'] = (processed_df['title'] + ' ' + processed_df['abstract']).str.strip()
    elif len(text_cols) == 1:
        processed_df['combined_text'] = processed_df[text_cols[0]].fillna('')
        processed_df['title'] = processed_df[text_cols[0]].fillna('')
        processed_df['abstract'] = ''
    else:
        raise ValueError("No suitable text columns found in the dataset")
    
    # Find label columns
    label_cols = []
    for col in processed_df.columns:
        if any(keyword in col.lower() for keyword in ['domain', 'class', 'label', 'category', 'type']):
            label_cols.append(col)
    
    # If no obvious label columns, look for categorical columns
    if not label_cols:
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object' and processed_df[col].nunique() < len(processed_df) * 0.5:
                unique_ratio = processed_df[col].nunique() / len(processed_df)
                if 0.01 < unique_ratio < 0.3:  # Good range for categories
                    label_cols.append(col)
    
    logger.info(f"Identified label columns: {label_cols}")
    
    if not label_cols:
        raise ValueError("No suitable label columns found in the dataset")
    
    # Process labels - use the first suitable label column
    label_col = label_cols[0]
    processed_df['raw_domains'] = processed_df[label_col].fillna('')
    
    # Clean and standardize domain labels
    def clean_domains(domain_str):
        if pd.isna(domain_str) or domain_str == '':
            return []
        
        domain_str = str(domain_str).lower().strip()
        
        # Handle different separators
        if '|' in domain_str:
            domains = domain_str.split('|')
        elif ',' in domain_str:
            domains = domain_str.split(',')
        elif ';' in domain_str:
            domains = domain_str.split(';')
        else:
            domains = [domain_str]
        
        # Clean each domain
        cleaned_domains = []
        for domain in domains:
            domain = domain.strip()
            if domain:
                # Standardize common medical domains
                domain_mapping = {
                    'neuro': 'neurological',
                    'cardio': 'cardiovascular',
                    'cardiac': 'cardiovascular',
                    'heart': 'cardiovascular',
                    'cancer': 'oncological',
                    'tumor': 'oncological',
                    'oncology': 'oncological',
                    'liver': 'hepatorenal',
                    'kidney': 'hepatorenal',
                    'renal': 'hepatorenal',
                    'hepatic': 'hepatorenal'
                }
                
                # Apply mapping
                for key, value in domain_mapping.items():
                    if key in domain:
                        domain = value
                        break
                
                cleaned_domains.append(domain)
        
        return cleaned_domains
    
    processed_df['domains'] = processed_df['raw_domains'].apply(clean_domains)
    
    # Remove samples with no domains or very short text
    processed_df = processed_df[processed_df['domains'].apply(len) > 0]
    processed_df = processed_df[processed_df['combined_text'].str.len() > 20]
    
    # Show domain distribution
    all_domains = []
    for domains in processed_df['domains']:
        all_domains.extend(domains)
    
    domain_counts = pd.Series(all_domains).value_counts()
    logger.info(f"Domain distribution after cleaning:\n{domain_counts}")
    
    # Keep only necessary columns
    final_df = processed_df[['combined_text', 'domains']].copy()
    
    # Add title and abstract columns if they exist
    if 'title' in processed_df.columns:
        final_df['title'] = processed_df['title']
    if 'abstract' in processed_df.columns:
        final_df['abstract'] = processed_df['abstract']
    
    logger.info(f"Final dataset: {len(final_df)} samples ready for training")
    
    return final_df

def main():
    """
    Main function to download and prepare the medical literature dataset
    """
    # Dataset URL provided by user
    dataset_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/challenge_data-18-ago-AqNGbTgyg33WLMnLzfqMS9tsB4at3D.csv"
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download dataset
    raw_data_path = 'data/raw_medical_data.csv'
    if not download_dataset(dataset_url, raw_data_path):
        logger.error("Failed to download dataset. Exiting.")
        return
    
    try:
        # Load the dataset
        logger.info("Loading downloaded dataset...")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(raw_data_path, encoding=encoding)
                logger.info(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("Could not load CSV with any encoding")
        
        # Analyze dataset structure
        analysis = analyze_dataset_structure(df)
        
        logger.info("Dataset Analysis:")
        logger.info(f"  Total samples: {analysis['total_samples']}")
        logger.info(f"  Columns: {analysis['columns']}")
        logger.info(f"  Potential text columns: {analysis['potential_text_columns']}")
        logger.info(f"  Potential label columns: {analysis['potential_label_columns']}")
        
        # Prepare data for training
        prepared_df = prepare_medical_data(df)
        
        # Save prepared dataset
        prepared_data_path = 'data/medical_literature_prepared.csv'
        prepared_df.to_csv(prepared_data_path, index=False, encoding='utf-8')
        
        logger.info(f"Prepared dataset saved to: {prepared_data_path}")
        
        # Save analysis results
        import json
        with open('data/dataset_analysis.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            analysis_json = {}
            for key, value in analysis.items():
                if isinstance(value, dict):
                    analysis_json[key] = {str(k): str(v) for k, v in value.items()}
                else:
                    analysis_json[key] = str(value)
            json.dump(analysis_json, f, indent=2)
        
        print("\n" + "="*50)
        print("DATA PREPARATION COMPLETED!")
        print("="*50)
        print(f"Raw data: {raw_data_path}")
        print(f"Prepared data: {prepared_data_path}")
        print(f"Total samples: {len(prepared_df)}")
        print(f"Ready for BioBERT training!")
        print("\nNext steps:")
        print(f"1. Run: python scripts/train_biobert.py --data_path {prepared_data_path}")
        print("2. Wait for training to complete (2-4 hours)")
        print("3. View results in the dashboard")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main()
