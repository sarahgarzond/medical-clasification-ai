import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def prepare_real_dataset():
    """Prepare the real dataset for training"""
    
    # Load the downloaded data
    df = pd.read_csv('data/challenge_data.csv')
    
    print(f"Original dataset shape: {df.shape}")
    
    # Clean and prepare data based on actual structure
    # This will be updated after we see the real structure
    
    # Remove rows with missing essential data
    df_clean = df.dropna(subset=['title', 'abstract']).copy() if 'title' in df.columns and 'abstract' in df.columns else df.copy()
    
    print(f"After cleaning: {df_clean.shape}")
    
    # Combine title and abstract for classification
    if 'title' in df_clean.columns and 'abstract' in df_clean.columns:
        df_clean['text'] = df_clean['title'].astype(str) + " " + df_clean['abstract'].astype(str)
    elif 'title' in df_clean.columns:
        df_clean['text'] = df_clean['title'].astype(str)
    else:
        # Find the main text column
        text_col = None
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object' and col not in ['group', 'category', 'class', 'label']:
                text_col = col
                break
        if text_col:
            df_clean['text'] = df_clean[text_col].astype(str)
    
    # Find target column
    target_col = None
    for col in ['group', 'category', 'class', 'label', 'target']:
        if col in df_clean.columns:
            target_col = col
            break
    
    if not target_col:
        print("Warning: No target column found. Using last column as target.")
        target_col = df_clean.columns[-1]
    
    print(f"Using '{target_col}' as target column")
    print(f"Classes found: {df_clean[target_col].unique()}")
    
    # Handle multi-label if needed (separated by | or ;)
    df_clean['labels'] = df_clean[target_col].astype(str)
    
    # Split data
    train_df, test_df = train_test_split(
        df_clean[['text', 'labels']], 
        test_size=0.2, 
        random_state=42,
        stratify=df_clean['labels'] if df_clean['labels'].nunique() > 1 else None
    )
    
    # Save prepared data
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train_real.csv', index=False)
    test_df.to_csv('data/test_real.csv', index=False)
    
    print(f"Training set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    print("Data saved to data/train_real.csv and data/test_real.csv")
    
    return train_df, test_df

if __name__ == "__main__":
    prepare_real_dataset()
