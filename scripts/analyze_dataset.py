import pandas as pd
import requests
import numpy as np
from collections import Counter

def download_and_analyze_csv():
    """Download and analyze the real dataset structure"""
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/challenge_data-18-ago-AqNGbTgyg33WLMnLzfqMS9tsB4at3D.csv"
    
    print("Downloading dataset...")
    response = requests.get(url)
    
    # Save locally
    with open('data/challenge_data.csv', 'wb') as f:
        f.write(response.content)
    
    # Analyze structure
    print("Analyzing dataset structure...")
    df = pd.read_csv('data/challenge_data.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Analyze target classes
    if 'group' in df.columns or 'category' in df.columns or 'class' in df.columns:
        target_col = None
        for col in ['group', 'category', 'class', 'label']:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            print(f"\nTarget column: {target_col}")
            print("Class distribution:")
            print(df[target_col].value_counts())
            
            # Check for multi-label
            sample_values = df[target_col].dropna().head(10)
            print(f"\nSample target values:")
            for val in sample_values:
                print(f"  {val}")
    
    # Analyze text columns
    text_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['group', 'category', 'class', 'label']:
            text_cols.append(col)
    
    print(f"\nPotential text columns: {text_cols}")
    
    for col in text_cols[:3]:  # Show first 3 text columns
        print(f"\nSample {col} values:")
        sample_texts = df[col].dropna().head(3)
        for i, text in enumerate(sample_texts):
            print(f"  {i+1}: {str(text)[:100]}...")
    
    return df

if __name__ == "__main__":
    df = download_and_analyze_csv()
