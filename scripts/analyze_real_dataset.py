import pandas as pd
import requests
import numpy as np
from collections import Counter
import os

def download_and_analyze_csv():
    """Download and analyze the real dataset structure"""
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/challenge_data-18-ago-AqNGbTgyg33WLMnLzfqMS9tsB4at3D.csv"
    
    print("🔄 Downloading dataset...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save locally
        with open('data/challenge_data.csv', 'wb') as f:
            f.write(response.content)
        
        print("✅ Dataset downloaded successfully!")
        
        # Analyze structure
        print("\n📊 Analyzing dataset structure...")
        df = pd.read_csv('data/challenge_data.csv')
        
        print(f"📏 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        print("\n🔍 First 5 rows:")
        print(df.head())
        
        print(f"\n📊 Data types:")
        print(df.dtypes)
        
        print(f"\n❌ Missing values:")
        missing_info = df.isnull().sum()
        print(missing_info[missing_info > 0] if missing_info.sum() > 0 else "No missing values found!")
        
        # Try to identify target column
        potential_targets = []
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_vals = df[col].nunique()
                if 2 <= unique_vals <= 20:  # Reasonable number of classes
                    potential_targets.append((col, unique_vals))
        
        print(f"\n🎯 Potential target columns:")
        for col, unique_count in potential_targets:
            print(f"  - {col}: {unique_count} unique values")
            sample_values = df[col].value_counts().head()
            print(f"    Sample values: {list(sample_values.index)}")
        
        # Analyze text columns
        text_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 20:  # Likely text content
                    text_cols.append((col, avg_length))
        
        print(f"\n📝 Potential text columns:")
        for col, avg_len in text_cols:
            print(f"  - {col}: avg length {avg_len:.1f} chars")
            sample_text = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
            print(f"    Sample: {str(sample_text)[:100]}...")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading or analyzing dataset: {e}")
        return None

if __name__ == "__main__":
    print("=== MEDICAL LITERATURE DATASET ANALYZER ===")
    df = download_and_analyze_csv()
    
    if df is not None:
        print(f"\n✅ Analysis complete! Dataset saved to 'data/challenge_data.csv'")
        print(f"📊 Ready to process {df.shape[0]} records with {df.shape[1]} columns")
    else:
        print("❌ Analysis failed. Please check the URL and try again.")
