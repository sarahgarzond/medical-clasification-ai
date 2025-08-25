"""
Main script to analyze and prepare the real dataset
Run this first to understand your data structure
"""

import os
import sys
sys.path.append('src')

from scripts.analyze_dataset import download_and_analyze_csv
from scripts.prepare_real_data import prepare_real_dataset

def main():
    print("=== MEDICAL LITERATURE CLASSIFIER ===")
    print("Step 1: Downloading and analyzing real dataset...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download and analyze
    df = download_and_analyze_csv()
    
    print("\n" + "="*50)
    print("Step 2: Preparing data for training...")
    
    # Prepare for training
    train_df, test_df = prepare_real_dataset()
    
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("1. Review the analysis above")
    print("2. Run: python main.py --train data/train_real.csv --test data/test_real.csv")
    print("3. Run: streamlit run demo.py (for interactive demo)")
    
if __name__ == "__main__":
    main()
