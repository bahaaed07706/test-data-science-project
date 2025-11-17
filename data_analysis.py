"""
Simple Data Analysis Script
Author: Bahaa Al-Deen Walid Ahmad Al-Eid
Description: A basic Python script demonstrating data analysis capabilities
"""

import pandas as pd
import numpy as np

def analyze_data(data):
    """
    Analyze a dataset and return basic statistics
    
    Args:
        data: pandas DataFrame
        
    Returns:
        dict: Dictionary containing statistical information
    """
    stats = {
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max()
    }
    return stats

def create_sample_dataset(rows=100):
    """
    Create a sample dataset for testing
    
    Args:
        rows: Number of rows to generate
        
    Returns:
        pandas DataFrame
    """
    np.random.seed(42)
    data = pd.DataFrame({
        'feature_1': np.random.randn(rows),
        'feature_2': np.random.randn(rows) * 10,
        'feature_3': np.random.randint(0, 100, rows)
    })
    return data

if __name__ == "__main__":
    # Create sample data
    sample_data = create_sample_dataset()
    
    # Analyze the data
    results = analyze_data(sample_data)
    
    print("Data Analysis Results:")
    print("=" * 50)
    for stat_name, stat_value in results.items():
        print(f"\n{stat_name.upper()}:")
        print(stat_value)
