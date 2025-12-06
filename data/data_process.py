import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_and_process_data(file_path='data/train.csv', save_cache='data/processed_cache.npz'):

    if save_cache and os.path.exists(save_cache):
        print(f"Found cached data at {save_cache}, loading directly...")
        data = np.load(save_cache)

        return data['X'], data['y']

    print(f"Cache not found. Loading raw data from {file_path}...")
   
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Original file not found: {file_path}")
        
    df = pd.read_csv(file_path)

    target_col = 'forward_returns'
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found!")
        
    df = df.fillna(0)
    
    y = df[target_col].values

    X = df.drop(columns=[target_col, 'date_id', 'time_id', 'investment_id'], errors='ignore')
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if save_cache:
        print(f"Saving processed data to {save_cache} ...")
        os.makedirs(os.path.dirname(save_cache), exist_ok=True)
        np.savez_compressed(save_cache, X=X_scaled, y=y)
        print("Data saved!")
    
    return X_scaled, y


if __name__ == "__main__":
    print(" Testing data loading...")

    try:
        X, y = load_and_process_data()
        print(f"Success! X shape: {X.shape}, y shape: {y.shape}")
        print(f"Sample data (first 5 rows of X):\n{X[:5]}")
    except Exception as e:
        print(f"Error during test: {e}")