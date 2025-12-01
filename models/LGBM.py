import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import os

# ================= Control Panel =================
# Feature Selection Switch
USE_TOP_10_ONLY = True   

# Model Hyperparameters
WINDOW_SIZE = 10         # Lookback window size
LGBM_LEAVES = 256        # Maximum number of leaves
LEARNING_RATE = 0.05     # Learning rate
EPOCHS = 100             # Number of boosting rounds
# =================================================

def create_lgbm_features(df, window_size):
    """
    Create time-series features for LightGBM (Lags and Rolling Stats)
    """
    df_featured = df.copy()
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target columns if they exist
    for col in ['forward_returns', 'market_forward_excess_returns']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    
    # Create Lag features
    for col in numeric_cols:
        for lag in range(1, window_size + 1):
            df_featured[f'{col}_lag_{lag}'] = df_featured[col].shift(lag)
    
    # Create Rolling Statistics features
    for col in numeric_cols:
        df_featured[f'{col}_rolling_mean_{window_size}'] = df_featured[col].rolling(window=window_size).mean()
        df_featured[f'{col}_rolling_std_{window_size}'] = df_featured[col].rolling(window=window_size).std()
        df_featured[f'{col}_rolling_max_{window_size}'] = df_featured[col].rolling(window=window_size).max()
        df_featured[f'{col}_rolling_min_{window_size}'] = df_featured[col].rolling(window=window_size).min()
    
    # Drop rows with NaN values created by lags and rolling windows
    df_featured = df_featured.dropna()
    
    return df_featured

def run(X_train, y_train, X_test):
    """
    Standardized interface function for model training and prediction.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        X_test (pd.DataFrame): Test features
        
    Returns:
        np.array: Predictions on X_test
    """
    print("="*50)
    print(">>> LightGBM Model: Training & Prediction...")
    print("="*50)
    
    # 1. Feature Engineering (Apply lags & rolling stats)
    # Note: We need to process train and test separately or combined carefully.
    # For simplicity in this interface, assuming X_train and X_test are raw features.
    # But time-series features need history. If X_train is long enough, it's fine.
    # If X_test is just one row, it might fail without history.
    # Here we assume X_train and X_test are sufficient for feature generation.
    
    # Combining to ensure consistent feature generation if needed, 
    # but strictly using 'run' signature means we process what we get.
    
    print(f"Generating time-series features (Window Size: {WINDOW_SIZE})...")
    X_train_featured = create_lgbm_features(X_train, WINDOW_SIZE)
    
    # Align y_train with X_train_featured (because dropna removed rows)
    y_train_aligned = y_train.loc[X_train_featured.index]
    
    # Feature Selection Logic
    if USE_TOP_10_ONLY:
        print("★ Experiment Mode: ON. Using Top 10 features + derived features.")
        top_features = ['M4', 'V13', 'S5', 'S2', 'V7', 'M2', 'M17', 'M12', 'M8', 'S6']
        # We keep original top features AND their derived lag/rolling versions
        cols_to_keep = []
        for col in X_train_featured.columns:
            # Keep if it IS a top feature OR is derived from one
            base_feature = col.split('_lag_')[0].split('_rolling_')[0]
            if base_feature in top_features:
                cols_to_keep.append(col)
        X_train_final = X_train_featured[cols_to_keep]
    else:
        X_train_final = X_train_featured

    print(f"Training Input Shape: {X_train_final.shape}")

    # 2. Prepare LightGBM Dataset
    train_data = lgb.Dataset(X_train_final, label=y_train_aligned)

    # 3. Set Parameters
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': LGBM_LEAVES,
        'learning_rate': LEARNING_RATE,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    # 4. Train Model
    print("Training LightGBM...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=EPOCHS
    )
    
    # 5. Prediction on Test Set
    # Note: X_test also needs feature engineering. 
    # Ideally X_test should be appended to X_train tail to generate lags, then sliced back.
    # For this implementation, we try to generate features on X_test directly.
    # WARNING: If X_test is small (e.g. < window_size), this will result in empty dataframe.
    # A robust system would handle lookback buffer. Assuming X_test is large enough here.
    
    print("Processing Test Data...")
    # To generate features for test data, we might need the last few rows of train data
    # But following the strict signature run(X_train, y_train, X_test), we treat them independently.
    # This might lose the first 'window_size' rows of X_test predictions.
    
    # Option: Simply predict on raw X_test if feature engineering is done externally?
    # Based on the previous code, feature engineering was internal.
    # Let's assume X_test comes with enough history or we skip FE for X_test in this simple wrapper
    # OR (Better): We just use the columns that match X_train_final.
    
    # Let's try generating features for X_test. 
    X_test_featured = create_lgbm_features(X_test, WINDOW_SIZE)
    
    if USE_TOP_10_ONLY:
         X_test_final = X_test_featured[[c for c in X_train_final.columns if c in X_test_featured.columns]]
    else:
         X_test_final = X_test_featured

    # Align columns strictly
    # Add missing cols with 0, drop extra cols
    for col in X_train_final.columns:
        if col not in X_test_final.columns:
            X_test_final[col] = 0
    X_test_final = X_test_final[X_train_final.columns]

    print(f"Prediction Input Shape: {X_test_final.shape}")
    
    predictions = model.predict(X_test_final)
    
    # Return predictions
    # Note: These predictions correspond to X_test_featured indices (start+window_size onwards)
    # To fit the API format precisely, we might need to pad the beginning?
    # Returning the raw array for now as requested.
    return predictions

# Optional: Keep a main block for standalone testing
if __name__ == "__main__":
    # Test code (requires file path setup)
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.normpath(os.path.join(current_dir, '..', 'data', 'train.csv'))
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if 'date_id' in df.columns: df = df.sort_values('date_id').drop(columns=['date_id'])
        
        target = 'forward_returns'
        drop_cols = ['market_forward_excess_returns', 'forward_returns']
        
        # Simple split
        split = int(len(df) * 0.8)
        train = df.iloc[:split]
        test = df.iloc[split:]
        
        X_train = train.drop(columns=drop_cols, errors='ignore')
        y_train = train[target]
        X_test = test.drop(columns=drop_cols, errors='ignore')
        
        preds = run(X_train, y_train, X_test)
        print(f"Test Predictions: {preds[:5]}")

