import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set(style="whitegrid")
# Removed SimHei font setting to avoid errors on systems without Chinese fonts
plt.rcParams['axes.unicode_minus'] = False

def run(X_train, y_train, X_test):
    """
    Standardized interface function for Ridge Regression.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        X_test (pd.DataFrame): Test features
        
    Returns:
        np.array: Predictions on X_test
    """
    print("="*50)
    print(">>> Ridge Regression: Training & Prediction...")
    print("="*50)
    
    # 1. Preprocessing: Handle Missing Values
    # Linear models cannot handle NaNs
    X_train = X_train.fillna(0)
    # Ensure X_test has the same columns as X_train (align features)
    # (In a strict pipeline, we might want to align columns explicitly here)
    X_test = X_test.fillna(0)
    
    # 2. Preprocessing: Standardization
    # Crucial for Ridge Regression to penalize coefficients fairly
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Train Model
    # alpha=1.0 is the regularization strength
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    # 4. Predict
    predictions = model.predict(X_test_scaled)
    
    return predictions

def evaluate_baseline_cv(df):
    """
    Performs Time Series Cross-Validation to evaluate baseline performance.
    (This logic was in the original script, preserved here for analysis)
    """
    print("\n[Baseline Analysis] Starting Time Series Cross-Validation...")
    
    # Prepare Data
    if 'date_id' in df.columns:
        df = df.sort_values('date_id')
    
    # Columns to exclude from features
    exclude_cols = ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate', 'is_scored']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    target_col = 'forward_returns'

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    # Time Series Split
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = Ridge(alpha=1.0)
    
    mse_scores = []
    ic_scores = [] 

    fold = 1
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Fit & Predict
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        # Metrics
        mse = mean_squared_error(y_val, y_pred)
        # Calculate IC (Information Coefficient / Correlation)
        ic = np.corrcoef(y_val, y_pred)[0, 1]
        
        # Handle NaN IC (constant prediction)
        if np.isnan(ic): ic = 0.0

        mse_scores.append(mse)
        ic_scores.append(ic)

        print(f"Fold {fold}: MSE = {mse:.6f}, IC = {ic:.4f}")
        fold += 1

    print("\n" + "="*30)
    print(">>> Baseline Results (Average)")
    print("="*30)
    print(f"Avg MSE: {np.mean(mse_scores):.6f}")
    print(f"Avg IC : {np.mean(ic_scores):.4f}")
    
    # Visualization
    print("\n[Visualization] Generating prediction plot...")
    # Train on full data for plotting
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    final_preds = model.predict(X_scaled)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y, final_preds, alpha=0.1, s=1, label='Data Points')
    plt.xlabel('True Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Ridge Regression: Predicted vs True')
    
    # Plot diagonal line
    min_val = min(y.min(), final_preds.min())
    max_val = max(y.max(), final_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('ridge_baseline_result.png')
    print("-> Plot saved: ridge_baseline_result.png")


if __name__ == "__main__":
    # 1. Setup Paths (Relative path logic)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming data is in ../data/train.csv
    data_path = os.path.join(current_dir, '..', 'data', 'train.csv')
    file_path = os.path.normpath(data_path)

    if os.path.exists(file_path):
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Run the baseline evaluation to see stats and generate plot
        evaluate_baseline_cv(df)
    else:
        print(f"Error: Data file not found at {file_path}")
