import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# ================= Control Panel =================
# Modify these parameters before running the script
# =================================================

# 1. Feature Selection Switch
# Set to True to use only Top 10 features, False to use all features
USE_TOP_10_ONLY = True   

# 2. Model Hyperparameters
WINDOW_SIZE = 10         # Lookback window size (e.g., 10 days)
LGBM_LEAVES = 256        # Maximum number of leaves
LEARNING_RATE = 0.05     # Learning rate

# 3. File Configuration
# Changed to relative path. Make sure train.csv is in the same folder.
FILE_PATH = r'C:\Users\36007\Desktop\CUHKSZ\2025-Fall\DDA3020\Homework\Project\Basic Resource\train.csv'
EPOCHS = 100             # Number of boosting rounds

# =================================================

def create_lgbm_features(df, window_size):
    """
    Create time-series features for LightGBM (Lags and Rolling Stats)
    """
    # Copy data to avoid modifying the original dataframe
    df_featured = df.copy()
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target columns from feature generation
    if 'forward_returns' in numeric_cols:
        numeric_cols.remove('forward_returns')
    if 'market_forward_excess_returns' in numeric_cols:
        numeric_cols.remove('market_forward_excess_returns')
    
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

def run_lgbm():
    print("="*50)
    print(">>> LightGBM Start: Building Gradient Boosting Model...")
    print("="*50)

    # 1. Load Data
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}. Please check the path.")
        return
    
    df = pd.read_csv(FILE_PATH)
    
    # Sort by date to ensure time-series integrity
    if 'date_id' in df.columns:
        df = df.sort_values('date_id')
        # date_ids are dropped for training but order is preserved
        df = df.drop(columns=['date_id'])

    # === Feature Selection ===
    target_col = 'forward_returns'
    
    if USE_TOP_10_ONLY:
        print(f"★ Experiment Mode: ON. Using only Top 10 features.")
        # Based on previous EDA results
        top_features = ['M4', 'V13', 'S5', 'S2', 'V7', 'M2', 'M17', 'M12', 'M8', 'S6']
        # Ensure target is included
        required_cols = top_features + [target_col]
        # Filter existing columns
        existing_cols = [c for c in required_cols if c in df.columns]
        df = df[existing_cols]
    else:
        print(f"★ Experiment Mode: OFF. Using ALL features.")

    # Drop future data / leakage columns if they exist
    drop_cols = ['market_forward_excess_returns']
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # 2. Feature Engineering
    print(f"Generating time-series features (Window Size: {WINDOW_SIZE})...")
    # Suppress fragmentation warning for cleaner output if desired, or just ignore it
    df_featured = create_lgbm_features(df, WINDOW_SIZE)
    
    # Separate Features (X) and Target (y)
    feature_cols = [col for col in df_featured.columns if col != target_col]
    X = df_featured[feature_cols]
    y = df_featured[target_col]
    
    print(f"Total Features: {X.shape[1]}")

    # 3. Time-Series Split (Strictly NO Shuffling)
    # Using 80% for training, 20% for validation (sequential)
    split_idx = int(len(X) * 0.8)
    
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training Set Size: {len(X_train)}, Validation Set Size: {len(X_val)}")

    # 4. Create LightGBM Dataset
    print("\n[Model] Preparing LightGBM Datasets...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 5. Set Parameters
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': LGBM_LEAVES,
        'learning_rate': LEARNING_RATE,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42  # Fixed seed for reproducibility
    }

    # 6. Train Model
    print("\n[Training] Training LightGBM...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=EPOCHS,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=True),
            lgb.log_evaluation(50)  # Log every 50 rounds
        ]
    )

    # 7. Prediction and Strategy Logic
    print("\n[Prediction] Generating predictions...")
    predictions = model.predict(X_val)
    
    # Strategy Logic: Convert regression output to portfolio weights (0, 1, 2)
    predicted_weights = []
    for pred in predictions:
        if pred > 0.001:   # If predicted return > 0.1%
            weight = 1.0
        elif pred > 0.005: # If predicted return > 0.5% (High confidence)
            weight = 2.0
        else:
            weight = 0.0
        predicted_weights.append(weight)
        
    print("Prediction Complete! First 10 weights:", predicted_weights[:10])
    
    # Save Model
    # Using .txt for compatibility, but .txt is sufficient for LGBM
    model.save_model('my_lgbm_model.txt')
    print("Model saved to 'my_lgbm_model.txt'")

    # 8. Evaluation
    # --- Calculate Directional Hit Rate ---
    true_direction = y_val > 0
    pred_direction = predictions > 0
    hit_rate = accuracy_score(true_direction, pred_direction)
    
    # --- Feature Importance Analysis ---
    feature_importance = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*50)
    print(f"★ LightGBM Evaluation Results ★")
    print(f"Directional Hit Rate: {hit_rate * 100:.2f}%")
    print(f"MSE: {mean_squared_error(y_val, predictions):.6f}")
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    print("="*50)

    # 9. Visualization
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Predictions vs True Returns
    plt.subplot(2, 1, 1)
    # Plotting first 100 points for clarity
    plt.plot(y_val.values[:100], label='True Returns', color='gray', alpha=0.7)
    plt.plot(predictions[:100], label='LGBM Predictions', color='blue', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title(f'LightGBM Prediction (Hit Rate: {hit_rate*100:.1f}%)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Feature Importance
    plt.subplot(2, 1, 2)
    top_20_features = feature_importance.head(20)
    plt.barh(range(len(top_20_features)), top_20_features['importance'])
    plt.yticks(range(len(top_20_features)), top_20_features['feature'])
    plt.title('Top 20 Feature Importance')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis() # Invert y-axis to show top features at the top
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lgbm_report_plot.png', dpi=300)
    print("-> Plot saved as: lgbm_report_plot.png")
    plt.show()

if __name__ == "__main__":
    run_lgbm()