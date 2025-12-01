import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import os

# ================= Configuration =================
# Top 10 features selected based on EDA correlation analysis
SELECTED_FEATURES = ['M4', 'V13', 'S5', 'S2', 'V7', 'M2', 'M17', 'M12', 'M8', 'S6']
# =================================================

def run(X_train, y_train, X_test):
    """
    Standardized interface function for MLP model training and prediction.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        X_test (pd.DataFrame): Test features
        
    Returns:
        np.array: Predictions on X_test
    """
    print("="*50)
    print(">>> MLP (Neural Network): Training & Prediction...")
    print("="*50)

    # 1. Feature Selection & Preprocessing (Train)
    # Extract only the selected features
    # We handle missing columns gracefully just in case
    X_train_selected = pd.DataFrame(index=X_train.index)
    for feat in SELECTED_FEATURES:
        if feat in X_train.columns:
            X_train_selected[feat] = X_train[feat]
        else:
            X_train_selected[feat] = 0.0
            
    # Fill NaNs with 0
    X_train_final = X_train_selected.fillna(0)
    
    # Data Standardization (Crucial for Neural Networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    
    print(f"Training Input Shape: {X_train_scaled.shape}")

    # 2. Define MLP Model (Using hyperparameters from original script)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),  # Two hidden layers
        activation='relu',            # ReLU activation
        solver='adam',                # Adam optimizer
        alpha=0.001,                  # L2 regularization
        learning_rate_init=0.001,     # Initial learning rate
        max_iter=200,                 # Maximum iterations
        early_stopping=True,          # Stop if validation score stops improving
        validation_fraction=0.1,      # 10% data for validation
        n_iter_no_change=10,
        random_state=42,              # Seed for reproducibility
        verbose=False                 # Suppress verbose output during ensemble run
    )

    # 3. Train Model
    print("Training MLP model...")
    model.fit(X_train_scaled, y_train)
    
    # 4. Preprocessing (Test) & Prediction
    print("Processing Test Data...")
    
    # Align test features with selected features
    X_test_selected = pd.DataFrame(index=X_test.index)
    for feat in SELECTED_FEATURES:
        if feat in X_test.columns:
            X_test_selected[feat] = X_test[feat]
        else:
            # If feature missing in test set, fill with 0
            X_test_selected[feat] = 0.0
            
    X_test_final = X_test_selected.fillna(0)
    
    # Use the same scaler fitted on training data
    X_test_scaled = scaler.transform(X_test_final)
    
    print(f"Prediction Input Shape: {X_test_scaled.shape}")
    
    # Predict
    predictions = model.predict(X_test_scaled)
    
    return predictions

# Optional: Keep a main block for standalone testing
if __name__ == "__main__":
    # Test code (requires file path setup)
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming data is in ../data/train.csv relative to this script
    file_path = os.path.normpath(os.path.join(current_dir, '..', 'data', 'train.csv'))
    
    if os.path.exists(file_path):
        print(f"Testing MLP module with data from: {file_path}")
        df = pd.read_csv(file_path)
        if 'date_id' in df.columns: df = df.sort_values('date_id').drop(columns=['date_id'])
        
        target = 'forward_returns'
        # Columns to drop to simulate raw X input
        drop_cols = ['market_forward_excess_returns', 'forward_returns']
        
        # Simple split
        split = int(len(df) * 0.8)
        train = df.iloc[:split]
        test = df.iloc[split:]
        
        X_train = train.drop(columns=drop_cols, errors='ignore')
        y_train = train[target]
        X_test = test.drop(columns=drop_cols, errors='ignore')
        
        preds = run(X_train, y_train, X_test)
        print(f"Test Predictions (First 5): {preds[:5]}")
    else:
        print("Test data not found, skipping standalone test.")
