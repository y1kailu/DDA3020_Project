import os
import pandas as pd
import polars as pl
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import kaggle_evaluation.default_inference_server

# Global variables to store the model and scaler in memory across API calls
model = None
scaler = None

# === Configuration ===
# Top 10 features selected based on EDA correlation analysis
SELECTED_FEATURES = ['M4', 'V13', 'S5', 'S2', 'V7', 'M2', 'M17', 'M12', 'M8', 'S6']
current_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.normpath(os.path.join(current_dir, '..', 'data', 'train.csv'))

def train_model():
    """
    Trains the MLP model using the provided training dataset.
    This is called once during the first inference request (Cold Start).
    """
    global model, scaler
    
    print("Loading training data...")
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"Error: Training file not found at {TRAIN_DATA_PATH}")
        return

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    
    # 1. Prepare Training Data
    # Extract only the selected features, filling NaNs with 0
    X_train = train_df[SELECTED_FEATURES].fillna(0)
    y_train = train_df['forward_returns']
    
    # 2. Data Standardization
    # Neural Networks are sensitive to feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 3. Define and Train MLP Model
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
        verbose=False                 # Suppress verbose output during inference
    )
    
    print("Starting MLP model training...")
    model.fit(X_train_scaled, y_train)
    
    train_score = model.score(X_train_scaled, y_train)
    print(f"Training complete. Training set R^2 score: {train_score:.4f}")

def predict(test: pl.DataFrame) -> float:
    """
    Inference function called by the Kaggle API for each timestep.
    
    Args:
        test: Polars DataFrame containing the features for the current timestep.
    Returns:
        float: A portfolio weight between 0 and 2.
    """
    global model, scaler
    
    try:
        # 1. Convert Polars DataFrame to Pandas
        test_data = test.to_pandas()
        
        # 2. Cold Start: Train model if it doesn't exist
        if model is None:
            print("First call detected. Initializing model...")
            train_model()
        
        # 3. Feature Alignment (Bug Fix)
        # Ensure input X_test has exactly the same columns as training data.
        # If a feature is missing in the test stream, fill it with 0.
        X_test = pd.DataFrame(index=test_data.index)
        for feat in SELECTED_FEATURES:
            if feat in test_data.columns:
                X_test[feat] = test_data[feat]
            else:
                X_test[feat] = 0.0
        
        # Handle any remaining NaNs
        X_test = X_test.fillna(0)
        
        # 4. Scale and Predict
        X_test_scaled = scaler.transform(X_test)
        predicted_return = model.predict(X_test_scaled)[0]
        
        # 5. Strategy Logic: Convert Return to Portfolio Weight
        # Base weight represents 100% invested in the index
        base_weight = 1.0
        adjustment = 0.0
        
        # Step-wise allocation strategy based on predicted return thresholds
        if predicted_return > 0.01:      # Strong Bullish signal (> 1%)
            adjustment = 0.8
        elif predicted_return > 0.005:   # Moderate Bullish signal (> 0.5%)
            adjustment = 0.4
        elif predicted_return > 0:       # Slight Bullish
            adjustment = 0.1
        elif predicted_return > -0.005:  # Slight Bearish
            adjustment = -0.1
        elif predicted_return > -0.01:   # Moderate Bearish
            adjustment = -0.4
        else:                            # Strong Bearish signal
            adjustment = -0.8
            
        # Calculate final weight and clip to valid range [0, 2]
        weight = base_weight + adjustment
        weight = max(0.0, min(2.0, weight))
        
        # Optional: Log prediction for debugging
        # print(f"Pred: {predicted_return:.5f} -> Weight: {weight}")
        
        return float(weight)
        
    except Exception as e:
        # Fallback: Return 1.0 (Market Benchmark) if any error occurs
        print(f"Prediction Error: {e}")
        return 1.0

# === Entry Point for Kaggle Evaluation API ===
if __name__ == "__main__":
    inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)
    data_dir = os.path.normpath(os.path.join(current_dir, '..', 'data'))

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway((data_dir,))
        
        print(f"Start local evaluation, data directory: {local_data_dir}")
        inference_server.run_local_gateway(
            (local_data_dir,)  

        )
