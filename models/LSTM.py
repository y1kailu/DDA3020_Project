import pandas as pd
import numpy as np
import os
import tensorflow as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ================= Control Panel =================
# Modify these parameters before running the script
# =================================================

# 1. Feature Selection Switch
# Set to True to use only Top 10 features, False to use all features
USE_TOP_10_ONLY = False   

# 2. Model Hyperparameters
WINDOW_SIZE = 5          # Lookback window size (e.g., 5 days)
LSTM_UNITS = 64          # Number of neurons in LSTM layer
DROPOUT_RATE = 0.4       # Dropout rate to prevent overfitting

# 3. Training Configuration
BATCH_SIZE = 32
EPOCHS = 20
# =================================================

def create_sequences(data, target, window_size):
    """
    Helper function: Convert 2D dataframe into 3D array for LSTM (Samples, TimeSteps, Features)
    """
    X, y = [], []
    # Start from window_size index because we need prior history
    for i in range(len(data) - window_size):
        # Take the past 'window_size' days as features
        X.append(data[i:(i + window_size)]) 
        # Take the target of the current day to predict
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

def run_lstm():
    print("="*50)
    print(">>> LSTM Start: Building Time-Series Neural Network...")
    print("="*50)

    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming data is in ../data/train.csv relative to models/ folder
    data_path = os.path.join(current_dir, '..', 'data', 'train.csv')
    file_path = os.path.normpath(data_path)

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Sort by date to ensure time-series integrity
    if 'date_id' in df.columns:
        df = df.sort_values('date_id')
        df = df.drop(columns=['date_id'])

    # === Feature Selection Logic ===
    target_col = 'forward_returns'
    
    if USE_TOP_10_ONLY:
        print(f"★ Experiment Mode: ON. Using only Top 10 features.")
        # Top 10 features identified in EDA
        top_features = ['M4', 'V13', 'S5', 'S2', 'V7', 'M2', 'M17', 'M12', 'M8', 'S6']
        # Ensure target is included
        required_cols = top_features + [target_col]
        # Filter dataframe (check if columns exist first)
        existing_cols = [c for c in required_cols if c in df.columns]
        df = df[existing_cols]
    else:
        print(f"★ Experiment Mode: OFF. Using ALL features.")
    # =====================================
    
    # Drop target and leakage columns from features
    drop_cols = [target_col, 'market_forward_excess_returns']
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    feature_df = df.drop(columns=drop_cols)
    target_df = df[target_col]

    print(f"Number of features: {feature_df.shape[1]}")
    
    # 3. Data Standardization
    # LSTM is sensitive to scale, standardization is crucial
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(feature_df)
    target_scaled = target_df.values # Target usually doesn't need scaling for regression, but can be done if needed

    # 4. Build Time-Series Sequences (Sliding Window)
    print(f"Building time windows (Lookback: {WINDOW_SIZE} days)...")
    X, y = create_sequences(feature_scaled, target_scaled, WINDOW_SIZE)
    print(f"Data shape - Input X: {X.shape}, Target y: {y.shape}")
    # X shape: (Samples, Window_Size, Features)

    # 5. Train/Validation Split (Sequential)
    # No shuffling allowed for time-series!
    split_idx = int(len(X) * 0.8) # 80% Train, 20% Val
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training Set: {len(X_train)}, Validation Set: {len(X_val)}")

    # 6. Build LSTM Architecture
    print("\n[Model] Constructing Neural Network...")
    model = Sequential()
    
    # LSTM Layer
    # return_sequences=False because we want one output after the sequence is processed
    model.add(LSTM(units=LSTM_UNITS, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    
    # Dropout Layer: Prevent overfitting
    model.add(Dropout(DROPOUT_RATE))
    
    # Output Layer: Predicts one continuous value (return)
    model.add(Dense(1))
    
    # Compile Model: MSE loss, Adam optimizer
    model.compile(optimizer='adam', loss='mse')
    
    model.summary()

    # 7. Train Model
    print("\n[Training] Starting training loop...")
    
    # Early Stopping: Stop if validation loss doesn't improve for 3 epochs
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # 8. Prediction and Strategy Logic
    print("\n[Prediction] Generating predictions...")
    predictions = model.predict(X_val)
    
    # Convert regression output to portfolio weights (0, 1, 2)
    predicted_weights = []
    for pred in predictions:
        p = pred[0]
        if p > 0.001:  # Strong signal threshold
            weight = 1.0
        elif p > 0.005: # Very strong signal
            weight = 2.0
        else:
            weight = 0.0
        predicted_weights.append(weight)
        
    print("Prediction complete! Sample weights:", predicted_weights[:10])
    
    # Save Model
    model.save('my_lstm_model.keras')
    print("Model saved to 'my_lstm_model.keras'")

    # ==========================================
    # Evaluation Module
    # ==========================================
    
    # --- 1. Calculate Directional Hit Rate ---
    # True if directions match (both positive or both negative)
    # Note: y_val is 1D array, predictions is 2D (N, 1)
    true_direction = y_val > 0
    pred_direction = predictions.flatten() > 0
    hit_rate = np.mean(true_direction == pred_direction)
    
    print("\n" + "="*30)
    print(f"★ Evaluation Metrics ★")
    print(f"Directional Hit Rate: {hit_rate * 100:.2f}%")
    print("="*30 + "\n")
    
    # --- 2. Visualization ---
    plt.figure(figsize=(12, 6))
    
    # Plot first 100 days for clarity
    plt.plot(y_val[:100], label='True Returns', color='gray', alpha=0.6)
    plt.plot(predictions[:100], label='LSTM Predictions', color='red', linewidth=2)
    
    # Zero line
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.title(f'LSTM Prediction Analysis (Hit Rate: {hit_rate*100:.1f}%)', fontsize=14)
    plt.xlabel('Time Steps (Days)', fontsize=12)
    plt.ylabel('Normalized Returns', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('lstm_report_plot.png', dpi=300)
    print("-> Plot saved as: lstm_report_plot.png")
    plt.show()

if __name__ == "__main__":
    run_lstm()
