import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')


def run(X_train, y_train, X_test, params=None):
    print(">>> XGBoost: Training & Prediction...")
    
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 4,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    if params:
        default_params.update(params)
    
    model = xgb.XGBRegressor(**default_params)
    
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    
    model.fit(X_train.fillna(0), y_train)
    return model.predict(X_test.fillna(0))

# Standalone test block
if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', 'train.csv')
    
    if os.path.exists(file_path):
        print("Testing XGBoost module...")
        df = pd.read_csv(file_path)
        if 'date_id' in df.columns: df = df.sort_values('date_id').drop(columns=['date_id'])
        
        target = 'forward_returns'
        drop_cols = ['market_forward_excess_returns', 'forward_returns', 'risk_free_rate', 'is_scored']
        
        split = int(len(df) * 0.8)
        train = df.iloc[:split]
        test = df.iloc[split:]
        
        X_train = train.drop(columns=[c for c in drop_cols if c in train.columns], errors='ignore')
        y_train = train[target]
        X_test = test.drop(columns=[c for c in drop_cols if c in test.columns], errors='ignore')
        
        preds = run(X_train, y_train, X_test)
        print(f"Test Predictions: {preds[:5]}")
    else:
        print("Test data not found.")
