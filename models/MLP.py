import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')

class FastXGBPredictor:
    def __init__(self):
        self.model = None
        self.feature_groups = {}
        self.best_params = None
        
    def create_feature_groups_by_importance(self, X, y, top_pct=0.1, mid_pct=0.5):
        """Create three feature groups based on feature importance from an initial XGB run"""
        print("Creating feature groups by importance...")
        
        feature_cols = X.columns.tolist()
        
        # Train initial model to get feature importance
        initial_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        initial_model.fit(X, y)
        
        # Extract importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': initial_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Split features into groups
        n_features = len(importance_df)
        n_top = int(n_features * top_pct)
        n_mid = int(n_features * mid_pct) - n_top
        
        top_features = importance_df.head(n_top)['feature'].tolist()
        mid_features = importance_df.iloc[n_top:n_top+n_mid]['feature'].tolist()
        low_features = importance_df.iloc[n_top+n_mid:]['feature'].tolist()
        
        self.feature_groups = {
            'top_10pct': top_features,
            'mid_10_50pct': mid_features,
            'low_50pct': low_features
        }
        
        print(f"Top {len(top_features)} features (Top 10%)")
        print(f"Mid {len(mid_features)} features (10-50%)") 
        print(f"Low {len(low_features)} features (Remaining 50%)")
        
        return self.feature_groups
    
    def tune_regularization_params(self, X_train, y_train, feature_group_name, features):
        """Tune regularization parameters for a specific feature group"""
        print(f"\nTuning {feature_group_name} with {len(features)} features...")
        
        X_subset = X_train[features]
        
        # Define parameter grid (Simplified for speed in run mode)
        param_combinations = []
        # Reduced grid for faster execution
        for reg_alpha in [0, 1]:  
            for reg_lambda in [1, 5]: 
                for max_depth in [3, 4]:
                    param_combinations.append({
                        'reg_alpha': reg_alpha,
                        'reg_lambda': reg_lambda,
                        'max_depth': max_depth
                    })
        
        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=3)
        best_score = float('inf')
        best_params = None
        
        for i, params in enumerate(param_combinations):
            try:
                model = xgb.XGBRegressor(
                    n_estimators=50, # Reduced for tuning speed
                    learning_rate=0.1,
                    reg_alpha=params['reg_alpha'],
                    reg_lambda=params['reg_lambda'],
                    max_depth=params['max_depth'],
                    random_state=42,
                    n_jobs=-1
                )
                
                # CV Loop
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_subset):
                    X_tr, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_val)
                    score = mean_squared_error(y_val, preds)
                    cv_scores.append(score)
                
                avg_score = np.mean(cv_scores)
                
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = params
                    best_params['score'] = avg_score
                    
            except Exception as e:
                continue
        
        return best_params
    
    def find_best_regularization_combinations(self, X_train, y_train):
        """Find the best parameters for the 3 feature groups"""
        print("Finding best regularization combinations...")
        
        # Tune parameters for each group
        group_results = {}
        for group_name, features in self.feature_groups.items():
            best_params = self.tune_regularization_params(X_train, y_train, group_name, features)
            if best_params:
                group_results[group_name] = best_params
                
        # Select best combinations
        all_combinations = []
        for group_name, params in group_results.items():
            params_copy = params.copy()
            params_copy['feature_group'] = group_name
            all_combinations.append(params_copy)
        
        # Sort by MSE score
        all_combinations.sort(key=lambda x: x['score'])
        best_combinations = all_combinations[:3]
        
        print(f"\nBest regularization combination found: {best_combinations[0]}")
        self.best_params = best_combinations
        return best_combinations
    
    def train_final_model(self, X_train, y_train, X_test):
        """Train final model using the best found parameters"""
        print("\nTraining final XGBoost model...")
        
        # Use the best group configuration
        best_group = self.best_params[0]['feature_group']
        features = self.feature_groups[best_group]
        params = self.best_params[0]
        
        print(f"Using feature group: {best_group} with {len(features)} features")
        
        # Align test features (fill missing with 0)
        X_test_selected = pd.DataFrame(index=X_test.index)
        for f in features:
            if f in X_test.columns:
                X_test_selected[f] = X_test[f]
            else:
                X_test_selected[f] = 0.0
                
        X_train_final = X_train[features]
        X_test_final = X_test_selected
        
        # Train final XGB
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            max_depth=params['max_depth'],
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_final, y_train)
        
        # Predict
        test_pred = self.model.predict(X_test_final)
        return test_pred

def run(X_train, y_train, X_test):
    """
    Standardized interface function for XGBoost model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        X_test (pd.DataFrame): Test features
        
    Returns:
        np.array: Predictions on X_test
    """
    print("="*50)
    print(">>> XGBoost (Advanced Tuning): Training & Prediction...")
    print("="*50)
    
    # Handle missing values for trees (though XGB handles NaN, 0 is safer here)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    predictor = FastXGBPredictor()
    
    # 1. Feature Selection & Grouping
    predictor.create_feature_groups_by_importance(X_train, y_train)
    
    # 2. Hyperparameter Tuning
    # Note: This might take time. In 'run' mode we use a reduced grid in the class above.
    predictor.find_best_regularization_combinations(X_train, y_train)
    
    # 3. Train & Predict
    test_predictions = predictor.train_final_model(X_train, y_train, X_test)
    
    return test_predictions

# Optional: Standalone test block
if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'Basic Resource', 'train.csv')
    
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
