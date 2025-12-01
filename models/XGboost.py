import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

class FastXGBPredictor:
    def __init__(self):
        self.model = None
        self.feature_groups = {}
        self.best_params = None
        
    def load_data(self, file_path):
        """Load data from csv file"""
        print(f"Loading data from {file_path}...")
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        print(f"Data shape: {df.shape}")
        return df
    
    def create_feature_groups_by_importance(self, train_df, target_col='forward_returns', top_pct=0.1, mid_pct=0.5):
        """Create three feature groups based on feature importance from an initial XGB run"""
        print("Creating feature groups by importance...")
        
        # Prepare features and target
        # Exclude non-feature columns
        feature_cols = [col for col in train_df.columns if col not in 
                       ['date_id', 'forward_returns', 'risk_free_rate', 
                        'market_forward_excess_returns', 'is_scored']]
        
        X = train_df[feature_cols].fillna(0)
        y = train_df[target_col]
        
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
        
        X_subset = X_train[features].fillna(0)
        
        # Define parameter grid
        param_combinations = []
        for reg_alpha in [0, 0.1, 0.5, 1, 2]:  # L1 regularization
            for reg_lambda in [0.1, 0.5, 1, 2, 5]:  # L2 regularization
                for max_depth in [3, 4, 5]:
                    param_combinations.append({
                        'reg_alpha': reg_alpha,
                        'reg_lambda': reg_lambda,
                        'max_depth': max_depth
                    })
        
        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=3)
        best_score = float('inf')
        best_params = None
        
        # Test first 10 combinations for speed (in this demo)
        # In production, you might want to test more
        for i, params in enumerate(param_combinations[:10]):
            try:
                model = xgb.XGBRegressor(
                    n_estimators=100,
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
    
    def find_best_regularization_combinations(self, train_df, target_col='forward_returns'):
        """Find the best parameters for the 3 feature groups"""
        print("Finding best regularization combinations...")
        
        X_train = train_df
        y_train = train_df[target_col]
        
        # Tune parameters for each group
        group_results = {}
        for group_name, features in self.feature_groups.items():
            best_params = self.tune_regularization_params(X_train, y_train, group_name, features)
            if best_params:
                group_results[group_name] = best_params
                print(f"{group_name}: {best_params}")
        
        # Select best combinations
        all_combinations = []
        for group_name, params in group_results.items():
            params_copy = params.copy()
            params_copy['feature_group'] = group_name
            all_combinations.append(params_copy)
        
        # Sort by MSE score
        all_combinations.sort(key=lambda x: x['score'])
        best_combinations = all_combinations[:3]
        
        print(f"\nBest 3 regularization combinations:")
        for i, combo in enumerate(best_combinations, 1):
            print(f"{i}. {combo}")
        
        self.best_params = best_combinations
        return best_combinations
    
    def train_final_model(self, train_df, test_df, target_col='forward_returns'):
        """Train final model using the best found parameters"""
        print("\nTraining final model with best parameters...")
        
        # Use the best group configuration
        best_group = self.best_params[0]['feature_group']
        features = self.feature_groups[best_group]
        params = self.best_params[0]
        
        print(f"Using feature group: {best_group} with {len(features)} features")
        print(f"Parameters: {params}")
        
        X_train = train_df[features].fillna(0)
        y_train = train_df[target_col]
        
        # Ensure test_df has the same features
        # If features are missing in test_df, fill with 0
        X_test = pd.DataFrame(index=test_df.index)
        for f in features:
            if f in test_df.columns:
                X_test[f] = test_df[f]
            else:
                X_test[f] = 0.0
        X_test = X_test.fillna(0)
        
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
        
        self.model.fit(X_train, y_train)
        
        # Predict
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        print(f"Training RMSE: {train_rmse:.6f}")
        
        return test_pred
    
    def calculate_positions(self, predictions, method='adaptive_sigmoid'):
        """Convert predictions to portfolio weights (0 to 2)"""
        if method == 'adaptive_sigmoid':
            # Adaptive sigmoid based on prediction distribution
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            
            # Adjust scale factor
            if pred_std > 0:
                scale_factor = 1 / (pred_std * 2)
            else:
                scale_factor = 10
                
            # Sigmoid mapping
            positions = 1 + (1 / (1 + np.exp(-(predictions - pred_mean) * scale_factor)) - 0.5) * 2
        
        elif method == 'volatility_adjusted':
            pred_vol = np.std(predictions)
            if pred_vol > 0:
                z_scores = (predictions - np.mean(predictions)) / pred_vol
                positions = np.clip(1 + z_scores * 0.5, 0, 2)
            else:
                positions = np.ones_like(predictions)
        
        else:
            # Simple clipping
            positions = np.clip(predictions * 5 + 1, 0, 2)
        
        # Final Clip to [0, 2]
        positions = np.clip(positions, 0, 2)
        
        print(f"Position stats - Min: {positions.min():.3f}, Max: {positions.max():.3f}, Mean: {positions.mean():.3f}")
        return positions

def main():
    """Main execution flow"""
    
    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming directory structure: project/models/this_script.py and project/data/train.csv
    data_dir = os.path.join(current_dir, 'Basic Resource')
    
    # Paths to csv files
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    predictor = FastXGBPredictor()
    
    # 2. Load Data
    train_df = predictor.load_data(train_path)
    test_df = predictor.load_data(test_path)
    
    if train_df is None or test_df is None:
        print("Aborting due to missing data.")
        return

    # 3. Feature Selection & Grouping
    feature_groups = predictor.create_feature_groups_by_importance(train_df)
    
    # 4. Hyperparameter Tuning
    best_combinations = predictor.find_best_regularization_combinations(train_df)
    
    # 5. Train & Predict
    test_predictions = predictor.train_final_model(train_df, test_df)
    
    # 6. Convert to Weights
    positions = predictor.calculate_positions(test_predictions, method='adaptive_sigmoid')
    
    # 7. Create Submission File
    # Ensure date_id exists in test_df, otherwise generate dummy index
    if 'date_id' in test_df.columns:
        date_ids = test_df['date_id']
    else:
        date_ids = range(len(positions))

    submission = pd.DataFrame({
        'date_id': date_ids,
        'weight': positions
    })
    
    output_file = 'submission_xgboost.csv'
    submission.to_csv(output_file, index=False)
    print(f"\nSubmission saved to {output_file} with {len(submission)} predictions")
    
    # Stats
    print("\nPosition distribution:")
    print(submission['weight'].describe())

if __name__ == "__main__":
    main()