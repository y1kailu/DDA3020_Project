import optuna
import numpy as np
import pandas as pd
import os
import sys
import traceback # 移到最上面，方便全局调用
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# --- 导入模型 ---
# 确保你的 models 文件夹下有这些文件，且 Ridge_Regression.py 没有空格
try:
    from models import LGBM, XGboost, Ridge_Regression, MLP, LSTM
    from data.data_process import load_and_process_data
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保：\n1. data_process.py 在 data/ 目录下\n2. 所有模型文件在 models/ 目录下\n3. Ridge Regression.py 已重命名为 Ridge_Regression.py")
    sys.exit(1)

# --- 全局配置 ---
N_TRIALS_TREE = 30   # 树模型尝试 30 次
N_TRIALS_NN = 10     # 神经网络尝试 10 次 (LSTM/MLP 较慢)
DATA_PATH = 'data/train.csv'

def save_config(model_name, best_params):
    """将最佳参数写入 configs/model_name_config.py"""
    os.makedirs('configs', exist_ok=True)
    file_path = f'configs/{model_name.lower()}_config.py'
    
    with open(file_path, 'w') as f:
        f.write(f"# Auto-generated config for {model_name}\n")
        f.write("params = {\n")
        for k, v in best_params.items():
            if isinstance(v, str):
                f.write(f"    '{k}': '{v}',\n")
            else:
                f.write(f"    '{k}': {v},\n")
        f.write("}\n")
    print(f"✅ Saved optimized config to {file_path}")

def objective(trial, model_name, X, y, model_module):
    """
    使用 TimeSeriesSplit 进行交叉验证调参
    """
    params = {}
    
    # === 1. 定义搜索空间 ===
    if model_name == 'LGBM':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'objective': 'regression',
            'metric': 'mse',
            'n_jobs': -1,
            'verbose': -1
        }
    elif model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'n_jobs': -1
        }
    elif model_name == 'Ridge':
        params = {
            'alpha': trial.suggest_float('alpha', 0.1, 100.0, log=True)
        }
    elif model_name == 'MLP':
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
            'epochs': 10 
        }
    elif model_name == 'LSTM':
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.4),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': 128,
            'epochs': 5 # LSTM 比较慢，调参时 Epoch 设小一点
        }

    # === 2. 交叉验证 ===
    # 3折交叉验证
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
  
        try:
            # 调用模型的 run 函数
            preds = model_module.run(X_train_cv, y_train_cv, X_val_cv, params)
            mse = mean_squared_error(y_val_cv, preds)
            cv_scores.append(mse)
        except Exception as e:
            # 如果某组参数导致模型崩溃（比如梯度爆炸），返回无穷大，让 Optuna 跳过
            print(f"⚠️ Error in trial: {e}")
            return float('inf')

    return np.mean(cv_scores)

def main():
    print("🚀 Loading Data for Auto-Tuning...")
    # 这里会使用 data_process 的缓存功能（如果上次跑过的话）
    X, y = load_and_process_data(DATA_PATH)
    
    # === 注册所有 5 个模型 ===
    models_map = {
        'Ridge': Ridge_Regression,
        'LGBM': LGBM,
        'XGBoost': XGboost,
        'MLP': MLP,
        'LSTM': LSTM  # <--- 之前这里漏了，加上它！
    }
    
    for model_name, model_module in models_map.items():
        print(f"\n{'='*40}")
        print(f"🤖 Tuning {model_name}...")
        print(f"{'='*40}")
        
        # 最小化 MSE
        study = optuna.create_study(direction='minimize')
        
        # 神经网络跑得慢，次数设少一点
        n_trials = N_TRIALS_NN if model_name in ['MLP', 'LSTM'] else N_TRIALS_TREE
        
        try:
            study.optimize(
                lambda trial: objective(trial, model_name, X, y, model_module), 
                n_trials=n_trials
            )
            
            print(f"🏆 Best MSE for {model_name}: {study.best_value:.6f}")
            print(f"🔧 Best Params: {study.best_params}")
            save_config(model_name, study.best_params)
            
        except Exception as e:
            print(f"❌ Error tuning {model_name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()