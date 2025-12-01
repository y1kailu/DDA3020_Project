import optuna
import os
import numpy as np
from sklearn.metrics import mean_squared_error

# 导入数据处理
from data.data_process import load_and_process_data

# 导入你的模型模块 (确保它们都在 models/ 文件夹下且有 run 函数)
from models import lgbm_model, xgboost_model, ridge_model, mlp_model, lstm_model

# 全局配置
N_TRIALS_TREE = 50   # 树模型尝试次数 (跑得快，可以多试)
N_TRIALS_NN = 15     # 神经网络尝试次数 (跑得慢，少试几次)
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

def objective(trial, model_name, X_train, y_train, X_test, y_test):
    """定义不同模型的搜索空间"""
    
    params = {}
    
    # === 1. LightGBM Search Space ===
    if model_name == 'LGBM':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'objective': 'regression',
            'metric': 'mse',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        # 调用模型训练 (注意：你的 run 函数需要能接收 params)
        y_pred = lgbm_model.run(X_train, y_train, X_test, params)

    # === 2. XGBoost Search Space ===
    elif model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_jobs': -1,
            'random_state': 42
        }
        y_pred = xgboost_model.run(X_train, y_train, X_test, params)

    # === 3. Ridge Regression Search Space ===
    elif model_name == 'Ridge':
        params = {
            'alpha': trial.suggest_float('alpha', 0.1, 1000.0, log=True)
        }
        y_pred = ridge_model.run(X_train, y_train, X_test, params)

    # === 4. MLP (ResNet) Search Space ===
    elif model_name == 'MLP':
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'num_blocks': trial.suggest_int('num_blocks', 1, 4),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
            'epochs': 20  # 固定 Epoch，靠早停或快速验证
        }
        y_pred = mlp_model.run(X_train, y_train, X_test, params)
    
    # === 5. LSTM Search Space ===
    elif model_name == 'LSTM':
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 1, 2),
            'dropout': trial.suggest_float('dropout', 0.1, 0.4),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': 128,
            'epochs': 15 
        }
        y_pred = lstm_model.run(X_train, y_train, X_test, params)

    # 计算 MSE (这里不用 Sharpe 是因为 Sharpe 很难直接优化，MSE 稳健)
    # 如果 y_pred 是 tensor 或 list，转为 numpy
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # 确保维度匹配
    y_pred = y_pred.flatten()
    
    mse = mean_squared_error(y_test, y_pred)
    return mse

def main():
    print("🚀 Loading Data for Auto-Tuning...")
    # 只需要加载一次数据
    X_train, X_test, y_train, y_test = load_and_process_data(DATA_PATH)
    
    # 定义要优化的模型
    models_to_tune = ['Ridge', 'LGBM', 'XGBoost', 'MLP', 'LSTM']
    
    for model_name in models_to_tune:
        print(f"\n===========================================")
        print(f"🤖 Tuning {model_name}...")
        print(f"===========================================")
        
        # 定义优化方向 (minimize MSE)
        study = optuna.create_study(direction='minimize')
        
        # 针对不同模型设置不同的 Trial 次数
        n_trials = N_TRIALS_NN if model_name in ['MLP', 'LSTM'] else N_TRIALS_TREE
        
        # 开始优化
        try:
            study.optimize(
                lambda trial: objective(trial, model_name, X_train, y_train, X_test, y_test), 
                n_trials=n_trials
            )
            
            print(f"🏆 Best MSE for {model_name}: {study.best_value:.6f}")
            print(f"🔧 Best Params: {study.best_params}")
            
            # 自动保存到 configs/
            save_config(model_name, study.best_params)
            
        except Exception as e:
            print(f"❌ Error tuning {model_name}: {e}")
            print("Skipping to next model...")

if __name__ == "__main__":
    main()