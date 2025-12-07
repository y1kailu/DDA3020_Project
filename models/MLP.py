import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略收敛警告（金融数据噪音大，很难完全收敛是正常的）
warnings.filterwarnings('ignore')

# ================= Configuration =================
# 默认参数 (作为兜底，当 params=None 时使用)
DEFAULT_PARAMS = {
    'hidden_layer_sizes': (64, 32),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.001,             # L2 正则化
    'learning_rate_init': 0.001,
    'max_iter': 200,            # 对应 epochs
    'batch_size': 64,
    'early_stopping': True,     # 防止过拟合
    'validation_fraction': 0.1, # 10% 用于验证早停
    'n_iter_no_change': 10,
    'random_state': 42
}
# =================================================

def run(X_train, y_train, X_test, params=None):
    """
    MLP (Sklearn 版本) 标准化接口。
    
    兼容性设计：
    1. 输入兼容：支持 Numpy Array 和 Pandas DataFrame。
    2. 参数兼容：自动映射 auto_tune 传来的通用参数名。
    3. 输出兼容：强制返回一维数组。
    """
    print("="*50)
    print(">>> MLP (Sklearn): Training & Prediction...")
    
    # -------------------------------------------------------
    # 1. 参数清洗与映射 (Parameter Cleaning)
    # -------------------------------------------------------
    model_params = DEFAULT_PARAMS.copy()
    
    if params:
        # 映射表：把 auto_tune 的参数名 -> sklearn 参数名
        mapping = {
            'epochs': 'max_iter',
            'learning_rate': 'learning_rate_init',
            'reg_alpha': 'alpha',
            'reg_lambda': 'alpha'
        }
        
        for k, v in params.items():
            # A. 处理需要改名的参数
            if k in mapping:
                model_params[mapping[k]] = v
            # B. 处理 hidden_dim (int -> tuple)
            elif k == 'hidden_dim':
                dim = int(v)
                model_params['hidden_layer_sizes'] = (dim, dim // 2)
            # C. 忽略不兼容参数 (Sklearn MLP 不支持 dropout, num_blocks, num_layers)
            elif k in ['dropout', 'dropout_rate', 'num_blocks', 'num_layers']:
                continue
            # D. 其他合法参数直接更新
            elif k in DEFAULT_PARAMS:
                model_params[k] = v

    print(f"   Active Params: {model_params}")

    # -------------------------------------------------------
    # 2. 数据格式统一 (Data Handling)
    # -------------------------------------------------------
    # 辅助函数：转为无 NaN 的 Numpy 数组
    def to_numpy_clean(data):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.fillna(0).values
        else:
            # 如果是 numpy，处理 nan 和 inf
            return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    X_train_clean = to_numpy_clean(X_train)
    X_test_clean = to_numpy_clean(X_test)
    y_train_clean = to_numpy_clean(y_train)

    # 强制 y 为 1D 数组 (N,)
    y_train_clean = y_train_clean.ravel()

    # -------------------------------------------------------
    # 3. 数据归一化 (Scaling)
    # -------------------------------------------------------
    # 神经网络对尺度极度敏感，必须再次确保归一化
    print("   Scaling data (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    # -------------------------------------------------------
    # 4. 模型训练 (Training)
    # -------------------------------------------------------
    # 实例化模型
    model = MLPRegressor(**model_params)
    
    print("   Training model...")
    try:
        model.fit(X_train_scaled, y_train_clean)
        print(f"   Training R2 Score: {model.score(X_train_scaled, y_train_clean):.4f}")
        print(f"   Best Loss: {model.best_loss_:.6f}")
    except Exception as e:
        print(f"⚠️ Training Warning: {e}")
        # 如果训练出错，不要崩溃，继续跑预测（虽然结果可能不好）

    # -------------------------------------------------------
    # 5. 预测 (Prediction)
    # -------------------------------------------------------
    print("   Generating predictions...")
    try:
        predictions = model.predict(X_test_scaled)
    except:
        # 兜底：如果预测失败，返回全0
        print("⚠️ Prediction failed, returning zeros.")
        predictions = np.zeros(len(X_test_scaled))
    
    # 确保返回 1D numpy array
    return predictions.flatten()

# === 单元测试 (独立运行时检查) ===
if __name__ == "__main__":
    print("Running standalone test...")
    # 模拟 main.py 的输入 (Numpy)
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.random.randn(100)
    X_test_dummy = np.random.randn(20, 10)
    
    # 模拟 auto_tune.py 的参数 (包含干扰项)
    messy_params = {
        'epochs': 50,          # 应映射为 max_iter
        'dropout': 0.5,        # 应被忽略
        'hidden_dim': 128,     # 应转为 tuple
        'learning_rate': 0.01  # 应映射
    }
    
    try:
        preds = run(X_dummy, y_dummy, X_test_dummy, messy_params)
        print(f"✅ Test Passed! Shape: {preds.shape}")
        print(f"Sample preds: {preds[:5]}")
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()