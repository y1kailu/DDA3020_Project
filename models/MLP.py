import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略收敛警告
warnings.filterwarnings('ignore')

# ================= Configuration =================
DEFAULT_PARAMS = {
    'hidden_layer_sizes': (64, 32),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.001,
    'learning_rate_init': 0.001,
    'max_iter': 200,
    'batch_size': 64,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 10,
    'random_state': 42
}
# =================================================

def run(X_train, y_train, X_test, params=None):
    """
    MLP (Sklearn) 防数值爆炸版。
    修复：
    1. 输入异常值截断 (Clip)。
    2. 目标变量 (y) 归一化。
    """
    print("="*50)
    print(">>> MLP (Sklearn): Training & Prediction...")
    
    # --- 1. 参数清洗 ---
    model_params = DEFAULT_PARAMS.copy()
    if params:
        mapping = {'epochs': 'max_iter', 'learning_rate': 'learning_rate_init', 
                   'reg_alpha': 'alpha', 'reg_lambda': 'alpha'}
        for k, v in params.items():
            if k in mapping: model_params[mapping[k]] = v
            elif k == 'hidden_dim': model_params['hidden_layer_sizes'] = (int(v), int(v)//2)
            elif k in DEFAULT_PARAMS: model_params[k] = v

    # --- 2. 数据格式统一 ---
    def to_numpy_clean(data):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.fillna(0).values
        else:
            return np.nan_to_num(data, nan=0.0)

    X_train_clean = to_numpy_clean(X_train)
    X_test_clean = to_numpy_clean(X_test)
    y_train_clean = to_numpy_clean(y_train).ravel()

    # --- 3. 特征归一化 & 异常值截断 (修复病根1) ---
    print("   Scaling features...")
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train_clean)
    X_test_scaled = scaler_x.transform(X_test_clean)

    # 【防爆盾 1】强制截断 5倍标准差之外的极端值
    # 这能防止某个 1e9 的异常值毁掉整个神经网络的权重
    X_train_scaled = np.clip(X_train_scaled, -5.0, 5.0)
    X_test_scaled = np.clip(X_test_scaled, -5.0, 5.0)

    # --- 4. 目标变量归一化 (修复病根2 - 关键！) ---
    # 神经网络不喜欢极小的 y (如 1e-4)。我们将 y 缩放到 0 附近，方差为 1。
    print("   Scaling target (y)...")
    scaler_y = StandardScaler()
    # reshape(-1, 1) 是因为 scaler 需要 2D 输入
    y_train_scaled = scaler_y.fit_transform(y_train_clean.reshape(-1, 1)).ravel()

    # --- 5. 模型训练 ---
    model = MLPRegressor(**model_params)
    
    print("   Training model...")
    try:
        # 注意：这里我们用的是 scaled y 进行训练
        model.fit(X_train_scaled, y_train_scaled)
        print(f"   Training Score (R2 on scaled y): {model.score(X_train_scaled, y_train_scaled):.4f}")
    except Exception as e:
        print(f"⚠️ Training Error: {e}")
        return np.zeros(len(X_test_scaled)) # 兜底

    # --- 6. 预测 & 反归一化 ---
    print("   Generating predictions...")
    try:
        # 预测出来的是 Scaled 的值 (比如 0.5, -0.2)
        pred_scaled = model.predict(X_test_scaled)
        
        # 【防爆盾 2】反归一化：把预测值还原回真实的收益率量级 (比如 0.005)
        # 这一步非常重要！否则 auto_tune 算 MSE 会错得离谱
        predictions = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        
    except Exception as e:
        print(f"⚠️ Prediction Error: {e}")
        predictions = np.zeros(len(X_test_scaled))
    
    return predictions.flatten()

# === 单元测试 ===
if __name__ == "__main__":
    print("Running test with extreme outliers...")
    # 模拟包含极端异常值的数据
    X_bad = np.random.randn(100, 5)
    X_bad[0, 0] = 1000000.0  # 制造一个超级异常值
    y_tiny = np.random.randn(100) * 0.001 # 模拟极小的收益率
    
    X_test = np.random.randn(20, 5)
    
    preds = run(X_bad, y_tiny, X_test)
    print(f"Preds mean: {preds.mean():.6f} (Should be close to 0.001 range, not 1000)")
    print(f"Test passed if shape is (20,): {preds.shape}")