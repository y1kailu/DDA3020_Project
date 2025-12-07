import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略一些不必要的 sklearn 警告，保持输出整洁
warnings.filterwarnings('ignore', category=UserWarning)

def run(X_train, y_train, X_test, params=None):
    """
    Standardized interface function for Ridge Regression.
    Compatible with auto_tune.py and main.py.
    
    Args:
        X_train (pd.DataFrame or np.array): Training features
        y_train (pd.Series or np.array): Training targets
        X_test (pd.DataFrame or np.array): Test features
        params (dict): Hyperparameters (optional). If None, uses RidgeCV.
        
    Returns:
        np.array: Predictions on X_test (1D array, shape=(N,))
    """
    print("="*50)
    print(">>> Ridge Regression: Training & Prediction...")
    
    # ==========================================
    # 1. 数据对齐与清洗 (Data Alignment & Cleaning)
    # ==========================================
    
    # 如果输入是 DataFrame，必须确保测试集的列与训练集完全一致
    if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        # 获取训练集的列名列表
        train_cols = X_train.columns.tolist()
        
        # 创建一个新的 DataFrame，确保列顺序和训练集一致
        # 1. 提取 X_test 中存在的列
        X_test_aligned = X_test.reindex(columns=train_cols, fill_value=0.0)
        
        # 2. 填充空值 (NaN -> 0)
        X_train_final = X_train.fillna(0.0)
        X_test_final = X_test_aligned.fillna(0.0)
    else:
        # 如果是 Numpy 数组，假设特征已经对齐，仅处理 NaN
        X_train_final = np.nan_to_num(X_train)
        X_test_final = np.nan_to_num(X_test)

    # 强制类型转换为 float32，防止 object 类型报错，也能节省内存
    # 使用 try-except 包装，以防数据中包含无法转换的字符
    try:
        X_train_final = np.array(X_train_final, dtype=np.float32)
        X_test_final = np.array(X_test_final, dtype=np.float32)
    except Exception as e:
        print(f"⚠️ Data Type Conversion Error: {e}")
        print("Trying to coerce data types...")
        # 强制转换，无法转换的变为 NaN，然后填 0
        if isinstance(X_train_final, pd.DataFrame):
             X_train_final = X_train_final.apply(pd.to_numeric, errors='coerce').fillna(0).values
             X_test_final = X_test_final.apply(pd.to_numeric, errors='coerce').fillna(0).values

    # 目标变量 y 的维度处理：强制铺平为 1D 数组 (N,)
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    elif isinstance(y_train, pd.DataFrame):
        y_train = y_train.values.flatten()
    else:
        y_train = np.array(y_train).flatten()
        
    # ==========================================
    # 2. 数据标准化 (Scaling) - 解决尺度灾难
    # ==========================================
    print("   Scaling data (StandardScaler)...")
    scaler = StandardScaler()
    # 严防数据泄露：只在 Train 上 fit，然后 transform Test
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)
    
    # ==========================================
    # 3. 模型初始化与训练 (Model Training)
    # ==========================================
    if params and 'alpha' in params:
        # 情况 A: auto_tune 传入了最佳参数
        alpha_val = float(params['alpha']) # 确保是 float
        print(f"   Training with fixed alpha: {alpha_val:.4f}")
        model = Ridge(alpha=alpha_val, random_state=42)
    else:
        # 情况 B: 默认模式，使用 RidgeCV 自动寻找最佳 alpha
        print("   No params provided. Using RidgeCV to auto-tune alpha...")
        # 搜索范围：10^-3 到 10^3
        alphas_to_test = np.logspace(-3, 3, 50) 
        model = RidgeCV(alphas=alphas_to_test, scoring='neg_mean_squared_error')
        
    model.fit(X_train_scaled, y_train)
    
    # 如果使用了 RidgeCV，打印出找到的最佳 alpha
    if isinstance(model, RidgeCV):
        print(f"   Best alpha found by RidgeCV: {model.alpha_:.4f}")

    # ==========================================
    # 4. 预测 (Prediction)
    # ==========================================
    print("   Generating predictions...")
    predictions = model.predict(X_test_scaled)
    
    # 再次确保返回的是 1D 数组
    predictions = predictions.flatten()
    
    print(f"   >>> Ridge Done. Output shape: {predictions.shape}")
    return predictions

# 单元测试代码 (独立运行时执行)
if __name__ == "__main__":
    print("Running Ridge standalone test...")
    
    # 1. 构造测试数据 (故意制造 Train 和 Test 列不一致的情况)
    X_train_dummy = pd.DataFrame({
        'Feature_A': np.random.rand(100),
        'Feature_B': np.random.rand(100) * 1000 # 大尺度特征
    })
    y_train_dummy = pd.Series(np.random.randn(100))
    
    # Test 数据少了一列 Feature_B，多了一列 Feature_C
    X_test_dummy = pd.DataFrame({
        'Feature_A': np.random.rand(20),
        'Feature_C': np.random.rand(20) 
    })

    print("Test Case: Column mismatch handling...")
    try:
        preds = run(X_train_dummy, y_train_dummy, X_test_dummy)
        print(f"✅ Test passed! Prediction shape: {preds.shape} (Expected: (20,))")
        print(f"   First 5 preds: {preds[:5]}")
    except Exception as e:
        print(f"❌ Test failed: {e}")