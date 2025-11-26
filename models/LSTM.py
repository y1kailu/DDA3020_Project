import pandas as pd
import numpy as np
import os
import tensorflow as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

# ================= 实验控制面板 (Control Panel) =================
# 每次跑实验前，只需要改这里的参数，然后保存运行即可！

# 1. 特征选择开关
USE_TOP_10_ONLY = False   # 【实验1开关】设为 True 就是只用Top10，设为 False 就是用全量数据

# 2. 模型参数调整
WINDOW_SIZE = 5          # 【实验2参数】回顾过去多少天？(比如改成 20)
LSTM_UNITS = 64          # 【实验3参数】神经元数量 (比如改成 128)
DROPOUT_RATE = 0.4       # 【实验3参数】丢弃率 (比如改成 0.4)

# 3. 固定参数 (通常不用变)
FILE_PATH = 'train_cleaned.csv'
BATCH_SIZE = 32
EPOCHS = 20
# ==============================================================

def create_sequences(data, target, window_size):
    """
    工具函数：把二维表格变成LSTM需要的三维数据 (样本数, 时间步, 特征数)
    """
    X, y = [], []
    # 从第 window_size 天开始，因为前几天没有足够的历史数据
    for i in range(len(data) - window_size):
        # 取出过去 window_size 天的所有特征
        X.append(data[i:(i + window_size)]) 
        # 取出第 i + window_size 天的目标值（我们要预测的那一天）
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

def run_lstm():
    print("="*50)
    print(">>> LSTM 模型启动：正在构建时间序列模型...")
    print("="*50)

    # 1. 加载数据
    if not os.path.exists(FILE_PATH):
        print(f"错误：找不到 {FILE_PATH}，请检查路径。")
        return
    
    df = pd.read_csv(FILE_PATH)
    
    # 按时间排序（绝对不能乱！）
    if 'date_id' in df.columns:
        df = df.sort_values('date_id')
        # 训练时不需要 date_id，把它设为索引或扔掉
        df = df.drop(columns=['date_id'])

    
    # === 新增：根据开关决定用哪些特征 ===
    target_col = 'forward_returns'
    
    if USE_TOP_10_ONLY:
        print(f"★ 实验模式：开启！仅使用 Top 10 特征进行训练")
        # 这里填入你之前分析出来的 Top 10 名字
        top_features = ['M4', 'V13', 'S5', 'S2', 'V7', 'M2', 'M17', 'M12', 'M8', 'S6']
        # 必须保留 target 否则没法训练
        required_cols = top_features + [target_col]
        # 过滤数据
        df = df[required_cols]
    else:
        print(f"★ 实验模式：使用全量特征进行训练")
    # =====================================

    
    # 还要把 market_forward_excess_returns 这种“未来数据”删掉，防止作弊
    drop_cols = [target_col, 'market_forward_excess_returns']
    # 确保这些列存在才删
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    feature_df = df.drop(columns=drop_cols)
    target_df = df[target_col]

    print(f"特征数量: {feature_df.shape[1]}")
    
    # 3. 数据标准化 (Standardization) —— LSTM 对数值范围非常敏感！
    # 我们需要把所有数据缩放到 0 附近
    scaler = StandardScaler()
    # fit_transform 会计算均值和方差并转换
    feature_scaled = scaler.fit_transform(feature_df)
    target_scaled = target_df.values # 目标值通常不需要缩放，或者也可以缩放

    # 4. 构建时间序列数据 (Sliding Window)
    print(f"正在构建时间窗口 (回顾过去 {WINDOW_SIZE} 天)...")
    X, y = create_sequences(feature_scaled, target_scaled, WINDOW_SIZE)
    print(f"构建完成！输入形状 X: {X.shape}, 标签形状 y: {y.shape}")
    # X 的形状应该是 (样本数, 10, 特征数)

    # 5. 切分训练集和验证集 (Time Series Split)
    # 严禁 shuffle！必须切最后一段作为验证集
    split_idx = int(len(X) * 0.8) # 80% 训练, 20% 验证
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")

    # 6. 搭建 LSTM 模型架构
    print("\n[Model] 正在搭建神经网络...")
    model = Sequential()
    
    # 第一层 LSTM
    # units=64: 神经元数量，可以调
    # return_sequences=False: 因为我们接全连接层输出结果，不需要返回序列
    # input_shape: (时间步, 特征数)
    model.add(LSTM(units=LSTM_UNITS, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    
    # Dropout层: 随机丢弃 20% 的神经元，防止过拟合（金融预测必备！）
    model.add(Dropout(DROPOUT_RATE))
    
    # 输出层: 输出 1 个数值 (预测的收益率)
    model.add(Dense(1))
    
    # 编译模型: 损失函数用 MSE (均方误差)，优化器用 Adam
    model.compile(optimizer='adam', loss='mse')
    
    model.summary() # 打印模型结构

    # 7. 训练模型
    print("\n[Training] 开始训练...")
    
    # 早停机制: 如果验证集 loss 在 3 轮内没有下降，就停止训练，防止过拟合
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # 8. 简单的策略转换 (把预测的收益率变成 0-2 的权重)
    # 这里我们做一个简单的逻辑：预测收益 > 0 就满仓(1)，收益很高就上杠杆(2)，否则空仓(0)
    print("\n[Prediction] 正在生成预测结果...")
    predictions = model.predict(X_val)
    
    # 简单的策略逻辑 (你可以修改这里！)
    predicted_weights = []
    for pred in predictions:
        p = pred[0]
        if p > 0.001:  # 预测涨幅超过 0.1%
            weight = 1.0
        elif p > 0.005: # 预测涨幅超过 0.5% (很有信心)
            weight = 2.0
        else:
            weight = 0.0
        predicted_weights.append(weight)
        
    print("预测完成！前10个预测权重:", predicted_weights[:10])
    
    # 保存模型，方便以后用
    model.save('my_lstm_model.keras')
    print("模型已保存为 my_lstm_model.keras")

# ==========================================
    # [升级版] 评估模块：解决乱码 + 计算命中率
    # ==========================================
    import matplotlib.pyplot as plt
    
    # --- 1. 计算方向命中率 (Hit Rate) ---
    # 逻辑：如果 (真实值>0) 和 (预测值>0) 是一样的，那就是猜对了
    true_direction = y_val > 0
    pred_direction = predictions > 0
    # np.mean 会自动把 True 当作 1，False 当作 0 来算平均值，结果就是百分比
    hit_rate = np.mean(true_direction == pred_direction)
    
    print("\n" + "="*30)
    print(f"★ 核心指标评估 (Evaluation) ★")
    print(f"方向命中率 (Hit Rate): {hit_rate * 100:.2f}%")
    print("="*30 + "\n")
    
    # --- 2. 画图 (使用英文标签，报告专用) ---
    plt.figure(figsize=(12, 6))
    
    # 只画前 100 天，不然线太密了看不清
    # 真实值用灰色，预测值用红色
    plt.plot(y_val[:100], label='True Returns', color='gray', alpha=0.6)
    plt.plot(predictions[:100], label='LSTM Predictions', color='red', linewidth=2)
    
    # 画一条水平的 0 线，方便看涨跌
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.title(f'LSTM Prediction Analysis (Hit Rate: {hit_rate*100:.1f}%)', fontsize=14)
    plt.xlabel('Time Steps (Days)', fontsize=12)
    plt.ylabel('Normalized Returns', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 保存高清图
    plt.savefig('lstm_report_plot.png', dpi=300)
    print("-> 已保存高清英文图表: lstm_report_plot.png")
    plt.show()

if __name__ == "__main__":
    run_lstm()