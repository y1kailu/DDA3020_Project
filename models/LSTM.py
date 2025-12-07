import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ================= Configuration =================
# 默认超参数 (如果 auto_tune 没有传入参数，则使用这些)
DEFAULT_PARAMS = {
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 15,
    'window_size': 10  # 时间窗口长度
}

# 选定的 Top 10 特征 (必须与 EDA 结果一致)
SELECTED_FEATURES = ['M4', 'V13', 'S5', 'S2', 'V7', 'M2', 'M17', 'M12', 'M8', 'S6']
# =================================================

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 定义 LSTM 层
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层输出预测值
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # 初始化隐状态 (h0, c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # 前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取序列最后一个时间步的输出进行预测
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(X, y, window_size):
    """
    将 2D 数据 (Rows, Features) 转换为 LSTM 需要的 3D 数据 (Samples, Window, Features)
    """
    Xs, ys = [], []
    # 确保输入是 numpy 数组
    if isinstance(X, pd.DataFrame): X = X.values
    if isinstance(y, pd.Series): y = y.values
    
    # 滑动窗口生成序列
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i + window_size)])
        if y is not None:
            ys.append(y[i + window_size])
            
    return np.array(Xs), np.array(ys)

def run(X_train, y_train, X_test, params=None):
    """
    标准化接口函数：供 main.py 和 auto_tune.py 调用
    """
    print("="*50)
    print(">>> LSTM (PyTorch): Training & Prediction...")
    
    # 1. 参数初始化
    if params is None:
        params = DEFAULT_PARAMS
    else:
        # 如果传入部分参数，补全默认值
        for k, v in DEFAULT_PARAMS.items():
            if k not in params:
                params[k] = v
    
    # 自动检测设备 (GPU 优先)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    print(f"   Params: {params}")

    # 2. 特征对齐 (防止接口欺诈)
    # 强制只使用 SELECTED_FEATURES，缺失的补 0
    def align_features(df_input):
        if not isinstance(df_input, pd.DataFrame):
            return df_input # 如果已经是 numpy，假设特征已对齐
        
        df_aligned = pd.DataFrame(index=df_input.index)
        for feat in SELECTED_FEATURES:
            if feat in df_input.columns:
                df_aligned[feat] = df_input[feat]
            else:
                df_aligned[feat] = 0.0
        return df_aligned.fillna(0)

    X_train_final = align_features(X_train)
    X_test_final = align_features(X_test)

    # 3. 数据标准化 (修复数据泄露)
    # 仅在训练集上 fit，然后 transform 测试集
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)
    
    # 强制转换为 float32 (PyTorch 默认格式)
    X_train_scaled = X_train_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)
    
    if isinstance(y_train, pd.Series):
        y_train = y_train.values.astype(np.float32)
    else:
        y_train = y_train.astype(np.float32)

    # 4. 构建时间序列 (3D Tensor)
    window_size = int(params['window_size'])
    
    # 安全检查：数据长度必须大于窗口长度
    if len(X_train_scaled) <= window_size:
        print("⚠️ Error: Training data too short for window size.")
        return np.zeros(len(X_test)) # 返回全0防止报错
        
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, window_size)
    
    # 为测试集创建序列 (注意：这里会丢失前 window_size 个数据)
    X_test_seq, _ = create_sequences(X_test_scaled, None, window_size)
    
    if len(X_test_seq) == 0:
         print("⚠️ Warning: Test data too short. Returning zeros.")
         return np.zeros(len(X_test))

    # 转换为 Tensor 并加载到 DataLoader
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_seq), 
        torch.from_numpy(y_train_seq)
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(params['batch_size']), 
        shuffle=False # 时间序列数据不打乱
    )

    # 5. 初始化模型
    input_dim = X_train_seq.shape[2]
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=int(params['hidden_dim']),
        num_layers=int(params['num_layers']),
        dropout=params['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # 6. 训练循环
    model.train() # 开启 Dropout
    for epoch in range(int(params['epochs'])):
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            # 修复设备不匹配
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            # squeeze(-1) 确保维度匹配: (batch, 1) -> (batch)
            loss = criterion(outputs.view(-1), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 每 5 轮打印一次日志
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{params['epochs']}, Loss: {train_loss/len(train_loader):.6f}")

    # 7. 预测 (修复验证模式)
    print("   Generating predictions...")
    model.eval() # 关闭 Dropout
    
    predictions_seq = []
    test_tensor = torch.from_numpy(X_test_seq).to(device)
    
    # 分批次预测以防止显存溢出 (虽然这里数据量不大，但更稳健)
    # 简单起见，如果数据不大，直接全量预测
    with torch.no_grad(): # 关闭梯度计算
        preds = model(test_tensor)
        predictions_seq = preds.cpu().numpy().flatten()
    
    # 8. 结果对齐 (修复长度不一致)
    # 因为滑动窗口吃掉了前 N 个数据，我们需要填充对齐，保证输出长度 == X_test 长度
    pad_length = len(X_test) - len(predictions_seq)
    
    if pad_length > 0:
        # 策略：用第一个预测值填充前面的空缺 (Backfill)
        # 或者用 0 填充。这里选择用第一个有效值填充，保持趋势连续性。
        first_val = predictions_seq[0] if len(predictions_seq) > 0 else 0.0
        padding = np.full(pad_length, first_val)
        final_predictions = np.concatenate([padding, predictions_seq])
    else:
        final_predictions = predictions_seq

    print(f"   >>> LSTM Done. Output shape: {final_predictions.shape}")
    
    # 9. 修复沉默输出
    return final_predictions

# 单元测试 (独立运行时执行)
if __name__ == "__main__":
    print("Running LSTM standalone test...")
    # 生成假数据测试流程
    dummy_X = pd.DataFrame(np.random.randn(100, 15), columns=[f'col_{i}' for i in range(15)])
    dummy_X['M4'] = dummy_X['col_0'] # 模拟真实特征名
    dummy_y = pd.Series(np.random.randn(100))
    
    dummy_X_test = pd.DataFrame(np.random.randn(30, 15), columns=[f'col_{i}' for i in range(15)])
    dummy_X_test['M4'] = dummy_X_test['col_0']

    preds = run(dummy_X, dummy_y, dummy_X_test)
    print(f"Test passed if shape is (30,): {preds.shape}")