import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_ridge_baseline():
    print("="*60)
    print(">>> 项目启动：Ridge Regression 基准模型训练 (Baseline)")
    print("="*60)

    # 1. 加载清洗后的数据
    # 请确保这里读取的是你上一步清洗完保存的文件
    file_path = 'train_cleaned.csv' 
    print(f"正在读取 {file_path} ...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("错误：找不到 train_cleaned.csv，请检查文件名或路径。")
        return

    # 2. 准备特征 (X) 和 目标 (y)
    print("\n[Step 1] 准备数据...")
    
    # 必须按时间排序，否则时间序列验证会失效
    if 'date_id' in df.columns:
        df = df.sort_values('date_id')
    
    # 排除掉不是特征的列
    # date_id: 日期索引
    # forward_returns: 预测目标
    # market_forward_excess_returns: 另一个目标变量，也不能当特征
    # risk_free_rate: 计算夏普比率用的，不是特征
    exclude_cols = ['date_id', 'forward_returns', 'market_forward_excess_returns', 'risk_free_rate']
    
    # 特征列 (X)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    # 目标列 (y)
    target_col = 'forward_returns'

    X = df[feature_cols]
    y = df[target_col]

    print(f"特征数量: {X.shape[1]}")
    print(f"训练样本数: {X.shape[0]}")

    # 3. 设置时间序列交叉验证 (Time Series Cross-Validation)
    # 这是金融预测的灵魂：只能用前 80% 预测后 20%，不能反过来
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    model = Ridge(alpha=1.0) # alpha是惩罚系数，默认1.0，越大惩罚越狠
    
    mse_scores = []
    ic_scores = [] # Information Coefficient (预测值和真实值的相关性)

    print(f"\n[Step 2] 开始 {n_splits} 折时间序列交叉验证...")

    fold = 1
    for train_index, val_index in tscv.split(X):
        # 切分数据
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # 数据标准化 (StandardScaler)
        # 注意：必须只在训练集上拟合(fit)，然后应用到验证集，防止数据泄露
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 训练模型
        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_val_scaled)

        # 评估
        mse = mean_squared_error(y_val, y_pred)
        # 计算 IC (Pearson Correlation)
        df_eval = pd.DataFrame({'real': y_val, 'pred': y_pred})
        ic = df_eval.corr().iloc[0, 1]

        mse_scores.append(mse)
        ic_scores.append(ic)

        print(f"Fold {fold}: MSE = {mse:.6f}, IC = {ic:.4f} (样本数: {len(y_val)})")
        fold += 1

    # 4. 总结结果
    print("\n" + "="*30)
    print(">>> 最终评估结果 (Average Performance)")
    print("="*30)
    print(f"平均 MSE (越小越好): {np.mean(mse_scores):.6f}")
    print(f"平均 IC  (越大越好): {np.mean(ic_scores):.4f}")
    
    if np.mean(ic_scores) > 0:
        print("结论：模型有预测能力！预测方向与真实方向正相关。")
    else:
        print("结论：模型预测能力较弱，可能需要更强的特征或非线性模型。")

    # 5. 全量训练并保存可视化
    print("\n[Step 3] 全量训练并生成预测分布图...")
    # 用所有数据最后训练一次
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    final_preds = model.predict(X_scaled)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y, final_preds, alpha=0.1, s=1)
    plt.xlabel('真实回报 (Real Returns)')
    plt.ylabel('预测回报 (Predicted Returns)')
    plt.title('Ridge Regression: 预测值 vs 真实值')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # 画一条对角线
    plt.tight_layout()
    plt.savefig('ridge_baseline_result.png')
    print("-> 已保存可视化结果: ridge_baseline_result.png")

if __name__ == "__main__":
    run_ridge_baseline()