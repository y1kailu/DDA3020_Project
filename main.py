import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 导入数据处理
from data.data_process import load_and_process_data

# 导入你的5个模型 (假设你已经把它们转成了 .py 并定义了 run 函数)
# 如果模型比较复杂，也可以在这里直接写 wrapper
from models import lgbm_model, lstm_model, mlp_model, ridge_model, xgboost_model

def evaluate_model(y_true, y_pred, model_name):
    """计算评价指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    sse = np.sum((y_true - y_pred) ** 2)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "Model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "SSE": sse,
        "R2": r2
    }

def plot_results(y_true, predictions_dict, save_dir="figures"):
    """绘图并保存"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 为了绘图清晰，只取前200个点 (如果是时间序列)
    sample_len = 200
    y_true_sample = y_true[:sample_len]
    
    # 1. 趋势对比图 (Trend Plot)
    plt.figure(figsize=(15, 6))
    plt.plot(y_true_sample.values, label='True Value', color='black', linewidth=2)
    
    for name, y_pred in predictions_dict.items():
        plt.plot(y_pred[:sample_len], label=name, alpha=0.7)
        
    plt.title("Model Prediction Trend (First 200 samples)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "trend_comparison.png"))
    plt.close()
    
    # 2. 散点图 (Scatter Plot) - 预测值 vs 真实值
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes[i]
        ax.scatter(y_true, y_pred, alpha=0.3, s=10)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--') # 对角线
        ax.set_title(f"{name} Scatter")
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted")
    
    # 隐藏多余的子图
    for j in range(len(predictions_dict), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scatter_comparison.png"))
    plt.close()

def main():
    print("Loading Data...")
    # 注意路径，取决于你在哪里运行 main.py
    X_train, X_test, y_train, y_test = load_and_process_data('data/train.csv')
    
    # 定义要运行的模型列表
    # 这里的 key 是名字，value 是导入的模块 (需要模块里有 run 函数)
    models = {
        "Ridge": ridge_model,
        "LGBM": lgbm_model,
        "XGBoost": xgboost_model,
        "MLP": mlp_model,
        "LSTM": lstm_model 
    }
    
    results = []
    predictions = {}
    
    print("Training & Evaluating Models...")
    for name, module in models.items():
        print(f"Running {name}...")
        try:
            # 假设每个模型文件里都有一个 run(X_train, y_train, X_test) 函数
            # 对于 LSTM，你可能需要确保输入维度是 (Batch, Seq, Feature)
            # 这里需要在各模型内部做好 reshape
            y_pred = module.run(X_train, y_train, X_test)
            
            # 确保 y_pred 是 1D array
            y_pred = y_pred.flatten() if hasattr(y_pred, 'flatten') else np.array(y_pred).flatten()
            
            # 记录结果
            metrics = evaluate_model(y_test, y_pred, name)
            results.append(metrics)
            predictions[name] = y_pred
            
        except Exception as e:
            print(f"Error running {name}: {e}")

    # 生成评价表格
    results_df = pd.DataFrame(results)
    print("\n=== Evaluation Results ===")
    print(results_df)
    results_df.to_csv("figures/model_metrics.csv", index=False)
    
    # 绘图
    print("Plotting figures...")
    plot_results(y_test, predictions)
    print("Done! Check 'figures/' folder.")

if __name__ == "__main__":
    main()