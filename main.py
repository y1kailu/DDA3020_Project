import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys
import importlib

# --- 1. 导入你的模块 (兼容不同文件夹结构) ---
try:
    # 尝试从 data 文件夹和 models 文件夹导入
    from data.data_process import load_and_process_data
    from models import LGBM, XGboost, Ridge_Regression, MLP, LSTM
except ImportError:
    # 如果所有文件都在根目录
    from data_process import load_and_process_data
    import LGBM, XGboost, Ridge_Regression, MLP, LSTM

# --- 全局绘图设置 ---
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans'] # 尝试显示中文，如果不行回退到英文
plt.rcParams['axes.unicode_minus'] = False 

DATA_PATH = 'data/train.csv'

def load_params(model_name):
    """
    尝试从 configs/ 文件夹加载 auto_tune 跑出来的最佳参数。
    如果找不到文件，返回空字典（使用模型默认参数）。
    """
    config_path = f"configs.{model_name.lower()}_config"
    try:
        if config_path in sys.modules:
            return sys.modules[config_path].params
        
        mod = importlib.import_module(config_path)
        print(f"✅ Loaded tuned params for {model_name}")
        return mod.params
    except ImportError:
        print(f"⚠️ No config found for {model_name}, using default params.")
        return {}

def plot_results(model_name, y_true, y_pred, r2_score_val):
    """
    画图函数：左边是折线对比（只画最后150个点），右边是散点图
    """
    plt.figure(figsize=(16, 6))
    
    # --- 子图 1: 折线图 (Line Plot) ---
    plt.subplot(1, 2, 1)
    # 只取最后 150 个点，让图表清晰可见
    subset_n = 150
    if len(y_true) > subset_n:
        y_true_plot = y_true[-subset_n:]
        y_pred_plot = y_pred[-subset_n:]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        
    plt.plot(range(len(y_true_plot)), y_true_plot, label='Actual (Truth)', color='black', alpha=0.7, linewidth=1.5)
    plt.plot(range(len(y_pred_plot)), y_pred_plot, label='Predicted', color='#ff4b4b', linestyle='--', linewidth=1.5)
    plt.title(f'{model_name}: Actual vs Predicted (Last {subset_n} Time Steps)', fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Returns', fontsize=12)
    plt.legend(fontsize=12)
    
    # --- 子图 2: 散点图 (Scatter Plot) ---
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5, s=10, color='#1f77b4', label='Data Points')
    
    # 画对角线 y=x (完美预测线)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit (y=x)')
    
    plt.title(f'{model_name}: Scatter Plot ($R^2$ = {r2_score_val:.4f})', fontsize=14)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    # 保存图片
    os.makedirs('results', exist_ok=True)
    save_path = f'results/{model_name}_performance.png'
    plt.savefig(save_path, dpi=150)
    print(f"   📊 Plot saved to {save_path}")
    plt.close() # 关闭画布，释放内存

def main():
    print("🚀 Starting Main Pipeline with Cross-Validation...")
    
    # 1. 加载数据
    X, y = load_and_process_data(DATA_PATH)
    
    # 2. 定义要运行的模型列表
    # 确保这里的名字和 keys 与 auto_tune 生成的 config 名字一致
    models_map = {
        'Ridge': Ridge_Regression,
        'LGBM': LGBM,
        'XGBoost': XGboost,
        'MLP': MLP,
        'LSTM': LSTM
    }
    
    # 3. 设置交叉验证 (Time Series Split)
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    final_metrics = []
    
    print(f"\n{'='*60}")
    print(f"🧪 Running {n_splits}-Fold Time Series Cross-Validation")
    print(f"{'='*60}\n")
    
    for name, model_module in models_map.items():
        print(f"▶️  Running Model: {name}")
        
        # 加载参数
        params = load_params(name)
        
        cv_rmse = []
        cv_r2 = []
        
        # 用于保存最后一折的数据来画图
        last_fold_y_true = None
        last_fold_y_pred = None
        
        fold = 1
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            try:
                # 运行模型 (使用统一接口 run)
                y_pred = model_module.run(X_train, y_train, X_test, params)
                
                # 计算指标
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                cv_rmse.append(rmse)
                cv_r2.append(r2)
                
                # 如果是最后一折，保存数据用于画图
                if fold == n_splits:
                    last_fold_y_true = y_test
                    last_fold_y_pred = y_pred
                
                # print(f"   Fold {fold}/{n_splits}: RMSE={rmse:.5f}, R2={r2:.5f}")
                fold += 1
                
            except Exception as e:
                print(f"   ❌ Error in Fold {fold}: {e}")
                # 遇到错误填入 NaN，防止程序崩溃
                cv_rmse.append(np.nan)
                cv_r2.append(np.nan)

        # 汇总当前模型结果
        avg_rmse = np.nanmean(cv_rmse)
        avg_r2 = np.nanmean(cv_r2)
        
        print(f"   ✅ Average RMSE: {avg_rmse:.5f} | Average R2: {avg_r2:.5f}")
        
        final_metrics.append({
            'Model': name,
            'CV RMSE': avg_rmse,
            'CV R2': avg_r2
        })
        
        # 画图 (只画最后一折的表现)
        if last_fold_y_true is not None:
            plot_results(name, last_fold_y_true, last_fold_y_pred, avg_r2)
        
        print("-" * 40)

    # --- 4. 输出最终表格 ---
    print(f"\n{'='*20} 🏆 Final Results Summary 🏆 {'='*20}")
    results_df = pd.DataFrame(final_metrics)
    
    # 按照 R2 排序 (R2 越高越好)
    results_df = results_df.sort_values(by='CV R2', ascending=False)
    
    print(results_df)
    
    # 保存表格
    results_df.to_csv('final_model_comparison.csv', index=False)
    print("\n✅ Results saved to 'final_model_comparison.csv'")
    print("✅ Plots saved in 'results/' folder")

if __name__ == "__main__":
    main()