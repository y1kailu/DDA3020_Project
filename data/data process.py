import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置绘图风格
sns.set(style="whitegrid")
# 尝试设置中文显示，如果乱码也没关系，不影响分析
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

def run_eda_pipeline():
    print("="*50)
    print(">>> 项目启动：Hull Tactical 数据探索 (升级版)")
    print("="*50)

    # --- 1. 加载数据 (请确保这里是你真实的绝对路径) ---
    # 注意：如果在 VS Code 终端直接运行，且文件在同一目录下，直接写文件名即可
    # 如果报错找不到文件，请把下面这行改成绝对路径，例如: r'C:\Users\...\train.csv'
    file_path = r'C:\Users\36007\Desktop\CUHKSZ\2025-Fall\DDA3020\Homework\train.csv'
    
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}。请检查路径！")
        return

    print(f"正在读取 {file_path} ...")
    df = pd.read_csv(file_path)
    
    # --- 2. 数据清洗 ---
    print("\n[Step 1] 清洗缺失数据...")
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
    df_clean = df.drop(columns=cols_to_drop)
    
    if 'date_id' in df_clean.columns:
        df_clean = df_clean.sort_values('date_id')
    
    df_clean = df_clean.fillna(method='ffill').fillna(0)
    print(f"清洗完成！剩余特征: {df_clean.shape[1]}")

    # --- 3. 主力特征筛选 ---
    print("\n[Step 2] 筛选 Top 10 特征...")
    target = 'forward_returns'
    target_cols = [c for c in df_clean.columns if c.startswith(('M', 'V', 'S'))]
    
    analysis_df = df_clean[target_cols + [target]]
    corr_matrix = analysis_df.corr()
    
    # 自动抓取 Top 10 特征的名字（不用你手动抄写了）
    # head(11) 是因为包含 target 本身，所以取出来后我们要去掉 target
    top_features_series = corr_matrix[target].abs().sort_values(ascending=False).head(11)
    print(top_features_series)
    
    # 获取这10个特征的名字（去掉 forward_returns 自己）
    top_10_names = top_features_series.index.tolist()
    if target in top_10_names:
        top_10_names.remove(target)
    
    print(f"\n自动识别出的 Top 10 特征: {top_10_names}")

    # --- 4. 进阶分析：Top 10 内部相关性 (这就是你要跑的新代码) ---
    print("\n[Step 3] 正在生成进阶分析图表...")
    print("正在检查这 10 个特征是不是'抱团'太紧...")

    plt.figure(figsize=(10, 8))
    
    # 提取这10个特征的数据
    subset_df = df_clean[top_10_names]
    # 计算它们内部的相关性
    corr_internal = subset_df.corr()
    
    # 画图
    sns.heatmap(corr_internal, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title('Correlation Among Top 10 Features (特征内部相关性)', fontsize=15)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('feature_internal_correlation.png')
    print("-> 已保存图片: feature_internal_correlation.png")
    plt.show()

    print("\n分析结束！请查看生成的图片。")

if __name__ == "__main__":
    run_eda_pipeline()