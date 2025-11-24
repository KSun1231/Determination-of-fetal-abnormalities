import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置中文字体和数学字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10

# 数据读取 - 确保使用正确的文件路径
script_dir = os.path.dirname(os.path.abspath(__file__))
boys_path = os.path.join(script_dir, "clean_data_csv", "boys_clean_1.csv")

# 检查文件是否存在
if not os.path.exists(boys_path):
    print(f"错误：找不到文件 {boys_path}")
    print(f"当前工作目录：{os.getcwd()}")
    print(f"脚本目录：{script_dir}")
    exit()

df_boys = pd.read_csv(boys_path)

# 处理男胎数据
df_male = df_boys.copy()
df_male["孕周"] = df_male["孕天"] / 7.0
df_male["BMI"] = df_male["孕妇BMI"]
df_male["Y浓度"] = df_male["Y染色体浓度"]
df_male["年龄"] = df_male["年龄"]
df_male = df_male.dropna(subset=["孕周","BMI","Y浓度","年龄"])

print(f"样本数量: {len(df_male)}")

# 多元线性回归分析
X_multi = df_male[["孕周","BMI","年龄"]]
X_multi = sm.add_constant(X_multi)
y = df_male["Y浓度"]
model_multi = sm.OLS(y, X_multi).fit()

print(f"多元回归R²: {model_multi.rsquared:.4f}")

# 创建子图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Y染色体浓度回归分析结果', fontsize=16)

# 多元回归拟合值 vs 实际值
axes[1,0].scatter(df_male["Y浓度"], model_multi.fittedvalues, alpha=0.6, color='red')
axes[1,0].plot([df_male["Y浓度"].min(), df_male["Y浓度"].max()], 
               [df_male["Y浓度"].min(), df_male["Y浓度"].max()], 'k--', alpha=0.8)
axes[1,0].set_xlabel("实际Y浓度")
axes[1,0].set_ylabel("预测Y浓度")
axes[1,0].set_title(f"多元回归拟合效果 ($R^2$={model_multi.rsquared:.4f})")
axes[1,0].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()