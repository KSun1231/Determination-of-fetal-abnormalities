import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats

# 设置中文字体和数学字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['mathtext.fontset'] = 'stix'  # 用来正常显示数学符号
plt.rcParams['font.size'] = 10  # 设置字体大小

# ========== 数据读取与预处理 ==========

# 确保使用正确的文件路径
script_dir = os.path.dirname(os.path.abspath(__file__))
boys_path = os.path.join(script_dir, "clean_data_csv", "boys_clean_1.csv")

df_boys = pd.read_csv(boys_path)

# 处理男胎数据，提取需要分析的变量
df_male = df_boys.copy()
df_male["孕周"] = df_male["孕天"] / 7.0
df_male["BMI"] = df_male["孕妇BMI"]
df_male["Y浓度"] = df_male["Y染色体浓度"]
df_male["年龄"] = df_male["年龄"]
df_male = df_male.dropna(subset=["孕周","BMI","Y浓度","年龄"])

print(f"样本: {len(df_male)}条")

# 相关性分析
corr_matrix = df_male[["孕周","BMI","Y浓度","年龄"]].corr()
y_correlations = corr_matrix["Y浓度"].sort_values(ascending=False)
print(f"Y浓度相关性: 孕周{y_correlations['孕周']:.3f}, BMI{y_correlations['BMI']:.3f}, 年龄{y_correlations['年龄']:.3f}")

# 数据可视化

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Y染色体浓度与各指标的关系分析', fontsize=16)

# Y浓度 vs 孕周
axes[0,0].scatter(df_male["孕周"], df_male["Y浓度"], alpha=0.6, color='blue')
axes[0,0].set_xlabel("孕周 (周)")
axes[0,0].set_ylabel("Y染色体浓度")
axes[0,0].set_title("Y染色体浓度 vs 孕周")
# 添加拟合线
z = np.polyfit(df_male["孕周"], df_male["Y浓度"], 1)
p = np.poly1d(z)
axes[0,0].plot(df_male["孕周"], p(df_male["孕周"]), "r--", alpha=0.8)

# Y浓度 vs BMI
axes[0,1].scatter(df_male["BMI"], df_male["Y浓度"], alpha=0.6, color='green')
axes[0,1].set_xlabel("BMI")
axes[0,1].set_ylabel("Y染色体浓度")
axes[0,1].set_title("Y染色体浓度 vs BMI")
# 添加拟合线
z = np.polyfit(df_male["BMI"], df_male["Y浓度"], 1)
p = np.poly1d(z)
axes[0,1].plot(df_male["BMI"], p(df_male["BMI"]), "r--", alpha=0.8)

# Y浓度 vs 年龄
axes[1,0].scatter(df_male["年龄"], df_male["Y浓度"], alpha=0.6, color='purple')
axes[1,0].set_xlabel("年龄 (岁)")
axes[1,0].set_ylabel("Y染色体浓度")
axes[1,0].set_title("Y染色体浓度 vs 年龄")
# 添加拟合线
z = np.polyfit(df_male["年龄"], df_male["Y浓度"], 1)
p = np.poly1d(z)
axes[1,0].plot(df_male["年龄"], p(df_male["年龄"]), "r--", alpha=0.8)

# 相关性热力图
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, ax=axes[1,1])
axes[1,1].set_title("相关性热力图")

plt.tight_layout()
plt.show()

# 一元线性回归分析
y = df_male["Y浓度"]

# 孕周与Y浓度的一元回归
X1 = sm.add_constant(df_male["孕周"])
model_week = sm.OLS(y, X1).fit()

# BMI与Y浓度的一元回归
X2 = sm.add_constant(df_male["BMI"])
model_bmi = sm.OLS(y, X2).fit()

# 年龄与Y浓度的一元回归
X3 = sm.add_constant(df_male["年龄"])
model_age = sm.OLS(y, X3).fit()

print(f"一元回归R²: 孕周{model_week.rsquared:.3f}, BMI{model_bmi.rsquared:.3f}, 年龄{model_age.rsquared:.3f}")

# 多元线性回归分析
X_multi = df_male[["孕周","BMI","年龄"]]
X_multi = sm.add_constant(X_multi)
model_multi = sm.OLS(y, X_multi).fit()
print(f"多元回归: R²={model_multi.rsquared:.3f}, 调整R²={model_multi.rsquared_adj:.3f}")

# 模型评估与统计检验

# 计算基本统计量用于检验
y_actual = y
y_predicted = model_multi.fittedvalues
y_mean = np.mean(y_actual)
TSS = np.sum((y_actual - y_mean)**2)
RSS = np.sum((y_actual - y_predicted)**2)
ESS = np.sum((y_predicted - y_mean)**2)
n = len(y_actual)
k = int(model_multi.df_model)
coefficients = model_multi.params

# 模型评估
confidence_intervals = model_multi.conf_int()
significant_vars = []

for i, coef_name in enumerate(model_multi.params.index):
    if coef_name == 'const':
        continue
    p_value = model_multi.pvalues[coef_name]
    if p_value < 0.05:
        significant_vars.append(coef_name)

print(f"模型指标: R²={model_multi.rsquared:.3f}, 调整R²={model_multi.rsquared_adj:.3f}, F={model_multi.fvalue:.2f}(p={model_multi.f_pvalue:.3f})")
print(f"显著变量: {', '.join(significant_vars)} ({len(significant_vars)}/{len(model_multi.params)-1})")

# 模型比较分析 
# 收集模型信息
models_info = {
    '孕周（一元）': {
        'model': model_week,
        'r2': model_week.rsquared,
        'adj_r2': model_week.rsquared_adj,
        'aic': model_week.aic,
        'bic': model_week.bic,
        'mse': np.mean(model_week.resid**2)
    },
    'BMI（一元）': {
        'model': model_bmi,
        'r2': model_bmi.rsquared,
        'adj_r2': model_bmi.rsquared_adj,
        'aic': model_bmi.aic,
        'bic': model_bmi.bic,
        'mse': np.mean(model_bmi.resid**2)
    },
    '年龄（一元）': {
        'model': model_age,
        'r2': model_age.rsquared,
        'adj_r2': model_age.rsquared_adj,
        'aic': model_age.aic,
        'bic': model_age.bic,
        'mse': np.mean(model_age.resid**2)
    },
    '多元回归': {
        'model': model_multi,
        'r2': model_multi.rsquared,
        'adj_r2': model_multi.rsquared_adj,
        'aic': model_multi.aic,
        'bic': model_multi.bic,
        'mse': np.mean(model_multi.resid**2)
    }
}

# 创建比较表格
comparison_df = pd.DataFrame({
    '模型': list(models_info.keys()),
    'R²': [info['r2'] for info in models_info.values()],
    '调整R²': [info['adj_r2'] for info in models_info.values()],
    'AIC': [info['aic'] for info in models_info.values()],
    'BIC': [info['bic'] for info in models_info.values()],
    'MSE': [info['mse'] for info in models_info.values()]
})

print(f"模型对比: 孕周一元{comparison_df.loc[0, 'R²']:.3f}, BMI一元{comparison_df.loc[1, 'R²']:.3f}, 年龄一元{comparison_df.loc[2, 'R²']:.3f}, 多元{comparison_df.loc[3, 'R²']:.3f}")

# 模型对比可视化

# 拟合效果对比图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('一元线性回归 vs 多元线性回归 拟合效果对比', fontsize=16)

# 孕周一元回归
axes[0,0].scatter(df_male["孕周"], df_male["Y浓度"], alpha=0.6, color='blue', label='实际值')
axes[0,0].plot(df_male["孕周"], model_week.fittedvalues, 'r-', alpha=0.8, label=f'拟合线 ($R^2$={model_week.rsquared:.4f})')
axes[0,0].set_xlabel("孕周 (周)")
axes[0,0].set_ylabel("Y染色体浓度")
axes[0,0].set_title("孕周 vs Y浓度 (一元回归)")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# BMI一元回归
axes[0,1].scatter(df_male["BMI"], df_male["Y浓度"], alpha=0.6, color='green', label='实际值')
axes[0,1].plot(df_male["BMI"], model_bmi.fittedvalues, 'r-', alpha=0.8, label=f'拟合线 ($R^2$={model_bmi.rsquared:.4f})')
axes[0,1].set_xlabel("BMI")
axes[0,1].set_ylabel("Y染色体浓度")
axes[0,1].set_title("BMI vs Y浓度 (一元回归)")
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 年龄一元回归
axes[0,2].scatter(df_male["年龄"], df_male["Y浓度"], alpha=0.6, color='purple', label='实际值')
axes[0,2].plot(df_male["年龄"], model_age.fittedvalues, 'r-', alpha=0.8, label=f'拟合线 ($R^2$={model_age.rsquared:.4f})')
axes[0,2].set_xlabel("年龄 (岁)")
axes[0,2].set_ylabel("Y染色体浓度")
axes[0,2].set_title("年龄 vs Y浓度 (一元回归)")
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# 多元回归拟合值 vs 实际值
axes[1,0].scatter(df_male["Y浓度"], model_multi.fittedvalues, alpha=0.6, color='red')
axes[1,0].plot([df_male["Y浓度"].min(), df_male["Y浓度"].max()], 
               [df_male["Y浓度"].min(), df_male["Y浓度"].max()], 'k--', alpha=0.8)
axes[1,0].set_xlabel("实际Y浓度")
axes[1,0].set_ylabel("预测Y浓度")
axes[1,0].set_title(f"多元回归拟合效果 ($R^2$={model_multi.rsquared:.4f})")
axes[1,0].grid(True, alpha=0.3)

# 模型性能对比柱状图
metrics = ['R²', '调整R²']  # DataFrame中的列名
metric_labels = ['$R^2$', '调整$R^2$']  # 显示用的标签
x_pos = np.arange(len(comparison_df))
width = 0.35

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    axes[1,1].bar(x_pos + i*width, comparison_df[metric], width, 
                  label=label, alpha=0.8)

axes[1,1].set_xlabel('模型')
axes[1,1].set_ylabel('值')
axes[1,1].set_title('$R^2$和调整$R^2$对比')
axes[1,1].set_xticks(x_pos + width/2)
axes[1,1].set_xticklabels(comparison_df['模型'], rotation=45)
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

# AIC/BIC对比
axes[1,2].bar(x_pos - width/2, comparison_df['AIC'], width, label='AIC', alpha=0.8)
axes[1,2].bar(x_pos + width/2, comparison_df['BIC'], width, label='BIC', alpha=0.8)
axes[1,2].set_xlabel('模型')
axes[1,2].set_ylabel('信息准则值')
axes[1,2].set_title('AIC/BIC对比 (越小越好)')
axes[1,2].set_xticks(x_pos)
axes[1,2].set_xticklabels(comparison_df['模型'], rotation=45)
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 残差分析与置信区间

# 多元回归残差分析与置信区间
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('多元回归残差分析与置信区间', fontsize=16)

# 基本残差信息
fitted_values = model_multi.fittedvalues
residuals = model_multi.resid

# 计算残差的标准差和置信区间
residual_std = np.std(residuals)
n = len(residuals)
alpha = 0.05  # 95%置信区间
t_critical = stats.t.ppf(1 - alpha/2, n-1)

# 残差的95%置信区间
residual_ci_lower = -t_critical * residual_std
residual_ci_upper = t_critical * residual_std

print(f"\n残差统计: σ={residual_std:.4f}, 异常点={np.sum((residuals < residual_ci_lower) | (residuals > residual_ci_upper))}个")

# 残差vs拟合值 (含置信区间)
axes[0,0].scatter(fitted_values, residuals, alpha=0.6, color='blue')
axes[0,0].axhline(y=0, color='red', linestyle='--', label='零线')
axes[0,0].axhline(y=residual_ci_upper, color='orange', linestyle='--', alpha=0.8, label=f'95%置信区间')
axes[0,0].axhline(y=residual_ci_lower, color='orange', linestyle='--', alpha=0.8)
axes[0,0].fill_between(fitted_values, residual_ci_lower, residual_ci_upper, alpha=0.1, color='orange')
axes[0,0].set_xlabel("拟合值")
axes[0,0].set_ylabel("残差")
axes[0,0].set_title("残差 vs 拟合值 (含95%置信区间)")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Q-Q图检验正态性
stats.probplot(residuals, dist="norm", plot=axes[0,1])
axes[0,1].set_title("残差正态性检验 (Q-Q图)")
axes[0,1].grid(True, alpha=0.3)

# 残差直方图与正态分布对比
axes[1,0].hist(residuals, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
# 叠加理论正态分布
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
y_norm = stats.norm.pdf(x_norm, loc=0, scale=residual_std)
axes[1,0].plot(x_norm, y_norm, 'r-', linewidth=2, label='理论正态分布')
axes[1,0].axvline(x=residual_ci_lower, color='orange', linestyle='--', alpha=0.8, label='95%置信区间')
axes[1,0].axvline(x=residual_ci_upper, color='orange', linestyle='--', alpha=0.8)
axes[1,0].set_xlabel("残差值")
axes[1,0].set_ylabel("密度")
axes[1,0].set_title("残差分布直方图")
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 标准化残差
standardized_residuals = residuals / residual_std
axes[1,1].scatter(fitted_values, standardized_residuals, alpha=0.6, color='green')
axes[1,1].axhline(y=0, color='red', linestyle='--', label='零线')
axes[1,1].axhline(y=2, color='orange', linestyle='--', alpha=0.8, label='±2σ')
axes[1,1].axhline(y=-2, color='orange', linestyle='--', alpha=0.8)
axes[1,1].axhline(y=3, color='red', linestyle=':', alpha=0.8, label='±3σ')
axes[1,1].axhline(y=-3, color='red', linestyle=':', alpha=0.8)
axes[1,1].set_xlabel("拟合值")
axes[1,1].set_ylabel("标准化残差")
axes[1,1].set_title("标准化残差图")
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 异常值检测
outliers_2sigma = np.abs(standardized_residuals) > 2
outliers_3sigma = np.abs(standardized_residuals) > 3
print(f"\n异常值: ±2σ外{np.sum(outliers_2sigma)}个, ±3σ外{np.sum(outliers_3sigma)}个")

# 预测区间计算

# 正确计算预测区间
X_matrix = X_multi.values  # 转换为numpy数组
XTX_inv = np.linalg.inv(X_matrix.T @ X_matrix)  # 使用@操作符
hat_matrix_diag = np.diag(X_matrix @ XTX_inv @ X_matrix.T)  # 帽子矩阵对角元素

# 预测标准误
prediction_se = residual_std * np.sqrt(1 + hat_matrix_diag)
prediction_ci_lower = fitted_values - t_critical * prediction_se
prediction_ci_upper = fitted_values + t_critical * prediction_se

print(f"\n预测区间: 平均宽度{np.mean(prediction_ci_upper - prediction_ci_lower):.4f}, 覆盖率{np.mean((df_male['Y浓度'] >= prediction_ci_lower) & (df_male['Y浓度'] <= prediction_ci_upper))*100:.1f}%")

# 预测区间可视化
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
# 按拟合值排序以便绘制
sort_indices = np.argsort(fitted_values)
fitted_sorted = fitted_values[sort_indices]
pred_lower_sorted = prediction_ci_lower[sort_indices]
pred_upper_sorted = prediction_ci_upper[sort_indices]
actual_sorted = df_male["Y浓度"].iloc[sort_indices]

ax.scatter(fitted_sorted, actual_sorted, alpha=0.6, color='blue', label='实际观测值')
ax.plot(fitted_sorted, fitted_sorted, 'r-', linewidth=2, label='完美预测线')
ax.fill_between(fitted_sorted, pred_lower_sorted, pred_upper_sorted, 
                alpha=0.2, color='orange', label='95%预测区间')
ax.plot(fitted_sorted, pred_lower_sorted, 'orange', linestyle='--', alpha=0.8)
ax.plot(fitted_sorted, pred_upper_sorted, 'orange', linestyle='--', alpha=0.8)

ax.set_xlabel('拟合值 (预测Y浓度)')
ax.set_ylabel('实际Y浓度')
ax.set_title('多元回归预测区间图')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# 残差分析对比（最佳一元 vs 多元）
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('残差分析对比', fontsize=16)

# 最佳一元回归模型的残差（选择R²最高的）
best_single_model = max([model_week, model_bmi, model_age], key=lambda x: x.rsquared)
best_single_name = ['孕周', 'BMI', '年龄'][np.argmax([model_week.rsquared, model_bmi.rsquared, model_age.rsquared])]

# 最佳一元回归残差vs拟合值
axes[0,0].scatter(best_single_model.fittedvalues, best_single_model.resid, alpha=0.6)
axes[0,0].axhline(y=0, color='red', linestyle='--')
axes[0,0].set_xlabel("拟合值")
axes[0,0].set_ylabel("残差")
axes[0,0].set_title(f"最佳一元回归({best_single_name})残差图")
axes[0,0].grid(True, alpha=0.3)

# 多元回归残差vs拟合值
axes[0,1].scatter(model_multi.fittedvalues, model_multi.resid, alpha=0.6)
axes[0,1].axhline(y=0, color='red', linestyle='--')
axes[0,1].set_xlabel("拟合值")
axes[0,1].set_ylabel("残差")
axes[0,1].set_title("多元回归残差图")
axes[0,1].grid(True, alpha=0.3)

# Q-Q图对比
stats.probplot(best_single_model.resid, dist="norm", plot=axes[1,0])
axes[1,0].set_title(f"最佳一元回归({best_single_name})Q-Q图")

stats.probplot(model_multi.resid, dist="norm", plot=axes[1,1])
axes[1,1].set_title("多元回归Q-Q图")

plt.tight_layout()
plt.show()

# 综合结论
best_r2 = comparison_df['R²'].max()
improvement = model_multi.rsquared - max(model_week.rsquared, model_bmi.rsquared, model_age.rsquared)
percentage_improvement = (improvement / max(model_week.rsquared, model_bmi.rsquared, model_age.rsquared)) * 100

print(f"\n综合结论:")
print(f"最佳模型: 多元回归 (R²={best_r2:.3f}, 提升{percentage_improvement:.1f}%)")
print(f"显著变量: {', '.join(significant_vars)} (均显著)")
print(f"结论: 推荐使用多元回归模型")
