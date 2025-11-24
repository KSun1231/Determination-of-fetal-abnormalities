import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from scipy import stats
import os

# 设置中文字体和数学字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['mathtext.fontset'] = 'stix'  # 用来正常显示数学符号
plt.rcParams['font.size'] = 10
# 修复R²符号显示问题
plt.rcParams['mathtext.default'] = 'regular'

print("多项式回归分析 - 临床特征与技术质量版本")

# 数据读取
script_dir = os.path.dirname(os.path.abspath(__file__))
boys_path = os.path.join(script_dir, "clean_data_csv", "boys_clean_full_variables.csv")

if not os.path.exists(boys_path):
    boys_path = os.path.join(script_dir, "clean_data_csv", "boys_clean_1.csv")

df_boys = pd.read_csv(boys_path)
print(f"样本: {len(df_boys)}条")

# 变量选择
available_vars = []
core_vars = ['Y染色体浓度', '孕天', '孕妇BMI', '年龄']

for var in core_vars:
    if var in df_boys.columns:
        available_vars.append(var)

extended_vars = [
    '身高', '体重', 'GC含量', '原始读段数', '唯一比对的读段数  ', '在参考基因组上比对的比例',
    '重复读段的比例', '被过滤掉读段数的比例', '怀孕次数', '生产次数', 'IVF妊娠'
]

for var in extended_vars:
    if var in df_boys.columns:
        available_vars.append(var)

# 数据处理
df_analysis = df_boys[available_vars].copy()

if 'IVF妊娠' in df_analysis.columns:
    df_analysis['IVF妊娠'] = (df_analysis['IVF妊娠'] == "IVF妊娠").astype(int)

for col in ['怀孕次数', '生产次数']:
    if col in df_analysis.columns:
        df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')

df_analysis['孕周'] = df_analysis['孕天'] / 7
df_analysis = df_analysis.dropna()

print(f"有效样本: {len(df_analysis)}条")

# 相关性分析
corr_matrix = df_analysis.corr()
y_correlations = corr_matrix['Y染色体浓度'].sort_values(ascending=False)

# 变量选择
core_vars_for_model = ['孕周', '孕妇BMI', '年龄']
high_corr_vars = []
for var, corr in y_correlations.items():
    if var != 'Y染色体浓度' and abs(corr) > 0.1:
        high_corr_vars.append(var)

# 构建多项式特征
modeling_vars = core_vars_for_model.copy()

for var in high_corr_vars:
    if var not in modeling_vars and len(modeling_vars) < 8:
        modeling_vars.append(var)

# 准备数据
y = df_analysis['Y染色体浓度'].values
X_raw = df_analysis[modeling_vars].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 构建模型特征
X_linear = np.column_stack([np.ones(len(df_analysis)), X_scaled])

# 多项式特征（前3个变量）
n_poly_vars = min(3, len(modeling_vars))
X_poly_base = X_scaled[:, :n_poly_vars]
X_poly_other = X_scaled[:, n_poly_vars:]

poly_features = []
for i in range(n_poly_vars):
    poly_features.append(X_poly_base[:, i])
    poly_features.append(X_poly_base[:, i]**2)

for i in range(n_poly_vars):
    for j in range(i+1, n_poly_vars):
        poly_features.append(X_poly_base[:, i] * X_poly_base[:, j])

X_poly = np.column_stack([np.ones(len(df_analysis))] + poly_features + [X_poly_other])

# 特征名称
linear_names = ['截距'] + modeling_vars
poly_names = ['截距']
for i in range(n_poly_vars):
    poly_names.append(f'{modeling_vars[i]}_线性')
    poly_names.append(f'{modeling_vars[i]}_二次')
for i in range(n_poly_vars):
    for j in range(i+1, n_poly_vars):
        poly_names.append(f'{modeling_vars[i]}_×_{modeling_vars[j]}')
for i in range(n_poly_vars, len(modeling_vars)):
    poly_names.append(modeling_vars[i])

# 模型拟合
model_linear_ols = sm.OLS(y, X_linear).fit()
model_poly_ols = sm.OLS(y, X_poly).fit()

core_vars_indices = [0, 1, 2, 3]
X_core = X_linear[:, core_vars_indices]
model_core = sm.OLS(y, X_core).fit()

model_full = sm.OLS(y, X_poly).fit()

models = {
    '线性模型': model_linear_ols,
    '多项式模型': model_poly_ols,
    '核心变量模型': model_core,
    '全变量模型': model_full
}

print(f"模型性能: 线性{model_linear_ols.rsquared:.3f}, 多项式{model_poly_ols.rsquared:.3f}, 核心{model_core.rsquared:.3f}, 全变量{model_full.rsquared:.3f}")

# 交叉验证
def cross_validate_model(X_data, y_data, cv_folds=5):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    r2_scores = []
    
    for train_idx, val_idx in kf.split(X_data):
        X_train, X_val = X_data[train_idx], X_data[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]
        
        try:
            temp_model = sm.OLS(y_train, X_train).fit()
            y_pred = temp_model.predict(X_val)
            r2_scores.append(r2_score(y_val, y_pred))
        except:
            continue
    
    return np.mean(r2_scores)

cv_results = {
    '线性模型': cross_validate_model(X_linear, y),
    '多项式模型': cross_validate_model(X_poly, y),
    '核心变量模型': cross_validate_model(X_core, y),
    '全变量模型': cross_validate_model(X_poly, y)
}

best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x])
best_model = models[best_model_name]

print(f"最佳模型: {best_model_name} (R²={best_model.rsquared:.3f}, CV-R²={cv_results[best_model_name]:.3f})")

# 创建图片保存目录
os.makedirs(os.path.join(script_dir, "plots"), exist_ok=True)

# 1. 相关性热力图
plt.figure(figsize=(10, 8))
top_vars = y_correlations.head(15).index.tolist()
corr_subset = df_analysis[top_vars].corr()
sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('变量相关性热力图')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "correlation_heatmap.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 2. 交叉验证性能对比
plt.figure(figsize=(8, 6))
model_names = list(cv_results.keys())
r2_means = list(cv_results.values())
x_pos = np.arange(len(model_names))
plt.bar(x_pos, r2_means, alpha=0.8, color=['blue', 'red', 'green', 'orange'])
plt.xlabel('模型类型')
plt.ylabel('交叉验证R²')
plt.title('交叉验证性能对比')
plt.xticks(x_pos, model_names, rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "cv_performance.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 3. 预测值vs实际值对比
plt.figure(figsize=(8, 6))
y_pred_linear = model_linear_ols.predict(X_linear)
y_pred_poly = model_poly_ols.predict(X_poly)
y_pred_core = model_core.predict(X_core)

plt.scatter(y_pred_linear, y, alpha=0.6, label='线性模型', s=30)
plt.scatter(y_pred_poly, y, alpha=0.6, label='多项式模型', s=30)
plt.scatter(y_pred_core, y, alpha=0.6, label='核心变量模型', s=30)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='完美预测线')
plt.xlabel('预测值')
plt.ylabel('实际值')
plt.title('预测值 vs 实际值')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "prediction_vs_actual.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 4. 残差分布对比
plt.figure(figsize=(8, 6))
residuals_linear = y - y_pred_linear
residuals_poly = y - y_pred_poly
residuals_core = y - y_pred_core

plt.hist(residuals_linear, bins=30, alpha=0.7, label='线性模型残差', density=True)
plt.hist(residuals_poly, bins=30, alpha=0.7, label='多项式模型残差', density=True)
plt.hist(residuals_core, bins=30, alpha=0.7, label='核心变量模型残差', density=True)
plt.xlabel('残差')
plt.ylabel('密度')
plt.title('残差分布对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "residual_distribution.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 5. 残差vs拟合值
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_linear, residuals_linear, alpha=0.6, label='线性模型', s=30)
plt.scatter(y_pred_poly, residuals_poly, alpha=0.6, label='多项式模型', s=30)
plt.scatter(y_pred_core, residuals_core, alpha=0.6, label='核心变量模型', s=30)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('拟合值')
plt.ylabel('残差')
plt.title('残差 vs 拟合值')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "residual_vs_fitted.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 6. Q-Q图（最佳模型）
plt.figure(figsize=(8, 6))
best_residuals = y - best_model.fittedvalues
stats.probplot(best_residuals, dist="norm", plot=plt)
plt.title(f'最佳模型({best_model_name})残差Q-Q图')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "qq_plot.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 7. 变量重要性
plt.figure(figsize=(8, 6))
coef_values = best_model.params[1:]

if best_model_name == '核心变量模型':
    coef_names = ['孕周', 'BMI', '年龄']
elif best_model_name == '线性模型':
    coef_names = modeling_vars[:len(coef_values)]
elif best_model_name == '多项式模型':
    coef_names = poly_names[1:len(coef_values)+1]
else:
    coef_names = modeling_vars[:min(8, len(coef_values))]

coef_abs = np.abs(coef_values)
sorted_indices = np.argsort(coef_abs)[::-1]
n_show = min(8, len(coef_values))

plt.barh(range(n_show), coef_values[sorted_indices[:n_show]])
plt.yticks(range(n_show), [coef_names[i] for i in sorted_indices[:n_show]])
plt.xlabel('回归系数')
plt.title(f'{best_model_name}变量重要性')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "variable_importance.eps"), dpi=300, bbox_inches='tight')
plt.close()

print("已保存可视化图片到 plots/ 文件夹")

# 模型诊断
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan

residuals = best_model.resid
jb_stat, jb_pvalue = jarque_bera(residuals)
bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, best_model.model.exog)

print(f"模型诊断: JB正态性检验p值={jb_pvalue:.4f}, BP异方差检验p值={bp_pvalue:.4f}")

# 预测示例
example_cases = [
    {"孕周": 12, "BMI": 25, "年龄": 28, "描述": "12周，正常BMI"},
    {"孕周": 16, "BMI": 30, "年龄": 30, "描述": "16周，偏高BMI"},
    {"孕周": 20, "BMI": 35, "年龄": 32, "描述": "20周，高BMI"},
    {"孕周": 24, "BMI": 28, "年龄": 29, "描述": "24周，轻度超重"}
]

print("\n预测示例:")
for case in example_cases:
    sample_idx = np.argmin(np.abs(df_analysis['孕周'] - case["孕周"]) + 
                          np.abs(df_analysis['孕妇BMI'] - case["BMI"]) + 
                          np.abs(df_analysis['年龄'] - case["年龄"]))
    
    pred_linear = model_linear_ols.fittedvalues[sample_idx]
    pred_core = model_core.fittedvalues[sample_idx]
    
    print(f"{case['描述']}: 线性模型{pred_linear*100:.2f}%, 核心变量模型{pred_core*100:.2f}%")

print(f"\n分析总结: 最佳模型{best_model_name}, CV-R²={cv_results[best_model_name]:.3f}")
