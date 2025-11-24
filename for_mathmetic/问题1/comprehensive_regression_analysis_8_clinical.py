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
# 修复R²符号显示问题
plt.rcParams['mathtext.default'] = 'regular'

print("Y染色体浓度8个临床指标多元线性回归分析")
print("=" * 50)

# 数据读取与预处理
script_dir = os.path.dirname(os.path.abspath(__file__))
boys_path = os.path.join(script_dir, "clean_data_csv", "boys_clean_8_clinical.csv")

if not os.path.exists(boys_path):
    boys_path = os.path.join(script_dir, "clean_data_csv", "boys_clean_1.csv")

df_boys = pd.read_csv(boys_path)
df_male = df_boys.copy()

# 基础指标
df_male["孕周"] = df_male["孕天"] / 7.0
df_male["BMI"] = df_male["孕妇BMI"]
df_male["Y浓度"] = df_male["Y染色体浓度"]
df_male["年龄"] = df_male["年龄"]
df_male["身高"] = df_male["身高"]
df_male["体重"] = df_male["体重"]

# 生育史指标
df_male["怀孕次数"] = pd.to_numeric(df_male["怀孕次数"], errors='coerce')
df_male["生产次数"] = pd.to_numeric(df_male["生产次数"], errors='coerce')
df_male["IVF妊娠"] = (df_male["IVF妊娠"] == 1).astype(int) if 'IVF妊娠' in df_male.columns else 0

# 定义8个临床指标
clinical_indicators = ['孕周', 'BMI', '年龄', '身高', '体重', '怀孕次数', '生产次数', 'IVF妊娠']

# 清理数据
df_male = df_male[clinical_indicators + ["Y浓度"]].dropna()

print(f"样本: {len(df_male)}条, 指标: {len(clinical_indicators)}个")

# 相关性分析
corr_matrix = df_male[clinical_indicators + ["Y浓度"]].corr()
y_correlations = corr_matrix["Y浓度"].sort_values(ascending=False)
corr_str = ', '.join([f'{var}({corr:.3f})' for var, corr in y_correlations.head(5).items() if var != 'Y浓度'])
print(f"Y浓度相关性: {corr_str}")

# 生成可视化图表

# 创建子图
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Y染色体浓度与8个临床指标的关系分析', fontsize=16)

# 绘制8个临床指标与Y浓度的散点图
for i, indicator in enumerate(clinical_indicators):
    row = i // 3
    col = i % 3
    
    axes[row, col].scatter(df_male[indicator], df_male["Y浓度"], alpha=0.6, color=plt.cm.tab10(i))
    axes[row, col].set_xlabel(f"{indicator}")
    axes[row, col].set_ylabel("Y染色体浓度")
    axes[row, col].set_title(f"Y浓度 vs {indicator}")
    
    # 添加拟合线（添加错误处理）
    try:
        # 检查数据是否有足够的变异性
        if df_male[indicator].std() > 0 and df_male["Y浓度"].std() > 0:
            z = np.polyfit(df_male[indicator], df_male["Y浓度"], 1)
            p = np.poly1d(z)
            axes[row, col].plot(df_male[indicator], p(df_male[indicator]), "r--", alpha=0.8)
    except (np.linalg.LinAlgError, ValueError):
        # 如果拟合失败，跳过拟合线
        pass
    
    # 添加相关系数
    corr = df_male[indicator].corr(df_male["Y浓度"])
    axes[row, col].text(0.05, 0.95, f'r={corr:.3f}', transform=axes[row, col].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 相关性热力图
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, ax=axes[2, 2], cbar_kws={'shrink': 0.8})
axes[2, 2].set_title("8个临床指标相关性热力图")

plt.tight_layout()
plt.show()

# 一元线性回归分析
y = df_male["Y浓度"]
univariate_models = {}
univariate_results = []

for indicator in clinical_indicators:
    X = sm.add_constant(df_male[indicator])
    model = sm.OLS(y, X).fit()
    univariate_models[indicator] = model
    univariate_results.append({
        '指标': indicator,
        'R²': model.rsquared,
        '调整R²': model.rsquared_adj,
        '系数': model.params[indicator],
        't值': model.tvalues[indicator],
        'p值': model.pvalues[indicator],
        'AIC': model.aic,
        'BIC': model.bic
    })

# 显示前5个最重要的指标
univariate_df = pd.DataFrame(univariate_results)
top_indicators = univariate_df.nlargest(5, 'R²')
top5_str = ', '.join([f'{row["指标"]}(R²={row["R²"]:.3f})' for _, row in top_indicators.iterrows()])
print(f"一元回归前5: {top5_str}")

# 多元线性回归分析
# 1. 三维基础模型
X_basic = df_male[["孕周", "BMI", "年龄"]]
X_basic = sm.add_constant(X_basic)
model_basic = sm.OLS(y, X_basic).fit()

# 2. 8个临床指标完整模型
X_full = df_male[clinical_indicators]
X_full = sm.add_constant(X_full)
model_full = sm.OLS(y, X_full).fit()

# 3. 逐步回归模型
def forward_selection(X, y, significance_level=0.05):
    """前向选择逐步回归"""
    included = []
    excluded = list(X.columns)
    
    while True:
        changed = False
        best_pvalue = 1
        best_feature = None
        
        for feature in excluded:
            if feature == 'const':
                continue
            temp_included = included + [feature]
            X_temp = X[temp_included]
            model = sm.OLS(y, X_temp).fit()
            pvalue = model.pvalues[feature]
            
            if pvalue < best_pvalue and pvalue < significance_level:
                best_pvalue = pvalue
                best_feature = feature
        
        if best_feature:
            included.append(best_feature)
            excluded.remove(best_feature)
            changed = True
        
        if not changed:
            break
    
    return included

# 执行逐步回归
selected_features = forward_selection(X_full, y)

if selected_features:
    X_stepwise = X_full[['const'] + selected_features]
    model_stepwise = sm.OLS(y, X_stepwise).fit()
else:
    model_stepwise = model_basic
    selected_features = ["孕周", "BMI", "年龄"]

print(f"模型性能: 基础{model_basic.rsquared:.3f}, 完整{model_full.rsquared:.3f}, 逐步{model_stepwise.rsquared:.3f}")

# 模型比较
models_info = {
    '三维基础模型': {'model': model_basic, 'r2': model_basic.rsquared, 'adj_r2': model_basic.rsquared_adj, 'aic': model_basic.aic, 'bic': model_basic.bic, 'variables': 3},
    '8指标完整模型': {'model': model_full, 'r2': model_full.rsquared, 'adj_r2': model_full.rsquared_adj, 'aic': model_full.aic, 'bic': model_full.bic, 'variables': 8},
    '逐步回归模型': {'model': model_stepwise, 'r2': model_stepwise.rsquared, 'adj_r2': model_stepwise.rsquared_adj, 'aic': model_stepwise.aic, 'bic': model_stepwise.bic, 'variables': len(selected_features)}
}

# 选择最佳模型
best_model_name = max(models_info.keys(), key=lambda x: models_info[x]['adj_r2'])
best_model = models_info[best_model_name]['model']
print(f"最佳模型: {best_model_name} (调整R²: {best_model.rsquared_adj:.3f})")

# 最佳模型详细分析
confidence_intervals = best_model.conf_int()
significant_vars = []

for i, coef_name in enumerate(best_model.params.index):
    if coef_name == 'const':
        continue
    p_value = best_model.pvalues[coef_name]
    if p_value < 0.05:
        significant_vars.append(coef_name)

sig_vars_str = ', '.join(significant_vars)
print(f"显著变量: {sig_vars_str} ({len(significant_vars)}/{len(best_model.params)-1})")

# 模型诊断
residuals = best_model.resid
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan

jb_stat, jb_pvalue = jarque_bera(residuals)
bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, best_model.model.exog)

print(f"模型诊断: 正态性p={jb_pvalue:.3f}, 异方差p={bp_pvalue:.3f}")

# 生成可视化图表

# 创建综合可视化
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('8个临床指标多元线性回归分析结果', fontsize=16, fontweight='bold')

# 1. 模型性能对比
model_names = list(models_info.keys())
r2_scores = [models_info[name]['r2'] for name in model_names]
adj_r2_scores = [models_info[name]['adj_r2'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

axes[0,0].bar(x - width/2, r2_scores, width, label='R$^2$', alpha=0.8)
axes[0,0].bar(x + width/2, adj_r2_scores, width, label='调整R$^2$', alpha=0.8)
axes[0,0].set_xlabel('模型')
axes[0,0].set_ylabel('R$^2$值')
axes[0,0].set_title('模型性能对比')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(model_names, rotation=45)
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. 变量重要性（基于系数绝对值）
coef_values = best_model.params[1:]  # 排除截距
coef_abs = np.abs(coef_values)
sorted_indices = np.argsort(coef_abs)[::-1]

axes[0,1].barh(range(len(coef_values)), coef_values.iloc[sorted_indices])
axes[0,1].set_yticks(range(len(coef_values)))
axes[0,1].set_yticklabels([coef_values.index[i] for i in sorted_indices])
axes[0,1].set_xlabel('回归系数')
axes[0,1].set_title(f'{best_model_name}变量重要性')
axes[0,1].grid(True, alpha=0.3)

# 3. 残差分析（带置信区间）
fitted_values = best_model.fittedvalues
axes[0,2].scatter(fitted_values, residuals, alpha=0.6)
axes[0,2].axhline(y=0, color='red', linestyle='--', linewidth=2, label='零线')

# 计算残差的置信区间
residual_std = residuals.std()
residual_mean = residuals.mean()
confidence_level = 0.95
z_score = 1.96  # 95%置信区间的z值

# 添加置信区间带
axes[0,2].axhline(y=residual_mean + z_score * residual_std, color='orange', linestyle=':', alpha=0.7, label=f'95%置信区间上界')
axes[0,2].axhline(y=residual_mean - z_score * residual_std, color='orange', linestyle=':', alpha=0.7, label=f'95%置信区间下界')

# 填充置信区间
axes[0,2].fill_between(fitted_values, 
                      residual_mean - z_score * residual_std, 
                      residual_mean + z_score * residual_std, 
                      alpha=0.1, color='orange', label='95%置信区间')

axes[0,2].set_xlabel('拟合值')
axes[0,2].set_ylabel('残差')
axes[0,2].set_title('残差 vs 拟合值（带置信区间）')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# 4. Q-Q图
stats.probplot(residuals, dist="norm", plot=axes[1,0])
axes[1,0].set_title('残差正态性检验 (Q-Q图)')
axes[1,0].grid(True, alpha=0.3)

# 5. 残差标准差分析
fitted_bins = pd.cut(fitted_values, bins=10)
residual_std_by_fitted = residuals.groupby(fitted_bins, observed=False).std()
bin_centers = [interval.mid for interval in residual_std_by_fitted.index]

axes[1,1].plot(bin_centers, residual_std_by_fitted.values, 'o-', linewidth=2, markersize=6, color='blue')
axes[1,1].axhline(y=residuals.std(), color='red', linestyle='--', linewidth=2, label=f'总体标准差: {residuals.std():.4f}')
axes[1,1].set_xlabel('拟合值区间中心')
axes[1,1].set_ylabel('残差标准差')
axes[1,1].set_title('残差标准差随拟合值变化')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

# 6. 预测 vs 实际
axes[1,2].scatter(y, fitted_values, alpha=0.6)
axes[1,2].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', alpha=0.8)
axes[1,2].set_xlabel('实际Y浓度')
axes[1,2].set_ylabel('预测Y浓度')
axes[1,2].set_title(f'预测效果 (R$^2$={best_model.rsquared:.4f})')
axes[1,2].grid(True, alpha=0.3)

# 7. 模型复杂度 vs 性能
complexity = [models_info[name]['variables'] for name in model_names]
axes[2,0].scatter(complexity, adj_r2_scores, s=100, alpha=0.7)
for i, name in enumerate(model_names):
    axes[2,0].annotate(name, (complexity[i], adj_r2_scores[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[2,0].set_xlabel('模型复杂度（变量数）')
axes[2,0].set_ylabel('调整R$^2$')
axes[2,0].set_title('复杂度 vs 性能')
axes[2,0].grid(True, alpha=0.3)

# 8. AIC/BIC对比
axes[2,1].bar(x - width/2, [models_info[name]['aic'] for name in model_names], width, label='AIC', alpha=0.8)
axes[2,1].bar(x + width/2, [models_info[name]['bic'] for name in model_names], width, label='BIC', alpha=0.8)
axes[2,1].set_xlabel('模型')
axes[2,1].set_ylabel('信息准则值')
axes[2,1].set_title('AIC/BIC对比')
axes[2,1].set_xticks(x)
axes[2,1].set_xticklabels(model_names, rotation=45)
axes[2,1].legend()
axes[2,1].grid(True, alpha=0.3)

# 9. 交叉验证（使用sklearn）
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# 准备交叉验证数据
X_cv = df_male[selected_features] if selected_features else df_male[["孕周", "BMI", "年龄"]]
y_cv = y

# 交叉验证
lr = LinearRegression()
cv_scores = cross_val_score(lr, X_cv, y_cv, cv=5, scoring='r2')

axes[2,2].bar(range(1, 6), cv_scores, alpha=0.7)
axes[2,2].axhline(y=cv_scores.mean(), color='red', linestyle='--', 
                  label=f'平均: {cv_scores.mean():.4f}')
axes[2,2].set_xlabel('交叉验证折数')
axes[2,2].set_ylabel('R$^2$')
axes[2,2].set_title('5折交叉验证结果')
axes[2,2].legend()
axes[2,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 最终结论
print(f"\n分析结论:")
print(f"  最佳模型: {best_model_name} (R²={best_model.rsquared:.3f}, 调整R²={best_model.rsquared_adj:.3f})")
print(f"  显著变量: {sig_vars_str} ({len(significant_vars)}/{len(best_model.params)-1})")
print(f"  模型诊断: 正态性p={jb_pvalue:.3f}, 异方差p={bp_pvalue:.3f}")

if best_model_name == '三维基础模型':
    print(f"  结论: 三维基础模型已足够，孕周、BMI、年龄是核心预测因子")
else:
    print(f"  结论: 扩展临床指标改善了模型性能，建议使用{best_model_name}")
