import pandas as pd
import statsmodels.api as sm
import numpy as np
import os
import json

# 设置文件路径
script_dir = os.path.dirname(os.path.abspath(__file__))
boys_path = os.path.join(script_dir, "clean_data_csv", "boys_clean_1.csv")

# 读取数据
df_boys = pd.read_csv(boys_path)

# 处理数据
df_male = df_boys.copy()
df_male["孕周"] = df_male["孕天"] / 7.0
df_male["BMI"] = df_male["孕妇BMI"]
df_male["Y浓度"] = df_male["Y染色体浓度"]
df_male["年龄"] = df_male["年龄"]
df_male = df_male.dropna(subset=["孕周","BMI","Y浓度","年龄"])

# 构建多元线性回归模型
y = df_male["Y浓度"]
X_multi = df_male[["孕周","BMI","年龄"]]
X_multi = sm.add_constant(X_multi)
model_multi = sm.OLS(y, X_multi).fit()

# 提取模型参数
regression_params = {
    'coefficients': {
        'const': float(model_multi.params['const']),
        '孕周': float(model_multi.params['孕周']),
        'BMI': float(model_multi.params['BMI']),
        '年龄': float(model_multi.params['年龄'])
    },
    'r_squared': float(model_multi.rsquared),
    'adj_r_squared': float(model_multi.rsquared_adj),
    'mse': float(np.mean(model_multi.resid**2)),
    'residual_std': float(np.std(model_multi.resid)),
    'f_value': float(model_multi.fvalue),
    'f_pvalue': float(model_multi.f_pvalue),
    'n_obs': int(model_multi.nobs),
    'df_model': int(model_multi.df_model),
    'df_resid': int(model_multi.df_resid)
}

# 计算预测标准误
X_matrix = X_multi.values
XTX_inv = np.linalg.inv(X_matrix.T @ X_matrix)
hat_matrix_diag = np.diag(X_matrix @ XTX_inv @ X_matrix.T)
prediction_se_base = regression_params['residual_std'] * np.sqrt(1 + hat_matrix_diag)

# 保存参数到JSON文件
output_path = os.path.join(script_dir, "..", "问题3", "regression_params.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(regression_params, f, ensure_ascii=False, indent=2)

print("多元线性回归模型参数:")
print(f"回归方程: Y浓度 = {regression_params['coefficients']['const']:.6f} + {regression_params['coefficients']['孕周']:.6f}×孕周 + {regression_params['coefficients']['BMI']:.6f}×BMI + {regression_params['coefficients']['年龄']:.6f}×年龄")
print(f"R² = {regression_params['r_squared']:.4f}")
print(f"调整R² = {regression_params['adj_r_squared']:.4f}")
print(f"均方误差 = {regression_params['mse']:.6f}")
print(f"残差标准差 = {regression_params['residual_std']:.6f}")
print(f"F统计量 = {regression_params['f_value']:.2f} (p = {regression_params['f_pvalue']:.6f})")
print(f"样本数 = {regression_params['n_obs']}")
print(f"\n参数已保存到: {output_path}")

