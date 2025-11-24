import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

# 新增：统一控制是否保存输出文件（图和CSV）。默认不保存，防止权限/编码问题。
SAVE_OUTPUTS = False
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("蒙特卡洛模拟分析")

# 数据加载
script_dir = os.path.dirname(os.path.abspath(__file__))
boys_csv_path = os.path.join(script_dir, "boys_clean_2_all_with_clusters.csv")
boys_data = pd.read_csv(boys_csv_path)
print(f"数据: {len(boys_data)} 条记录")

# 建立回归预测模型
regression_data = boys_data.dropna(subset=['Y染色体浓度', '孕天', '孕妇BMI', '年龄']).copy()
regression_data['孕周'] = regression_data['孕天'] / 7

X_reg = sm.add_constant(regression_data[['孕周', '孕妇BMI', '年龄']])
y_reg = regression_data['Y染色体浓度']
regression_model = sm.OLS(y_reg, X_reg).fit()

coef = regression_model.params.values
sigma = regression_model.resid.std()

print(f"回归模型: R² = {regression_model.rsquared:.4f}")

# 定义预测Y浓度的函数
def predict_y_concentration_vectorized(weeks, bmis, ages):
    """向量化预测Y染色体浓度"""
    X_sim = np.column_stack([
        np.ones(len(weeks)),  # 常数项
        weeks,                # 孕周
        bmis,                 # BMI
        ages                  # 年龄
    ])
    mu = X_sim @ coef
    # 添加随机误差
    y_pred = mu + np.random.normal(0, sigma, size=len(weeks))
    # 确保Y浓度在合理范围内 [0, 1]
    y_pred = np.clip(y_pred, 0, 1)
    return y_pred

# 数据准备
cluster_vars = ['年龄', '体重', '身高']
X_cluster = boys_data[cluster_vars].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# 检查聚类结果
if '聚类组' in boys_data.columns:
    best_n_clusters = boys_data['聚类组'].nunique()
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    print(f"聚类组数: {best_n_clusters}")
else:
    print("未发现聚类结果，请先运行 clustering.py")
    exit(1)

# 创建图片保存目录
os.makedirs(os.path.join(script_dir, "plots"), exist_ok=True)

# 可视化聚类结果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(boys_data['身高'], boys_data['体重'], 
                     c=boys_data['聚类组'], cmap='viridis', 
                     s=50, alpha=0.8)
plt.colorbar(scatter, label='聚类组')
plt.xlabel('身高 (cm)')
plt.ylabel('体重 (kg)')
plt.title('基于身高和体重的聚类结果')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "clustering_height_weight.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 蒙特卡洛模拟
y_threshold = 0.04
error_levels = {
    '低': {'孕天': 0.01, '身高': 0.01, '体重': 0.01, '年龄': 0.01, 'Y染色体浓度': 0.01},
    '中': {'孕天': 0.03, '身高': 0.03, '体重': 0.03, '年龄': 0.03, 'Y染色体浓度': 0.03},
    '高': {'孕天': 0.05, '身高': 0.05, '体重': 0.05, '年龄': 0.05, 'Y染色体浓度': 0.05}
}

min_week, max_week = 10, 24
weeks = np.arange(min_week, max_week + 1)
days = weeks * 7

def calculate_risk(p_detect, week):
    fail_risk = (1 - p_detect) * 100
    delay_risk = (week - min_week) / (max_week - min_week) * 50
    return fail_risk + delay_risk

n_simulations = 1000
results = {}

for cluster in range(best_n_clusters):
    cluster_data = boys_data[boys_data['聚类组'] == cluster]
    results[cluster] = {}
    print(f"模拟聚类组 {cluster} (样本数: {len(cluster_data)})")
    
    for error_level, errors in error_levels.items():
        results[cluster][error_level] = {
            'p_detect': np.zeros(len(days)),
            'risk': np.zeros(len(days))
        }
        
        for i, day in enumerate(days):
            p_detect_samples = []
            risk_samples = []
            
            for _ in range(n_simulations):
                sim_data = cluster_data.copy()
                sim_data['模拟孕天'] = day
                sim_data['模拟孕周'] = day / 7
                
                # 添加误差
                for var, error in errors.items():
                    if var == '孕天':
                        error_factor = np.random.normal(0, error, size=len(sim_data))
                        sim_data['模拟孕天'] *= (1 + error_factor)
                        sim_data['模拟孕周'] = sim_data['模拟孕天'] / 7
                    elif var == '身高':
                        error_factor = np.random.normal(0, error, size=len(sim_data))
                        sim_data['模拟身高'] = sim_data['身高'] * (1 + error_factor)
                    elif var == '体重':
                        error_factor = np.random.normal(0, error, size=len(sim_data))
                        sim_data['模拟体重'] = sim_data['体重'] * (1 + error_factor)
                    elif var == '年龄':
                        error_factor = np.random.normal(0, error, size=len(sim_data))
                        sim_data['模拟年龄'] = sim_data['年龄'] * (1 + error_factor)
                
                # 计算BMI
                if '模拟身高' in sim_data.columns and '模拟体重' in sim_data.columns:
                    sim_data['模拟BMI'] = sim_data['模拟体重'] / (sim_data['模拟身高'] / 100) ** 2
                else:
                    sim_data['模拟BMI'] = sim_data['孕妇BMI']
                
                # 预测Y浓度
                sim_data['模拟Y浓度'] = predict_y_concentration_vectorized(
                    weeks=sim_data['模拟孕周'].values,
                    bmis=sim_data['模拟BMI'].values,
                    ages=sim_data['模拟年龄'].values if '模拟年龄' in sim_data.columns else sim_data['年龄'].values
                )
                
                p_detect = (sim_data['模拟Y浓度'] >= y_threshold).mean()
                p_detect_samples.append(p_detect)
                risk = calculate_risk(p_detect, day/7)
                risk_samples.append(risk)
            
            results[cluster][error_level]['p_detect'][i] = np.mean(p_detect_samples)
            results[cluster][error_level]['risk'][i] = np.mean(risk_samples)

# 结果分析与推荐时点
recommendations = pd.DataFrame(columns=['聚类组', '误差水平', '推荐孕周', '达标率', '风险值'])

for cluster in range(best_n_clusters):
    for error_level in error_levels.keys():
        risk_values = results[cluster][error_level]['risk']
        best_idx = np.argmin(risk_values)
        best_week = days[best_idx] / 7
        p_detect = results[cluster][error_level]['p_detect'][best_idx]
        risk = risk_values[best_idx]
        
        recommendations = pd.concat([recommendations, pd.DataFrame([{
            '聚类组': cluster,
            '误差水平': error_level,
            '推荐孕周': best_week,
            '达标率': p_detect,
            '风险值': risk
        }])], ignore_index=True)
        
        print(f"聚类组 {cluster}, 误差水平 {error_level}: 推荐孕周 = {best_week:.1f}w, 达标率 = {p_detect:.2f}")

# 保存推荐结果
recommendations.to_csv(os.path.join(script_dir, '蒙特卡洛推荐时点.csv'), index=False, encoding='utf-8-sig')

# 可视化结果
# 1. 误差率 vs 推荐孕周曲线
plt.figure(figsize=(8, 6))
error_values = [0.01, 0.03, 0.05]
markers = ['o', 's', '^']

for i, cluster in enumerate(range(best_n_clusters)):
    recommended_weeks = []
    for error_level in error_levels.keys():
        cluster_data = recommendations[(recommendations['聚类组'] == cluster) & 
                                      (recommendations['误差水平'] == error_level)]
        recommended_weeks.append(cluster_data['推荐孕周'].values[0])
    
    plt.plot(error_values, recommended_weeks, marker=markers[i % len(markers)], 
             linewidth=2, markersize=8, label=f'聚类组 {cluster}')

plt.xlabel('误差率')
plt.ylabel('推荐孕周')
plt.title('误差率 vs 推荐孕周关系')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "error_vs_week.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 2. 孕周 vs 达标率曲线
plt.figure(figsize=(8, 6))
for cluster in range(best_n_clusters):
    p_detect = results[cluster]['中']['p_detect']
    plt.plot(weeks, p_detect, linewidth=2, label=f'聚类组 {cluster}')

plt.axhline(y=0.95, color='r', linestyle='--', label='95%达标率')
plt.xlabel('孕周')
plt.ylabel('达标率')
plt.title('孕周 vs 达标率关系 (中等误差水平)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "week_vs_detection.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 3. 3D可视化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

cluster_centers = kmeans.cluster_centers_
cluster_centers_original = scaler.inverse_transform(cluster_centers)

for i in range(best_n_clusters):
    recommended_week = recommendations[(recommendations['聚类组'] == i) & 
                                      (recommendations['误差水平'] == '中')]['推荐孕周'].values[0]
    mean_age = cluster_centers_original[i, 2]
    
    ax.scatter(cluster_centers_original[i, 0], cluster_centers_original[i, 1], recommended_week,
               s=200, c=f'C{i}', marker='*', label=f'聚类组 {i}')
    
    ax.text(cluster_centers_original[i, 0], cluster_centers_original[i, 1], recommended_week,
            f'组{i}: {recommended_week:.1f}w', fontsize=10)

ax.set_xlabel('身高 (cm)')
ax.set_ylabel('体重 (kg)')
ax.set_zlabel('推荐孕周')
ax.set_title('身高、体重与推荐孕周的3D关系')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "3d_recommendation.eps"), dpi=300, bbox_inches='tight')
plt.close()

print("已保存可视化图片到 plots/ 文件夹")