import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
# 新增：用于构造基于脚本目录的绝对路径
import os
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("K-means聚类BMI分组分析")

# 数据加载
script_dir = os.path.dirname(os.path.abspath(__file__))
boys_csv_path = os.path.join(script_dir, "clean_data_csv", "boys_clean_2.csv")
boys_data = pd.read_csv(boys_csv_path)
print(f"数据: {len(boys_data)} 条记录")

# K-means聚类分组

def kmeans_bmi_grouping(df):
    """使用K-means对BMI进行聚类"""
    
    # 准备聚类数据
    clustering_data = df[['孕妇BMI']].copy()
    
    # 标准化数据
    scaler = StandardScaler()
    bmi_scaled = scaler.fit_transform(clustering_data)
    
    # 确定最优聚类数
    k_range = range(2, 8)
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(bmi_scaled)
        silhouette_scores.append(silhouette_score(bmi_scaled, cluster_labels))
    
    # 找到最优K值
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"最优聚类数: {best_k} (轮廓系数: {max(silhouette_scores):.3f})")
    
    # 使用最优K值进行聚类
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_copy = df.copy()
    df_copy['BMI_聚类分组'] = kmeans_final.fit_predict(bmi_scaled)
    
    # 分析各组特征
    group_analysis = {}
    
    for group in sorted(df_copy['BMI_聚类分组'].unique()):
        group_data = df_copy[df_copy['BMI_聚类分组'] == group]
        
        analysis = {
            'count': len(group_data),
            'bmi_range': (group_data['孕妇BMI'].min(), group_data['孕妇BMI'].max()),
            'bmi_mean': group_data['孕妇BMI'].mean(),
            'weeks_mean': group_data['孕天'].mean() / 7,
            'optimal_week': group_data['孕天'].median() / 7
        }
        
        group_analysis[group] = analysis
        print(f"群组{group}: BMI {analysis['bmi_range'][0]:.1f}-{analysis['bmi_range'][1]:.1f}, 推荐{analysis['optimal_week']:.1f}周, n={analysis['count']}")
    
    return df_copy, group_analysis, best_k, max(silhouette_scores)

boys_kmeans, kmeans_analysis, optimal_k, silhouette_score_value = kmeans_bmi_grouping(boys_data)

# 风险评估
def calculate_risk_metrics(df, group_analysis):
    """计算各组风险指标"""
    risk_analysis = {}
    
    for group in group_analysis.keys():
        group_data = df[df['BMI_聚类分组'] == group]
        if len(group_data) == 0:
            continue
            
        # 简化的风险计算
        timing_risk = group_data['孕天'].std() / 7
        y_cv = group_data['Y染色体浓度'].std() / group_data['Y染色体浓度'].mean()
        composite_risk = timing_risk * 20 + y_cv * 25
        
        risk_analysis[group] = {'composite_risk': composite_risk}
        print(f"群组{group}: 风险 {composite_risk:.1f}")
    
    return risk_analysis

kmeans_risks = calculate_risk_metrics(boys_kmeans, kmeans_analysis)

# 误差分析

def find_optimal_weeks_with_errors(df, group_analysis, error_rates=[0.0, 0.01, 0.03, 0.05]):
    """对每组在不同误差水平下求解最佳检测孕周"""
    optimal_results = {}
    
    for group in group_analysis.keys():
        group_data = df[df['BMI_聚类分组'] == group]
        group_results = {}
        
        for error_rate in error_rates:
            # 模拟带误差的Y浓度数据
            np.random.seed(42)
            if error_rate > 0:
                y_with_error = group_data['Y染色体浓度'] * (1 + np.random.normal(0, error_rate, len(group_data)))
                y_with_error = np.maximum(y_with_error, 0)
            else:
                y_with_error = group_data['Y染色体浓度'].copy()
            
            # 使用合理的孕周范围
            week_range = np.arange(11, 25, 0.5)
            week_risks = []
            
            for week in week_range:
                week_window = group_data[(group_data['孕天'] >= (week-1)*7) & 
                                       (group_data['孕天'] <= (week+1)*7)]
                
                if len(week_window) < 3:
                    week_risks.append(float('inf'))
                    continue
                
                y_values = y_with_error[week_window.index]
                timing_risk = week_window['孕天'].std() / 7
                y_cv = y_values.std() / y_values.mean() if y_values.mean() > 0 else 0
                timing_penalty = abs(week - 13) * 3
                
                composite_risk = y_cv * 50 + timing_penalty + timing_risk * 10
                
                if np.isnan(composite_risk) or np.isinf(composite_risk):
                    week_risks.append(float('inf'))
                else:
                    week_risks.append(composite_risk)
            
            # 找到风险最小的孕周
            valid_risks = [r for r in week_risks if r != float('inf')]
            if len(valid_risks) == 0:
                optimal_week = 13.0
                min_risk = 100
            else:
                min_risk_idx = np.argmin(week_risks)
                optimal_week = week_range[min_risk_idx]
                min_risk = week_risks[min_risk_idx]
            
            group_results[error_rate] = {
                'optimal_week': optimal_week,
                'min_risk': min_risk,
                'week_risks': week_risks
            }
        
        optimal_results[group] = group_results
    
    return optimal_results

optimal_weeks_results = find_optimal_weeks_with_errors(boys_kmeans, kmeans_analysis)

# 简化的结果输出
print(f"\n误差分析结果:")
for group in sorted(optimal_weeks_results.keys()):
    group_data = optimal_weeks_results[group]
    no_error_week = group_data[0.0]['optimal_week']
    error_1_week = group_data[0.01]['optimal_week']
    error_3_week = group_data[0.03]['optimal_week']
    error_5_week = group_data[0.05]['optimal_week']
    print(f"群组{group}: 无误差{no_error_week:.1f}周, 1%{error_1_week:.1f}周, 3%{error_3_week:.1f}周, 5%{error_5_week:.1f}周")

# 创建图片保存目录
os.makedirs(os.path.join(script_dir, "plots"), exist_ok=True)

# 1. 聚类结果可视化
plt.figure(figsize=(8, 6))
for i, (group, analysis) in enumerate(kmeans_analysis.items()):
    plt.scatter([analysis['bmi_mean']], [analysis['weeks_mean']], 
               s=analysis['count']*3, alpha=0.7, label=f'群组{group}({analysis["count"]})', 
               color=plt.cm.Set2(i))

plt.xlabel('平均BMI')
plt.ylabel('平均达标孕周')
plt.title('K-means聚类结果')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "kmeans_clustering.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 2. 风险评分
plt.figure(figsize=(8, 6))
groups = list(kmeans_analysis.keys())
risk_scores = [kmeans_risks[g]['composite_risk'] for g in groups]
colors = ['green' if r < 50 else 'orange' if r < 80 else 'red' for r in risk_scores]

bars = plt.bar([f'群组{g}' for g in groups], risk_scores, color=colors, alpha=0.7)
plt.ylabel('综合风险评分')
plt.title('各群组风险评估')
plt.grid(True, alpha=0.3)

for bar, risk in zip(bars, risk_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{risk:.1f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "risk_assessment.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 3. 误差影响分析
plt.figure(figsize=(8, 6))
for group in groups:
    risk_changes = []
    for error_rate in [0.01, 0.03, 0.05]:
        no_error_risk = optimal_weeks_results[group][0.0]['min_risk']
        error_risk = optimal_weeks_results[group][error_rate]['min_risk']
        risk_change = ((error_risk - no_error_risk) / no_error_risk * 100) if no_error_risk > 0 else 0
        risk_changes.append(risk_change)
    
    plt.plot([1, 3, 5], risk_changes, 'o-', linewidth=2, markersize=6, 
            label=f'群组{group}')

plt.xlabel('检测误差率 (%)')
plt.ylabel('风险值变化 (%)')
plt.title('误差对风险值的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "error_impact.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 4. BMI分布散点图
plt.figure(figsize=(8, 6))
bmi_values = boys_kmeans['孕妇BMI'].values
weeks_values = boys_kmeans['孕天'].values / 7
cluster_labels = boys_kmeans['BMI_聚类分组'].values

for i, group in enumerate(sorted(np.unique(cluster_labels))):
    mask = (cluster_labels == group)
    plt.scatter(bmi_values[mask], weeks_values[mask], 
              alpha=0.6, label=f'群组{group}', color=plt.cm.Set2(i), s=30)

plt.xlabel('BMI')
plt.ylabel('达标孕周')
plt.title('BMI-孕周分布')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "bmi_distribution.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 最终推荐
print(f"\n推荐策略:")
print(f"聚类数: {optimal_k}, 轮廓系数: {silhouette_score_value:.3f}")

for group, analysis in kmeans_analysis.items():
    risk_level = "低" if kmeans_risks[group]['composite_risk'] < 50 else "中" if kmeans_risks[group]['composite_risk'] < 80 else "高"
    print(f"群组{group}({risk_level}): BMI {analysis['bmi_range'][0]:.1f}-{analysis['bmi_range'][1]:.1f}, 推荐{analysis['optimal_week']:.1f}周, n={analysis['count']}")

# 保存结果
boys_kmeans.to_csv('boys_kmeans_grouping_simple.csv', index=False, encoding='utf-8-sig')
print(f"\n已保存K-means聚类结果到 boys_kmeans_grouping_simple.csv")
print(f"已保存可视化图片到 plots/ 文件夹")
