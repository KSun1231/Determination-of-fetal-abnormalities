import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("K-means聚类 vs 经验分组效果对比分析")

# 数据加载
script_dir = os.path.dirname(os.path.abspath(__file__))
boys_csv_path = os.path.join(script_dir, "clean_data_csv", "boys_clean_2.csv")
boys_data = pd.read_csv(boys_csv_path)
print(f"数据: {len(boys_data)} 条记录")

# K-means聚类分析

def kmeans_analysis(df):
    X = df[['孕妇BMI']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    k_range = range(2, 8)
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
    
    best_k = k_range[np.argmax(silhouette_scores)]
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_scaled)
    
    df_kmeans = df.copy()
    df_kmeans['聚类分组'] = cluster_labels
    
    kmeans_groups = {}
    for group in range(best_k):
        group_data = df_kmeans[df_kmeans['聚类分组'] == group]
        kmeans_groups[group] = {
            'count': len(group_data),
            'bmi_mean': group_data['孕妇BMI'].mean(),
            'bmi_std': group_data['孕妇BMI'].std(),
            'bmi_range': (group_data['孕妇BMI'].min(), group_data['孕妇BMI'].max()),
            'weeks_mean': group_data['孕天'].mean() / 7,
            'y_concentration_mean': group_data['Y染色体浓度'].mean() * 100
        }
    
    return df_kmeans, kmeans_groups, best_k, max(silhouette_scores)

df_kmeans, kmeans_groups, optimal_k, best_silhouette = kmeans_analysis(boys_data)

print(f"K-means: {optimal_k}组, 轮廓系数={best_silhouette:.3f}")

# 经验分组分析

def fixed_bmi_grouping(df):
    bins = [0, 28, 32, 36, 40, 50]
    labels = ['<28', '28-32', '32-36', '36-40', '≥40']
    
    df_fixed = df.copy()
    df_fixed['经验分组'] = pd.cut(df_fixed['孕妇BMI'], bins=bins, labels=labels, right=False)
    
    fixed_groups = {}
    for group in labels:
        group_data = df_fixed[df_fixed['经验分组'] == group]
        if len(group_data) > 0:
            fixed_groups[group] = {
                'count': len(group_data),
                'bmi_mean': group_data['孕妇BMI'].mean(),
                'bmi_std': group_data['孕妇BMI'].std(),
                'bmi_range': (group_data['孕妇BMI'].min(), group_data['孕妇BMI'].max()),
                'weeks_mean': group_data['孕天'].mean() / 7,
                'y_concentration_mean': group_data['Y染色体浓度'].mean() * 100
            }
    
    return df_fixed, fixed_groups

df_fixed, fixed_groups = fixed_bmi_grouping(boys_data)

print(f"经验分组: {len(fixed_groups)}组")

# 聚类质量评估

def evaluate_clustering_quality(df, group_col, method_name):
    X = df[['孕妇BMI']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if group_col == '聚类分组':
        labels = df[group_col].values
    else:
        unique_groups = df[group_col].dropna().unique()
        group_to_num = {group: i for i, group in enumerate(unique_groups)}
        labels = df[group_col].map(group_to_num).values
        
        valid_mask = ~pd.isna(labels)
        labels = labels[valid_mask].astype(int)
        X_scaled = X_scaled[valid_mask]
    
    silhouette = silhouette_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    
    n_groups = len(np.unique(labels))
    group_means = []
    group_sizes = []
    
    for group in np.unique(labels):
        group_data = X_scaled[labels == group]
        group_means.append(group_data.mean(axis=0))
        group_sizes.append(len(group_data))
    
    group_means = np.array(group_means)
    overall_mean = X_scaled.mean(axis=0)
    
    wcss = 0
    for group in np.unique(labels):
        group_data = X_scaled[labels == group]
        wcss += np.sum((group_data - group_means[group])**2)
    
    bcss = 0
    for i, group in enumerate(np.unique(labels)):
        bcss += group_sizes[i] * np.sum((group_means[i] - overall_mean)**2)
    
    variance_ratio = bcss / (bcss + wcss) if (bcss + wcss) > 0 else 0
    
    print(f"{method_name}: 轮廓系数={silhouette:.3f}, 分组数={n_groups}")
    
    return {
        'silhouette': silhouette,
        'calinski': calinski,
        'davies_bouldin': davies_bouldin,
        'variance_ratio': variance_ratio,
        'n_groups': n_groups,
        'wcss': wcss,
        'bcss': bcss
    }

kmeans_quality = evaluate_clustering_quality(df_kmeans, '聚类分组', 'K-means')
fixed_quality = evaluate_clustering_quality(df_fixed, '经验分组', '经验分组')

# 组内一致性分析

def analyze_group_consistency(df, group_col, method_name):
    groups = df[group_col].dropna().unique()
    consistency_metrics = {}
    
    for group in groups:
        group_data = df[df[group_col] == group]
        
        if len(group_data) < 2:
            continue
            
        bmi_cv = group_data['孕妇BMI'].std() / group_data['孕妇BMI'].mean()
        
        consistency_metrics[group] = {
            'bmi_cv': bmi_cv,
            'count': len(group_data)
        }
    
    return consistency_metrics

kmeans_consistency = analyze_group_consistency(df_kmeans, '聚类分组', 'K-means')
fixed_consistency = analyze_group_consistency(df_fixed, '经验分组', '经验分组')

# 综合对比分析
kmeans_avg_cv = np.mean([metrics['bmi_cv'] for metrics in kmeans_consistency.values()])
fixed_avg_cv = np.mean([metrics['bmi_cv'] for metrics in fixed_consistency.values()])

print(f"\n对比结果:")
print(f"轮廓系数: K-means={kmeans_quality['silhouette']:.3f} vs 经验分组={fixed_quality['silhouette']:.3f}")
print(f"BMI变异系数: K-means={kmeans_avg_cv:.3f} vs 经验分组={fixed_avg_cv:.3f}")
print(f"分组数量: K-means={kmeans_quality['n_groups']} vs 经验分组={fixed_quality['n_groups']}")

# 创建图片保存目录
os.makedirs(os.path.join(script_dir, "plots"), exist_ok=True)

# 1. 聚类质量指标对比
plt.figure(figsize=(8, 6))
metrics = ['轮廓系数', 'Calinski-Harabasz', 'Davies-Bouldin', '方差比']
kmeans_values = [kmeans_quality['silhouette'], kmeans_quality['calinski']/1000, 
                 kmeans_quality['davies_bouldin'], kmeans_quality['variance_ratio']]
fixed_values = [fixed_quality['silhouette'], fixed_quality['calinski']/1000, 
                fixed_quality['davies_bouldin'], fixed_quality['variance_ratio']]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, kmeans_values, width, label='K-means', alpha=0.8)
plt.bar(x + width/2, fixed_values, width, label='经验分组', alpha=0.8)

plt.xlabel('评估指标')
plt.ylabel('指标值')
plt.title('聚类质量指标对比')
plt.xticks(x, metrics, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "clustering_quality_metrics.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 2. 组内一致性对比
plt.figure(figsize=(8, 6))
kmeans_cv = [kmeans_consistency[g]['bmi_cv'] for g in kmeans_consistency.keys()]
fixed_cv = [fixed_consistency[g]['bmi_cv'] for g in fixed_consistency.keys()]

kmeans_groups_list = list(kmeans_consistency.keys())
fixed_groups_list = list(fixed_consistency.keys())

plt.plot(kmeans_groups_list, kmeans_cv, 'o-', label='K-means', linewidth=2, markersize=6)
plt.plot(fixed_groups_list, fixed_cv, 's-', label='经验分组', linewidth=2, markersize=6)

plt.xlabel('分组')
plt.ylabel('BMI变异系数')
plt.title('组内BMI一致性对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "group_consistency.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 3. 样本分布对比
plt.figure(figsize=(8, 6))
kmeans_counts = [kmeans_groups[g]['count'] for g in kmeans_groups.keys()]
fixed_counts = [fixed_groups[g]['count'] for g in fixed_groups.keys()]

kmeans_groups_labels = [f'群组{g}' for g in kmeans_groups.keys()]
fixed_groups_labels = [str(g) for g in fixed_groups.keys()]

x1 = np.arange(len(kmeans_groups_labels))
x2 = np.arange(len(fixed_groups_labels))

plt.bar(x1 - width/2, kmeans_counts, width, label='K-means', alpha=0.8)
plt.bar(x2 + width/2, fixed_counts, width, label='经验分组', alpha=0.8)

plt.xlabel('分组')
plt.ylabel('样本数量')
plt.title('样本分布对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "sample_distribution.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 4. 综合评估
plt.figure(figsize=(10, 6))
plt.axis('off')

evaluation_text = "综合评估结果\n\n"
evaluation_text += f"K-means聚类:\n"
evaluation_text += f"• 轮廓系数: {kmeans_quality['silhouette']:.3f}\n"
evaluation_text += f"• 分组数量: {kmeans_quality['n_groups']}\n"
evaluation_text += f"• 平均BMI变异: {kmeans_avg_cv:.3f}\n\n"

evaluation_text += f"经验分组:\n"
evaluation_text += f"• 轮廓系数: {fixed_quality['silhouette']:.3f}\n"
evaluation_text += f"• 分组数量: {fixed_quality['n_groups']}\n"
evaluation_text += f"• 平均BMI变异: {fixed_avg_cv:.3f}\n\n"

if kmeans_quality['silhouette'] > fixed_quality['silhouette'] and kmeans_avg_cv < fixed_avg_cv:
    evaluation_text += "推荐: K-means聚类\n• 更好的聚类质量\n• 更高的组内一致性"
elif fixed_quality['silhouette'] > kmeans_quality['silhouette'] and fixed_avg_cv < kmeans_avg_cv:
    evaluation_text += "推荐: 经验分组\n• 更好的聚类质量\n• 更高的组内一致性"
else:
    evaluation_text += "两种方法各有优势\n• 需要根据具体需求选择"

plt.text(0.05, 0.95, evaluation_text, transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "evaluation_summary.eps"), dpi=300, bbox_inches='tight')
plt.close()

print("已保存可视化图片到 plots/ 文件夹")

# 最终结论
print(f"\n结论:")
if kmeans_quality['silhouette'] > fixed_quality['silhouette'] and kmeans_avg_cv < fixed_avg_cv:
    print(f"K-means聚类更优")
elif fixed_quality['silhouette'] > kmeans_quality['silhouette'] and fixed_avg_cv < kmeans_avg_cv:
    print(f"经验分组更优")
else:
    print(f"两种方法各有优势")
