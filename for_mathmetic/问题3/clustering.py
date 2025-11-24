import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def perform_clustering(data, max_clusters=10, visualize=True):
    """
    执行K-means聚类 - 限定年龄、体重、身高三个维度
    """
    cluster_vars = ['年龄', '体重', '身高']
    
    # 检查必需的列是否存在
    missing_vars = [var for var in cluster_vars if var not in data.columns]
    if missing_vars:
        raise ValueError(f"数据中缺少必需的聚类变量: {missing_vars}")
    
    print(f"聚类变量: {', '.join(cluster_vars)}")
    
    # 准备聚类数据
    X_cluster = data[cluster_vars].copy()
    
    # 检查并处理缺失值
    if X_cluster.isnull().any().any():
        print("发现缺失值，将删除包含缺失值的记录")
        X_cluster = X_cluster.dropna()
        data = data.loc[X_cluster.index].copy()
    
    print(f"有效样本数: {len(X_cluster)}")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # 确定最佳聚类数 - 改进算法
    silhouette_scores = []
    inertia_scores = []
    print("测试不同聚类数的性能指标:")
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        inertia_scores.append(kmeans.inertia_)
        print(f"  {n_clusters}组: 轮廓系数 {silhouette_avg:.3f}, 惯性 {kmeans.inertia_:.1f}")
    
    # 综合考虑轮廓系数和聚类数量的平衡
    # 选择轮廓系数在前三名且聚类数适中的方案
    silhouette_array = np.array(silhouette_scores)
    top_indices = np.argsort(silhouette_array)[-3:]  # 前三名
    
    # 优先选择3-5个聚类组的方案
    preferred_clusters = []
    for idx in reversed(top_indices):
        n_clusters = idx + 2
        if 3 <= n_clusters <= 5:
            preferred_clusters.append((n_clusters, silhouette_array[idx]))
    
    if preferred_clusters:
        best_n_clusters, best_silhouette = preferred_clusters[0]
        print(f"选择最佳聚类数: {best_n_clusters} (轮廓系数: {best_silhouette:.3f}, 平衡考虑)")
    else:
        # 如果3-5范围内没有好的选择，则选择轮廓系数最高的
        best_n_clusters = np.argmax(silhouette_scores) + 2
        best_silhouette = max(silhouette_scores)
        print(f"选择最佳聚类数: {best_n_clusters} (轮廓系数: {best_silhouette:.3f}, 最优轮廓系数)")
    
    # 分析数据分布
    print(f"\n数据分布分析:")
    print(f"年龄: {X_cluster['年龄'].min():.1f}-{X_cluster['年龄'].max():.1f}岁 (标准差: {X_cluster['年龄'].std():.1f})")
    print(f"体重: {X_cluster['体重'].min():.1f}-{X_cluster['体重'].max():.1f}kg (标准差: {X_cluster['体重'].std():.1f})")
    print(f"身高: {X_cluster['身高'].min():.1f}-{X_cluster['身高'].max():.1f}cm (标准差: {X_cluster['身高'].std():.1f})")
    
    # 应用聚类
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    data['聚类组'] = kmeans.fit_predict(X_scaled)
    
    # 聚类总结 - 简化版本
    cluster_counts = data['聚类组'].value_counts().sort_index()
    
    # 创建简化的聚类总结
    cluster_summary = pd.DataFrame()
    for var in cluster_vars:
        cluster_summary[f'{var}_mean'] = data.groupby('聚类组')[var].mean()
        cluster_summary[f'{var}_std'] = data.groupby('聚类组')[var].std()
    
    # 添加其他重要指标
    if 'Y染色体浓度' in data.columns:
        cluster_summary['Y浓度_mean'] = data.groupby('聚类组')['Y染色体浓度'].mean()
        cluster_summary['Y浓度_std'] = data.groupby('聚类组')['Y染色体浓度'].std()
    
    if '孕天' in data.columns:
        cluster_summary['孕天_mean'] = data.groupby('聚类组')['孕天'].mean()
        cluster_summary['孕天_std'] = data.groupby('聚类组')['孕天'].std()
    
    # 添加样本数量
    cluster_summary['样本数'] = cluster_counts
    
    print(f"聚类结果: {best_n_clusters}组, 轮廓系数{best_silhouette:.3f}, 样本分布{dict(cluster_counts)}")
    
    # 显示各聚类组特征
    for i in range(best_n_clusters):
        group_data = data[data['聚类组'] == i]
        age_mean = group_data['年龄'].mean()
        weight_mean = group_data['体重'].mean()
        height_mean = group_data['身高'].mean()
        print(f"  组{i}: 年龄{age_mean:.1f}岁, 体重{weight_mean:.1f}kg, 身高{height_mean:.1f}cm, n={len(group_data)}")
    
    # 可视化
    if visualize:
        # 创建图片保存目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(script_dir, "plots"), exist_ok=True)
        
        # 1. 3D散点图 - 年龄、体重、身高
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data['年龄'], data['体重'], data['身高'], 
                           c=data['聚类组'], cmap='viridis', s=50, alpha=0.7)
        ax.set_xlabel('年龄')
        ax.set_ylabel('体重 (kg)')
        ax.set_zlabel('身高 (cm)')
        ax.set_title('3D聚类结果 - 年龄、体重、身高')
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, "plots", "3d_clustering.eps"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 年龄 vs 体重
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(data['年龄'], data['体重'], 
                            c=data['聚类组'], cmap='viridis', s=50, alpha=0.7)
        plt.xlabel('年龄')
        plt.ylabel('体重 (kg)')
        plt.title('年龄 vs 体重')
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, "plots", "age_weight.eps"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 年龄 vs 身高
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(data['年龄'], data['身高'], 
                            c=data['聚类组'], cmap='viridis', s=50, alpha=0.7)
        plt.xlabel('年龄')
        plt.ylabel('身高 (cm)')
        plt.title('年龄 vs 身高')
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, "plots", "age_height.eps"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 体重 vs 身高
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(data['体重'], data['身高'], 
                            c=data['聚类组'], cmap='viridis', s=50, alpha=0.7)
        plt.xlabel('体重 (kg)')
        plt.ylabel('身高 (cm)')
        plt.title('体重 vs 身高')
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, "plots", "weight_height.eps"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 如果有Y染色体浓度数据，创建Y浓度分析图
        if 'Y染色体浓度' in data.columns:
            # Y浓度 vs 年龄
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(data['年龄'], data['Y染色体浓度'] * 100, 
                                c=data['聚类组'], cmap='viridis', s=50, alpha=0.7)
            plt.xlabel('年龄')
            plt.ylabel('Y染色体浓度 (%)')
            plt.title('Y浓度 vs 年龄')
            plt.grid(True, alpha=0.3)
            plt.colorbar(scatter)
            plt.tight_layout()
            plt.savefig(os.path.join(script_dir, "plots", "y_concentration_age.eps"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Y浓度 vs 体重
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(data['体重'], data['Y染色体浓度'] * 100, 
                                c=data['聚类组'], cmap='viridis', s=50, alpha=0.7)
            plt.xlabel('体重 (kg)')
            plt.ylabel('Y染色体浓度 (%)')
            plt.title('Y浓度 vs 体重')
            plt.grid(True, alpha=0.3)
            plt.colorbar(scatter)
            plt.tight_layout()
            plt.savefig(os.path.join(script_dir, "plots", "y_concentration_weight.eps"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Y浓度 vs 身高
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(data['身高'], data['Y染色体浓度'] * 100, 
                                c=data['聚类组'], cmap='viridis', s=50, alpha=0.7)
            plt.xlabel('身高 (cm)')
            plt.ylabel('Y染色体浓度 (%)')
            plt.title('Y浓度 vs 身高')
            plt.grid(True, alpha=0.3)
            plt.colorbar(scatter)
            plt.tight_layout()
            plt.savefig(os.path.join(script_dir, "plots", "y_concentration_height.eps"), dpi=300, bbox_inches='tight')
            plt.close()
        
        print("已保存可视化图片到 plots/ 文件夹")
    
    return data, kmeans, best_n_clusters, cluster_summary, scaler

# 主函数测试
if __name__ == "__main__":
    print("年龄、体重、身高三维聚类分析")

    # 读取数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "问题2", "clean_data_csv", "boys_clean_2_all.csv")
    data = pd.read_csv(data_path)
    print(f"数据: {len(data)}条记录")
            
    # 执行聚类
    clustered_data, kmeans, n_clusters, summary, scaler = perform_clustering(data, max_clusters=8, visualize=True)

    # 保存带聚类结果的数据
    output_filename = 'boys_clean_2_all_with_clusters.csv'
    output_data_path = os.path.join(script_dir, output_filename)
    clustered_data.to_csv(output_data_path, index=False, encoding='utf-8-sig')
    print(f"\n聚类完成: 生成{n_clusters}个聚类组")
    print(f"已保存带聚类结果的数据文件: {output_data_path}")