import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
# 新增：用于构造基于脚本目录的绝对路径
import os
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("基于固定BMI区间的分组分析")

# 数据加载
script_dir = os.path.dirname(os.path.abspath(__file__))
boys_csv_path = os.path.join(script_dir, "clean_data_csv", "boys_clean_2.csv")
boys_data = pd.read_csv(boys_csv_path)
print(f"数据: {len(boys_data)} 条记录")

# 基于固定BMI区间分组
def fixed_bmi_grouping_analysis(df):
    """基于固定BMI区间进行分组和风险分析"""
    
    # 定义BMI区间
    bins = [0, 28, 32, 36, 40, 50]
    labels = ['<28', '28-32', '32-36', '36-40', '≥40']
    
    df_copy = df.copy()
    df_copy['BMI_分组'] = pd.cut(df_copy['孕妇BMI'], bins=bins, labels=labels, right=False)
    
    # 分析各组特征
    group_analysis = {}
    
    for group in labels:
        group_data = df_copy[df_copy['BMI_分组'] == group]
        
        if len(group_data) > 0:
            # 基本统计
            bmi_mean = group_data['孕妇BMI'].mean()
            bmi_std = group_data['孕妇BMI'].std()
            bmi_min = group_data['孕妇BMI'].min()
            bmi_max = group_data['孕妇BMI'].max()
            
            age_mean = group_data['年龄'].mean()
            age_std = group_data['年龄'].std()
            
            weeks_mean = group_data['孕天'].mean() / 7
            weeks_std = group_data['孕天'].std() / 7
            weeks_median = group_data['孕天'].median() / 7
            weeks_q25 = group_data['孕天'].quantile(0.25) / 7
            weeks_q75 = group_data['孕天'].quantile(0.75) / 7
            
            y_mean = group_data['Y染色体浓度'].mean() * 100
            y_std = group_data['Y染色体浓度'].std() * 100
            y_median = group_data['Y染色体浓度'].median() * 100
            
            # 风险指标计算
            timing_risk = weeks_std  # 检测时间变异风险
            y_cv = (group_data['Y染色体浓度'].std() / group_data['Y染色体浓度'].mean())  # Y浓度变异系数
            early_detection_rate = len(group_data[group_data['孕天'] < 84]) / len(group_data) * 100  # 12周前检测率
            sample_size_risk = 1 / (1 + len(group_data) / 50)  # 样本数量风险
            
            # 推荐NIPT时点（基于中位数，考虑风险最小化）
            optimal_week = weeks_median
            
            # 时间窗口建议（基于四分位数）
            time_window_lower = weeks_q25
            time_window_upper = weeks_q75
            
            analysis = {
                'count': len(group_data),
                'bmi_range': (bmi_min, bmi_max),
                'bmi_mean': bmi_mean,
                'bmi_std': bmi_std,
                'age_mean': age_mean,
                'age_std': age_std,
                'weeks_mean': weeks_mean,
                'weeks_std': weeks_std,
                'weeks_median': weeks_median,
                'weeks_q25': time_window_lower,
                'weeks_q75': time_window_upper,
                'y_mean': y_mean,
                'y_std': y_std,
                'y_median': y_median,
                'optimal_week': optimal_week,
                'time_window': (time_window_lower, time_window_upper),
                'timing_risk': timing_risk,
                'y_cv': y_cv,
                'early_detection_rate': early_detection_rate,
                'sample_size_risk': sample_size_risk,
                'composite_risk': timing_risk * 20 + y_cv * 25 + early_detection_rate * 0.3 + sample_size_risk * 15
            }
            
            group_analysis[group] = analysis
            
            sample_adequacy = '充足' if len(group_data) >= 50 else '一般' if len(group_data) >= 20 else '不足'
            print(f"{group}组: BMI {bmi_min:.1f}-{bmi_max:.1f}, 推荐{optimal_week:.1f}周, n={analysis['count']}({sample_adequacy}), 风险{analysis['composite_risk']:.1f}")
    
    return df_copy, group_analysis

boys_grouped, group_analysis = fixed_bmi_grouping_analysis(boys_data)

# 检测误差影响分析
def analyze_detection_errors(df, group_analysis, error_rates=[0.01, 0.02, 0.05, 0.10]):
    """分析不同检测误差率对各组检测成功率的影响"""
    
    error_impact = {}
    
    for error_rate in error_rates:
        group_impacts = {}
        
        for group, analysis in group_analysis.items():
            group_data = df[df['BMI_分组'] == group]
            
            if len(group_data) == 0:
                continue
            
            # 原始检测成功率 (Y浓度≥4%)
            original_success_rate = len(group_data[group_data['Y染色体浓度'] >= 0.04]) / len(group_data)
            
            # 模拟检测误差
            np.random.seed(42)  # 确保结果可重现
            y_concentrations = group_data['Y染色体浓度'].values
            
            # 添加检测误差
            errors = np.random.normal(0, error_rate, len(y_concentrations))
            y_with_error = y_concentrations * (1 + errors)
            y_with_error = np.maximum(y_with_error, 0)  # 确保浓度非负
            
            # 计算误差影响下的检测成功率
            error_success_rate = np.sum(y_with_error >= 0.04) / len(y_with_error)
            
            # 计算影响程度
            success_rate_change = (error_success_rate - original_success_rate) * 100
            relative_change = abs(success_rate_change / original_success_rate * 100) if original_success_rate > 0 else 0
            
            group_impacts[group] = {
                'original_rate': original_success_rate * 100,
                'error_rate': error_success_rate * 100,
                'absolute_change': success_rate_change,
                'relative_change': relative_change,
                'robustness': '强' if relative_change < 5 else '中等' if relative_change < 15 else '弱'
            }
        
        error_impact[error_rate] = group_impacts
    
    return error_impact

error_impact_analysis = analyze_detection_errors(boys_grouped, group_analysis)

# 最终推荐策略
def generate_recommendations(group_analysis, error_impact):
    """生成最终的NIPT时点推荐策略"""
    
    recommendations = {}
    
    for group, analysis in group_analysis.items():
        # 综合考虑风险和误差敏感性
        avg_error_sensitivity = np.mean([error_impact[rate][group]['relative_change'] 
                                       for rate in error_impact.keys() if group in error_impact[rate]])
        
        # 风险等级评估
        risk_level = "低风险" if analysis['composite_risk'] < 50 else "中等风险" if analysis['composite_risk'] < 80 else "高风险"
        
        # 样本充足性评估
        sample_adequacy = "充足" if analysis['count'] >= 50 else "一般" if analysis['count'] >= 20 else "不足"
        
        recommendations[group] = {
            'bmi_range': analysis['bmi_range'],
            'optimal_week': analysis['optimal_week'],
            'time_window': analysis['time_window'],
            'sample_size': analysis['count'],
            'risk_level': risk_level,
            'error_sensitivity': avg_error_sensitivity,
            'sample_adequacy': sample_adequacy,
            'special_notes': []
        }
        
        # 特殊注意事项
        if analysis['early_detection_rate'] > 20:
            recommendations[group]['special_notes'].append("注意早期检测风险较高")
        if analysis['timing_risk'] > 3:
            recommendations[group]['special_notes'].append("检测时间变异较大，建议更频繁监测")
        if avg_error_sensitivity > 15:
            recommendations[group]['special_notes'].append("对检测误差敏感，需提高检测精度")
        if analysis['count'] < 20:
            recommendations[group]['special_notes'].append("样本数量不足，建议扩大样本验证")
        
        # 不打印每个组的详细推荐信息
    
    return recommendations

final_recommendations = generate_recommendations(group_analysis, error_impact_analysis)

# 生成分析图表

# 创建可视化图表
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('基于经验的BMI固定区间分组NIPT时点优化分析', fontsize=16, y=0.98)

# 5.1 各组BMI分布和推荐时点
ax = axes[0, 0]
groups = list(group_analysis.keys())
bmi_means = [group_analysis[g]['bmi_mean'] for g in groups]
optimal_weeks = [group_analysis[g]['optimal_week'] for g in groups]
sample_sizes = [group_analysis[g]['count'] for g in groups]

scatter = ax.scatter(bmi_means, optimal_weeks, s=[s*3 for s in sample_sizes], 
                    alpha=0.7, c=range(len(groups)), cmap='viridis')

for i, (group, bmi, week) in enumerate(zip(groups, bmi_means, optimal_weeks)):
    ax.annotate(f'{group}\n({sample_sizes[i]}人)', (bmi, week), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('平均BMI', fontsize=12)
ax.set_ylabel('推荐NIPT时点 (周)', fontsize=12)
ax.set_title('各组推荐NIPT时点', fontsize=14)
ax.grid(True, alpha=0.3)

# 5.2 风险评分对比
ax = axes[0, 1]
risk_scores = [group_analysis[g]['composite_risk'] for g in groups]
colors = ['green' if r < 50 else 'orange' if r < 80 else 'red' for r in risk_scores]

bars = ax.bar(groups, risk_scores, color=colors, alpha=0.7)
ax.set_ylabel('综合风险评分', fontsize=12)
ax.set_title('各组风险评估', fontsize=14)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

# 添加数值标签
for bar, risk in zip(bars, risk_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{risk:.1f}', ha='center', va='bottom', fontsize=10)

# 5.3 检测时间窗口
ax = axes[0, 2]
time_windows_lower = [group_analysis[g]['time_window'][0] for g in groups]
time_windows_upper = [group_analysis[g]['time_window'][1] for g in groups]
time_windows_width = [upper - lower for lower, upper in zip(time_windows_lower, time_windows_upper)]

bars = ax.barh(range(len(groups)), time_windows_width, 
               left=time_windows_lower, alpha=0.7, color='lightblue')
ax.scatter(optimal_weeks, range(len(groups)), color='red', s=50, zorder=5, label='推荐时点')

ax.set_yticks(range(len(groups)))
ax.set_yticklabels(groups)
ax.set_xlabel('孕周', fontsize=12)
ax.set_title('建议检测时间窗口', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# 5.4 误差敏感性分析
ax = axes[1, 0]
error_rates = list(error_impact_analysis.keys())
for i, group in enumerate(groups):
    sensitivities = [error_impact_analysis[rate][group]['relative_change'] 
                    for rate in error_rates if group in error_impact_analysis[rate]]
    if sensitivities:
        ax.plot([r*100 for r in error_rates[:len(sensitivities)]], sensitivities, 
                'o-', label=group, alpha=0.8)

ax.set_xlabel('检测误差率 (%)', fontsize=12)
ax.set_ylabel('相对变化 (%)', fontsize=12)
ax.set_title('误差敏感性分析', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# 5.5 样本分布
ax = axes[1, 1]
sample_counts = [group_analysis[g]['count'] for g in groups]
colors = ['green' if c >= 50 else 'orange' if c >= 20 else 'red' for c in sample_counts]

bars = ax.bar(groups, sample_counts, color=colors, alpha=0.7)
ax.set_ylabel('样本数量', fontsize=12)
ax.set_title('各组样本分布', fontsize=14)
ax.tick_params(axis='x', rotation=45)
ax.axhline(y=50, color='green', linestyle='--', alpha=0.7, label='充足样本线')
ax.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='最低样本线')
ax.legend()
ax.grid(True, alpha=0.3)

# 添加数值标签
for bar, count in zip(bars, sample_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{count}', ha='center', va='bottom', fontsize=10)

# 5.6 综合推荐总结
ax = axes[1, 2]
ax.axis('off')

summary_text = "最终推荐策略总结\n\n"
for group, rec in final_recommendations.items():
    risk_color = "🟢" if rec['risk_level'] == "低风险" else "🟡" if rec['risk_level'] == "中等风险" else "🔴"
    summary_text += f"{risk_color} {group}: {rec['optimal_week']:.1f}周\n"
    summary_text += f"   BMI {rec['bmi_range'][0]:.1f}-{rec['bmi_range'][1]:.1f}\n"
    summary_text += f"   窗口 {rec['time_window'][0]:.1f}-{rec['time_window'][1]:.1f}周\n\n"

summary_text += "\n风险等级:\n🟢 低风险  🟡 中等风险  🔴 高风险"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

plt.tight_layout()
plt.show()

# 生成最终报告
total_samples = sum([analysis['count'] for analysis in group_analysis.values()])
avg_risk = np.mean([analysis['composite_risk'] for analysis in group_analysis.values()])

print(f"\n推荐策略:")
for group, rec in final_recommendations.items():
    print(f"  {group}组: BMI {rec['bmi_range'][0]:.1f}-{rec['bmi_range'][1]:.1f}, 推荐{rec['optimal_week']:.1f}周, n={rec['sample_size']}({rec['sample_adequacy']}), 风险{rec['risk_level']}")

print(f"总体: {total_samples}人, 平均风险{avg_risk:.1f}, {len(group_analysis)}组")

# 保存结果
boys_grouped.to_csv('boys_normal_analysis.csv', index=False, encoding='utf-8-sig')
print(f"已保存: boys_normal_analysis.csv")