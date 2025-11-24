import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("数据清理与质量分析")

# 读取数据
boys_df = pd.read_csv("boys_with_days.csv")
print(f"数据: 男胎{len(boys_df)}条")

# 数据基本信息分析

def analyze_basic_info(df, name):
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    missing_info = f"{len(missing_counts)}个字段有缺失值" if len(missing_counts) > 0 else "无缺失值"
    
    duplicate_mothers = df['孕妇代码'].value_counts()
    multiple_tests = duplicate_mothers[duplicate_mothers > 1]
    multiple_info = f"{len(multiple_tests)}人多次检测" if len(multiple_tests) > 0 else "无多次检测"
    
    print(f"{name}: {df.shape[0]}条记录, {missing_info}, {multiple_info}")

analyze_basic_info(boys_df, "男胎")

# 异常值检测

def detect_outliers(df, name):
    numeric_cols = ['年龄', '身高', '体重', '孕妇BMI', '孕天']
    if 'Y染色体浓度' in df.columns:
        numeric_cols.append('Y染色体浓度')
    if 'X染色体浓度' in df.columns:
        numeric_cols.append('X染色体浓度')
    
    outliers_info = {}
    total_outliers = 0
    
    for col in numeric_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min_value': df[col].min(),
                'max_value': df[col].max()
            }
            total_outliers += len(outliers)
    
    print(f"{name}: {total_outliers}个异常值")
    return outliers_info

boys_outliers = detect_outliers(boys_df, "男胎")

# Y染色体浓度达标分析
def preview_y_concentration(df):
    if 'Y染色体浓度' not in df.columns:
        return None
    
    df = df.copy()
    df['Y浓度_百分比'] = df['Y染色体浓度'] * 100
    target_threshold = 4.0
    reached_target = df[df['Y浓度_百分比'] >= target_threshold]
    
    if len(reached_target) > 0:
        print(f"Y浓度达标: {len(reached_target)}/{len(df)} ({len(reached_target)/len(df)*100:.1f}%), 孕周{reached_target['孕天'].min()/7:.1f}-{reached_target['孕天'].max()/7:.1f}周")
    
    return df

preview_y_concentration(boys_df)


# 数据清理

def clean_data(df, name, outliers_info):
    original_count = len(df)
    df_cleaned = df.copy()
    
    # 测序质量控制
    quality_before = len(df_cleaned)
    
    # GC含量控制 (使用染色体特异性范围，基于数据分布的合理范围)
    gc_thresholds = {
        '13号染色体的GC含量': (0.35, 0.41),  # 基于实际分布：36.65%-40.29%
        '18号染色体的GC含量': (0.37, 0.42),  # 基于实际分布：37.85%-41.22%
        '21号染色体的GC含量': (0.38, 0.43)   # 基于实际分布：38.52%-42.51%
    }
    
    for gc_col, (min_gc, max_gc) in gc_thresholds.items():
        if gc_col in df_cleaned.columns:
            before_filter = len(df_cleaned)
            df_cleaned = df_cleaned[(df_cleaned[gc_col] >= min_gc) & (df_cleaned[gc_col] <= max_gc)]
            removed_count = before_filter - len(df_cleaned)
            # 不打印每个GC含量的详细删除信息
    
    # 读段数质量控制
    if '唯一比对的读段数' in df_cleaned.columns:
        read_threshold = df_cleaned['唯一比对的读段数'].quantile(0.05)
        before_filter = len(df_cleaned)
        df_cleaned = df_cleaned[df_cleaned['唯一比对的读段数'] >= read_threshold]
    
    # 过滤比例控制
    if '被过滤掉读段数的比例' in df_cleaned.columns:
        filter_threshold = 0.5
        before_filter = len(df_cleaned)
        df_cleaned = df_cleaned[df_cleaned['被过滤掉读段数的比例'] <= filter_threshold]
    
    # 计算总的质量控制删除数量
    quality_removed = quality_before - len(df_cleaned)
    
    # 5.2 处理异常值（使用医学友好的阈值）
    removed_outliers = 0
    
    # 年龄异常值处理
    if '年龄' in df_cleaned.columns:
        age_outliers = df_cleaned[(df_cleaned['年龄'] < 15) | (df_cleaned['年龄'] > 50)]
        if len(age_outliers) > 0:
            df_cleaned = df_cleaned[(df_cleaned['年龄'] >= 15) & (df_cleaned['年龄'] <= 50)]
            removed_outliers += len(age_outliers)
    
    # BMI异常值处理
    if '孕妇BMI' in df_cleaned.columns:
        bmi_outliers = df_cleaned[(df_cleaned['孕妇BMI'] < 15) | (df_cleaned['孕妇BMI'] > 70)]
        if len(bmi_outliers) > 0:
            df_cleaned = df_cleaned[(df_cleaned['孕妇BMI'] >= 15) & (df_cleaned['孕妇BMI'] <= 70)]
            removed_outliers += len(bmi_outliers)
    
    # 孕周异常值处理
    if '孕天' in df_cleaned.columns:
        week_outliers = df_cleaned[(df_cleaned['孕天'] < 70) | (df_cleaned['孕天'] > 280)]
        if len(week_outliers) > 0:
            df_cleaned = df_cleaned[(df_cleaned['孕天'] >= 70) & (df_cleaned['孕天'] <= 280)]
            removed_outliers += len(week_outliers)
    
    # 5.3 处理缺失值（优先使用中位数填补，更稳健的策略）
    numeric_cols = ['年龄', '身高', '体重', '孕妇BMI', '孕天']
    filled_missing = 0
    rows_deleted_for_missing = 0
    original_length = len(df_cleaned)
    
    # 分两步处理：1) 先填补，2) 再删除
    for col in numeric_cols:
        if col in df_cleaned.columns:
            missing_count = df_cleaned[col].isnull().sum()
            if missing_count > 0:
                missing_rate = missing_count / len(df_cleaned)
                
                # 对于关键建模变量（年龄、BMI、孕天），更谨慎处理
                if col in ['年龄', '孕妇BMI', '孕天']:
                    if missing_rate < 0.05:  # 缺失率小于5%，删除记录
                        df_cleaned = df_cleaned[df_cleaned[col].notna()]
                    else:  # 缺失率较高，使用中位数填补
                        median_value = df_cleaned[col].median()
                        df_cleaned[col] = df_cleaned[col].fillna(median_value)
                        filled_missing += missing_count
                else:  # 非关键变量，同样策略
                    if missing_rate < 0.05:  # 缺失率小于5%，删除记录
                        df_cleaned = df_cleaned[df_cleaned[col].notna()]
                    else:  # 缺失率较高，使用中位数填补
                        median_value = df_cleaned[col].median()
                        df_cleaned[col] = df_cleaned[col].fillna(median_value)
                        filled_missing += missing_count
    
    # 计算实际删除的行数
    rows_deleted_for_missing = original_length - len(df_cleaned)
    
    print(f"{name}: {original_count}→{len(df_cleaned)}条 (删除{original_count-len(df_cleaned)}条, 填补{filled_missing}条)")
    
    return df_cleaned

# 首先清理列名中的多余空格
boys_df.columns = boys_df.columns.str.strip()

boys_cleaned = clean_data(boys_df, "男胎", boys_outliers)


# 最早达标时间处理
def get_earliest_qualified_records(df, name):
    """为每个孕妇保留最早达到Y染色体浓度≥4%的记录"""
    if 'Y染色体浓度' not in df.columns:
        return df.copy()
    
    # 筛选出Y浓度≥4%的记录
    qualified_records = df[df['Y染色体浓度'] >= 0.04].copy()
    
    if len(qualified_records) == 0:
        return pd.DataFrame()
    
    # 按孕妇代码分组，保留每人最早的达标时间
    qualified_records = qualified_records.sort_values(['孕妇代码', '孕天'])
    earliest_qualified = qualified_records.groupby('孕妇代码').first().reset_index()
    
    print(f"{name}: {len(df)}→{len(earliest_qualified)}条 (保留最早达标记录)")
    
    # 统计达标时间分布
    if len(earliest_qualified) > 0:
        earliest_weeks = earliest_qualified['孕天'] / 7
        print(f"  达标孕周: {earliest_weeks.min():.1f}-{earliest_weeks.max():.1f}周 (平均{earliest_weeks.mean():.1f}周)")
    
    return earliest_qualified

# 处理男胎数据
boys_final = get_earliest_qualified_records(boys_cleaned, "男胎")

print(f"最终: 男胎{len(boys_final)}条")

# 清理后数据统计
def final_statistics(df, name):
    if 'Y染色体浓度' in df.columns:
        y_concentration_pct = df['Y染色体浓度'] * 100
        qualified = df[df['Y染色体浓度'] >= 0.04]
        print(f"{name}: {len(df)}条, BMI {df['孕妇BMI'].min():.1f}-{df['孕妇BMI'].max():.1f}, 孕周{df['孕天'].min()/7:.1f}-{df['孕天'].max()/7:.1f}周, Y浓度{y_concentration_pct.min():.1f}%-{y_concentration_pct.max():.1f}%")
    else:
        print(f"{name}: {len(df)}条, BMI {df['孕妇BMI'].min():.1f}-{df['孕妇BMI'].max():.1f}, 孕周{df['孕天'].min()/7:.1f}-{df['孕天'].max()/7:.1f}周")

final_statistics(boys_final, "男胎")


# 保存清理后的数据
script_dir = os.path.dirname(os.path.abspath(__file__))

# 保存两份数据
# 1. 全量数据（清洗后但未按Y≥4%过滤）
boys_all_path = os.path.join(script_dir, "clean_data_csv", "boys_clean_2_all.csv")

# 2. 最早达标数据（用于描述性统计）
boys_path = os.path.join(script_dir, "clean_data_csv", "boys_clean_2.csv")

os.makedirs(os.path.join(script_dir, "clean_data_csv"), exist_ok=True)

# 保存全量数据（boys_cleaned 是清洗后但未过滤Y浓度的）
boys_cleaned.to_csv(boys_all_path, index=False, encoding='utf-8-sig')

# 保存最早达标数据
boys_final.to_csv(boys_path, index=False, encoding='utf-8-sig')

print(f"已保存全量数据: boys_clean_2_all.csv({len(boys_cleaned)}条)")
print(f"已保存达标数据: boys_clean_2.csv({len(boys_final)}条)")

# 生成数据清理对比图表

# 创建更大的图表布局来容纳新的散点图
fig, axes = plt.subplots(3, 3, figsize=(24, 22))
fig.suptitle('数据清理前后对比分析', fontsize=20, y=0.97)

# 9.1 样本数量对比
categories = ['原始数据', '清理后', '最早达标']
boys_counts = [len(boys_df), len(boys_cleaned), len(boys_final)]

x = np.arange(len(categories))

axes[0,0].bar(x, boys_counts, label='男胎', alpha=0.8)
axes[0,0].set_xlabel('数据状态', fontsize=12)
axes[0,0].set_ylabel('样本数量', fontsize=12)
axes[0,0].set_title('样本数量对比', fontsize=14, pad=20)
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(categories, fontsize=10)
axes[0,0].legend(fontsize=10)
axes[0,0].grid(True, alpha=0.3)

# 9.2 年龄分布对比
if len(boys_final) > 0:
    axes[0,1].hist(boys_df['年龄'].dropna(), bins=20, alpha=0.5, label='原始数据', color='blue')
    axes[0,1].hist(boys_final['年龄'].dropna(), bins=20, alpha=0.7, label='最早达标', color='red')
    axes[0,1].set_xlabel('年龄', fontsize=12)
    axes[0,1].set_ylabel('频数', fontsize=12)
    axes[0,1].set_title('男胎年龄分布对比', fontsize=14, pad=20)
    axes[0,1].legend(fontsize=10)
    axes[0,1].grid(True, alpha=0.3)

# 9.3 BMI分布对比
if len(boys_final) > 0:
    axes[0,2].hist(boys_df['孕妇BMI'].dropna(), bins=20, alpha=0.5, label='原始数据', color='blue')
    axes[0,2].hist(boys_final['孕妇BMI'].dropna(), bins=20, alpha=0.7, label='最早达标', color='red')
    axes[0,2].set_xlabel('BMI', fontsize=12)
    axes[0,2].set_ylabel('频数', fontsize=12)
    axes[0,2].set_title('男胎BMI分布对比', fontsize=14, pad=20)
    axes[0,2].legend(fontsize=10)
    axes[0,2].grid(True, alpha=0.3)

# 9.4 孕周分布对比
if len(boys_final) > 0:
    axes[1,0].hist(boys_df['孕天'].dropna()/7, bins=20, alpha=0.5, label='原始数据', color='blue')
    axes[1,0].hist(boys_final['孕天'].dropna()/7, bins=20, alpha=0.7, label='最早达标', color='red')
    axes[1,0].set_xlabel('孕周 (周)', fontsize=12)
    axes[1,0].set_ylabel('频数', fontsize=12)
    axes[1,0].set_title('男胎孕周分布对比', fontsize=14, pad=20)
    axes[1,0].legend(fontsize=10)
    axes[1,0].grid(True, alpha=0.3)

# 9.5 Y浓度分布对比
if len(boys_final) > 0 and 'Y染色体浓度' in boys_final.columns:
    y_conc_before = boys_df['Y染色体浓度'].dropna() * 100
    y_conc_after = boys_final['Y染色体浓度'].dropna() * 100
    
    axes[1,1].hist(y_conc_before, bins=30, alpha=0.5, label='原始数据', color='blue')
    axes[1,1].hist(y_conc_after, bins=30, alpha=0.7, label='最早达标', color='red')
    axes[1,1].axvline(x=4.0, color='green', linestyle='--', linewidth=2, label='4%达标线')
    axes[1,1].set_xlabel('Y染色体浓度 (%)', fontsize=12)
    axes[1,1].set_ylabel('频数', fontsize=12)
    axes[1,1].set_title('Y染色体浓度分布对比', fontsize=14, pad=20)
    axes[1,1].legend(fontsize=10)
    axes[1,1].grid(True, alpha=0.3)

# 9.6 达标时间分布
if len(boys_final) > 0:
    earliest_weeks = boys_final['孕天'] / 7
    axes[1,2].hist(earliest_weeks, bins=15, alpha=0.7, color='green')
    axes[1,2].set_xlabel('孕周 (周)', fontsize=12)
    axes[1,2].set_ylabel('频数', fontsize=12)
    axes[1,2].set_title('最早达标时间分布', fontsize=14, pad=20)
    axes[1,2].grid(True, alpha=0.3)

# 9.7 BMI与达标孕周关系散点图
if len(boys_final) > 0:
    bmi_values = boys_final['孕妇BMI']
    weeks_values = boys_final['孕天'] / 7
    
    # 计算相关性
    correlation = np.corrcoef(bmi_values, weeks_values)[0, 1]
    
    # 绘制散点图
    axes[2,0].scatter(bmi_values, weeks_values, alpha=0.6, color='purple', s=50)
    
    # 添加趋势线
    z = np.polyfit(bmi_values, weeks_values, 1)
    p = np.poly1d(z)
    axes[2,0].plot(bmi_values, p(bmi_values), "r--", alpha=0.8, linewidth=2)
    
    axes[2,0].set_xlabel('BMI', fontsize=12)
    axes[2,0].set_ylabel('达标孕周 (周)', fontsize=12)
    axes[2,0].set_title(f'BMI与达标孕周关系 (r={correlation:.3f})', fontsize=14, pad=20)
    axes[2,0].grid(True, alpha=0.3)
    
    # 添加相关性说明文本
    if correlation > 0.3:
        corr_text = "正相关 (较强)"
    elif correlation > 0.1:
        corr_text = "正相关 (较弱)"
    elif correlation > -0.1:
        corr_text = "无明显相关"
    elif correlation > -0.3:
        corr_text = "负相关 (较弱)"
    else:
        corr_text = "负相关 (较强)"
    
    axes[2,0].text(0.05, 0.95, f'相关性: {corr_text}', transform=axes[2,0].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   verticalalignment='top', fontsize=10)

# 9.8 BMI分布箱线图（按达标孕周分组）
if len(boys_final) > 0:
    # 将达标孕周分组
    boys_final_copy = boys_final.copy()
    boys_final_copy['孕周_分组'] = pd.cut(boys_final_copy['孕天']/7, 
                                       bins=[0, 12, 15, 18, 30], 
                                       labels=['<12周', '12-15周', '15-18周', '≥18周'])
    
    # 按分组绘制BMI箱线图
    bmi_by_weeks = [boys_final_copy[boys_final_copy['孕周_分组'] == group]['孕妇BMI'].dropna() 
                    for group in ['<12周', '12-15周', '15-18周', '≥18周']]
    
    # 过滤掉空的分组
    valid_groups = []
    valid_labels = []
    for i, (group_data, label) in enumerate(zip(bmi_by_weeks, ['<12周', '12-15周', '15-18周', '≥18周'])):
        if len(group_data) > 0:
            valid_groups.append(group_data)
            valid_labels.append(f'{label}\n(n={len(group_data)})')
    
    if valid_groups:
        bp = axes[2,1].boxplot(valid_groups, tick_labels=valid_labels, patch_artist=True)
        
        # 设置箱线图颜色
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(valid_groups)]):
            patch.set_facecolor(color)
        
        axes[2,1].set_ylabel('BMI', fontsize=12)
        axes[2,1].set_xlabel('达标孕周分组', fontsize=12)
        axes[2,1].set_title('不同达标时间组的BMI分布', fontsize=14, pad=20)
        axes[2,1].grid(True, alpha=0.3)

# 9.9 Y浓度与BMI关系散点图
if len(boys_final) > 0 and 'Y染色体浓度' in boys_final.columns:
    y_conc_pct = boys_final['Y染色体浓度'] * 100
    bmi_values = boys_final['孕妇BMI']
    
    # 计算相关性
    y_bmi_correlation = np.corrcoef(bmi_values, y_conc_pct)[0, 1]
    
    # 绘制散点图
    scatter = axes[2,2].scatter(bmi_values, y_conc_pct, 
                               c=boys_final['孕天']/7, cmap='viridis', 
                               alpha=0.7, s=50)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=axes[2,2])
    cbar.set_label('达标孕周', rotation=270, labelpad=15)
    
    # 添加趋势线
    z = np.polyfit(bmi_values, y_conc_pct, 1)
    p = np.poly1d(z)
    axes[2,2].plot(bmi_values, p(bmi_values), "r--", alpha=0.8, linewidth=2)
    
    axes[2,2].axhline(y=4, color='red', linestyle=':', alpha=0.7, label='4%达标线')
    axes[2,2].set_xlabel('BMI', fontsize=12)
    axes[2,2].set_ylabel('Y染色体浓度 (%)', fontsize=12)
    axes[2,2].set_title(f'BMI与Y浓度关系 (r={y_bmi_correlation:.3f})', fontsize=14, pad=20)
    axes[2,2].grid(True, alpha=0.3)
    axes[2,2].legend(fontsize=10)

# 调整子图间距，避免重叠
plt.tight_layout(pad=4.0, h_pad=4.0, w_pad=4.0)
plt.subplots_adjust(top=0.93, bottom=0.06, left=0.06, right=0.94, hspace=0.45, wspace=0.35)
plt.show()

# 相关性统计分析
if len(boys_final) > 0:
    bmi_values = boys_final['孕妇BMI']
    weeks_values = boys_final['孕天'] / 7
    age_values = boys_final['年龄']
    
    # 计算关键相关性
    bmi_week_corr = np.corrcoef(bmi_values, weeks_values)[0, 1]
    from scipy.stats import pearsonr
    corr_coef, p_value = pearsonr(bmi_values, weeks_values)
    
    print(f"BMI与达标孕周相关性: r={bmi_week_corr:.3f}, p={p_value:.3f} ({'显著' if p_value < 0.05 else '不显著'})")
    
    if 'Y染色体浓度' in boys_final.columns:
        y_conc_pct = boys_final['Y染色体浓度'] * 100
        bmi_y_corr = np.corrcoef(bmi_values, y_conc_pct)[0, 1]
        week_y_corr = np.corrcoef(weeks_values, y_conc_pct)[0, 1]
        print(f"BMI与Y浓度: r={bmi_y_corr:.3f}, 孕周与Y浓度: r={week_y_corr:.3f}")
    
    # 按BMI分组统计
    bmi_groups = pd.cut(bmi_values, bins=[0, 25, 30, 35, 50], 
                       labels=['正常(<25)', '超重(25-30)', '肥胖I(30-35)', '肥胖II(≥35)'])
    
    group_stats = []
    for group in bmi_groups.cat.categories:
        group_data = boys_final[bmi_groups == group]
        if len(group_data) > 0:
            mean_weeks = group_data['孕天'].mean() / 7
            group_stats.append(f"{group}:{len(group_data)}人({mean_weeks:.1f}周)")
    
    print(f"BMI分组达标时间: {', '.join(group_stats)}")

print(f"\n数据清理完成")
print(f"男胎: {len(boys_df)}→{len(boys_cleaned)}→{len(boys_final)}条")
if len(boys_final) > 0:
    avg_weeks = boys_final['孕天'].mean() / 7
    print(f"男胎平均达标孕周: {avg_weeks:.1f}周")