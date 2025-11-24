import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("数据清理与质量分析 - 8个临床指标版本")

# 读取数据
boys_df = pd.read_csv("boys_with_days.csv")
girls_df = pd.read_csv("girls_with_days.csv")

print(f"数据: 男胎{len(boys_df)}条, 女胎{len(girls_df)}条")

# 数据基本信息分析
def analyze_basic_info(df, name):
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    missing_info = f"缺失{len(missing_counts)}列" if len(missing_counts) > 0 else "无缺失值"
    
    duplicate_mothers = df['孕妇代码'].value_counts()
    multiple_tests = duplicate_mothers[duplicate_mothers > 1]
    multiple_info = f"多次检测{len(multiple_tests)}人" if len(multiple_tests) > 0 else "无多次检测"
    
    print(f"{name}: {df.shape[0]}条记录, {missing_info}, {multiple_info}")

analyze_basic_info(boys_df, "男胎")
analyze_basic_info(girls_df, "女胎")

# 8个临床指标定义
clinical_indicators = {
    '基础指标': ['年龄', '身高', '体重', '孕妇BMI'],
    '孕周指标': ['孕天', '孕周'],
    '生育史指标': ['怀孕次数', '生产次数', 'IVF妊娠']
}

all_clinical_vars = []
for indicators in clinical_indicators.values():
    all_clinical_vars.extend(indicators)

available_indicators = [var for var in all_clinical_vars if var in boys_df.columns]
missing_indicators = [var for var in all_clinical_vars if var not in boys_df.columns]

print(f"临床指标: 可用{len(available_indicators)}/{len(all_clinical_vars)}个")
if missing_indicators:
    print(f"缺失: {', '.join(missing_indicators)}")

# 异常值检测
def detect_clinical_outliers(df, name):
    outlier_count = 0
    for category, indicators in clinical_indicators.items():
        for col in indicators:
            if col in df.columns and df[col].notna().sum() > 0:
                df_temp = df.copy()
                df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                
                Q1 = df_temp[col].quantile(0.25)
                Q3 = df_temp[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df_temp[(df_temp[col] < lower_bound) | (df_temp[col] > upper_bound)]
                outlier_count += len(outliers)
    
    print(f"{name}异常值: {outlier_count}个")

detect_clinical_outliers(boys_df, "男胎")
detect_clinical_outliers(girls_df, "女胎")

# Y染色体浓度达标分析
def preview_y_concentration(df):
    if 'Y染色体浓度' not in df.columns:
        return None
    
    df = df.copy()
    reached_target = df[df['Y染色体浓度'] >= 0.04]
    rate = len(reached_target)/len(df)*100
    print(f"Y浓度达标: {len(reached_target)}/{len(df)} ({rate:.1f}%)")
    return df

preview_y_concentration(boys_df)

# 数据清理
def clean_data_8_clinical(df, name):
    df_cleaned = df.copy()
    
    # 测序质量控制
    quality_before = len(df_cleaned)
    
    # GC含量控制
    gc_thresholds = {
        '13号染色体的GC含量': (0.35, 0.41),
        '18号染色体的GC含量': (0.37, 0.42),
        '21号染色体的GC含量': (0.38, 0.43)
    }
    
    for gc_col, (min_gc, max_gc) in gc_thresholds.items():
        if gc_col in df_cleaned.columns:
            df_cleaned = df_cleaned[(df_cleaned[gc_col] >= min_gc) & (df_cleaned[gc_col] <= max_gc)]
    
    # 读段数质量控制
    if '唯一比对的读段数  ' in df_cleaned.columns:
        read_threshold = df_cleaned['唯一比对的读段数  '].quantile(0.05)
        df_cleaned = df_cleaned[df_cleaned['唯一比对的读段数  '] >= read_threshold]
    
    # 过滤比例控制
    if '被过滤掉读段数的比例' in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned['被过滤掉读段数的比例'] <= 0.5]
    
    # Z值质量控制
    z_value_cols = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值', 'Y染色体的Z值']
    for z_col in z_value_cols:
        if z_col in df_cleaned.columns:
            df_cleaned = df_cleaned[(df_cleaned[z_col] >= -3) & (df_cleaned[z_col] <= 3)]
    
    quality_removed = quality_before - len(df_cleaned)
    
    # 临床指标异常值处理
    removed_outliers = 0
    
    # 年龄异常值处理
    if '年龄' in df_cleaned.columns:
        age_outliers = df_cleaned[(df_cleaned['年龄'] < 15) | (df_cleaned['年龄'] > 50)]
        df_cleaned = df_cleaned[(df_cleaned['年龄'] >= 15) & (df_cleaned['年龄'] <= 50)]
        removed_outliers += len(age_outliers)
    
    # 身高异常值处理
    if '身高' in df_cleaned.columns:
        height_outliers = df_cleaned[(df_cleaned['身高'] < 140) | (df_cleaned['身高'] > 200)]
        df_cleaned = df_cleaned[(df_cleaned['身高'] >= 140) & (df_cleaned['身高'] <= 200)]
        removed_outliers += len(height_outliers)
    
    # 体重异常值处理
    if '体重' in df_cleaned.columns:
        weight_outliers = df_cleaned[(df_cleaned['体重'] < 35) | (df_cleaned['体重'] > 150)]
        df_cleaned = df_cleaned[(df_cleaned['体重'] >= 35) & (df_cleaned['体重'] <= 150)]
        removed_outliers += len(weight_outliers)
    
    # BMI异常值处理
    if '孕妇BMI' in df_cleaned.columns:
        bmi_outliers = df_cleaned[(df_cleaned['孕妇BMI'] < 15) | (df_cleaned['孕妇BMI'] > 70)]
        df_cleaned = df_cleaned[(df_cleaned['孕妇BMI'] >= 15) & (df_cleaned['孕妇BMI'] <= 70)]
        removed_outliers += len(bmi_outliers)
    
    # 孕周异常值处理
    if '孕天' in df_cleaned.columns:
        df_cleaned['孕天'] = pd.to_numeric(df_cleaned['孕天'], errors='coerce')
        week_outliers = df_cleaned[(df_cleaned['孕天'] < 70) | (df_cleaned['孕天'] > 280)]
        df_cleaned = df_cleaned[(df_cleaned['孕天'] >= 70) & (df_cleaned['孕天'] <= 280)]
        removed_outliers += len(week_outliers)
    
    # 生育史异常值处理
    if '怀孕次数' in df_cleaned.columns:
        df_cleaned['怀孕次数'] = pd.to_numeric(df_cleaned['怀孕次数'], errors='coerce')
        pregnancy_outliers = df_cleaned[(df_cleaned['怀孕次数'] < 0) | (df_cleaned['怀孕次数'] > 20)]
        df_cleaned = df_cleaned[(df_cleaned['怀孕次数'] >= 0) & (df_cleaned['怀孕次数'] <= 20)]
        removed_outliers += len(pregnancy_outliers)
    
    if '生产次数' in df_cleaned.columns:
        df_cleaned['生产次数'] = pd.to_numeric(df_cleaned['生产次数'], errors='coerce')
        delivery_outliers = df_cleaned[(df_cleaned['生产次数'] < 0) | (df_cleaned['生产次数'] > 15)]
        df_cleaned = df_cleaned[(df_cleaned['生产次数'] >= 0) & (df_cleaned['生产次数'] <= 15)]
        removed_outliers += len(delivery_outliers)
    
    # IVF妊娠处理
    if 'IVF妊娠' in df_cleaned.columns:
        df_cleaned['IVF妊娠'] = (df_cleaned['IVF妊娠'] == "IVF妊娠").astype(int)
    
    # 处理缺失值
    clinical_numeric_cols = ['年龄', '身高', '体重', '孕妇BMI', '孕天', '怀孕次数', '生产次数']
    filled_missing = 0
    rows_deleted_for_missing = 0
    original_length = len(df_cleaned)
    
    for col in clinical_numeric_cols:
        if col in df_cleaned.columns:
            missing_count = df_cleaned[col].isnull().sum()
            if missing_count > 0:
                missing_rate = missing_count / len(df_cleaned)
                
                if col in ['年龄', '孕妇BMI', '孕天']:
                    if missing_rate < 0.05:
                        df_cleaned = df_cleaned[df_cleaned[col].notna()]
                    else:
                        median_value = df_cleaned[col].median()
                        df_cleaned[col] = df_cleaned[col].fillna(median_value)
                        filled_missing += missing_count
                else:
                    if missing_rate < 0.05:
                        df_cleaned = df_cleaned[df_cleaned[col].notna()]
                    else:
                        median_value = df_cleaned[col].median()
                        df_cleaned[col] = df_cleaned[col].fillna(median_value)
                        filled_missing += missing_count
    
    rows_deleted_for_missing = original_length - len(df_cleaned)
    
    print(f"{name}清理: 质量{quality_removed}条, 异常值{removed_outliers}条, 缺失值删除{rows_deleted_for_missing}条, 填补{filled_missing}条, 最终{len(df_cleaned)}条")
    
    return df_cleaned

boys_cleaned = clean_data_8_clinical(boys_df, "男胎")
girls_cleaned = clean_data_8_clinical(girls_df, "女胎")

# 第一题数据处理
if 'Y染色体浓度' in boys_cleaned.columns:
    boys_before_filter = len(boys_cleaned)
    boys_final = boys_cleaned[boys_cleaned['Y染色体浓度'] >= 0.04].copy()
    boys_filtered_count = boys_before_filter - len(boys_final)
    print(f"Y浓度过滤: {boys_before_filter}→{len(boys_final)}条, 删除{boys_filtered_count}条")
else:
    boys_final = boys_cleaned.copy()

girls_final = girls_cleaned.copy()

boys_mothers = boys_final['孕妇代码'].value_counts()
multiple_boys = boys_mothers[boys_mothers > 1]
girls_mothers = girls_final['孕妇代码'].value_counts()  
multiple_girls = girls_mothers[girls_mothers > 1]

print(f"多次检测: 男胎{len(multiple_boys)}人, 女胎{len(multiple_girls)}人")

# 清理后统计
def final_clinical_statistics(df, name):
    stats = []
    if '年龄' in df.columns:
        stats.append(f"年龄{df['年龄'].min():.0f}-{df['年龄'].max():.0f}岁")
    if '孕妇BMI' in df.columns:
        stats.append(f"BMI{df['孕妇BMI'].min():.1f}-{df['孕妇BMI'].max():.1f}")
    if '孕天' in df.columns:
        stats.append(f"孕周{df['孕天'].min()/7:.1f}-{df['孕天'].max()/7:.1f}周")
    if 'Y染色体浓度' in df.columns:
        y_pct = df['Y染色体浓度'] * 100
        qualified = df[df['Y染色体浓度'] >= 0.04]
        stats.append(f"Y浓度{y_pct.min():.2f}-{y_pct.max():.2f}%")
        stats.append(f"达标率{len(qualified)/len(df)*100:.1f}%")
    
    print(f"{name}: {len(df)}条, {', '.join(stats)}")

final_clinical_statistics(boys_final, "男胎")
final_clinical_statistics(girls_final, "女胎")

# 保存清理后的数据
script_dir = os.path.dirname(os.path.abspath(__file__))
boys_path = os.path.join(script_dir, "clean_data_csv", "boys_clean_8_clinical.csv")
girls_path = os.path.join(script_dir, "clean_data_csv", "girls_clean_8_clinical.csv")

os.makedirs(os.path.join(script_dir, "clean_data_csv"), exist_ok=True)

boys_final.to_csv(boys_path, index=False, encoding='utf-8-sig')
girls_final.to_csv(girls_path, index=False, encoding='utf-8-sig')

print(f"已保存: boys_clean_8_clinical.csv({len(boys_final)}条), girls_clean_8_clinical.csv({len(girls_final)}条)")

# ========== 10. 8个临床指标数据清理前后对比可视化 ==========
print("\n=== 10. 生成8个临床指标数据清理对比图表 ===")

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('8个临床指标数据清理前后对比分析', fontsize=16)

# 10.1 样本数量对比
categories = ['清理前', '清理后']
boys_counts = [len(boys_df), len(boys_final)]
girls_counts = [len(girls_df), len(girls_final)]

x = np.arange(len(categories))
width = 0.35

axes[0,0].bar(x - width/2, boys_counts, width, label='男胎', alpha=0.8)
axes[0,0].bar(x + width/2, girls_counts, width, label='女胎', alpha=0.8)
axes[0,0].set_xlabel('数据状态')
axes[0,0].set_ylabel('样本数量')
axes[0,0].set_title('样本数量对比')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(categories)
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 10.2 年龄分布对比
axes[0,1].hist(boys_df['年龄'].dropna(), bins=20, alpha=0.5, label='清理前', color='blue')
axes[0,1].hist(boys_final['年龄'].dropna(), bins=20, alpha=0.7, label='清理后', color='red')
axes[0,1].set_xlabel('年龄')
axes[0,1].set_ylabel('频数')
axes[0,1].set_title('男胎年龄分布对比')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 10.3 BMI分布对比
axes[0,2].hist(boys_df['孕妇BMI'].dropna(), bins=20, alpha=0.5, label='清理前', color='blue')
axes[0,2].hist(boys_final['孕妇BMI'].dropna(), bins=20, alpha=0.7, label='清理后', color='red')
axes[0,2].set_xlabel('BMI')
axes[0,2].set_ylabel('频数')
axes[0,2].set_title('男胎BMI分布对比')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# 10.4 身高分布对比
axes[1,0].hist(boys_df['身高'].dropna(), bins=20, alpha=0.5, label='清理前', color='blue')
axes[1,0].hist(boys_final['身高'].dropna(), bins=20, alpha=0.7, label='清理后', color='red')
axes[1,0].set_xlabel('身高 (cm)')
axes[1,0].set_ylabel('频数')
axes[1,0].set_title('男胎身高分布对比')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 10.5 体重分布对比
axes[1,1].hist(boys_df['体重'].dropna(), bins=20, alpha=0.5, label='清理前', color='blue')
axes[1,1].hist(boys_final['体重'].dropna(), bins=20, alpha=0.7, label='清理后', color='red')
axes[1,1].set_xlabel('体重 (kg)')
axes[1,1].set_ylabel('频数')
axes[1,1].set_title('男胎体重分布对比')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

# 10.6 孕周分布对比
axes[1,2].hist(boys_df['孕天'].dropna()/7, bins=20, alpha=0.5, label='清理前', color='blue')
axes[1,2].hist(boys_final['孕天'].dropna()/7, bins=20, alpha=0.7, label='清理后', color='red')
axes[1,2].set_xlabel('孕周 (周)')
axes[1,2].set_ylabel('频数')
axes[1,2].set_title('男胎孕周分布对比')
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3)

# 10.7 怀孕次数分布对比
if '怀孕次数' in boys_final.columns:
    pregnancy_before = pd.to_numeric(boys_df['怀孕次数'], errors='coerce').dropna()
    pregnancy_after = boys_final['怀孕次数'].dropna()
    axes[2,0].hist(pregnancy_before, bins=15, alpha=0.5, label='清理前', color='blue')
    axes[2,0].hist(pregnancy_after, bins=15, alpha=0.7, label='清理后', color='red')
    axes[2,0].set_xlabel('怀孕次数')
    axes[2,0].set_ylabel('频数')
    axes[2,0].set_title('男胎怀孕次数分布对比')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)

# 10.8 生产次数分布对比
if '生产次数' in boys_final.columns:
    delivery_before = pd.to_numeric(boys_df['生产次数'], errors='coerce').dropna()
    delivery_after = boys_final['生产次数'].dropna()
    axes[2,1].hist(delivery_before, bins=15, alpha=0.5, label='清理前', color='blue')
    axes[2,1].hist(delivery_after, bins=15, alpha=0.7, label='清理后', color='red')
    axes[2,1].set_xlabel('生产次数')
    axes[2,1].set_ylabel('频数')
    axes[2,1].set_title('男胎生产次数分布对比')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)

# 10.9 Y浓度分布对比
y_conc_before = boys_df['Y染色体浓度'].dropna() * 100
y_conc_after = boys_final['Y染色体浓度'].dropna() * 100

axes[2,2].hist(y_conc_before, bins=30, alpha=0.5, label='清理前', color='blue')
axes[2,2].hist(y_conc_after, bins=30, alpha=0.7, label='清理后', color='red')
axes[2,2].axvline(x=4.0, color='green', linestyle='--', linewidth=2, label='4%达标线')
axes[2,2].set_xlabel('Y染色体浓度 (%)')
axes[2,2].set_ylabel('频数')
axes[2,2].set_title('Y染色体浓度分布对比')
axes[2,2].legend()
axes[2,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n清理完成:")
print(f"男胎: {len(boys_df)}→{len(boys_final)}条, 女胎: {len(girls_df)}→{len(girls_final)}条")
if 'Y染色体浓度' in boys_df.columns:
    before_qualified = len(boys_df[boys_df['Y染色体浓度'] >= 0.04])
    after_qualified = len(boys_final[boys_final['Y染色体浓度'] >= 0.04])
    before_rate = before_qualified / len(boys_df) * 100
    after_rate = after_qualified / len(boys_final) * 100
    print(f"Y浓度达标率: {before_rate:.1f}%→{after_rate:.1f}%")
print(f"8个临床指标处理完成: 基础指标4个, 孕周指标2个, 生育史指标3个")
