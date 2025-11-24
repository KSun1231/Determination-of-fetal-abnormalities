import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("女胎数据清理与质量分析")

# 数据加载
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'girls_with_days.csv')

df = pd.read_csv(data_path)
print(f"原始数据: {len(df)} 条记录")

# 异常值检测与处理
rf_features = ['GC含量', '在参考基因组上比对的比例', '重复读段的比例', '孕妇BMI', 
               '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']

numeric_cols = ['年龄', '身高', '体重', '孕天', '原始读段数', 
                '唯一比对的读段数', 'X染色体浓度', '13号染色体的GC含量', 
                '18号染色体的GC含量', '21号染色体的GC含量', '被过滤掉读段数的比例']

numeric_cols = [col for col in numeric_cols if col in df.columns]
rf_features = [col for col in rf_features if col in df.columns]

# 使用IQR方法检测异常值
outlier_summary = {}
outlier_indices = set()

for col in numeric_cols:
    if df[col].dtype in ['int64', 'float64']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        
        if outlier_count > 0:
            outlier_summary[col] = {
                'count': outlier_count,
                'percentage': outlier_count / len(df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            outlier_indices.update(outliers.index.tolist())

print(f"检测到异常值: {len(outlier_indices)} 个样本涉及 {len(outlier_summary)} 个变量")

# 异常值填充策略
df_cleaned = df.copy()

fill_summary = {}
for col, info in outlier_summary.items():
    median_value = df[col].median()
    outlier_mask = (df[col] < info['lower_bound']) | (df[col] > info['upper_bound'])
    outlier_count = outlier_mask.sum()
    
    df_cleaned.loc[outlier_mask, col] = median_value
    fill_summary[col] = {
        'outlier_count': outlier_count,
        'median_value': median_value
    }
    print(f"{col}: 填充{outlier_count}个异常值")

# 染色体非整倍体数据转换
def create_abnormal_label(ab_value):
    if pd.isna(ab_value) or ab_value == '':
        return 0
    else:
        return 1

df_cleaned['异常标签'] = df_cleaned['染色体的非整倍体'].apply(create_abnormal_label)

abnormal_count = df_cleaned['异常标签'].sum()
normal_count = len(df_cleaned) - abnormal_count
print(f"异常标签转换: 异常{abnormal_count}条, 正常{normal_count}条")

# 数据质量检查
missing_data = df_cleaned.isnull().sum()
missing_count = missing_data.sum()
print(f"缺失值: {missing_count} 个")

# 创建图片保存目录
os.makedirs(os.path.join(script_dir, "plots"), exist_ok=True)

# 1. 年龄分布对比
plt.figure(figsize=(8, 6))
plt.hist(df['年龄'], bins=20, alpha=0.7, label='原始数据', color='lightblue')
plt.hist(df_cleaned['年龄'], bins=20, alpha=0.7, label='清理后', color='orange')
plt.xlabel('年龄 (岁)')
plt.ylabel('频数')
plt.title('年龄分布对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "age_distribution.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 2. BMI分布对比
plt.figure(figsize=(8, 6))
plt.hist(df['孕妇BMI'], bins=20, alpha=0.7, label='原始数据', color='lightgreen')
plt.hist(df_cleaned['孕妇BMI'], bins=20, alpha=0.7, label='清理后', color='red')
plt.xlabel('BMI')
plt.ylabel('频数')
plt.title('BMI分布对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "bmi_distribution.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 3. X染色体浓度分布对比
plt.figure(figsize=(8, 6))
plt.hist(df['X染色体浓度'], bins=20, alpha=0.7, label='原始数据', color='lightcoral')
plt.hist(df_cleaned['X染色体浓度'], bins=20, alpha=0.7, label='清理后', color='purple')
plt.xlabel('X染色体浓度')
plt.ylabel('频数')
plt.title('X染色体浓度分布对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "x_chromosome_distribution.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 4. 染色体非整倍体分布
plt.figure(figsize=(8, 6))
abnormal_labels = df_cleaned['异常标签'].value_counts()
plt.pie(abnormal_labels.values, labels=['无异常(0)', '有异常(1)'], 
        autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
plt.title('染色体非整倍体分布')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "abnormal_distribution.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 5. 孕天分布
plt.figure(figsize=(8, 6))
plt.hist(df_cleaned['孕天'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('孕天')
plt.ylabel('频数')
plt.title('孕天分布')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "pregnancy_days.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 6. 异常值统计
if outlier_summary:
    plt.figure(figsize=(8, 6))
    outlier_cols = list(outlier_summary.keys())[:5]
    outlier_counts = [outlier_summary[col]['count'] for col in outlier_cols]
    plt.bar(range(len(outlier_cols)), outlier_counts, color='salmon', alpha=0.7)
    plt.xlabel('变量')
    plt.ylabel('异常值数量')
    plt.title('各变量异常值数量')
    plt.xticks(range(len(outlier_cols)), outlier_cols, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "plots", "outlier_counts.eps"), dpi=300, bbox_inches='tight')
    plt.close()

print("已保存可视化图片到 plots/ 文件夹")

# 保存清理后的数据
output_file = 'girls_cleaned.csv'
df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n数据清理完成!")
print(f"原始数据: {len(df)} 条记录")
print(f"清理后数据: {len(df_cleaned)} 条记录")
print(f"异常值处理: {len(outlier_indices)} 个样本")
print(f"异常标签转换: 异常{abnormal_count}条, 正常{normal_count}条")
print(f"已保存清理后数据: {output_file}")
print(f"已保存可视化图片到 plots/ 文件夹")
