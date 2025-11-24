import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("随机森林异常值预测模型分析")

# 数据加载
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'girls_cleaned.csv')

df = pd.read_csv(data_path)
print(f"数据: {len(df)} 条记录")

abnormal_count = df['异常标签'].sum()
normal_count = len(df) - abnormal_count
print(f"目标变量分布: 异常{abnormal_count}条, 正常{normal_count}条")

# 特征选择与数据准备
rf_features = ['GC含量', '在参考基因组上比对的比例', '重复读段的比例', '孕妇BMI', 
               '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']

available_features = [col for col in rf_features if col in df.columns]

X = df[available_features].copy()
y = df['异常标签'].copy()

# 检查缺失值
missing_values = X.isnull().sum()
if missing_values.sum() > 0:
    for col in available_features:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
    print(f"缺失值处理: {missing_values.sum()} 个")

print(f"特征矩阵形状: {X.shape}")

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"训练集: {X_train.shape[0]} 条记录")
print(f"测试集: {X_test.shape[0]} 条记录")

# 构建SMOTE与随机森林管道
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(
        random_state=42,
        class_weight='balanced'
    ))
])

# 模型训练与评估
pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
y_train_proba = pipeline.predict_proba(X_train)[:, 1]
y_test_proba = pipeline.predict_proba(X_test)[:, 1]

# 模型评估
train_accuracy = accuracy_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"训练集性能: 准确率={train_accuracy:.3f}, AUC={train_auc:.3f}")
print(f"测试集性能: 准确率={test_accuracy:.3f}, AUC={test_auc:.3f}")

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
print(f"5折交叉验证AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': pipeline.named_steps['rf'].feature_importances_
}).sort_values('importance', ascending=False)

print("特征重要性排序:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# 超参数优化
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 20, None],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2]
}

print("开始网格搜索优化...")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证AUC: {grid_search.best_score_:.3f}")

best_pipeline = grid_search.best_estimator_
y_test_pred_best = best_pipeline.predict(X_test)
y_test_proba_best = best_pipeline.predict_proba(X_test)[:, 1]

best_accuracy = accuracy_score(y_test, y_test_pred_best)
best_auc = roc_auc_score(y_test, y_test_proba_best)

print(f"优化后测试集性能: 准确率={best_accuracy:.3f}, AUC={best_auc:.3f}")

# 创建图片保存目录
os.makedirs(os.path.join(script_dir, "plots"), exist_ok=True)

# 1. 特征重要性
plt.figure(figsize=(8, 6))
plt.barh(range(len(feature_importance)), feature_importance['importance'], color='skyblue')
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('重要性')
plt.title('特征重要性')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "feature_importance.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 2. ROC曲线
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_test_proba_best)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {best_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('ROC曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "roc_curve.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 3. 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_test_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "confusion_matrix.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 4. 预测概率分布
plt.figure(figsize=(8, 6))
plt.hist(y_test_proba_best[y_test == 0], bins=20, alpha=0.7, label='正常样本', color='lightblue')
plt.hist(y_test_proba_best[y_test == 1], bins=20, alpha=0.7, label='异常样本', color='lightcoral')
plt.xlabel('预测概率')
plt.ylabel('频数')
plt.title('预测概率分布')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "probability_distribution.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 5. 精确率-召回率曲线
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, y_test_proba_best)
ap_score = average_precision_score(y_test, y_test_proba_best)
plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR曲线 (AP = {ap_score:.3f})')
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('精确率-召回率曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "precision_recall_curve.eps"), dpi=300, bbox_inches='tight')
plt.close()

# 6. 交叉验证分数
plt.figure(figsize=(8, 6))
cv_scores_best = cross_val_score(best_pipeline, X, y, cv=5, scoring='roc_auc')
plt.boxplot([cv_scores, cv_scores_best], labels=['原始模型', '优化模型'])
plt.ylabel('AUC分数')
plt.title('交叉验证性能对比')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plots", "cv_performance.eps"), dpi=300, bbox_inches='tight')
plt.close()

print("已保存可视化图片到 plots/ 文件夹")

# 模型保存
import joblib
model_output_path = os.path.join(script_dir, 'random_forest_model.pkl')
joblib.dump(best_pipeline, model_output_path)

feature_output_path = os.path.join(script_dir, 'feature_importance.csv')
feature_importance.to_csv(feature_output_path, index=False, encoding='utf-8-sig')

print(f"\n模型总结:")
print(f"使用特征: {len(available_features)} 个")
print(f"训练样本: {len(X_train)} 条")
print(f"测试样本: {len(X_test)} 条")
print(f"测试集AUC: {best_auc:.3f}")
print(f"测试集准确率: {best_accuracy:.3f}")
print(f"交叉验证AUC: {cv_scores_best.mean():.3f} ± {cv_scores_best.std():.3f}")

print(f"\n最重要的特征:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

print(f"\n已保存模型和特征重要性文件")
print(f"已保存可视化图片到 plots/ 文件夹")
