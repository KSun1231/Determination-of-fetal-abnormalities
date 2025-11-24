# 问题3 蒙特卡洛分析改进总结

## 🔧 **主要改进**

### 1. **数据使用改进**
- **之前**: 使用过滤后的达标数据（Y≥4%）
- **现在**: 使用全量清洗数据（未按Y≥4%过滤）
- **优势**: 更真实反映实际检测场景

### 2. **Y浓度预测模型**
- **之前**: 直接使用原始Y浓度加随机误差
- **现在**: 基于回归模型预测Y浓度
- **模型**: Y浓度 = const + β₁×孕周 + β₂×BMI + β₃×年龄 + ε
- **优势**: Y浓度与孕周/体征真正相关

### 3. **聚类特征一致性修复**
- **问题**: clustering.py和monte_carlo_analysis.py使用不同的特征顺序
- **修复**: 统一使用['年龄', '体重', '身高']顺序
- **结果**: 聚类标签一致性100%

### 4. **BMI处理优化**
- **问题**: 列名不匹配（'模拟BMI' vs '孕妇BMI'）
- **修复**: 正确处理BMI计算和误差传播
- **逻辑**: 
  - 如果有身高体重误差 → 重新计算BMI
  - 否则直接对BMI添加误差

## 📊 **数据文件结构**

### 问题2生成的数据文件：
```
问题2/clean_data_csv/
├── boys_clean_2_all.csv     # 全量数据（1027条，未按Y≥4%过滤）
├── boys_clean_2.csv         # 达标数据（260条，Y≥4%且最早达标）
├── girls_clean_2_all.csv    # 女胎全量数据
└── girls_clean_2.csv        # 女胎达标数据
```

### 问题3使用的数据文件：
```
问题3/
├── boys_clean_2_with_clusters.csv  # 带聚类结果的数据
├── regression_params.json          # 回归模型参数
└── 各种图片文件...
```

## 🔄 **数据加载优先级**

蒙特卡洛分析按以下顺序查找数据文件：
1. `../问题2/clean_data_csv/boys_clean_2_all.csv` （首选：全量数据）
2. `boys_clean_2_with_clusters.csv` （带聚类结果）
3. `boys_with_days.csv` （原始数据）
4. `../问题2/clean_data_csv/boys_clean_2.csv` （兜底：达标数据）

## 🧮 **蒙特卡洛模拟流程**

### 改进前：
1. 读取达标数据（Y≥4%）
2. 对原始Y浓度添加误差
3. 计算达标率（大部分都是100%）

### 改进后：
1. 读取全量数据（包含Y<4%的样本）
2. 在全量数据上拟合回归模型
3. 对每个目标孕周：
   - 生成模拟的孕周、BMI、年龄（带误差）
   - 使用回归模型预测Y浓度
   - 添加预测误差
   - 计算达标率（更真实的变化）

## 🎯 **使用方法**

### 完整流程：
```bash
# 1. 数据清洗（生成全量+达标两份数据）
cd 问题2
python data_cleaning_analysis_2.py

# 2. 聚类分析（使用达标数据）
cd ../问题3
python clustering.py

# 3. 蒙特卡洛分析（使用全量数据+回归预测）
python monte_carlo_analysis.py
```

### 单独运行蒙特卡洛分析：
```bash
cd 问题3
python monte_carlo_analysis.py
```

## 🔬 **技术细节**

### 回归模型建立：
```python
# 使用全量数据拟合
regression_data = boys_data.dropna(subset=['Y染色体浓度', '孕天', '孕妇BMI', '年龄'])
X = sm.add_constant(regression_data[['孕周', '孕妇BMI', '年龄']])
y = regression_data['Y染色体浓度']
model = sm.OLS(y, X).fit()
```

### Y浓度预测：
```python
def predict_y_concentration_vectorized(weeks, bmis, ages):
    X_sim = np.column_stack([np.ones(len(weeks)), weeks, bmis, ages])
    mu = X_sim @ coef
    y_pred = mu + np.random.normal(0, sigma, size=len(weeks))
    return np.clip(y_pred, 0, 1)
```

## ✅ **改进效果**

1. **更真实的模拟**：Y浓度与孕周真正相关
2. **更丰富的数据**：使用全量数据而非过滤数据
3. **更准确的聚类**：修复特征顺序一致性问题
4. **更稳定的代码**：修复列名和路径问题

## 🎉 **结论**

这些改进使得蒙特卡洛分析更加科学和真实，能够更好地反映实际的检测场景和Y浓度变化规律。
