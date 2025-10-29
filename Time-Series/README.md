# 时间序列分析工具包

一个完整的Python时间序列分析工具包，包含数据加载、预处理、模型训练、预测和可视化功能。

## 功能特性

### 📊 数据预处理 (`data_preprocessing.py`)
- 支持CSV和Excel文件加载
- 缺失值检测和处理
- 时间序列特征工程（滞后特征、移动平均、时间特征）
- 数据标准化（StandardScaler, MinMaxScaler）
- 序列数据创建（用于深度学习）
- 时间序列数据分割

### 🤖 模型训练 (`models.py`)
包含多种机器学习、传统时间序列和深度学习模型：

#### 机器学习模型
- 线性回归 (Linear Regression)
- 岭回归 (Ridge Regression)
- 随机森林 (Random Forest)
- 梯度提升 (Gradient Boosting)

#### 传统时间序列模型
- ARIMA
- SARIMA
- 指数平滑 (Exponential Smoothing)

#### 深度学习模型
- LSTM (长短期记忆网络)
- GRU (门控循环单元)

### 📈 可视化 (`visualization.py`)
- 时间序列图
- 时间序列分解图
- 相关性矩阵热力图
- 特征重要性图
- 预测结果对比图
- 残差分析图
- 模型性能对比图
- 滚动统计图
- ACF/PACF图

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行演示

#### 快速演示模式
```bash
python main.py demo
```

#### 完整分析模式
```bash
python main.py complete
```

#### 查看帮助
```bash
python main.py help
```

### 3. 在自己的数据上使用

```python
from data_preprocessing import TimeSeriesPreprocessor
from models import TimeSeriesModels
from visualization import TimeSeriesVisualizer

# 初始化组件
preprocessor = TimeSeriesPreprocessor()
models = TimeSeriesModels()
visualizer = TimeSeriesVisualizer()

# 加载数据
data = preprocessor.load_data('your_data.csv', 'date_column', 'target_column')

# 数据预处理
processed_data = preprocessor.create_features(window_size=30)
normalized_data = preprocessor.normalize_data(processed_data)

# 数据分割
train_data, test_data = preprocessor.train_test_split_time_series(normalized_data, test_size=0.2)

# 准备特征和标签
feature_cols = [col for col in normalized_data.columns if col != 'target_column']
X_train = train_data[feature_cols]
y_train = train_data['target_column']
X_test = test_data[feature_cols]
y_test = test_data['target_column']

# 训练模型
models.train_random_forest(X_train, y_train, X_test, y_test)
models.train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq)

# 可视化结果
visualizer.plot_predictions_vs_actual(y_test, models.results['random_forest']['predictions'])
```

## 文件结构

```
Time-Series/
├── data_preprocessing.py      # 数据预处理模块
├── models.py                  # 模型训练模块
├── visualization.py           # 可视化模块
├── main.py                    # 主程序
├── requirements.txt           # 依赖包列表
└── README.md                  # 说明文档
```

## 示例输出

运行完整分析后，将生成以下文件：

### 图表文件
- `original_series.png` - 原始时间序列图
- `correlation_matrix.png` - 相关性矩阵热力图
- `ml_predictions_comparison.png` - 机器学习模型预测对比
- `time_series_predictions.png` - 时间序列预测结果
- `residuals_analysis.png` - 残差分析图
- `model_comparison.png` - 模型性能对比
- `feature_importance.png` - 特征重要性图

### 结果文件
- `model_comparison_results.csv` - 模型比较结果
- `best_model_info.json` - 最佳模型信息

## 高级用法

### 自定义模型参数

```python
# 自定义ARIMA参数
models.train_arima(train_data, test_data, order=(2,1,2))

# 自定义LSTM参数
models.train_lstm(X_train, y_train, X_test, y_test, epochs=200, batch_size=16)
```

### 批量模型训练

```python
# 定义要训练的模型列表
model_list = ['linear_regression', 'random_forest', 'lstm']

for model_name in model_list:
    if model_name == 'linear_regression':
        models.train_linear_regression(X_train, y_train, X_test, y_test)
    elif model_name == 'random_forest':
        models.train_random_forest(X_train, y_train, X_test, y_test)
    elif model_name == 'lstm':
        models.train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq)
```

### 自定义可视化

```python
# 自定义图表样式
visualizer = TimeSeriesVisualizer(figsize=(15, 10))
visualizer.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# 绘制特定图表
visualizer.plot_rolling_statistics(data, 'target_column', window=60)
visualizer.plot_acf_pacf(data, 'target_column', lags=50)
```

## 注意事项

1. **依赖安装**: 某些模型（如ARIMA、LSTM）需要额外的库，如果不想使用可以跳过相关安装
2. **数据格式**: 确保时间序列数据有正确的日期列和目标列
3. **内存使用**: 深度学习模型可能需要较多内存，对于大数据集建议使用较小的batch size
4. **训练时间**: 深度学习模型训练时间较长，可以通过减少epochs来加快训练

## 故障排除

### 常见问题

1. **中文显示问题**: 如果图表中文显示异常，请确保系统有中文字体
2. **内存不足**: 减少batch size或使用更小的数据集
3. **模型训练失败**: 检查数据格式和缺失值处理

### 性能优化

1. 对于大数据集，使用较小的窗口大小和序列长度
2. 深度学习模型可以先在小数据集上测试
3. 使用GPU加速深度学习模型训练（如果有的话）

## 扩展功能

这个工具包设计为模块化结构，易于扩展：

- 可以添加新的预处理功能到 `data_preprocessing.py`
- 可以添加新的模型到 `models.py`
- 可以添加新的可视化图表到 `visualization.py`

欢迎贡献代码和提出改进建议！

## 许可证

MIT License - 详见LICENSE文件