# 时间序列分析工具包

这是一个功能完整的时间序列分析工具包，包含数据加载、预处理、模型训练、预测和可视化等功能。

## 🎯 功能特性

### 📊 数据处理
- **数据加载**: 支持CSV、Excel、Parquet格式
- **数据预处理**: 缺失值处理、异常值检测与处理、数据标准化
- **特征工程**: 滞后特征、滚动窗口特征、时间特征、循环编码
- **数据划分**: 时间序列友好的训练/验证/测试集划分

### 🤖 预测模型
#### 传统统计模型
- **移动平均 (MA)**: 简单移动平均模型
- **指数平滑 (ES)**: 指数平滑模型
- **自回归 (AR)**: 自回归模型
- **ARIMA**: 自回归积分滑动平均模型

#### 机器学习模型
- **随机森林**: 基于决策树的集成模型
- **梯度提升**: 梯度提升回归模型
- **线性模型**: 线性回归、岭回归、Lasso回归

#### 深度学习模型
- **LSTM**: 长短期记忆网络
- **GRU**: 门控循环单元

### 📈 可视化
- **时间序列图**: 原始数据和预测结果可视化
- **残差分析**: 残差时间序列、直方图、Q-Q图、自相关图
- **季节性分解**: 趋势、季节性、残差成分分解
- **特征重要性**: 模型特征重要性可视化
- **模型对比**: 多模型性能对比图
- **相关性分析**: 相关系数矩阵热力图

### ⚙️ 配置管理
- **集中配置**: 统一的配置文件管理
- **模型选择**: 灵活选择要训练的模型
- **参数调整**: 易于修改模型参数
- **批量处理**: 支持批量模型训练和评估

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖库
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels torch scipy

# 或者使用requirements.txt文件
pip install -r requirements.txt
```

### 2. 基本使用

```python
from main import TimeSeriesPipeline

# 创建分析管道
pipeline = TimeSeriesPipeline()

# 运行完整分析
pipeline.run_complete_analysis(
    file_path='your_data.csv',
    model_names=['MA', 'ES', 'AR', 'RF']  # 选择要使用的模型
)

# 获取最佳模型
best_model, best_score = pipeline.get_best_model()
print(f"最佳模型: {best_model}, RMSE: {best_score:.4f}")
```

### 3. 模块单独使用

#### 数据加载和预处理

```python
from data_loader import TimeSeriesDataLoader

# 创建数据加载器
loader = TimeSeriesDataLoader()

# 加载数据
data = loader.load_data('data.csv', date_column='date', target_column='value')

# 数据预处理
data = loader.handle_missing_values(method='interpolate')
data = loader.handle_outliers(method='clip')

# 创建特征
data = loader.create_lag_features('value', lag_orders=[1, 3, 7])
data = loader.create_rolling_features('value', windows=[7, 30])
data = loader.create_time_features()

# 划分数据集
X_train, X_valid, X_test, y_train, y_valid, y_test = loader.split_data('value')
```

#### 模型训练和预测

```python
from models import ModelTrainer, create_model

# 创建模型训练器
trainer = ModelTrainer()

# 添加模型
trainer.add_model(create_model('MA', window_size=30))
trainer.add_model(create_model('ES', alpha=0.3))
trainer.add_model(create_model('RF', n_estimators=100))

# 训练模型
trainer.train_models(X_train, y_train, X_valid, y_valid)

# 进行预测
predictions = trainer.predict_all(X_test)

# 获取结果汇总
results = trainer.get_results_summary()
```

#### 可视化

```python
from visualization import TimeSeriesVisualizer

# 创建可视化器
visualizer = TimeSeriesVisualizer()

# 绘制时间序列图
visualizer.plot_time_series(data)

# 绘制预测结果对比
visualizer.plot_predictions_comparison(actual, predictions_dict)

# 绘制残差分析
visualizer.plot_residuals_analysis(actual, predicted, "模型名称")

# 季节性分解
visualizer.plot_seasonal_decomposition(data, period=12)
```

## 📁 文件结构

```
Time-Series/
├── data_loader.py          # 数据加载和预处理
├── models.py               # 预测模型
├── visualization.py        # 可视化工具
├── main.py                 # 主程序
├── config.py               # 配置文件
├── utils.py                # 工具函数
├── README.md               # 使用说明
└── requirements.txt        # 依赖库列表
```

## ⚙️ 配置说明

### 默认配置

系统提供了一套完整的默认配置，涵盖数据、模型、训练、可视化等各个方面。

```python
from config import Config

# 查看配置摘要
Config.print_config_summary()

# 创建配置文件
Config.create_default_config_file('my_config.json')

# 加载配置
config = Config.load_config_from_file('my_config.json')
```

### 主要配置项

#### 数据配置
- `date_column`: 日期列名
- `target_column`: 目标列名
- `missing_value_method`: 缺失值处理方法
- `outlier_method`: 异常值处理方法

#### 模型配置
- `models`: 启用的模型列表
- `model_parameters`: 模型特定参数
- `cross_validation`: 交叉验证设置
- `hyperparameter_tuning`: 超参数调优

#### 可视化配置
- `save_plots`: 是否保存图表
- `output_dir`: 输出目录
- `plot_formats`: 图表格式

## 📊 示例数据

系统提供示例数据生成功能：

```python
from data_loader import create_sample_data

# 创建示例数据
sample_data = create_sample_data(
    start_date='2020-01-01',
    end_date='2023-12-31',
    freq='D',
    save_path='sample_data.csv'
)
```

## 🔧 高级功能

### 自定义模型

```python
from models import BaseTimeSeriesModel

class MyCustomModel(BaseTimeSeriesModel):
    def __init__(self, param1=1.0):
        super().__init__()
        self.param1 = param1
        self.name = f"Custom_{param1}"
    
    def fit(self, X, y):
        # 训练模型
        self.is_fitted = True
        return self
    
    def predict(self, X):
        # 进行预测
        if not self.is_fitted:
            print("模型尚未训练")
            return None
        # 返回预测结果
        return predictions
```

### 批量模型比较

```python
# 批量训练和比较多个模型
model_names = ['MA', 'ES', 'AR', 'ARIMA', 'RF', 'GB', 'LSTM']
pipeline.train_models(model_names=model_names)

# 获取结果汇总
results = pipeline.get_results_summary()
print(results)
```

### 自定义可视化

```python
from visualization import TimeSeriesVisualizer

# 创建自定义图表
visualizer = TimeSeriesVisualizer(figsize=(15, 10))

# 添加自定义样式
plt.style.use('seaborn-darkgrid')
```

## 📈 评估指标

系统支持多种评估指标：

- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **MAPE**: 平均绝对百分比误差
- **R²**: 决定系数
- **MASE**: 平均绝对缩放误差

## 🎨 可视化输出

系统生成多种可视化图表：

1. **原始数据图**: 时间序列趋势展示
2. **预测对比图**: 多模型预测结果对比
3. **残差分析图**: 模型残差诊断
4. **特征重要性图**: 模型特征重要性
5. **性能对比图**: 模型性能指标对比
6. **季节性分解图**: 时间序列成分分解

## 🔍 故障排除

### 常见问题

1. **依赖库安装失败**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **中文显示问题**
   ```python
   # 系统会自动尝试设置中文字体
   # 如果仍然有问题，可以手动设置
   import matplotlib.pyplot as plt
   plt.rcParams['font.sans-serif'] = ['SimHei']
   plt.rcParams['axes.unicode_minus'] = False
   ```

3. **内存不足**
   - 减少批量大小
   - 使用数据采样
   - 选择更简单的模型

4. **模型训练时间过长**
   - 减少模型数量
   - 使用较小的参数
   - 启用早停机制

### 性能优化

1. **数据预处理优化**
   - 使用向量化操作
   - 避免循环
   - 使用合适的数据类型

2. **模型训练优化**
   - 使用并行处理
   - 启用GPU加速（深度学习模型）
   - 使用早停机制

3. **内存优化**
   - 及时清理不需要的变量
   - 使用数据生成器
   - 分批处理大数据

## 🤝 贡献指南

欢迎贡献代码和改进建议！

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

MIT License

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件

## 🙏 致谢

感谢所有开源库的贡献者，包括但不限于：

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Statsmodels
- PyTorch

---

**祝您使用愉快！** 🎉