#!/usr/bin/env python3
"""
时间序列分析主程序
包含完整的数据加载、预处理、模型训练、预测和可视化流程
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_preprocessing import TimeSeriesPreprocessor, generate_sample_data
from models import TimeSeriesModels
from visualization import TimeSeriesVisualizer

def create_sample_dataset():
    """创建示例数据集"""
    print("=" * 50)
    print("创建示例数据集")
    print("=" * 50)
    
    # 生成示例数据
    np.random.seed(42)
    
    # 生成日期范围
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    n_points = len(dates)
    
    # 生成具有趋势、季节性和噪声的时间序列数据
    trend = np.linspace(100, 200, n_points)  # 趋势
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_points) / 365.25)  # 年度季节性
    weekly_seasonal = 5 * np.sin(2 * np.pi * np.arange(n_points) / 7)  # 周季节性
    noise = np.random.normal(0, 3, n_points)  # 噪声
    
    # 主时间序列
    main_series = trend + seasonal + weekly_seasonal + noise
    
    # 相关特征
    temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(n_points) / 365.25) + np.random.normal(0, 2, n_points)
    humidity = 60 + 15 * np.sin(2 * np.pi * np.arange(n_points) / 365.25) + np.random.normal(0, 3, n_points)
    pressure = 1013 + 5 * np.sin(2 * np.pi * np.arange(n_points) / 365.25) + np.random.normal(0, 1, n_points)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'main_series': main_series,
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure
    })
    
    # 保存数据
    df.to_csv('sample_dataset.csv', index=False)
    print(f"示例数据已创建，包含 {len(df)} 条记录")
    print(f"数据范围: {dates[0]} 到 {dates[-1]}")
    print(f"数据已保存到: sample_dataset.csv")
    
    return df

def run_complete_analysis():
    """运行完整的时间序列分析流程"""
    
    print("\n" + "=" * 60)
    print("时间序列分析完整流程")
    print("=" * 60 + "\n")
    
    # 步骤1: 数据加载和预处理
    print("步骤1: 数据加载和预处理")
    print("-" * 40)
    
    # 创建或加载数据
    if not os.path.exists('sample_dataset.csv'):
        data = create_sample_dataset()
    else:
        data = pd.read_csv('sample_dataset.csv')
        print(f"已加载现有数据，包含 {len(data)} 条记录")
    
    # 初始化预处理器
    preprocessor = TimeSeriesPreprocessor()
    
    # 加载数据
    processed_data = preprocessor.load_data('sample_dataset.csv', 'date', 'main_series')
    
    # 检查缺失值
    missing_info = preprocessor.check_missing_values()
    
    # 填充缺失值（如果有）
    if missing_info is not None and (missing_info['缺失数量'] > 0).any():
        processed_data = preprocessor.fill_missing_values('interpolate')
    
    # 创建特征
    data_with_features = preprocessor.create_features(window_size=30)
    
    # 标准化数据
    normalized_data = preprocessor.normalize_data(data_with_features, method='standard', fit_scaler=True)
    
    print(f"预处理完成，数据形状: {normalized_data.shape}")
    
    # 步骤2: 数据分割
    print("\n步骤2: 数据分割")
    print("-" * 40)
    
    # 时间序列分割
    train_data, test_data = preprocessor.train_test_split_time_series(normalized_data, test_size=0.2)
    
    # 准备特征和标签
    feature_cols = [col for col in normalized_data.columns if col != 'main_series']
    X_train = train_data[feature_cols]
    y_train = train_data['main_series']
    X_test = test_data[feature_cols]
    y_test = test_data['main_series']
    
    print(f"训练集特征形状: {X_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")
    
    # 步骤3: 可视化数据
    print("\n步骤3: 数据可视化")
    print("-" * 40)
    
    visualizer = TimeSeriesVisualizer()
    
    # 绘制原始时间序列
    original_data = preprocessor.data
    visualizer.plot_time_series(original_data, ['main_series'], 
                               title="原始时间序列数据", 
                               save_path="Time-Series/original_series.png")
    
    # 绘制相关性矩阵
    visualizer.plot_correlation_matrix(normalized_data, 
                                      save_path="Time-Series/correlation_matrix.png")
    
    # 步骤4: 模型训练
    print("\n步骤4: 模型训练")
    print("-" * 40)
    
    models = TimeSeriesModels()
    
    # 训练机器学习模型
    print("\n训练机器学习模型...")
    models.train_linear_regression(X_train, y_train, X_test, y_test)
    models.train_ridge_regression(X_train, y_train, X_test, y_test)
    models.train_random_forest(X_train, y_train, X_test, y_test)
    models.train_gradient_boosting(X_train, y_train, X_test, y_test)
    
    # 训练传统时间序列模型（使用原始数据）
    print("\n训练传统时间序列模型...")
    train_size = int(len(original_data) * 0.8)
    train_series = original_data['main_series'][:train_size]
    test_series = original_data['main_series'][train_size:]
    
    models.train_arima(train_series, test_series, order=(1,1,1))
    models.train_sarima(train_series, test_series, order=(1,1,1), seasonal_order=(1,1,1,365))
    models.train_exponential_smoothing(train_series, test_series, trend='add', seasonal='add', seasonal_periods=365)
    
    # 训练深度学习模型（创建序列数据）
    print("\n训练深度学习模型...")
    sequence_length = 60
    X_seq, y_seq = preprocessor.create_sequences(normalized_data, 'main_series', sequence_length)
    
    # 分割序列数据
    train_size_seq = int(len(X_seq) * 0.8)
    X_train_seq, X_test_seq = X_seq[:train_size_seq], X_seq[train_size_seq:]
    y_train_seq, y_test_seq = y_seq[:train_size_seq], y_seq[train_size_seq:]
    
    models.train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq, epochs=50, batch_size=32)
    models.train_gru(X_train_seq, y_train_seq, X_test_seq, y_test_seq, epochs=50, batch_size=32)
    
    # 步骤5: 模型评估和比较
    print("\n步骤5: 模型评估和比较")
    print("-" * 40)
    
    # 比较所有模型
    comparison = models.compare_models()
    
    # 获取最佳模型
    best_model_name, best_model_result = models.get_best_model(metric='R2')
    
    # 步骤6: 可视化结果
    print("\n步骤6: 结果可视化")
    print("-" * 40)
    
    # 准备预测结果用于可视化
    predictions_dict = {}
    for model_name, result in models.results.items():
        predictions_dict[model_name] = result['predictions']
    
    # 绘制预测结果对比
    visualizer.plot_predictions_vs_actual(y_test, 
                                         {k: v for k, v in predictions_dict.items() 
                                          if k in ['linear_regression', 'ridge_regression', 'random_forest', 'gradient_boosting']},
                                         title="机器学习模型预测结果对比",
                                         save_path="Time-Series/ml_predictions_comparison.png")
    
    # 绘制时间序列预测结果
    visualizer.plot_time_series_prediction(train_data['main_series'], 
                                          test_data['main_series'],
                                          {k: v for k, v in predictions_dict.items() 
                                           if k in ['linear_regression', 'random_forest', 'gradient_boosting']},
                                          title="时间序列预测结果对比",
                                          save_path="Time-Series/time_series_predictions.png")
    
    # 绘制残差分析
    visualizer.plot_residuals(y_test, 
                             {k: v for k, v in predictions_dict.items() 
                              if k in ['linear_regression', 'random_forest', 'gradient_boosting']},
                             title="残差分析",
                             save_path="Time-Series/residuals_analysis.png")
    
    # 绘制模型性能对比
    metrics_dict = {name: result['metrics'] for name, result in models.results.items()}
    visualizer.plot_model_comparison(metrics_dict, 
                                    title="模型性能对比",
                                    save_path="Time-Series/model_comparison.png")
    
    # 特征重要性（随机森林）
    if 'random_forest' in models.trained_models:
        visualizer.plot_feature_importance(models.trained_models['random_forest'],
                                          feature_cols,
                                          title="随机森林特征重要性",
                                          save_path="Time-Series/feature_importance.png")
    
    # 步骤7: 保存结果
    print("\n步骤7: 保存分析结果")
    print("-" * 40)
    
    # 保存模型比较结果
    comparison.to_csv('Time-Series/model_comparison_results.csv')
    print(f"模型比较结果已保存到: Time-Series/model_comparison_results.csv")
    
    # 保存最佳模型信息
    best_model_info = {
        'best_model': best_model_name,
        'metrics': best_model_result['metrics'],
        'all_models_comparison': metrics_dict
    }
    
    import json
    with open('Time-Series/best_model_info.json', 'w', encoding='utf-8') as f:
        json.dump(best_model_info, f, ensure_ascii=False, indent=2)
    
    print(f"最佳模型信息已保存到: Time-Series/best_model_info.json")
    
    # 最终总结
    print("\n" + "=" * 60)
    print("时间序列分析完成总结")
    print("=" * 60)
    print(f"最佳模型: {best_model_name}")
    print(f"最佳模型R²分数: {best_model_result['metrics']['R2']:.4f}")
    print(f"最佳模型RMSE: {best_model_result['metrics']['RMSE']:.4f}")
    print(f"所有图表已保存到 Time-Series 文件夹")
    print(f"模型比较结果已保存到 model_comparison_results.csv")
    print(f"最佳模型信息已保存到 best_model_info.json")
    print("=" * 60)

def run_quick_demo():
    """快速演示模式"""
    print("\n" + "=" * 50)
    print("快速演示模式")
    print("=" * 50)
    
    # 生成简单数据
    data = generate_sample_data(periods=500, freq='D')
    data.to_csv('demo_data.csv', index=False)
    
    # 初始化组件
    preprocessor = TimeSeriesPreprocessor()
    models = TimeSeriesModels()
    visualizer = TimeSeriesVisualizer()
    
    # 加载和预处理数据
    processed_data = preprocessor.load_data('demo_data.csv', 'date', 'value')
    data_with_features = preprocessor.create_features(window_size=7)
    
    # 数据分割
    train_data, test_data = preprocessor.train_test_split_time_series(data_with_features, test_size=0.2)
    
    feature_cols = [col for col in data_with_features.columns if col != 'value']
    X_train = train_data[feature_cols]
    y_train = train_data['value']
    X_test = test_data[feature_cols]
    y_test = test_data['value']
    
    # 训练几个简单模型
    print("训练模型...")
    models.train_linear_regression(X_train, y_train, X_test, y_test)
    models.train_random_forest(X_train, y_train, X_test, y_test)
    
    # 可视化结果
    predictions_dict = {
        'Linear Regression': models.results['linear_regression']['predictions'],
        'Random Forest': models.results['random_forest']['predictions']
    }
    
    visualizer.plot_predictions_vs_actual(y_test, predictions_dict)
    
    # 比较模型
    comparison = models.compare_models()
    
    print("\n演示完成!")
    print("可用文件: demo_data.csv")

def print_usage():
    """打印使用说明"""
    print("""
时间序列分析工具使用说明:

1. 完整分析模式:
   python main.py complete
   
   运行完整的分析流程，包括:
   - 创建示例数据
   - 数据预处理
   - 多种模型训练
   - 结果可视化
   - 模型比较

2. 快速演示模式:
   python main.py demo
   
   快速演示基本功能

3. 查看帮助:
   python main.py help

功能模块:
- data_preprocessing.py: 数据预处理
- models.py: 模型训练
- visualization.py: 结果可视化
""")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == 'complete':
            run_complete_analysis()
        elif command == 'demo':
            run_quick_demo()
        elif command == 'help':
            print_usage()
        else:
            print(f"未知命令: {command}")
            print_usage()
    else:
        print("请指定运行模式:")
        print_usage()
        print("\n默认运行快速演示模式...")
        run_quick_demo()

if __name__ == "__main__":
    main()