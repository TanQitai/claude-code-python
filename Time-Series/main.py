"""
时间序列分析主程序
整合数据加载、模型训练和结果可视化
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import TimeSeriesDataLoader, create_sample_data
from models import ModelTrainer, create_model
from visualization import TimeSeriesVisualizer, create_summary_report

class TimeSeriesPipeline:
    """时间序列分析管道"""
    
    def __init__(self, config=None):
        """
        初始化管道
        
        Args:
            config: 配置字典
        """
        self.config = config or self._get_default_config()
        self.data_loader = TimeSeriesDataLoader()
        self.model_trainer = ModelTrainer()
        self.visualizer = TimeSeriesVisualizer()
        
        # 存储结果
        self.data = None
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
        self.predictions = {}
        self.results_summary = None
        
    def _get_default_config(self):
        """获取默认配置"""
        return {
            # 数据配置
            'data': {
                'file_path': 'sample_data.csv',
                'date_column': 'date',
                'target_column': 'value',
                'missing_value_method': 'interpolate',
                'outlier_method': 'clip',
                'scale_method': 'minmax'
            },
            # 特征工程配置
            'features': {
                'lag_orders': [1, 3, 7, 14],
                'rolling_windows': [7, 14, 30],
                'rolling_statistics': ['mean', 'std'],
                'create_time_features': True
            },
            # 数据划分配置
            'split': {
                'train_ratio': 0.7,
                'valid_ratio': 0.2,
                'time_based': True
            },
            # 模型配置
            'models': {
                'MA': {'window_size': 30},
                'ES': {'alpha': 0.3},
                'AR': {'lag_order': 7},
                'ARIMA': {'order': (1, 1, 1)},
                'RF': {'n_estimators': 100},
                'GB': {'n_estimators': 100},
                'LSTM': {'hidden_size': 50, 'num_epochs': 50}
            },
            # 可视化配置
            'visualization': {
                'save_plots': True,
                'output_dir': 'output_plots/',
                'plot_format': 'png'
            }
        }
    
    def load_and_preprocess_data(self, file_path=None):
        """
        加载和预处理数据
        
        Args:
            file_path: 数据文件路径，如果为None则使用配置中的路径
        """
        print("=== 数据加载和预处理 ===")
        
        # 使用指定的文件路径或配置中的路径
        if file_path is not None:
            self.config['data']['file_path'] = file_path
        
        data_config = self.config['data']
        
        # 加载数据
        self.data = self.data_loader.load_data(
            data_config['file_path'],
            date_column=data_config.get('date_column'),
            target_column=data_config.get('target_column')
        )
        
        if self.data is None:
            print("数据加载失败")
            return False
        
        # 处理缺失值
        self.data = self.data_loader.handle_missing_values(
            method=data_config['missing_value_method']
        )
        
        if self.data is None:
            print("缺失值处理失败")
            return False
        
        # 处理异常值
        self.data = self.data_loader.handle_outliers(
            method=data_config['outlier_method']
        )
        
        # 创建特征
        success = self._create_features()
        if not success:
            return False
        
        # 数据标准化
        if data_config.get('scale_method'):
            scaled_data = self.data_loader.scale_data(
                method=data_config['scale_method']
            )
            if scaled_data is not None:
                print("数据标准化完成")
        
        # 划分数据集
        success = self._split_data()
        if not success:
            return False
        
        print("数据加载和预处理完成")
        return True
    
    def _create_features(self):
        """创建特征"""
        print("创建特征...")
        
        features_config = self.config['features']
        target_column = self.config['data']['target_column']
        
        try:
            # 创建滞后特征
            if features_config.get('lag_orders'):
                self.data = self.data_loader.create_lag_features(
                    target_column, features_config['lag_orders']
                )
            
            # 创建滚动特征
            if features_config.get('rolling_windows'):
                self.data = self.data_loader.create_rolling_features(
                    target_column, 
                    features_config['rolling_windows'],
                    features_config.get('rolling_statistics', ['mean'])
                )
            
            # 创建时间特征
            if features_config.get('create_time_features', False):
                self.data = self.data_loader.create_time_features()
            
            # 删除包含NaN的行
            self.data = self.data.dropna()
            
            print(f"特征创建完成，数据形状: {self.data.shape}")
            return True
            
        except Exception as e:
            print(f"特征创建失败: {str(e)}")
            return False
    
    def _split_data(self):
        """划分数据集"""
        print("划分数据集...")
        
        split_config = self.config['split']
        target_column = self.config['data']['target_column']
        
        try:
            split_result = self.data_loader.split_data(
                target_column,
                train_ratio=split_config['train_ratio'],
                valid_ratio=split_config['valid_ratio'],
                time_based=split_config.get('time_based', True)
            )
            
            if split_result is None:
                return False
            
            self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = split_result
            
            print(f"数据集划分完成:")
            print(f"  训练集: {len(self.X_train)} 样本")
            print(f"  验证集: {len(self.X_valid)} 样本")
            print(f"  测试集: {len(self.X_test)} 样本")
            
            return True
            
        except Exception as e:
            print(f"数据集划分失败: {str(e)}")
            return False
    
    def train_models(self, model_names=None):
        """
        训练模型
        
        Args:
            model_names: 要训练的模型名称列表，如果为None则训练所有配置的模型
        """
        print("\n=== 模型训练 ===")
        
        if self.X_train is None or self.y_train is None:
            print("请先加载和预处理数据")
            return False
        
        models_config = self.config['models']
        
        # 如果指定了模型名称，只训练这些模型
        if model_names is not None:
            models_to_train = {name: config for name, config in models_config.items() 
                             if name in model_names}
        else:
            models_to_train = models_config
        
        # 添加模型到训练器
        for model_name, model_config in models_to_train.items():
            try:
                model = create_model(model_name, **model_config)
                self.model_trainer.add_model(model, name=model_name)
                print(f"添加模型: {model_name}")
            except Exception as e:
                print(f"添加模型 {model_name} 失败: {str(e)}")
                continue
        
        # 训练模型
        try:
            self.model_trainer.train_models(
                self.X_train, self.y_train, 
                self.X_valid, self.y_valid
            )
            print("模型训练完成")
            return True
            
        except Exception as e:
            print(f"模型训练失败: {str(e)}")
            return False
    
    def make_predictions(self):
        """进行预测"""
        print("\n=== 模型预测 ===")
        
        if self.X_test is None:
            print("测试数据不存在")
            return False
        
        try:
            # 在测试集上进行预测
            self.predictions = self.model_trainer.predict_all(self.X_test)
            
            if not self.predictions:
                print("没有模型可以预测")
                return False
            
            print("预测完成，模型预测结果:")
            for model_name in self.predictions.keys():
                pred_length = len(self.predictions[model_name])
                print(f"  {model_name}: {pred_length} 个预测值")
            
            return True
            
        except Exception as e:
            print(f"预测失败: {str(e)}")
            return False
    
    def evaluate_models(self):
        """评估模型"""
        print("\n=== 模型评估 ===")
        
        if not self.predictions or self.y_test is None:
            print("没有预测结果或测试数据")
            return False
        
        try:
            # 评估每个模型的预测结果
            evaluation_results = {}
            
            for model_name, predictions in self.predictions.items():
                if len(predictions) == len(self.y_test):
                    model = self.model_trainer.models[model_name]
                    metrics = model.evaluate(self.y_test, predictions)
                    evaluation_results[model_name] = metrics
                    
                    print(f"\n{model_name} 测试集评估结果:")
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value:.4f}")
            
            # 创建结果汇总表
            if evaluation_results:
                self.results_summary = pd.DataFrame(evaluation_results).T
                print(f"\n模型评估汇总:")
                print(self.results_summary.round(4))
            
            return True
            
        except Exception as e:
            print(f"模型评估失败: {str(e)}")
            return False
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("\n=== 创建可视化图表 ===")
        
        vis_config = self.config['visualization']
        output_dir = vis_config['output_dir']
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 1. 原始数据图
            print("创建原始数据图...")
            target_column = self.config['data']['target_column']
            if target_column in self.data.columns:
                self.visualizer.plot_time_series(
                    self.data[target_column],
                    title="原始时间序列数据",
                    save_path=f"{output_dir}original_data.png" if vis_config['save_plots'] else None
                )
            
            # 2. 预测结果对比图
            print("创建预测结果对比图...")
            if self.predictions and self.y_test is not None:
                actual_series = pd.Series(self.y_test.values, 
                                        index=self.y_test.index if hasattr(self.y_test, 'index') 
                                        else range(len(self.y_test)))
                
                self.visualizer.plot_predictions_comparison(
                    actual_series, self.predictions,
                    title="模型预测结果对比",
                    save_path=f"{output_dir}predictions_comparison.png" if vis_config['save_plots'] else None
                )
            
            # 3. 模型性能汇总图
            print("创建模型性能汇总图...")
            if self.results_summary is not None:
                # 重置索引以便绘图
                results_df = self.results_summary.reset_index()
                results_df.rename(columns={'index': 'model'}, inplace=True)
                
                self.visualizer.plot_model_performance_summary(
                    results_df,
                    title="模型性能对比",
                    save_path=f"{output_dir}model_performance.png" if vis_config['save_plots'] else None
                )
            
            # 4. 残差分析图
            print("创建残差分析图...")
            if self.predictions and self.y_test is not None:
                for model_name, predictions in self.predictions.items():
                    if len(predictions) == len(self.y_test):
                        actual_series = pd.Series(self.y_test.values, 
                                                index=self.y_test.index if hasattr(self.y_test, 'index') 
                                                else range(len(self.y_test)))
                        
                        self.visualizer.plot_residuals_analysis(
                            actual_series, predictions, model_name,
                            save_path=f"{output_dir}residuals_{model_name}.png" if vis_config['save_plots'] else None
                        )
            
            # 5. 特征重要性图（仅适用于树模型）
            print("创建特征重要性图...")
            for model_name, model in self.model_trainer.models.items():
                if hasattr(model, 'get_feature_importance'):
                    importance_df = model.get_feature_importance(self.X_train.columns)
                    if importance_df is not None:
                        self.visualizer.plot_feature_importance(
                            importance_df,
                            title=f"{model_name} 特征重要性",
                            save_path=f"{output_dir}feature_importance_{model_name}.png" if vis_config['save_plots'] else None
                        )
            
            print(f"可视化图表创建完成，保存到 {output_dir}")
            return True
            
        except Exception as e:
            print(f"创建可视化图表失败: {str(e)}")
            return False
    
    def get_best_model(self, metric='RMSE'):
        """获取最佳模型"""
        best_model, best_score = self.model_trainer.get_best_model(metric=metric)
        
        if best_model:
            print(f"\n基于 {metric} 的最佳模型: {best_model} (得分: {best_score:.4f})")
            return best_model, best_score
        else:
            print("无法确定最佳模型")
            return None, None
    
    def get_results_summary(self):
        """获取结果汇总"""
        return self.model_trainer.get_results_summary()
    
    def run_complete_analysis(self, file_path=None, model_names=None):
        """
        运行完整的时间序列分析流程
        
        Args:
            file_path: 数据文件路径
            model_names: 要训练的模型名称列表
        """
        print("=" * 50)
        print("开始完整的时间序列分析")
        print("=" * 50)
        
        # 1. 加载和预处理数据
        success = self.load_and_preprocess_data(file_path)
        if not success:
            print("数据加载和预处理失败，分析终止")
            return False
        
        # 2. 训练模型
        success = self.train_models(model_names)
        if not success:
            print("模型训练失败，分析终止")
            return False
        
        # 3. 进行预测
        success = self.make_predictions()
        if not success:
            print("模型预测失败，分析终止")
            return False
        
        # 4. 评估模型
        success = self.evaluate_models()
        if not success:
            print("模型评估失败，分析终止")
            return False
        
        # 5. 创建可视化图表
        success = self.create_visualizations()
        if not success:
            print("可视化创建失败，但分析已完成")
        
        # 6. 获取最佳模型
        best_model, best_score = self.get_best_model()
        
        print("\n" + "=" * 50)
        print("时间序列分析完成！")
        print("=" * 50)
        
        return True
    
    def generate_report(self):
        """生成分析报告"""
        print("\n=== 生成分析报告 ===")
        
        if self.results_summary is None:
            print("没有分析结果可以生成报告")
            return
        
        report = []
        report.append("时间序列分析报告")
        report.append("=" * 30)
        
        # 数据信息
        if self.data is not None:
            report.append(f"\n1. 数据信息:")
            report.append(f"   - 数据形状: {self.data.shape}")
            report.append(f"   - 时间范围: {self.data.index.min()} 到 {self.data.index.max()}")
        
        # 模型评估结果
        if self.results_summary is not None:
            report.append(f"\n2. 模型评估结果:")
            report.append(self.results_summary.round(4).to_string())
        
        # 最佳模型
        best_model, best_score = self.get_best_model()
        if best_model:
            report.append(f"\n3. 最佳模型:")
            report.append(f"   - 模型名称: {best_model}")
            report.append(f"   - RMSE得分: {best_score:.4f}")
        
        # 保存报告
        output_dir = self.config['visualization']['output_dir']
        report_path = f"{output_dir}analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"分析报告已保存到: {report_path}")
        print("\n报告内容:")
        print('\n'.join(report))

def main():
    """主函数"""
    print("=== 时间序列分析系统 ===")
    
    # 创建时间序列分析管道
    pipeline = TimeSeriesPipeline()
    
    # 选项1：使用示例数据
    print("\n选项1：创建示例数据")
    sample_data = create_sample_data(
        start_date='2020-01-01',
        end_date='2023-12-31',
        freq='D',
        save_path='Time-Series/sample_data.csv'
    )
    
    # 运行完整分析
    success = pipeline.run_complete_analysis(
        file_path='Time-Series/sample_data.csv',
        model_names=['MA', 'ES', 'AR', 'RF', 'GB']  # 可以根据需要选择模型
    )
    
    if success:
        # 生成详细报告
        pipeline.generate_report()
        
        print("\n您也可以尝试:")
        print("1. 使用自己的数据文件替换 'sample_data.csv'")
        print("2. 修改配置文件以调整模型参数")
        print("3. 尝试其他模型组合")
        print("4. 查看 output_plots/ 目录中的可视化结果")
    else:
        print("分析过程中出现错误，请检查错误信息")

if __name__ == "__main__":
    main()