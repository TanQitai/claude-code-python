import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class TimeSeriesVisualizer:
    """时间序列可视化类"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    def plot_time_series(self, data, columns=None, title="时间序列图", save_path=None):
        """
        绘制时间序列图
        
        Args:
            data: 时间序列数据 (DataFrame)
            columns: 要绘制的列名列表
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=self.figsize)
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        for i, col in enumerate(columns):
            plt.plot(data.index, data[col], label=col, color=self.colors[i % len(self.colors)], linewidth=2)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('值', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"时间序列图已保存到: {save_path}")
        
        plt.show()
    
    def plot_decomposition(self, data, column, model='additive', period=365, save_path=None):
        """
        绘制时间序列分解图
        
        Args:
            data: 时间序列数据
            column: 要分解的列名
            model: 分解模型 ('additive', 'multiplicative')
            period: 季节周期
            save_path: 保存路径
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            decomposition = seasonal_decompose(data[column], model=model, period=period)
            
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            # 原始数据
            decomposition.observed.plot(ax=axes[0], title='原始数据', color=self.colors[0])
            axes[0].set_ylabel('值')
            
            # 趋势
            decomposition.trend.plot(ax=axes[1], title='趋势', color=self.colors[1])
            axes[1].set_ylabel('趋势')
            
            # 季节性
            decomposition.seasonal.plot(ax=axes[2], title='季节性', color=self.colors[2])
            axes[2].set_ylabel('季节性')
            
            # 残差
            decomposition.resid.plot(ax=axes[3], title='残差', color=self.colors[3])
            axes[3].set_ylabel('残差')
            
            plt.suptitle(f'{column} 时间序列分解 ({model}模型)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"时间序列分解图已保存到: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("statsmodels 未安装，无法绘制分解图")
        except Exception as e:
            print(f"时间序列分解失败: {e}")
    
    def plot_correlation_matrix(self, data, save_path=None):
        """
        绘制相关性矩阵热力图
        
        Args:
            data: 数据
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        
        corr_matrix = data.select_dtypes(include=[np.number]).corr()
        
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8})
        
        plt.title('特征相关性矩阵', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相关性矩阵图已保存到: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, title="特征重要性", save_path=None):
        """
        绘制特征重要性图
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            print("模型没有特征重要性属性")
            return
        
        # 排序
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), importances[indices], color=self.colors[0])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征重要性图已保存到: {save_path}")
        
        plt.show()
    
    def plot_predictions_vs_actual(self, y_true, y_pred_dict, title="预测结果对比", save_path=None):
        """
        绘制预测值vs实际值对比图
        
        Args:
            y_true: 实际值
            y_pred_dict: 预测值字典 {模型名称: 预测值}
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(15, 10))
        
        # 创建子图
        n_models = len(y_pred_dict)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        for i, (model_name, y_pred) in enumerate(y_pred_dict.items()):
            plt.subplot(rows, cols, i + 1)
            
            # 散点图
            plt.scatter(y_true, y_pred, alpha=0.6, color=self.colors[i % len(self.colors)])
            
            # 对角线
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.title(f'{model_name}')
            plt.grid(True, alpha=0.3)
            
            # 添加R2分数
            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)
            plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测结果对比图已保存到: {save_path}")
        
        plt.show()
    
    def plot_time_series_prediction(self, train_data, test_data, predictions_dict, 
                                   title="时间序列预测结果", save_path=None):
        """
        绘制时间序列预测结果
        
        Args:
            train_data: 训练数据
            test_data: 测试数据 (实际值)
            predictions_dict: 预测值字典 {模型名称: 预测值}
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(15, 8))
        
        # 绘制训练数据
        plt.plot(train_data.index, train_data.values, label='训练数据', 
                color='blue', alpha=0.7, linewidth=2)
        
        # 绘制测试数据
        plt.plot(test_data.index, test_data.values, label='实际值', 
                color='black', linewidth=2)
        
        # 绘制预测值
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            plt.plot(test_data.index, predictions, 
                    label=f'{model_name} 预测', 
                    color=self.colors[i % len(self.colors)], 
                    linestyle='--', linewidth=2)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('时间')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"时间序列预测图已保存到: {save_path}")
        
        plt.show()
    
    def plot_residuals(self, y_true, y_pred_dict, title="残差分析", save_path=None):
        """
        绘制残差分析图
        
        Args:
            y_true: 实际值
            y_pred_dict: 预测值字典 {模型名称: 预测值}
            title: 图表标题
            save_path: 保存路径
        """
        n_models = len(y_pred_dict)
        fig, axes = plt.subplots(n_models, 2, figsize=(15, 5 * n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, y_pred) in enumerate(y_pred_dict.items()):
            residuals = y_true - y_pred
            
            # 残差时间序列图
            axes[i, 0].plot(residuals, color=self.colors[i % len(self.colors)], alpha=0.7)
            axes[i, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[i, 0].set_title(f'{model_name} - 残差时间序列')
            axes[i, 0].set_xlabel('样本')
            axes[i, 0].set_ylabel('残差')
            axes[i, 0].grid(True, alpha=0.3)
            
            # 残差直方图
            axes[i, 1].hist(residuals, bins=30, color=self.colors[i % len(self.colors)], 
                           alpha=0.7, edgecolor='black')
            axes[i, 1].set_title(f'{model_name} - 残差分布')
            axes[i, 1].set_xlabel('残差')
            axes[i, 1].set_ylabel('频率')
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"残差分析图已保存到: {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, metrics_dict, title="模型性能对比", save_path=None):
        """
        绘制模型性能对比图
        
        Args:
            metrics_dict: 指标字典 {模型名称: {指标名称: 指标值}}
            title: 图表标题
            save_path: 保存路径
        """
        # 转换为DataFrame
        comparison_df = pd.DataFrame(metrics_dict).T
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['R2', 'RMSE', 'MAE', 'MSE']
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                bars = axes[i].bar(comparison_df.index, comparison_df[metric], 
                                  color=self.colors[:len(comparison_df)])
                axes[i].set_title(f'{metric} 指标对比')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
                
                axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"模型性能对比图已保存到: {save_path}")
        
        plt.show()
    
    def plot_rolling_statistics(self, data, column, window=30, save_path=None):
        """
        绘制滚动统计图
        
        Args:
            data: 时间序列数据
            column: 要分析的列名
            window: 滚动窗口大小
            save_path: 保存路径
        """
        plt.figure(figsize=(15, 10))
        
        # 计算滚动统计量
        rolling_mean = data[column].rolling(window=window).mean()
        rolling_std = data[column].rolling(window=window).std()
        rolling_min = data[column].rolling(window=window).min()
        rolling_max = data[column].rolling(window=window).max()
        
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data[column], label='原始数据', alpha=0.7, color=self.colors[0])
        plt.plot(data.index, rolling_mean, label=f'{window}期滚动均值', 
                color=self.colors[1], linewidth=2)
        plt.fill_between(data.index, rolling_min, rolling_max, alpha=0.2, 
                        color=self.colors[2], label=f'{window}期滚动范围')
        plt.title(f'{column} - 滚动统计量')
        plt.xlabel('时间')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(data.index, rolling_std, label=f'{window}期滚动标准差', 
                color=self.colors[3], linewidth=2)
        plt.title(f'{column} - 滚动标准差')
        plt.xlabel('时间')
        plt.ylabel('标准差')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"滚动统计图已保存到: {save_path}")
        
        plt.show()
    
    def plot_acf_pacf(self, data, column, lags=40, save_path=None):
        """
        绘制ACF和PACF图
        
        Args:
            data: 时间序列数据
            column: 要分析的列名
            lags: 滞后阶数
            save_path: 保存路径
        """
        try:
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # ACF图
            plot_acf(data[column], lags=lags, ax=axes[0])
            axes[0].set_title(f'{column} - 自相关函数 (ACF)')
            
            # PACF图
            plot_pacf(data[column], lags=lags, ax=axes[1])
            axes[1].set_title(f'{column} - 偏自相关函数 (PACF)')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"ACF/PACF图已保存到: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("statsmodels 未安装，无法绘制ACF/PACF图")
        except Exception as e:
            print(f"ACF/PACF图绘制失败: {e}")

if __name__ == "__main__":
    print("时间序列可视化模块加载完成")
    print("可用图表类型:")
    print("- 时间序列图")
    print("- 时间序列分解图")
    print("- 相关性矩阵热力图")
    print("- 特征重要性图")
    print("- 预测结果对比图")
    print("- 时间序列预测结果图")
    print("- 残差分析图")
    print("- 模型性能对比图")
    print("- 滚动统计图")
    print("- ACF/PACF图")