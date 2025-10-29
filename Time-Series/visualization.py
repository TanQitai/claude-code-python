"""
时间序列可视化模块
包含各种可视化函数和图表生成工具
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class TimeSeriesVisualizer:
    """时间序列可视化器"""
    
    def __init__(self, figsize=(12, 8), style='whitegrid'):
        self.figsize = figsize
        self.style = style
        sns.set_style(style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_time_series(self, data, columns=None, title="时间序列图", 
                        xlabel="时间", ylabel="数值", figsize=None, save_path=None):
        """
        绘制时间序列图
        
        Args:
            data: 时间序列数据 (pd.DataFrame 或 pd.Series)
            columns: 要绘制的列名列表
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            figsize: 图形大小
            save_path: 保存路径
        """
        if figsize is None:
            figsize = self.figsize
            
        plt.figure(figsize=figsize)
        
        if isinstance(data, pd.Series):
            plt.plot(data.index, data.values, linewidth=2, label=data.name)
        elif isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.columns
            
            for i, col in enumerate(columns):
                if col in data.columns:
                    color = self.colors[i % len(self.colors)]
                    plt.plot(data.index, data[col], linewidth=2, 
                           label=col, color=color)
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions_comparison(self, actual, predictions_dict, title="预测结果对比",
                                  figsize=(15, 8), save_path=None):
        """
        绘制多模型预测结果对比图
        
        Args:
            actual: 实际值
            predictions_dict: 预测结果字典 {模型名称: 预测值}
            title: 图表标题
            figsize: 图形大小
            save_path: 保存路径
        """
        plt.figure(figsize=figsize)
        
        # 绘制实际值
        plt.plot(actual.index, actual.values, label='实际值', 
                linewidth=3, color='black', alpha=0.8)
        
        # 绘制各模型预测值
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            color = self.colors[i % len(self.colors)]
            plt.plot(actual.index[:len(predictions)], predictions, 
                    label=f'{model_name}', linewidth=2, 
                    color=color, alpha=0.8, linestyle='--')
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('数值', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_residuals_analysis(self, actual, predicted, model_name="模型",
                              figsize=(15, 10), save_path=None):
        """
        绘制残差分析图
        
        Args:
            actual: 实际值
            predicted: 预测值
            model_name: 模型名称
            figsize: 图形大小
            save_path: 保存路径
        """
        residuals = actual - predicted
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig)
        
        # 1. 残差时间序列图
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(residuals, color='red', alpha=0.7, linewidth=1)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title(f'{model_name} - 残差时间序列图')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('残差')
        ax1.grid(True, alpha=0.3)
        
        # 2. 残差直方图
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(residuals, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_title(f'{model_name} - 残差直方图')
        ax2.set_xlabel('残差')
        ax2.set_ylabel('频次')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q图
        ax3 = fig.add_subplot(gs[0, 2])
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title(f'{model_name} - Q-Q图')
        ax3.grid(True, alpha=0.3)
        
        # 4. 残差自相关图
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_autocorrelation(residuals, ax=ax4, lags=20)
        ax4.set_title(f'{model_name} - 残差自相关图')
        
        # 5. 预测值 vs 实际值散点图
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(actual, predicted, alpha=0.6, color='blue')
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax5.set_xlabel('实际值')
        ax5.set_ylabel('预测值')
        ax5.set_title(f'{model_name} - 预测值 vs 实际值')
        ax5.grid(True, alpha=0.3)
        
        # 6. 残差平方图（检测异方差性）
        ax6 = fig.add_subplot(gs[1, 2])
        residuals_squared = residuals ** 2
        ax6.plot(residuals_squared, color='purple', alpha=0.7, linewidth=1)
        ax6.set_title(f'{model_name} - 残差平方图')
        ax6.set_xlabel('时间')
        ax6.set_ylabel('残差平方')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - 残差分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_seasonal_decomposition(self, data, period=12, model='additive',
                                  figsize=(15, 12), save_path=None):
        """
        绘制季节性分解图
        
        Args:
            data: 时间序列数据
            period: 季节周期
            model: 分解模型 ('additive', 'multiplicative')
            figsize: 图形大小
            save_path: 保存路径
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            if isinstance(data, pd.DataFrame):
                ts_data = data.iloc[:, 0]
            else:
                ts_data = data
            
            # 执行季节性分解
            decomposition = seasonal_decompose(ts_data, model=model, period=period)
            
            fig, axes = plt.subplots(4, 1, figsize=figsize)
            
            # 原始数据
            decomposition.observed.plot(ax=axes[0], title='原始时间序列', color='blue')
            axes[0].set_ylabel('数值')
            axes[0].grid(True, alpha=0.3)
            
            # 趋势成分
            decomposition.trend.plot(ax=axes[1], title='趋势成分', color='red')
            axes[1].set_ylabel('趋势')
            axes[1].grid(True, alpha=0.3)
            
            # 季节成分
            decomposition.seasonal.plot(ax=axes[2], title='季节成分', color='green')
            axes[2].set_ylabel('季节')
            axes[2].grid(True, alpha=0.3)
            
            # 残差成分
            decomposition.resid.plot(ax=axes[3], title='残差成分', color='purple')
            axes[3].set_ylabel('残差')
            axes[3].set_xlabel('时间')
            axes[3].grid(True, alpha=0.3)
            
            plt.suptitle(f'季节性分解 ({model}模型)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
            return decomposition
            
        except ImportError:
            print("请安装 statsmodels 库: pip install statsmodels")
            return None
        except Exception as e:
            print(f"季节性分解失败: {str(e)}")
            return None
    
    def plot_feature_importance(self, importance_df, title="特征重要性",
                              figsize=(10, 8), save_path=None):
        """
        绘制特征重要性图
        
        Args:
            importance_df: 特征重要性数据框 (feature, importance)
            title: 图表标题
            figsize: 图形大小
            save_path: 保存路径
        """
        plt.figure(figsize=figsize)
        
        # 按重要性排序
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        # 绘制水平条形图
        bars = plt.barh(range(len(importance_df)), importance_df['importance'], 
                       color='skyblue', alpha=0.8)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('重要性', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, data, title="相关系数矩阵", 
                              figsize=(10, 8), save_path=None):
        """
        绘制相关系数矩阵热力图
        
        Args:
            data: 数据框
            title: 图表标题
            figsize: 图形大小
            save_path: 保存路径
        """
        plt.figure(figsize=figsize)
        
        # 计算相关系数矩阵
        corr_matrix = data.corr()
        
        # 绘制热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_distribution_comparison(self, train_data, test_data, 
                                   title="训练集与测试集分布对比",
                                   figsize=(12, 8), save_path=None):
        """
        绘制训练集与测试集分布对比图
        
        Args:
            train_data: 训练数据
            test_data: 测试数据
            title: 图表标题
            figsize: 图形大小
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 分布直方图
        axes[0, 0].hist(train_data, bins=30, alpha=0.7, label='训练集', color='blue')
        axes[0, 0].hist(test_data, bins=30, alpha=0.7, label='测试集', color='red')
        axes[0, 0].set_title('分布直方图')
        axes[0, 0].set_xlabel('数值')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 箱线图
        data_to_plot = [train_data, test_data]
        box_plot = axes[0, 1].boxplot(data_to_plot, labels=['训练集', '测试集'], 
                                     patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        axes[0, 1].set_title('箱线图')
        axes[0, 1].set_ylabel('数值')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 密度图
        axes[1, 0].hist(train_data, bins=30, density=True, alpha=0.7, 
                       label='训练集', color='blue', histtype='step', linewidth=2)
        axes[1, 0].hist(test_data, bins=30, density=True, alpha=0.7, 
                       label='测试集', color='red', histtype='step', linewidth=2)
        axes[1, 0].set_title('密度图')
        axes[1, 0].set_xlabel('数值')
        axes[1, 0].set_ylabel('密度')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q图
        from scipy import stats
        stats.probplot(train_data, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('训练集Q-Q图')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_rolling_statistics(self, data, window=30, 
                              title="滚动统计量", figsize=(15, 10), save_path=None):
        """
        绘制滚动统计量图
        
        Args:
            data: 时间序列数据
            window: 滚动窗口大小
            title: 图表标题
            figsize: 图形大小
            save_path: 保存路径
        """
        if isinstance(data, pd.DataFrame):
            ts_data = data.iloc[:, 0]
        else:
            ts_data = data
        
        # 计算滚动统计量
        rolling_mean = ts_data.rolling(window=window).mean()
        rolling_std = ts_data.rolling(window=window).std()
        rolling_min = ts_data.rolling(window=window).min()
        rolling_max = ts_data.rolling(window=window).max()
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # 原始数据
        axes[0].plot(ts_data.index, ts_data.values, color='blue', alpha=0.7)
        axes[0].set_title('原始时间序列')
        axes[0].set_ylabel('数值')
        axes[0].grid(True, alpha=0.3)
        
        # 滚动均值
        axes[1].plot(ts_data.index, rolling_mean.values, color='red', linewidth=2)
        axes[1].set_title(f'滚动均值 (窗口: {window})')
        axes[1].set_ylabel('均值')
        axes[1].grid(True, alpha=0.3)
        
        # 滚动标准差
        axes[2].plot(ts_data.index, rolling_std.values, color='green', linewidth=2)
        axes[2].set_title(f'滚动标准差 (窗口: {window})')
        axes[2].set_ylabel('标准差')
        axes[2].grid(True, alpha=0.3)
        
        # 滚动最小值和最大值
        axes[3].plot(ts_data.index, rolling_min.values, color='orange', 
                    linewidth=2, label='滚动最小值')
        axes[3].plot(ts_data.index, rolling_max.values, color='purple', 
                    linewidth=2, label='滚动最大值')
        axes[3].fill_between(ts_data.index, rolling_min.values, rolling_max.values, 
                           alpha=0.3, color='gray')
        axes[3].set_title(f'滚动最小值和最大值 (窗口: {window})')
        axes[3].set_ylabel('数值')
        axes[3].set_xlabel('时间')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_acf_pacf(self, data, lags=20, title="自相关和偏自相关图",
                     figsize=(12, 6), save_path=None):
        """
        绘制ACF和PACF图
        
        Args:
            data: 时间序列数据
            lags: 滞后阶数
            title: 图表标题
            figsize: 图形大小
            save_path: 保存路径
        """
        try:
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # ACF图
            plot_acf(data, ax=axes[0], lags=lags)
            axes[0].set_title('自相关函数 (ACF)')
            axes[0].grid(True, alpha=0.3)
            
            # PACF图
            plot_pacf(data, ax=axes[1], lags=lags)
            axes[1].set_title('偏自相关函数 (PACF)')
            axes[1].grid(True, alpha=0.3)
            
            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            print("请安装 statsmodels 库: pip install statsmodels")
        except Exception as e:
            print(f"ACF/PACF绘图失败: {str(e)}")
    
    def _plot_autocorrelation(self, data, ax=None, lags=20):
        """绘制自相关图（内部函数）"""
        try:
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(data, ax=ax, lags=lags)
        except ImportError:
            # 如果没有statsmodels，使用简单实现
            autocorr = []
            for lag in range(lags + 1):
                if lag == 0:
                    autocorr.append(1.0)
                else:
                    corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                    autocorr.append(corr if not np.isnan(corr) else 0)
            
            ax.bar(range(lags + 1), autocorr, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('滞后')
            ax.set_ylabel('自相关系数')
    
    def plot_model_performance_summary(self, results_df, title="模型性能汇总",
                                     figsize=(12, 8), save_path=None):
        """
        绘制模型性能汇总图
        
        Args:
            results_df: 结果数据框
            title: 图表标题
            figsize: 图形大小
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
        
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            
            if metric in results_df.columns:
                ax = axes[row, col]
                
                # 条形图
                bars = ax.bar(results_df['model'], results_df[metric], 
                            color=self.colors[:len(results_df)], alpha=0.8)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
                
                ax.set_title(f'{metric} 对比')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_plots_to_pdf(self, plot_functions, filename='time_series_analysis.pdf'):
        """
        将多个图表保存到PDF文件
        
        Args:
            plot_functions: 绘图函数列表
            filename: PDF文件名
        """
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            
            with PdfPages(filename) as pdf:
                for plot_func in plot_functions:
                    fig = plot_func()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            
            print(f"图表已保存到 {filename}")
            
        except ImportError:
            print("请安装 matplotlib.backends.backend_pdf")
        except Exception as e:
            print(f"保存PDF失败: {str(e)}")

# 实用函数
def set_plot_style(style='whitegrid', font_scale=1.2):
    """设置绘图样式"""
    sns.set_style(style)
    sns.set(font_scale=font_scale)

def create_summary_report(data, predictions_dict, model_results, output_dir='plots/'):
    """
    创建可视化汇总报告
    
    Args:
        data: 原始数据
        predictions_dict: 预测结果字典
        model_results: 模型结果数据框
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = TimeSeriesVisualizer()
    
    # 1. 原始数据图
    visualizer.plot_time_series(data, save_path=f'{output_dir}original_data.png')
    
    # 2. 预测结果对比图
    if predictions_dict:
        actual = data.iloc[-len(list(predictions_dict.values())[0]):]
        visualizer.plot_predictions_comparison(
            actual, predictions_dict, 
            save_path=f'{output_dir}predictions_comparison.png'
        )
    
    # 3. 模型性能汇总图
    if model_results is not None:
        visualizer.plot_model_performance_summary(
            model_results, save_path=f'{output_dir}model_performance.png'
        )
    
    # 4. 相关性矩阵图
    if isinstance(data, pd.DataFrame) and data.shape[1] > 1:
        visualizer.plot_correlation_matrix(
            data, save_path=f'{output_dir}correlation_matrix.png'
        )
    
    print(f"可视化报告已保存到 {output_dir} 目录")

if __name__ == "__main__":
    # 测试可视化功能
    print("=== 测试时间序列可视化功能 ===")
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 365
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    trend = np.linspace(100, 200, n_samples)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
    noise = np.random.normal(0, 5, n_samples)
    values = trend + seasonal + noise
    
    data = pd.DataFrame({
        'value': values,
        'trend': trend,
        'seasonal': seasonal
    }, index=dates)
    
    # 创建可视化器
    visualizer = TimeSeriesVisualizer()
    
    # 测试各种绘图功能
    print("1. 绘制时间序列图...")
    visualizer.plot_time_series(data['value'], title="示例时间序列数据")
    
    print("2. 绘制多变量时间序列图...")
    visualizer.plot_time_series(data, columns=['value', 'trend'], 
                               title="多变量时间序列")
    
    print("3. 绘制滚动统计量图...")
    visualizer.plot_rolling_statistics(data['value'], window=30)
    
    print("4. 绘制相关性矩阵图...")
    visualizer.plot_correlation_matrix(data)
    
    print("5. 绘制预测结果对比图...")
    # 创建一些模拟预测结果
    predictions_dict = {
        '移动平均': values[-30:] + np.random.normal(0, 2, 30),
        '指数平滑': values[-30:] + np.random.normal(0, 3, 30)
    }
    actual = pd.Series(values[-30:], index=dates[-30:])
    visualizer.plot_predictions_comparison(actual, predictions_dict)
    
    print("6. 绘制残差分析图...")
    predicted = values[-30:] + np.random.normal(0, 2, 30)
    visualizer.plot_residuals_analysis(actual, predicted, "测试模型")
    
    print("\n=== 可视化功能测试完成 ===")