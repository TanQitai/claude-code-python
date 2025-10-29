"""
时间序列数据加载与预处理模块
包含数据加载、清洗、特征工程等功能
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataLoader:
    """时间序列数据加载器"""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.scaler = None
        self.feature_columns = []
        
    def load_data(self, file_path, date_column=None, target_column=None, 
                  file_type='auto', encoding='utf-8'):
        """
        加载时间序列数据
        
        Args:
            file_path: 数据文件路径
            date_column: 日期列名
            target_column: 目标列名
            file_type: 文件类型 ('csv', 'excel', 'auto')
            encoding: 文件编码
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            # 自动检测文件类型
            if file_type == 'auto':
                if file_path.endswith('.csv'):
                    file_type = 'csv'
                elif file_path.endswith(('.xlsx', '.xls')):
                    file_type = 'excel'
                else:
                    raise ValueError("无法识别的文件类型，请指定file_type参数")
            
            # 加载数据
            if file_type == 'csv':
                self.data = pd.read_csv(file_path, encoding=encoding)
            elif file_type == 'excel':
                self.data = pd.read_excel(file_path, encoding=encoding)
            
            print(f"数据加载成功，形状: {self.data.shape}")
            print(f"列名: {list(self.data.columns)}")
            
            # 处理日期列
            if date_column and date_column in self.data.columns:
                self.data[date_column] = pd.to_datetime(self.data[date_column])
                self.data.set_index(date_column, inplace=True)
                self.data.sort_index(inplace=True)
                print(f"时间范围: {self.data.index.min()} 到 {self.data.index.max()}")
            
            # 如果指定了目标列，确保其为数值类型
            if target_column and target_column in self.data.columns:
                self.data[target_column] = pd.to_numeric(self.data[target_column], errors='coerce')
            
            # 显示数据基本信息
            print(f"\n数据基本信息:")
            print(f"数据类型:\n{self.data.dtypes}")
            print(f"\n缺失值统计:\n{self.data.isnull().sum()}")
            print(f"\n数据描述统计:\n{self.data.describe()}")
            
            return self.data
            
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            return None
    
    def handle_missing_values(self, method='interpolate', fill_value=None):
        """
        处理缺失值
        
        Args:
            method: 处理方法 ('interpolate', 'ffill', 'bfill', 'drop', 'fill', 'mean', 'median')
            fill_value: 填充值（仅当method='fill'时有效）
        """
        if self.data is None:
            print("请先加载数据")
            return None
            
        print(f"使用 {method} 方法处理缺失值...")
        
        missing_before = self.data.isnull().sum().sum()
        
        if method == 'interpolate':
            # 时间序列插值
            self.data = self.data.interpolate(method='time')
        elif method == 'ffill':
            self.data = self.data.fillna(method='ffill')
        elif method == 'bfill':
            self.data = self.data.fillna(method='bfill')
        elif method == 'drop':
            self.data = self.data.dropna()
        elif method == 'fill':
            if fill_value is not None:
                self.data = self.data.fillna(fill_value)
            else:
                print("请提供fill_value参数")
                return None
        elif method == 'mean':
            self.data = self.data.fillna(self.data.mean())
        elif method == 'median':
            self.data = self.data.fillna(self.data.median())
        else:
            print(f"不支持的处理方法: {method}")
            return None
        
        missing_after = self.data.isnull().sum().sum()
        print(f"缺失值处理完成: {missing_before} -> {missing_after}")
        
        return self.data
    
    def detect_outliers(self, method='iqr', threshold=1.5, columns=None):
        """
        检测异常值
        
        Args:
            method: 检测方法 ('iqr', 'zscore', 'quantile')
            threshold: 阈值
            columns: 需要检测的列，None表示所有数值列
            
        Returns:
            dict: 异常值信息
        """
        if self.data is None:
            print("请先加载数据")
            return None
            
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_info = {}
        
        for column in columns:
            if method == 'iqr':
                Q1 = self.data[column].quantile(0.25)
                Q3 = self.data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = self.data[(self.data[column] < lower_bound) | 
                                   (self.data[column] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.data[column].dropna()))
                outliers = self.data[z_scores > threshold]
                
            elif method == 'quantile':
                lower_bound = self.data[column].quantile(threshold/100)
                upper_bound = self.data[column].quantile(1-threshold/100)
                outliers = self.data[(self.data[column] < lower_bound) | 
                                   (self.data[column] > upper_bound)]
            
            outliers_info[column] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(self.data) * 100,
                'lower_bound': lower_bound if 'lower_bound' in locals() else None,
                'upper_bound': upper_bound if 'upper_bound' in locals() else None
            }
            
            print(f"{column}: 异常值数量 = {outliers_info[column]['count']}, "
                  f"占比 = {outliers_info[column]['percentage']:.2f}%")
        
        return outliers_info
    
    def handle_outliers(self, method='clip', threshold=1.5, columns=None):
        """
        处理异常值
        
        Args:
            method: 处理方法 ('clip', 'remove', 'median', 'mean')
            threshold: 阈值
            columns: 需要处理的列，None表示所有数值列
        """
        if self.data is None:
            print("请先加载数据")
            return None
            
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"使用 {method} 方法处理异常值...")
        
        for column in columns:
            # IQR方法检测异常值
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            if method == 'clip':
                self.data[column] = self.data[column].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'remove':
                mask = (self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)
                self.data = self.data[mask]
            elif method == 'median':
                median_value = self.data[column].median()
                mask = (self.data[column] < lower_bound) | (self.data[column] > upper_bound)
                self.data.loc[mask, column] = median_value
            elif method == 'mean':
                mean_value = self.data[column].mean()
                mask = (self.data[column] < lower_bound) | (self.data[column] > upper_bound)
                self.data.loc[mask, column] = mean_value
        
        print(f"异常值处理完成")
        return self.data
    
    def create_lag_features(self, target_column, lag_orders=[1, 2, 3, 7, 14]):
        """
        创建滞后特征
        
        Args:
            target_column: 目标列名
            lag_orders: 滞后阶数列表
            
        Returns:
            pd.DataFrame: 包含滞后特征的数据
        """
        if self.data is None:
            print("请先加载数据")
            return None
            
        print(f"创建滞后特征，阶数: {lag_orders}")
        
        df_features = self.data.copy()
        
        for lag in lag_orders:
            df_features[f'{target_column}_lag_{lag}'] = df_features[target_column].shift(lag)
        
        self.feature_columns.extend([f'{target_column}_lag_{lag}' for lag in lag_orders])
        
        return df_features
    
    def create_rolling_features(self, target_column, windows=[3, 7, 14, 30], 
                              statistics=['mean', 'std', 'min', 'max']):
        """
        创建滚动窗口特征
        
        Args:
            target_column: 目标列名
            windows: 窗口大小列表
            statistics: 统计量列表
            
        Returns:
            pd.DataFrame: 包含滚动特征的数据
        """
        if self.data is None:
            print("请先加载数据")
            return None
            
        print(f"创建滚动特征，窗口: {windows}, 统计量: {statistics}")
        
        df_features = self.data.copy()
        
        for window in windows:
            for stat in statistics:
                if stat == 'mean':
                    df_features[f'{target_column}_ma_{window}'] = \
                        df_features[target_column].rolling(window=window).mean()
                elif stat == 'std':
                    df_features[f'{target_column}_std_{window}'] = \
                        df_features[target_column].rolling(window=window).std()
                elif stat == 'min':
                    df_features[f'{target_column}_min_{window}'] = \
                        df_features[target_column].rolling(window=window).min()
                elif stat == 'max':
                    df_features[f'{target_column}_max_{window}'] = \
                        df_features[target_column].rolling(window=window).max()
                
                self.feature_columns.append(f'{target_column}_{stat}_{window}')
        
        return df_features
    
    def create_time_features(self, datetime_index=None):
        """
        创建时间特征
        
        Args:
            datetime_index: 时间索引，如果为None则使用数据的索引
            
        Returns:
            pd.DataFrame: 包含时间特征的数据
        """
        if self.data is None:
            print("请先加载数据")
            return None
            
        print("创建时间特征...")
        
        df_features = self.data.copy()
        
        # 使用数据的索引或指定的datetime_index
        if datetime_index is None:
            if isinstance(df_features.index, pd.DatetimeIndex):
                dt_index = df_features.index
            else:
                print("数据索引不是DatetimeIndex，请提供datetime_index参数")
                return df_features
        else:
            dt_index = pd.to_datetime(datetime_index)
        
        # 基础时间特征
        df_features['hour'] = dt_index.hour
        df_features['day'] = dt_index.day
        df_features['month'] = dt_index.month
        df_features['year'] = dt_index.year
        df_features['dayofweek'] = dt_index.dayofweek
        df_features['quarter'] = dt_index.quarter
        df_features['dayofyear'] = dt_index.dayofyear
        df_features['weekofyear'] = dt_index.isocalendar().week
        
        # 周期性特征（正弦和余弦变换）
        df_features['hour_sin'] = np.sin(2 * np.pi * dt_index.hour / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * dt_index.hour / 24)
        df_features['day_sin'] = np.sin(2 * np.pi * dt_index.day / 31)
        df_features['day_cos'] = np.cos(2 * np.pi * dt_index.day / 31)
        df_features['month_sin'] = np.sin(2 * np.pi * dt_index.month / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * dt_index.month / 12)
        df_features['dayofweek_sin'] = np.sin(2 * np.pi * dt_index.dayofweek / 7)
        df_features['dayofweek_cos'] = np.cos(2 * np.pi * dt_index.dayofweek / 7)
        
        # 添加特征列名
        time_features = ['hour', 'day', 'month', 'year', 'dayofweek', 'quarter', 
                        'dayofyear', 'weekofyear', 'hour_sin', 'hour_cos', 
                        'day_sin', 'day_cos', 'month_sin', 'month_cos', 
                        'dayofweek_sin', 'dayofweek_cos']
        self.feature_columns.extend(time_features)
        
        return df_features
    
    def scale_data(self, method='minmax', columns=None):
        """
        数据标准化
        
        Args:
            method: 标准化方法 ('minmax', 'standard', 'robust')
            columns: 需要标准化的列，None表示所有数值列
            
        Returns:
            np.array: 标准化后的数据
        """
        if self.data is None:
            print("请先加载数据")
            return None
            
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"使用 {method} 方法标准化数据...")
        
        # 选择标准化器
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            print(f"不支持的标准化方法: {method}")
            return None
        
        # 标准化数据
        scaled_data = scaler.fit_transform(self.data[columns])
        
        # 创建标准化后的DataFrame
        self.processed_data = pd.DataFrame(scaled_data, 
                                         columns=columns, 
                                         index=self.data.index)
        
        # 保存标准化器
        self.scaler = scaler
        
        print(f"数据标准化完成")
        return self.processed_data
    
    def split_data(self, target_column, train_ratio=0.8, valid_ratio=0.1, 
                   shuffle=False, time_based=True):
        """
        划分训练集、验证集和测试集
        
        Args:
            target_column: 目标列名
            train_ratio: 训练集比例
            valid_ratio: 验证集比例
            shuffle: 是否随机打乱
            time_based: 是否按时间顺序划分
            
        Returns:
            tuple: (X_train, X_valid, X_test, y_train, y_valid, y_test)
        """
        if self.data is None:
            print("请先加载数据")
            return None
            
        print(f"划分数据集 - 训练集: {train_ratio}, 验证集: {valid_ratio}, 测试集: {1-train_ratio-valid_ratio}")
        
        # 分离特征和标签
        if target_column not in self.data.columns:
            print(f"目标列 {target_column} 不存在")
            return None
        
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # 计算划分索引
        n_samples = len(self.data)
        train_size = int(n_samples * train_ratio)
        valid_size = int(n_samples * valid_ratio)
        
        if time_based:
            # 按时间顺序划分
            X_train = X[:train_size]
            X_valid = X[train_size:train_size+valid_size]
            X_test = X[train_size+valid_size:]
            y_train = y[:train_size]
            y_valid = y[train_size:train_size+valid_size]
            y_test = y[train_size+valid_size:]
        else:
            # 随机划分
            from sklearn.model_selection import train_test_split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=1-train_ratio-valid_ratio, shuffle=shuffle, random_state=42)
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_temp, y_temp, test_size=valid_ratio/(train_ratio+valid_ratio), 
                shuffle=shuffle, random_state=42)
        
        print(f"数据集划分完成:")
        print(f"训练集: {len(X_train)} 样本")
        print(f"验证集: {len(X_valid)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
    def get_data_info(self):
        """获取数据信息"""
        if self.data is None:
            print("请先加载数据")
            return None
            
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'time_range': (self.data.index.min(), self.data.index.max()) if isinstance(self.data.index, pd.DatetimeIndex) else None
        }
        
        return info

# 辅助函数
def create_sample_data(start_date='2020-01-01', end_date='2023-12-31', 
                      freq='D', save_path='sample_data.csv'):
    """
    创建示例时间序列数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        freq: 频率
        save_path: 保存路径
        
    Returns:
        pd.DataFrame: 示例数据
    """
    print("创建示例时间序列数据...")
    
    # 创建日期范围
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_samples = len(dates)
    
    # 生成合成时间序列数据（包含趋势、季节性和噪声）
    np.random.seed(42)
    
    # 趋势成分
    trend = np.linspace(100, 200, n_samples)
    
    # 季节性成分
    if freq == 'D':
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)  # 年度周期
        seasonal += 5 * np.sin(2 * np.pi * np.arange(n_samples) / 30.44)   # 月度周期
    elif freq == 'H':
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24)      # 日周期
    else:
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)  # 默认年度周期
    
    # 随机噪声
    noise = np.random.normal(0, 5, n_samples)
    
    # 综合时间序列
    values = trend + seasonal + noise
    
    # 添加一些异常值
    outlier_indices = np.random.choice(n_samples, size=int(0.01 * n_samples), replace=False)
    values[outlier_indices] += np.random.normal(0, 20, len(outlier_indices))
    
    # 添加一些缺失值
    missing_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    values[missing_indices] = np.nan
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'temperature': values + np.random.normal(0, 3, n_samples),  # 相关变量
        'humidity': 50 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25) + np.random.normal(0, 5, n_samples)
    })
    
    # 保存数据
    df.to_csv(save_path, index=False)
    print(f"示例数据已保存到: {save_path}")
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {dates.min()} 到 {dates.max()}")
    
    return df

if __name__ == "__main__":
    # 测试数据加载器
    print("=== 测试 TimeSeriesDataLoader ===")
    
    # 创建示例数据
    sample_data = create_sample_data(save_path='Time-Series/sample_data.csv')
    
    # 创建数据加载器实例
    loader = TimeSeriesDataLoader()
    
    # 加载数据
    data = loader.load_data('Time-Series/sample_data.csv', 
                           date_column='date', 
                           target_column='value')
    
    if data is not None:
        # 处理缺失值
        data = loader.handle_missing_values(method='interpolate')
        
        # 检测异常值
        outliers_info = loader.detect_outliers(method='iqr')
        
        # 处理异常值
        data = loader.handle_outliers(method='clip')
        
        # 创建滞后特征
        data_with_lag = loader.create_lag_features('value', lag_orders=[1, 3, 7])
        
        # 创建滚动特征
        data_with_rolling = loader.create_rolling_features('value', windows=[7, 30])
        
        # 创建时间特征
        data_with_time = loader.create_time_features()
        
        # 数据标准化
        scaled_data = loader.scale_data(method='minmax')
        
        # 划分数据集
        split_result = loader.split_data('value', train_ratio=0.7, valid_ratio=0.2)
        
        # 获取数据信息
        data_info = loader.get_data_info()
        print(f"\n数据信息: {data_info}")
        
        print("\n=== 数据加载器测试完成 ===")