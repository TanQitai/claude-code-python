import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesPreprocessor:
    """时间序列数据预处理类"""
    
    def __init__(self):
        self.scaler = None
        self.data = None
        
    def load_data(self, file_path, date_column=None, target_column=None):
        """
        加载时间序列数据
        
        Args:
            file_path: 数据文件路径
            date_column: 日期列名
            target_column: 目标列名
        """
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("不支持的文件格式，请使用CSV或Excel文件")
            
            print(f"数据加载成功，形状: {self.data.shape}")
            print(f"列名: {list(self.data.columns)}")
            
            if date_column:
                self.data[date_column] = pd.to_datetime(self.data[date_column])
                self.data.set_index(date_column, inplace=True)
                print(f"已设置 {date_column} 为索引")
            
            return self.data
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def check_missing_values(self):
        """检查缺失值"""
        if self.data is None:
            print("请先加载数据")
            return None
        
        missing_info = self.data.isnull().sum()
        missing_percent = (missing_info / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            '缺失数量': missing_info,
            '缺失比例(%)': missing_percent
        })
        
        print("缺失值检查:")
        print(missing_df[missing_df['缺失数量'] > 0])
        
        return missing_df
    
    def fill_missing_values(self, method='interpolate'):
        """
        填充缺失值
        
        Args:
            method: 填充方法 ('interpolate', 'forward_fill', 'backward_fill', 'mean')
        """
        if self.data is None:
            print("请先加载数据")
            return None
        
        if method == 'interpolate':
            self.data = self.data.interpolate()
        elif method == 'forward_fill':
            self.data = self.data.fillna(method='ffill')
        elif method == 'backward_fill':
            self.data = self.data.fillna(method='bfill')
        elif method == 'mean':
            self.data = self.data.fillna(self.data.mean())
        
        print(f"已使用 {method} 方法填充缺失值")
        return self.data
    
    def create_features(self, window_size=30):
        """
        创建时间序列特征
        
        Args:
            window_size: 滑动窗口大小
        """
        if self.data is None:
            print("请先加载数据")
            return None
        
        df = self.data.copy()
        
        # 添加时间特征
        if hasattr(df.index, 'year'):
            df['year'] = df.index.year
            df['month'] = df.index.month
            df['day'] = df.index.day
            df['dayofweek'] = df.index.dayofweek
            df['quarter'] = df.index.quarter
        
        # 添加滞后特征
        for lag in range(1, window_size + 1):
            for col in df.select_dtypes(include=[np.number]).columns:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # 添加移动平均
        for col in df.select_dtypes(include=[np.number]).columns:
            if not col.startswith(f'{col}_lag_'):
                df[f'{col}_ma_{window_size}'] = df[col].rolling(window=window_size).mean()
                df[f'{col}_std_{window_size}'] = df[col].rolling(window=window_size).std()
        
        # 删除包含NaN的行
        df = df.dropna()
        
        print(f"特征工程完成，新形状: {df.shape}")
        return df
    
    def normalize_data(self, data, method='standard', fit_scaler=True):
        """
        数据标准化
        
        Args:
            data: 要标准化的数据
            method: 标准化方法 ('standard', 'minmax')
            fit_scaler: 是否拟合标准化器
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            if fit_scaler or self.scaler is None:
                self.scaler = StandardScaler()
                data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
            else:
                data[numeric_cols] = self.scaler.transform(data[numeric_cols])
        
        elif method == 'minmax':
            if fit_scaler or self.scaler is None:
                self.scaler = MinMaxScaler()
                data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
            else:
                data[numeric_cols] = self.scaler.transform(data[numeric_cols])
        
        print(f"已使用 {method} 方法标准化数据")
        return data
    
    def create_sequences(self, data, target_col, sequence_length, prediction_length=1):
        """
        创建序列数据用于深度学习模型
        
        Args:
            data: 输入数据
            target_col: 目标列名
            sequence_length: 序列长度
            prediction_length: 预测长度
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_length + 1):
            X.append(data.iloc[i:(i + sequence_length)].values)
            y.append(data[target_col].iloc[i + sequence_length:i + sequence_length + prediction_length].values)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"序列数据创建完成:")
        print(f"X形状: {X.shape}")
        print(f"y形状: {y.shape}")
        
        return X, y
    
    def train_test_split_time_series(self, data, test_size=0.2, random_state=42):
        """
        时间序列数据分割（按时间顺序）
        
        Args:
            data: 输入数据
            test_size: 测试集比例
            random_state: 随机种子
        """
        split_index = int(len(data) * (1 - test_size))
        
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]
        
        print(f"数据分割完成:")
        print(f"训练集大小: {len(train_data)}")
        print(f"测试集大小: {len(test_data)}")
        
        return train_data, test_data

def generate_sample_data(start_date='2020-01-01', periods=1000, freq='D'):
    """
    生成示例时间序列数据
    
    Args:
        start_date: 开始日期
        periods: 数据点数量
        freq: 频率 ('D'表示日，'H'表示小时等)
    """
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # 生成具有趋势、季节性和噪声的数据
    trend = np.linspace(100, 200, periods)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(periods) / 365.25)
    noise = np.random.normal(0, 5, periods)
    
    values = trend + seasonal + noise
    
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(periods) / 365.25) + np.random.normal(0, 2, periods),
        'humidity': 60 + 20 * np.sin(2 * np.pi * np.arange(periods) / 365.25) + np.random.normal(0, 5, periods)
    })
    
    return df

if __name__ == "__main__":
    # 示例用法
    preprocessor = TimeSeriesPreprocessor()
    
    # 生成示例数据
    sample_data = generate_sample_data()
    sample_data.to_csv('sample_time_series.csv', index=False)
    print("示例数据已保存到 sample_time_series.csv")
    
    # 加载数据
    data = preprocessor.load_data('sample_time_series.csv', 'date', 'value')
    
    # 检查缺失值
    preprocessor.check_missing_values()
    
    # 创建特征
    data_with_features = preprocessor.create_features(window_size=7)
    
    print("数据预处理完成!")