"""
时间序列预测模型模块
包含传统统计模型、机器学习模型和深度学习模型
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("警告: statsmodels 未安装，传统统计模型将不可用")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("警告: PyTorch 未安装，深度学习模型将不可用")

class BaseTimeSeriesModel:
    """时间序列模型基类"""
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.training_history = None
        
    def fit(self, X, y):
        """训练模型"""
        raise NotImplementedError("子类必须实现fit方法")
    
    def predict(self, X):
        """预测"""
        raise NotImplementedError("子类必须实现predict方法")
    
    def evaluate(self, y_true, y_pred):
        """评估模型"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

class MovingAverageModel(BaseTimeSeriesModel):
    """移动平均模型"""
    
    def __init__(self, window_size=12):
        super().__init__()
        self.window_size = window_size
        self.name = f"MA_{window_size}"
        
    def fit(self, X, y):
        """移动平均模型不需要训练"""
        self.is_fitted = True
        return self
        
    def predict(self, X):
        """使用移动平均进行预测"""
        if isinstance(X, pd.Series):
            values = X.values
        else:
            values = X.flatten() if hasattr(X, 'flatten') else np.array(X)
        
        # 计算移动平均
        ma_values = pd.Series(values).rolling(window=self.window_size).mean()
        return ma_values.values

class ExponentialSmoothingModel(BaseTimeSeriesModel):
    """指数平滑模型"""
    
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.name = f"ES_{alpha}"
        
    def fit(self, X, y):
        """指数平滑模型不需要训练"""
        self.is_fitted = True
        return self
        
    def predict(self, X):
        """使用指数平滑进行预测"""
        if isinstance(X, pd.Series):
            values = X.values
        else:
            values = X.flatten() if hasattr(X, 'flatten') else np.array(X)
        
        # 计算指数平滑
        exp_smooth = pd.Series(values).ewm(alpha=self.alpha, adjust=False).mean()
        return exp_smooth.values

class ARModel(BaseTimeSeriesModel):
    """自回归模型"""
    
    def __init__(self, lag_order=5):
        super().__init__()
        self.lag_order = lag_order
        self.name = f"AR_{lag_order}"
        self.sklearn_model = LinearRegression()
        
    def fit(self, X, y):
        """训练自回归模型"""
        # 创建滞后特征
        X_features, y_target = self._create_lag_features(X, y)
        
        if len(X_features) > 0:
            self.sklearn_model.fit(X_features, y_target)
            self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """使用自回归模型预测"""
        if not self.is_fitted:
            print("模型尚未训练")
            return None
            
        X_features, _ = self._create_lag_features(X, None)
        
        if len(X_features) > 0:
            predictions = self.sklearn_model.predict(X_features)
            return predictions
        else:
            return np.array([])
    
    def _create_lag_features(self, X, y):
        """创建滞后特征"""
        if isinstance(X, pd.Series):
            values = X.values
        elif isinstance(X, pd.DataFrame):
            values = X.iloc[:, 0].values
        else:
            values = np.array(X).flatten()
        
        X_features, y_target = [], []
        
        for i in range(self.lag_order, len(values)):
            X_features.append(values[i-self.lag_order:i])
            if y is not None:
                if isinstance(y, pd.Series):
                    y_target.append(y.iloc[i])
                else:
                    y_target.append(y[i])
        
        X_features = np.array(X_features)
        y_target = np.array(y_target) if y is not None else None
        
        return X_features, y_target

class ARIMAModel(BaseTimeSeriesModel):
    """ARIMA模型"""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=None):
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        if seasonal_order:
            self.name = f"SARIMA_{order}_{seasonal_order}"
        else:
            self.name = f"ARIMA_{order}"
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels 库未安装，无法使用ARIMA模型")
    
    def fit(self, X, y=None):
        """训练ARIMA模型"""
        if isinstance(X, pd.Series):
            ts_data = X
        else:
            ts_data = pd.Series(X.flatten() if hasattr(X, 'flatten') else X)
        
        try:
            if self.seasonal_order:
                self.model = SARIMAX(ts_data, order=self.order, 
                                   seasonal_order=self.seasonal_order)
            else:
                self.model = ARIMA(ts_data, order=self.order)
            
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
        except Exception as e:
            print(f"ARIMA模型训练失败: {str(e)}")
            
        return self
    
    def predict(self, X, steps=None):
        """使用ARIMA模型预测"""
        if not self.is_fitted:
            print("模型尚未训练")
            return None
        
        if steps is None:
            if hasattr(X, '__len__'):
                steps = len(X)
            else:
                steps = 1
        
        try:
            predictions = self.fitted_model.forecast(steps=steps)
            return predictions.values if hasattr(predictions, 'values') else predictions
        except Exception as e:
            print(f"ARIMA预测失败: {str(e)}")
            return None

class RandomForestTSModel(BaseTimeSeriesModel):
    """随机森林时间序列模型"""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.name = f"RF_{n_estimators}"
        
    def fit(self, X, y):
        """训练随机森林模型"""
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_importance = self.model.feature_importances_
        return self
        
    def predict(self, X):
        """使用随机森林模型预测"""
        if not self.is_fitted:
            print("模型尚未训练")
            return None
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names=None):
        """获取特征重要性"""
        if not self.is_fitted:
            print("模型尚未训练")
            return None
            
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

class GradientBoostingTSModel(BaseTimeSeriesModel):
    """梯度提升时间序列模型"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        super().__init__()
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        self.name = f"GB_{n_estimators}_{learning_rate}"
        
    def fit(self, X, y):
        """训练梯度提升模型"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        """使用梯度提升模型预测"""
        if not self.is_fitted:
            print("模型尚未训练")
            return None
        return self.model.predict(X)

class LSTMModel(BaseTimeSeriesModel):
    """LSTM深度学习模型"""
    
    def __init__(self, hidden_size=50, num_layers=2, dropout=0.2, 
                 learning_rate=0.001, num_epochs=100, batch_size=32):
        super().__init__()
        
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch 库未安装，无法使用LSTM模型")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.name = f"LSTM_{hidden_size}_{num_layers}"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
    def fit(self, X, y):
        """训练LSTM模型"""
        # 准备序列数据
        X_seq, y_seq = self._prepare_sequences(X, y)
        
        if len(X_seq) == 0:
            print("序列数据准备失败")
            return self
        
        # 创建数据加载器
        dataset = TensorDataset(X_seq, y_seq)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 初始化模型
        input_size = X_seq.shape[2]
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 训练模型
        self.training_history = {'loss': []}
        
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.training_history['loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.6f}')
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """使用LSTM模型预测"""
        if not self.is_fitted:
            print("模型尚未训练")
            return None
        
        # 准备序列数据
        X_seq, _ = self._prepare_sequences(X, None)
        
        if len(X_seq) == 0:
            return np.array([])
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            X_seq = X_seq.to(self.device)
            predictions = self.model(X_seq)
            predictions = predictions.cpu().numpy()
        
        return predictions.flatten()
    
    def _prepare_sequences(self, X, y, sequence_length=10):
        """准备LSTM所需的序列数据"""
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_values = X.values
        else:
            X_values = np.array(X)
        
        # 确保二维数组
        if X_values.ndim == 1:
            X_values = X_values.reshape(-1, 1)
        
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X_values)):
            X_seq.append(X_values[i-sequence_length:i])
            if y is not None:
                y_seq.append(y.iloc[i] if isinstance(y, pd.Series) else y[i])
        
        X_seq = torch.FloatTensor(X_seq)
        if y is not None:
            y_seq = torch.FloatTensor(y_seq).unsqueeze(1)
        else:
            y_seq = None
        
        return X_seq, y_seq

class LSTMNetwork(nn.Module):
    """LSTM网络结构"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        last_output = self.dropout(last_output)
        
        # 全连接层
        output = self.fc(last_output)
        
        return output

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def add_model(self, model, name=None):
        """添加模型"""
        if name is None:
            name = model.name
        self.models[name] = model
        print(f"添加模型: {name}")
        
    def train_models(self, X_train, y_train, X_valid=None, y_valid=None):
        """训练所有模型"""
        print("开始训练所有模型...")
        
        for name, model in self.models.items():
            print(f"\n训练模型: {name}")
            try:
                # 训练模型
                model.fit(X_train, y_train)
                
                # 在训练集上评估
                train_pred = model.predict(X_train)
                if train_pred is not None and len(train_pred) > 0:
                    train_metrics = model.evaluate(y_train, train_pred)
                    self.results[name] = {'train_metrics': train_metrics}
                    
                    print(f"训练集评估结果:")
                    for metric, value in train_metrics.items():
                        print(f"  {metric}: {value:.4f}")
                
                # 在验证集上评估
                if X_valid is not None and y_valid is not None:
                    valid_pred = model.predict(X_valid)
                    if valid_pred is not None and len(valid_pred) > 0:
                        valid_metrics = model.evaluate(y_valid, valid_pred)
                        self.results[name]['valid_metrics'] = valid_metrics
                        
                        print(f"验证集评估结果:")
                        for metric, value in valid_metrics.items():
                            print(f"  {metric}: {value:.4f}")
                            
            except Exception as e:
                print(f"模型 {name} 训练失败: {str(e)}")
                continue
        
        print("\n所有模型训练完成")
        
    def predict_all(self, X_test):
        """使用所有模型进行预测"""
        predictions = {}
        
        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    pred = model.predict(X_test)
                    if pred is not None and len(pred) > 0:
                        predictions[name] = pred
                except Exception as e:
                    print(f"模型 {name} 预测失败: {str(e)}")
                    continue
        
        return predictions
    
    def get_best_model(self, metric='RMSE', dataset='valid'):
        """获取最佳模型"""
        if dataset == 'valid':
            key = 'valid_metrics'
        else:
            key = 'train_metrics'
        
        if not self.results:
            print("还没有模型结果")
            return None
        
        best_model = None
        best_score = float('inf')
        
        for name, results in self.results.items():
            if key in results and metric in results[key]:
                score = results[key][metric]
                if score < best_score:
                    best_score = score
                    best_model = name
        
        return best_model, best_score
    
    def get_results_summary(self):
        """获取结果摘要"""
        if not self.results:
            print("还没有模型结果")
            return None
        
        summary = pd.DataFrame()
        
        for name, results in self.results.items():
            row = {'model': name}
            
            if 'train_metrics' in results:
                for metric, value in results['train_metrics'].items():
                    row[f'train_{metric}'] = value
            
            if 'valid_metrics' in results:
                for metric, value in results['valid_metrics'].items():
                    row[f'valid_{metric}'] = value
            
            summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
        
        return summary

# 模型工厂
def create_model(model_type, **kwargs):
    """
    创建模型的工厂函数
    
    Args:
        model_type: 模型类型 ('MA', 'ES', 'AR', 'ARIMA', 'RF', 'GB', 'LSTM')
        **kwargs: 模型参数
        
    Returns:
        BaseTimeSeriesModel: 模型实例
    """
    if model_type == 'MA':
        return MovingAverageModel(**kwargs)
    elif model_type == 'ES':
        return ExponentialSmoothingModel(**kwargs)
    elif model_type == 'AR':
        return ARModel(**kwargs)
    elif model_type == 'ARIMA':
        return ARIMAModel(**kwargs)
    elif model_type == 'RF':
        return RandomForestTSModel(**kwargs)
    elif model_type == 'GB':
        return GradientBoostingTSModel(**kwargs)
    elif model_type == 'LSTM':
        return LSTMModel(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

if __name__ == "__main__":
    # 测试模型
    print("=== 测试时间序列模型 ===")
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    trend = np.linspace(100, 200, n_samples)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
    noise = np.random.normal(0, 5, n_samples)
    y = trend + seasonal + noise
    
    # 创建特征矩阵
    X = np.column_stack([
        trend,
        seasonal,
        np.random.normal(0, 1, n_samples),
        np.random.normal(0, 1, n_samples)
    ])
    
    # 划分训练集和测试集
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 创建模型训练器
    trainer = ModelTrainer()
    
    # 添加模型
    trainer.add_model(create_model('MA', window_size=30))
    trainer.add_model(create_model('ES', alpha=0.3))
    trainer.add_model(create_model('AR', lag_order=7))
    
    if STATSMODELS_AVAILABLE:
        trainer.add_model(create_model('ARIMA', order=(1, 1, 1)))
    
    trainer.add_model(create_model('RF', n_estimators=100))
    trainer.add_model(create_model('GB', n_estimators=100))
    
    if PYTORCH_AVAILABLE:
        trainer.add_model(create_model('LSTM', hidden_size=50, num_epochs=50))
    
    # 训练模型
    trainer.train_models(X_train, y_train)
    
    # 预测
    predictions = trainer.predict_all(X_test)
    
    # 评估预测结果
    for name, pred in predictions.items():
        if len(pred) == len(y_test):
            metrics = trainer.models[name].evaluate(y_test, pred)
            print(f"\n{name} 测试集评估:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    # 获取最佳模型
    best_model, best_score = trainer.get_best_model()
    print(f"\n最佳模型: {best_model}, 最佳RMSE: {best_score:.4f}")
    
    # 获取结果摘要
    results_summary = trainer.get_results_summary()
    if results_summary is not None:
        print("\n结果摘要:")
        print(results_summary)
    
    print("\n=== 模型测试完成 ===")