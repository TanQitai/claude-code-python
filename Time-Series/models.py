import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
    print("警告: statsmodels 未安装，ARIMA/SARIMAX/ExponentialSmoothing 模型不可用")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("警告: TensorFlow 未安装，深度学习模型不可用")

class TimeSeriesModels:
    """时间序列模型集合"""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.results = {}
        
    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    # ========== 传统机器学习模型 ==========
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """训练线性回归模型"""
        print("训练线性回归模型...")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        self.trained_models['linear_regression'] = model
        self.results['linear_regression'] = {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics
        }
        
        print(f"线性回归模型训练完成 - R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
        return model, metrics
    
    def train_ridge_regression(self, X_train, y_train, X_test, y_test):
        """训练岭回归模型"""
        print("训练岭回归模型...")
        
        # 超参数调优
        param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        model = Ridge()
        
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        self.trained_models['ridge_regression'] = best_model
        self.results['ridge_regression'] = {
            'model': best_model,
            'predictions': y_pred,
            'metrics': metrics,
            'best_params': grid_search.best_params_
        }
        
        print(f"岭回归模型训练完成 - R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
        print(f"最佳参数: {grid_search.best_params_}")
        return best_model, metrics
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """训练随机森林模型"""
        print("训练随机森林模型...")
        
        # 超参数调优
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        model = RandomForestRegressor(random_state=42)
        
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        self.trained_models['random_forest'] = best_model
        self.results['random_forest'] = {
            'model': best_model,
            'predictions': y_pred,
            'metrics': metrics,
            'best_params': grid_search.best_params_
        }
        
        print(f"随机森林模型训练完成 - R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
        print(f"最佳参数: {grid_search.best_params_}")
        return best_model, metrics
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """训练梯度提升模型"""
        print("训练梯度提升模型...")
        
        # 超参数调优
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.01],
            'max_depth': [3, 5, 7]
        }
        model = GradientBoostingRegressor(random_state=42)
        
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        self.trained_models['gradient_boosting'] = best_model
        self.results['gradient_boosting'] = {
            'model': best_model,
            'predictions': y_pred,
            'metrics': metrics,
            'best_params': grid_search.best_params_
        }
        
        print(f"梯度提升模型训练完成 - R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
        print(f"最佳参数: {grid_search.best_params_}")
        return best_model, metrics
    
    # ========== 传统时间序列模型 ==========
    
    def train_arima(self, train_data, test_data, order=(1,1,1)):
        """训练ARIMA模型"""
        if not STATSMODELS_AVAILABLE:
            print("statsmodels 未安装，无法使用ARIMA模型")
            return None, {}
        
        print(f"训练ARIMA模型 (order={order})...")
        
        try:
            model = ARIMA(train_data, order=order)
            fitted_model = model.fit()
            
            # 预测
            forecast = fitted_model.forecast(steps=len(test_data))
            
            metrics = self.calculate_metrics(test_data, forecast)
            
            self.trained_models['arima'] = fitted_model
            self.results['arima'] = {
                'model': fitted_model,
                'predictions': forecast,
                'metrics': metrics
            }
            
            print(f"ARIMA模型训练完成 - R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            return fitted_model, metrics
            
        except Exception as e:
            print(f"ARIMA模型训练失败: {e}")
            return None, {}
    
    def train_sarima(self, train_data, test_data, order=(1,1,1), seasonal_order=(1,1,1,12)):
        """训练SARIMA模型"""
        if not STATSMODELS_AVAILABLE:
            print("statsmodels 未安装，无法使用SARIMA模型")
            return None, {}
        
        print(f"训练SARIMA模型 (order={order}, seasonal_order={seasonal_order})...")
        
        try:
            model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit()
            
            # 预测
            forecast = fitted_model.forecast(steps=len(test_data))
            
            metrics = self.calculate_metrics(test_data, forecast)
            
            self.trained_models['sarima'] = fitted_model
            self.results['sarima'] = {
                'model': fitted_model,
                'predictions': forecast,
                'metrics': metrics
            }
            
            print(f"SARIMA模型训练完成 - R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            return fitted_model, metrics
            
        except Exception as e:
            print(f"SARIMA模型训练失败: {e}")
            return None, {}
    
    def train_exponential_smoothing(self, train_data, test_data, trend='add', seasonal='add', seasonal_periods=12):
        """训练指数平滑模型"""
        if not STATSMODELS_AVAILABLE:
            print("statsmodels 未安装，无法使用指数平滑模型")
            return None, {}
        
        print(f"训练指数平滑模型 (trend={trend}, seasonal={seasonal})...")
        
        try:
            model = ExponentialSmoothing(
                train_data, 
                trend=trend, 
                seasonal=seasonal, 
                seasonal_periods=seasonal_periods
            )
            fitted_model = model.fit()
            
            # 预测
            forecast = fitted_model.forecast(steps=len(test_data))
            
            metrics = self.calculate_metrics(test_data, forecast)
            
            self.trained_models['exponential_smoothing'] = fitted_model
            self.results['exponential_smoothing'] = {
                'model': fitted_model,
                'predictions': forecast,
                'metrics': metrics
            }
            
            print(f"指数平滑模型训练完成 - R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            return fitted_model, metrics
            
        except Exception as e:
            print(f"指数平滑模型训练失败: {e}")
            return None, {}
    
    # ========== 深度学习模型 ==========
    
    def build_lstm_model(self, input_shape):
        """构建LSTM模型"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow 未安装，无法使用LSTM模型")
            return None
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def build_gru_model(self, input_shape):
        """构建GRU模型"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow 未安装，无法使用GRU模型")
            return None
        
        model = Sequential([
            GRU(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train_lstm(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """训练LSTM模型"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow 未安装，无法使用LSTM模型")
            return None, {}
        
        print("训练LSTM模型...")
        
        try:
            model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
            if model is None:
                return None, {}
            
            # 回调函数
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
            
            # 训练模型
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # 预测
            y_pred = model.predict(X_test)
            
            metrics = self.calculate_metrics(y_test, y_pred)
            
            self.trained_models['lstm'] = model
            self.results['lstm'] = {
                'model': model,
                'predictions': y_pred,
                'metrics': metrics,
                'history': history
            }
            
            print(f"LSTM模型训练完成 - R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            return model, metrics
            
        except Exception as e:
            print(f"LSTM模型训练失败: {e}")
            return None, {}
    
    def train_gru(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """训练GRU模型"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow 未安装，无法使用GRU模型")
            return None, {}
        
        print("训练GRU模型...")
        
        try:
            model = self.build_gru_model((X_train.shape[1], X_train.shape[2]))
            if model is None:
                return None, {}
            
            # 回调函数
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
            
            # 训练模型
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # 预测
            y_pred = model.predict(X_test)
            
            metrics = self.calculate_metrics(y_test, y_pred)
            
            self.trained_models['gru'] = model
            self.results['gru'] = {
                'model': model,
                'predictions': y_pred,
                'metrics': metrics,
                'history': history
            }
            
            print(f"GRU模型训练完成 - R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            return model, metrics
            
        except Exception as e:
            print(f"GRU模型训练失败: {e}")
            return None, {}
    
    def get_best_model(self, metric='R2'):
        """获取最佳模型"""
        if not self.results:
            print("还没有训练任何模型")
            return None
        
        best_model_name = None
        best_score = float('-inf') if metric in ['R2'] else float('inf')
        
        for model_name, result in self.results.items():
            score = result['metrics'][metric]
            if (metric in ['R2'] and score > best_score) or (metric not in ['R2'] and score < best_score):
                best_score = score
                best_model_name = model_name
        
        print(f"最佳模型: {best_model_name} ({metric}: {best_score:.4f})")
        return best_model_name, self.results[best_model_name]
    
    def compare_models(self):
        """比较所有模型的性能"""
        if not self.results:
            print("还没有训练任何模型")
            return None
        
        comparison = pd.DataFrame()
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison[model_name] = pd.Series(metrics)
        
        print("模型性能比较:")
        print(comparison.round(4))
        
        return comparison

if __name__ == "__main__":
    print("时间序列模型模块加载完成")
    print("可用模型:")
    print("- 机器学习模型: Linear Regression, Ridge Regression, Random Forest, Gradient Boosting")
    if STATSMODELS_AVAILABLE:
        print("- 传统时间序列模型: ARIMA, SARIMA, Exponential Smoothing")
    if TENSORFLOW_AVAILABLE:
        print("- 深度学习模型: LSTM, GRU")