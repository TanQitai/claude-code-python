"""
时间序列分析配置文件
集中管理所有配置参数
"""

import os

class Config:
    """配置类"""
    
    # 基础配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
    MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
    
    # 数据配置
    DATA_CONFIG = {
        'default_file': 'sample_data.csv',
        'date_formats': ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y'],
        'supported_formats': ['.csv', '.xlsx', '.xls', '.parquet'],
        'encoding': 'utf-8',
        'chunk_size': 10000
    }
    
    # 预处理配置
    PREPROCESSING_CONFIG = {
        'missing_value_methods': ['interpolate', 'ffill', 'bfill', 'drop', 'fill', 'mean', 'median'],
        'default_missing_method': 'interpolate',
        'outlier_detection_methods': ['iqr', 'zscore', 'quantile'],
        'default_outlier_method': 'iqr',
        'outlier_threshold': 1.5,
        'scaling_methods': ['minmax', 'standard', 'robust'],
        'default_scaling_method': 'minmax'
    }
    
    # 特征工程配置
    FEATURE_CONFIG = {
        'lag_features': {
            'enabled': True,
            'lag_orders': [1, 2, 3, 7, 14, 30],
            'max_lag': 60
        },
        'rolling_features': {
            'enabled': True,
            'windows': [3, 7, 14, 30, 60],
            'statistics': ['mean', 'std', 'min', 'max', 'median'],
            'min_periods': 1
        },
        'time_features': {
            'enabled': True,
            'features': ['year', 'month', 'day', 'hour', 'dayofweek', 'quarter', 'dayofyear'],
            'cyclical_encoding': True
        },
        'external_features': {
            'enabled': False,
            'weather_data': False,
            'holiday_data': False,
            'economic_data': False
        }
    }
    
    # 数据划分配置
    SPLIT_CONFIG = {
        'train_ratio': 0.7,
        'validation_ratio': 0.2,
        'test_ratio': 0.1,
        'time_based_split': True,
        'shuffle': False,
        'stratify': False,
        'random_state': 42
    }
    
    # 模型配置
    MODEL_CONFIG = {
        'traditional_models': {
            'MA': {
                'enabled': True,
                'parameters': {
                    'window_size': 30
                }
            },
            'ES': {
                'enabled': True,
                'parameters': {
                    'alpha': 0.3
                }
            },
            'AR': {
                'enabled': True,
                'parameters': {
                    'lag_order': 7
                }
            },
            'ARIMA': {
                'enabled': True,
                'parameters': {
                    'order': (1, 1, 1),
                    'seasonal_order': None
                },
                'auto_selection': False,
                'max_p': 5,
                'max_d': 2,
                'max_q': 5
            }
        },
        'machine_learning_models': {
            'LinearRegression': {
                'enabled': True,
                'parameters': {}
            },
            'Ridge': {
                'enabled': True,
                'parameters': {
                    'alpha': 1.0
                }
            },
            'Lasso': {
                'enabled': True,
                'parameters': {
                    'alpha': 1.0
                }
            },
            'RandomForest': {
                'enabled': True,
                'parameters': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'random_state': 42
                }
            },
            'GradientBoosting': {
                'enabled': True,
                'parameters': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3
                }
            },
            'SVR': {
                'enabled': False,
                'parameters': {
                    'kernel': 'rbf',
                    'C': 1.0
                }
            }
        },
        'deep_learning_models': {
            'LSTM': {
                'enabled': True,
                'parameters': {
                    'hidden_size': 50,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'learning_rate': 0.001,
                    'num_epochs': 100,
                    'batch_size': 32,
                    'sequence_length': 10
                }
            },
            'GRU': {
                'enabled': False,
                'parameters': {
                    'hidden_size': 50,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'learning_rate': 0.001,
                    'num_epochs': 100,
                    'batch_size': 32
                }
            }
        }
    }
    
    # 训练配置
    TRAINING_CONFIG = {
        'cross_validation': {
            'enabled': True,
            'cv_folds': 5,
            'scoring': 'neg_mean_squared_error'
        },
        'early_stopping': {
            'enabled': True,
            'patience': 10,
            'min_delta': 0.001
        },
        'hyperparameter_tuning': {
            'enabled': False,
            'method': 'grid_search',  # 'grid_search', 'random_search', 'bayesian'
            'n_iter': 50,
            'cv_folds': 3
        },
        'ensemble_methods': {
            'enabled': False,
            'methods': ['voting', 'stacking', 'blending']
        }
    }
    
    # 评估指标配置
    EVALUATION_CONFIG = {
        'metrics': ['RMSE', 'MAE', 'MAPE', 'R2', 'MASE'],
        'primary_metric': 'RMSE',
        'custom_metrics': {
            'MASE': {
                'enabled': True,
                'seasonal_period': 12
            },
            'SMAPE': {
                'enabled': False
            }
        }
    }
    
    # 可视化配置
    VISUALIZATION_CONFIG = {
        'plot_settings': {
            'figsize': (12, 8),
            'dpi': 300,
            'style': 'whitegrid',
            'color_palette': 'Set2',
            'font_scale': 1.2
        },
        'plots_to_create': {
            'time_series_plot': True,
            'seasonal_decomposition': True,
            'correlation_matrix': True,
            'feature_importance': True,
            'prediction_comparison': True,
            'residual_analysis': True,
            'model_performance': True,
            'rolling_statistics': True,
            'acf_pacf': True
        },
        'output_formats': ['png', 'pdf'],
        'save_plots': True
    }
    
    # 输出配置
    OUTPUT_CONFIG = {
        'save_models': True,
        'save_predictions': True,
        'save_metrics': True,
        'save_plots': True,
        'save_reports': True,
        'compression': False,
        'output_structure': {
            'models': 'models/',
            'predictions': 'predictions/',
            'metrics': 'metrics/',
            'plots': 'plots/',
            'reports': 'reports/'
        }
    }
    
    # 日志配置
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_enabled': True,
        'console_enabled': True,
        'log_file': 'time_series_analysis.log',
        'max_file_size': '10MB',
        'backup_count': 5
    }
    
    # 性能配置
    PERFORMANCE_CONFIG = {
        'n_jobs': -1,  # 使用所有CPU核心
        'memory_limit': '4GB',
        'time_limit': '2h',
        'chunk_size': 10000,
        'use_gpu': False,
        'parallel_processing': True
    }
    
    # 验证配置
    VALIDATION_CONFIG = {
        'data_validation': {
            'check_missing_values': True,
            'check_outliers': True,
            'check_stationarity': False,
            'check_seasonality': False
        },
        'model_validation': {
            'cross_validation': True,
            'backtesting': True,
            'walk_forward_validation': True
        },
        'prediction_validation': {
            'prediction_intervals': True,
            'confidence_level': 0.95
        }
    }
    
    @classmethod
    def create_default_config_file(cls, filename='config.json'):
        """创建默认配置文件"""
        import json
        
        config_dict = {
            'base_dir': cls.BASE_DIR,
            'data_dir': cls.DATA_DIR,
            'output_dir': cls.OUTPUT_DIR,
            'data_config': cls.DATA_CONFIG,
            'preprocessing_config': cls.PREPROCESSING_CONFIG,
            'feature_config': cls.FEATURE_CONFIG,
            'split_config': cls.SPLIT_CONFIG,
            'model_config': cls.MODEL_CONFIG,
            'training_config': cls.TRAINING_CONFIG,
            'evaluation_config': cls.EVALUATION_CONFIG,
            'visualization_config': cls.VISUALIZATION_CONFIG,
            'output_config': cls.OUTPUT_CONFIG,
            'logging_config': cls.LOGGING_CONFIG,
            'performance_config': cls.PERFORMANCE_CONFIG,
            'validation_config': cls.VALIDATION_CONFIG
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        
        print(f"默认配置文件已创建: {filename}")
        return filename
    
    @classmethod
    def load_config_from_file(cls, filename):
        """从文件加载配置"""
        import json
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"配置已加载: {filename}")
            return config
        except FileNotFoundError:
            print(f"配置文件不存在: {filename}")
            return None
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            return None
    
    @classmethod
    def get_model_config(cls, model_type, model_name):
        """获取特定模型的配置"""
        if model_type in cls.MODEL_CONFIG and model_name in cls.MODEL_CONFIG[model_type]:
            return cls.MODEL_CONFIG[model_type][model_name]
        return None
    
    @classmethod
    def update_model_config(cls, model_type, model_name, new_config):
        """更新模型配置"""
        if model_type in cls.MODEL_CONFIG and model_name in cls.MODEL_CONFIG[model_type]:
            cls.MODEL_CONFIG[model_type][model_name].update(new_config)
            print(f"模型配置已更新: {model_type}.{model_name}")
        else:
            print(f"模型不存在: {model_type}.{model_name}")
    
    @classmethod
    def get_enabled_models(cls):
        """获取所有启用的模型"""
        enabled_models = {}
        
        for model_type, models in cls.MODEL_CONFIG.items():
            enabled_models[model_type] = {
                name: config for name, config in models.items()
                if config.get('enabled', False)
            }
        
        return enabled_models
    
    @classmethod
    def print_config_summary(cls):
        """打印配置摘要"""
        print("=== 时间序列分析配置摘要 ===")
        print(f"基础目录: {cls.BASE_DIR}")
        print(f"数据目录: {cls.DATA_DIR}")
        print(f"输出目录: {cls.OUTPUT_DIR}")
        
        print(f"\n启用的模型:")
        enabled_models = cls.get_enabled_models()
        for model_type, models in enabled_models.items():
            if models:
                print(f"  {model_type}:")
                for model_name in models.keys():
                    print(f"    - {model_name}")
        
        print(f"\n数据划分比例:")
        print(f"  训练集: {cls.SPLIT_CONFIG['train_ratio'] * 100}%")
        print(f"  验证集: {cls.SPLIT_CONFIG['validation_ratio'] * 100}%")
        print(f"  测试集: {cls.SPLIT_CONFIG['test_ratio'] * 100}%")
        
        print(f"\n评估指标: {cls.EVALUATION_CONFIG['metrics']}")
        print(f"主要指标: {cls.EVALUATION_CONFIG['primary_metric']}")
        
        print(f"\n可视化配置:")
        print(f"  图形大小: {cls.VISUALIZATION_CONFIG['plot_settings']['figsize']}")
        print(f"  保存图表: {cls.VISUALIZATION_CONFIG['save_plots']}")
        print(f"  输出格式: {cls.VISUALIZATION_CONFIG['output_formats']}")

if __name__ == "__main__":
    # 打印配置摘要
    Config.print_config_summary()
    
    # 创建默认配置文件
    config_file = Config.create_default_config_file()
    
    # 测试加载配置
    loaded_config = Config.load_config_from_file(config_file)
    if loaded_config:
        print(f"\n成功加载配置文件，包含 {len(loaded_config)} 个主要配置项")
    
    # 示例：更新模型配置
    Config.update_model_config('traditional_models', 'MA', {'parameters': {'window_size': 20}})
    
    print("\n配置文件系统测试完成！")