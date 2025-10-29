"""
时间序列分析工具函数模块
包含各种辅助函数和工具
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def save_object(obj, filepath):
    """
    保存对象到文件
    
    Args:
        obj: 要保存的对象
        filepath: 文件路径
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f"对象已保存到: {filepath}")
        return True
    except Exception as e:
        print(f"保存对象失败: {str(e)}")
        return False

def load_object(filepath):
    """
    从文件加载对象
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的对象
    """
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f"对象已从 {filepath} 加载")
        return obj
    except Exception as e:
        print(f"加载对象失败: {str(e)}")
        return None

def save_dataframe(df, filepath, format='auto'):
    """
    保存DataFrame到文件
    
    Args:
        df: DataFrame
        filepath: 文件路径
        format: 文件格式 ('csv', 'excel', 'parquet', 'json', 'auto')
    """
    if format == 'auto':
        format = filepath.split('.')[-1].lower()
    
    try:
        if format == 'csv':
            df.to_csv(filepath, index=True)
        elif format in ['xlsx', 'excel']:
            df.to_excel(filepath, index=True)
        elif format == 'parquet':
            df.to_parquet(filepath, index=True)
        elif format == 'json':
            df.to_json(filepath, orient='records', date_format='iso')
        else:
            print(f"不支持的格式: {format}")
            return False
        
        print(f"DataFrame已保存到: {filepath}")
        return True
        
    except Exception as e:
        print(f"保存DataFrame失败: {str(e)}")
        return False

def load_dataframe(filepath, format='auto'):
    """
    从文件加载DataFrame
    
    Args:
        filepath: 文件路径
        format: 文件格式 ('csv', 'excel', 'parquet', 'json', 'auto')
        
    Returns:
        DataFrame
    """
    if format == 'auto':
        format = filepath.split('.')[-1].lower()
    
    try:
        if format == 'csv':
            df = pd.read_csv(filepath, parse_dates=True, index_col=0)
        elif format in ['xlsx', 'excel']:
            df = pd.read_excel(filepath, parse_dates=True, index_col=0)
        elif format == 'parquet':
            df = pd.read_parquet(filepath)
        elif format == 'json':
            df = pd.read_json(filepath)
        else:
            print(f"不支持的格式: {format}")
            return None
        
        print(f"DataFrame已从 {filepath} 加载")
        return df
        
    except Exception as e:
        print(f"加载DataFrame失败: {str(e)}")
        return None

def create_directory_structure(base_dir):
    """
    创建目录结构
    
    Args:
        base_dir: 基础目录路径
    """
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'models',
        'models/trained',
        'models/checkpoints',
        'output',
        'output/predictions',
        'output/metrics',
        'output/plots',
        'output/reports',
        'logs',
        'config',
        'notebooks'
    ]
    
    created_dirs = []
    for directory in directories:
        full_path = os.path.join(base_dir, directory)
        try:
            os.makedirs(full_path, exist_ok=True)
            created_dirs.append(full_path)
        except Exception as e:
            print(f"创建目录失败 {full_path}: {str(e)}")
    
    print(f"目录结构创建完成，共创建 {len(created_dirs)} 个目录")
    return created_dirs

def check_dependencies():
    """
    检查依赖库
    
    Returns:
        dict: 依赖库检查结果
    """
    dependencies = {
        'numpy': False,
        'pandas': False,
        'matplotlib': False,
        'seaborn': False,
        'scikit-learn': False,
        'statsmodels': False,
        'torch': False,
        'scipy': False
    }
    
    # 检查numpy
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    # 检查pandas
    try:
        import pandas
        dependencies['pandas'] = True
    except ImportError:
        pass
    
    # 检查matplotlib
    try:
        import matplotlib
        dependencies['matplotlib'] = True
    except ImportError:
        pass
    
    # 检查seaborn
    try:
        import seaborn
        dependencies['seaborn'] = True
    except ImportError:
        pass
    
    # 检查scikit-learn
    try:
        import sklearn
        dependencies['scikit-learn'] = True
    except ImportError:
        pass
    
    # 检查statsmodels
    try:
        import statsmodels
        dependencies['statsmodels'] = True
    except ImportError:
        pass
    
    # 检查torch
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        pass
    
    # 检查scipy
    try:
        import scipy
        dependencies['scipy'] = True
    except ImportError:
        pass
    
    # 打印结果
    print("依赖库检查:")
    for lib, installed in dependencies.items():
        status = "✓" if installed else "✗"
        print(f"  {lib}: {status}")
    
    return dependencies

def install_requirements():
    """
    安装必需的依赖库
    """
    requirements = [
        'numpy>=1.19.0',
        'pandas>=1.3.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'scikit-learn>=0.24.0',
        'statsmodels>=0.12.0',
        'torch>=1.9.0',
        'scipy>=1.7.0'
    ]
    
    print("开始安装依赖库...")
    for requirement in requirements:
        try:
            os.system(f"pip install {requirement}")
            print(f"安装成功: {requirement}")
        except Exception as e:
            print(f"安装失败 {requirement}: {str(e)}")
    
    print("依赖库安装完成")

def generate_timestamp():
    """
    生成时间戳
    
    Returns:
        str: 时间戳字符串
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_filename(prefix="file", extension="txt", timestamp=True):
    """
    生成文件名
    
    Args:
        prefix: 文件名前缀
        extension: 文件扩展名
        timestamp: 是否添加时间戳
        
    Returns:
        str: 文件名
    """
    if timestamp:
        return f"{prefix}_{generate_timestamp()}.{extension}"
    else:
        return f"{prefix}.{extension}"

def validate_file_path(filepath):
    """
    验证文件路径
    
    Args:
        filepath: 文件路径
        
    Returns:
        bool: 是否有效
    """
    try:
        path = Path(filepath)
        
        # 检查文件是否存在
        if path.exists():
            return True
        
        # 检查目录是否存在或可创建
        if path.parent.exists() or path.parent.mkdir(parents=True, exist_ok=True):
            return True
        
        return False
        
    except Exception:
        return False

def get_file_info(filepath):
    """
    获取文件信息
    
    Args:
        filepath: 文件路径
        
    Returns:
        dict: 文件信息
    """
    try:
        path = Path(filepath)
        
        if not path.exists():
            return None
        
        stat = path.stat()
        
        return {
            'name': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'size': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': datetime.datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.datetime.fromtimestamp(stat.st_mtime),
            'is_file': path.is_file(),
            'is_dir': path.is_dir(),
            'absolute_path': str(path.absolute())
        }
        
    except Exception as e:
        print(f"获取文件信息失败: {str(e)}")
        return None

def list_files(directory, extensions=None, recursive=False):
    """
    列出目录中的文件
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名列表
        recursive: 是否递归
        
    Returns:
        list: 文件路径列表
    """
    try:
        path = Path(directory)
        
        if not path.exists():
            print(f"目录不存在: {directory}")
            return []
        
        files = []
        
        if recursive:
            pattern = "**/*" if extensions is None else "**/*"
            for ext in (extensions or ['']):
                files.extend(path.glob(f"*{ext}"))
        else:
            for item in path.iterdir():
                if item.is_file():
                    if extensions is None or item.suffix.lower() in extensions:
                        files.append(item)
        
        return sorted(files)
        
    except Exception as e:
        print(f"列出文件失败: {str(e)}")
        return []

def setup_logging(log_file='time_series_analysis.log', level='INFO'):
    """
    设置日志
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
        
    Returns:
        logger对象
    """
    import logging
    from logging.handlers import RotatingFileHandler
    
    # 创建logger
    logger = logging.getLogger('TimeSeriesAnalysis')
    logger.setLevel(getattr(logging, level.upper()))
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 文件处理器
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def calculate_running_time(func):
    """
    计算函数运行时间的装饰器
    
    Args:
        func: 函数
        
    Returns:
        包装函数
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        running_time = end_time - start_time
        
        print(f"函数 {func.__name__} 运行时间: {running_time:.2f} 秒")
        return result
    
    return wrapper

def memory_usage(func):
    """
    监控内存使用的装饰器
    
    Args:
        func: 函数
        
    Returns:
        包装函数
    """
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before
        
        print(f"函数 {func.__name__} 内存使用: {mem_used:.2f} MB")
        return result
    
    return wrapper

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    打印进度条
    
    Args:
        iteration: 当前迭代
        total: 总迭代
        prefix: 前缀
        suffix: 后缀
        decimals: 小数位数
        length: 进度条长度
        fill: 填充字符
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    
    if iteration == total:
        print()

def format_large_number(num):
    """
    格式化大数字
    
    Args:
        num: 数字
        
    Returns:
        str: 格式化的数字字符串
    """
    if abs(num) >= 1e9:
        return f"{num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"

def safe_divide(numerator, denominator, default=0):
    """
    安全除法
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 默认值
        
    Returns:
        除法结果
    """
    try:
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default

def create_sample_config():
    """创建示例配置文件"""
    config = {
        "project_name": "时间序列分析项目",
        "description": "这是一个示例时间序列分析配置",
        "data_path": "data/sample_data.csv",
        "output_path": "output/",
        "models": ["MA", "ES", "AR", "ARIMA", "RF"],
        "evaluation_metrics": ["RMSE", "MAE", "MAPE"],
        "visualization": {
            "save_plots": True,
            "plot_format": "png"
        }
    }
    
    config_path = "sample_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print(f"示例配置文件已创建: {config_path}")
    return config_path

if __name__ == "__main__":
    print("=== 测试工具函数 ===")
    
    # 测试依赖检查
    print("\n1. 检查依赖库:")
    dependencies = check_dependencies()
    
    # 测试目录创建
    print("\n2. 创建目录结构:")
    test_dir = "test_directory"
    created_dirs = create_directory_structure(test_dir)
    print(f"创建了 {len(created_dirs)} 个目录")
    
    # 测试文件操作
    print("\n3. 测试文件操作:")
    
    # 创建示例数据
    sample_data = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100)
    })
    
    # 保存数据
    csv_path = f"{test_dir}/data/sample_data.csv"
    save_dataframe(sample_data, csv_path)
    
    # 加载数据
    loaded_data = load_dataframe(csv_path)
    if loaded_data is not None:
        print(f"成功加载数据，形状: {loaded_data.shape}")
    
    # 测试对象保存和加载
    test_obj = {"model": "test", "accuracy": 0.95}
    obj_path = f"{test_dir}/models/test_model.pkl"
    save_object(test_obj, obj_path)
    loaded_obj = load_object(obj_path)
    print(f"加载的对象: {loaded_obj}")
    
    # 测试其他工具函数
    print(f"\n4. 时间戳: {generate_timestamp()}")
    print(f"文件名: {generate_filename('test', 'txt')}")
    print(f"大数字格式化: {format_large_number(1234567890)}")
    
    # 测试进度条
    print("\n5. 测试进度条:")
    for i in range(101):
        print_progress_bar(i, 100, prefix='进度:', suffix='完成')
    
    # 创建示例配置
    config_path = create_sample_config()
    
    print(f"\n=== 工具函数测试完成 ===")
    print(f"测试文件保存在 {test_dir}/ 目录中")
    print(f"示例配置文件: {config_path}")