"""配置模块"""
import os
from dotenv import load_dotenv

load_dotenv()

# K2 API配置
K2_API_KEY = os.getenv("K2_API_KEY", "")
K2_BASE_URL = os.getenv("K2_BASE_URL", "https://api.openai.com/v1")
K2_MODEL = os.getenv("K2_MODEL", "gpt-4")

# 系统配置
MAX_PARALLEL_TASKS = int(os.getenv("MAX_PARALLEL_TASKS", "5"))
BASH_TIMEOUT = int(os.getenv("BASH_TIMEOUT", "30"))

# 调试模式
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# 流式输出模式（如果遇到大代码文件解析问题，可以设置为 False）
STREAM_MODE = os.getenv("STREAM_MODE", "True").lower() == "true"

