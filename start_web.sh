#!/bin/bash
# 快速启动 Web UI

echo "=========================================="
echo "🚀 Claude Code Python - Web UI"
echo "=========================================="
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    echo "请先安装 Python 3.7+"
    exit 1
fi

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv .venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source .venv/bin/activate

# 安装依赖
echo "📥 检查依赖..."
pip install -q -r requirements.txt

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  警告: 未找到 .env 文件"
    echo "请创建 .env 文件并配置 API Key："
    echo ""
    echo "K2_API_KEY=your_api_key_here"
    echo "K2_BASE_URL=https://api.openai.com/v1"
    echo "K2_MODEL=gpt-4"
    echo ""
    read -p "是否继续启动？(y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 创建日志目录
mkdir -p logs

# 启动服务
echo ""
echo "=========================================="
echo "✅ 启动 Web UI 服务器..."
echo "=========================================="
echo ""

python3 web_ui.py

