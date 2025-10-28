# 🤖 Claude Code Python - AI Agent 系统

一个强大的 AI Agent 系统，支持自然语言交互、工具调用和 Web UI 界面。

## ✨ 特性

- 🔧 **丰富的工具系统**：文件操作、代码搜索、项目分析、命令执行等
- 💬 **多轮对话**：支持上下文记忆和连续对话
- 🌊 **流式输出**：实时显示 Agent 思考和执行过程
- 🌐 **Web UI**：简约高效的 Web 界面
- 📝 **CLI 模式**：支持命令行交互
- 🔒 **安全性**：路径安全检查、文件大小限制、类型白名单

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

创建 `.env` 文件并配置：

```bash
K2_API_KEY=your_api_key_here
K2_BASE_URL=https://api.openai.com/v1
K2_MODEL=gpt-4
```

### 3. 启动方式

#### 方式一：Web UI（推荐）

```bash
python web_ui.py
```

然后在浏览器中访问：`http://localhost:5000`

#### 方式二：命令行 CLI

```bash
python agent_cli.py
```

## 🎨 Web UI 截图

Web UI 提供了：
- ✅ 实时流式输出
- ✅ 工具调用可视化
- ✅ 简约现代的界面
- ✅ 响应式设计，支持移动端
- ✅ 键盘快捷键支持

### 快捷键

- `Enter` - 发送消息
- `Shift + Enter` - 换行
- `Esc` - 清空输入框

## 🔧 可用工具

| 工具 | 功能 | 优先级 |
|------|------|--------|
| `search_replace_tool` | 精确代码编辑 | ⭐⭐⭐⭐⭐ |
| `grep_tool` | 代码搜索 | ⭐⭐⭐⭐⭐ |
| `project_analysis` | 项目分析 | ⭐⭐⭐⭐⭐ |
| `list_dir` | 目录列表 | ⭐⭐⭐⭐ |
| `glob_file_search` | 文件名搜索 | ⭐⭐⭐⭐ |
| `file_tool` | 文件操作 | ⭐⭐⭐ |
| `bash_tool` | Bash 命令 | ⭐⭐⭐ |
| `task_tool` | 并行任务 | ⭐⭐⭐ |

## 💡 使用示例

### 项目分析
```
分析当前项目的代码结构
```

### 文件操作
```
创建一个名为 test.py 的文件，实现冒泡排序算法
```

### 代码搜索
```
查找所有包含 "Agent" 类的 Python 文件
```

### 代码重构
```
把所有文件中的 old_function 改成 new_function
```

### 系统任务
```
检查系统资源使用情况
```

## 📁 项目结构

```
claude-code-python/
├── agent.py              # 核心 Agent 逻辑
├── agent_cli.py          # CLI 界面
├── web_ui.py             # Web UI 服务器
├── llm_client.py         # LLM 客户端
├── config.py             # 配置管理
├── requirements.txt      # 依赖声明
├── tools/                # 工具模块
│   ├── base.py
│   ├── bash_tool.py
│   ├── file_tool.py
│   ├── grep_tool.py
│   ├── list_dir_tool.py
│   ├── glob_search_tool.py
│   ├── project_analysis_tool.py
│   ├── search_replace_tool.py
│   ├── task_tool.py
│   └── safe_file_handler.py
├── templates/            # Web UI 模板
│   └── index.html
└── static/               # 静态资源
    ├── style.css
    └── app.js
```

## ⚙️ 配置选项

在 `.env` 文件中可配置：

```bash
# LLM 配置
K2_API_KEY=your_api_key
K2_BASE_URL=https://api.openai.com/v1
K2_MODEL=kimi-k2-0905-preview

# 系统配置
MAX_PARALLEL_TASKS=5
BASH_TIMEOUT=30
DEBUG=False
```

## 🔐 安全特性

- ✅ 路径遍历检测
- ✅ 文件大小限制（默认 10MB）
- ✅ 文件类型白名单
- ✅ 工作目录限制
- ✅ 危险路径黑名单

## 📝 开发

### 添加新工具

1. 在 `tools/` 目录创建新工具文件
2. 继承 `BaseTool` 类
3. 实现必要的方法
4. 在 `tools/__init__.py` 中导出
5. 在 `AgentManager.create_main_agent()` 中注册

示例：

```python
from tools.base import BaseTool, ToolParameter

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"
    
    @property
    def description(self) -> str:
        return "我的工具描述"
    
    @property
    def parameters(self) -> Dict[str, ToolParameter]:
        return {
            "param1": ToolParameter(
                type="string",
                description="参数描述",
                required=True
            )
        }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        # 实现工具逻辑
        return {
            "success": True,
            "message": "执行成功"
        }
```



**享受使用 Claude Code Python！** 🎉

