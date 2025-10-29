"""智能Agent系统 - 支持工具调用和sub-agent"""
import json
import asyncio
from typing import List, Dict, Any, Optional, Iterator
from loguru import logger
from llm_client import K2Client
from tools import (
    BaseTool, BashTool, TaskTool, FileTool,
    SearchReplaceTool, GrepTool, SafeFileHandler,
    ListDirTool, GlobSearchTool, ProjectAnalysisTool
)


class Agent:
    """
    智能Agent
    
    功能：
    1. 接收任务指令
    2. 调用LLM进行推理
    3. 使用工具执行操作
    4. 支持创建sub-agent执行子任务
    """
    
    # 系统提示词（类常量，避免重复定义）
    SYSTEM_PROMPT = """你是一个强大的 AI Agent，拥有实际执行能力。你必须主动使用工具来完成任务。

🎯 核心原则 - 先执行，后回答：
• 用户问什么 → 先用工具查看/执行 → 再基于结果回答
• 不要凭空回答，要用工具获取实际数据
• 不要教用户怎么做，要直接去做
• 优先使用专用工具，而不是bash命令

🔧 可用工具（按推荐优先级）：

1. search_replace_tool - 精确代码编辑 ⭐⭐⭐⭐⭐
   • 在文件中搜索并替换文本
   • 支持单次或全局替换（replace_all）
   • 自动保留格式和缩进
   • 适用于：代码重构、修改配置、批量替换

2. grep_tool - 代码搜索 ⭐⭐⭐⭐⭐
   • 搜索文件内容，支持正则表达式
   • 显示匹配行和上下文
   • 自动排除无用目录
   • 适用于：查找函数、搜索变量、代码审查

3. list_dir - 目录列表 ⭐⭐⭐⭐
   • 列出目录内容，比ls更清晰
   • 支持递归（recursive=true）
   • 自动排除.git, __pycache__等
   • 显示文件大小和类型
   • 适用于：查看项目结构、浏览文件

4. glob_file_search - 文件名搜索 ⭐⭐⭐⭐
   • 按文件名搜索，支持通配符
   • 例如：'*.py', 'test_*.py', '**/*.js'
   • 比find命令更快更友好
   • 适用于：查找特定文件、批量定位

5. project_analysis - 项目分析 ⭐⭐⭐⭐⭐
   • action="summary": 项目概览（推荐）
   • action="structure": 目录树
   • action="statistics": 详细统计
   • action="dependencies": Python导入分析
   • action="large_files": 大文件检测
   • 适用于：理解项目、生成报告

6. file_tool - 文件操作
   • create: 创建文件并写入内容（⚠️ 单个文件内容不要超过100行）
   • read: 读取文件内容
   • append: 追加内容（用于分批写入大文件）
   • delete: 删除文件
   • exists: 检查文件是否存在
   
   ⚠️ 重要：创建大文件的正确方式
   • 如果代码超过100行，必须分多次写入
   • 方法1：先 create 写入前半部分，再 append 追加后半部分
   • 方法2：使用 bash_tool 创建文件：echo "代码" > file.py
   • 方法3：拆分为多个小文件（最推荐）

7. bash_tool - 执行bash命令
   • 查看文件：cat, head, tail
   • 系统信息：pwd, whoami, df, free
   • 统计分析：wc, awk, sed
   • ⚠️ 建议：优先使用专用工具，bash作为补充

8. task_tool - 并行任务管理
   • create: 创建多个子任务
   • execute_parallel: 并行执行
   • get_status: 查询状态

📋 工具选择决策树：

查看目录？
  → 用 list_dir（不要用ls）

查找文件名？
  → 用 glob_file_search（不要用find）

搜索代码内容？
  → 用 grep_tool（不要用bash grep）

修改代码？
  → 用 search_replace_tool（不要用sed）

了解项目？
  → 用 project_analysis action="summary"

查看文件内容？
  → 用 file_tool read

创建/修改文件？
  → 用 file_tool create 或 search_replace_tool

💡 典型场景处理方式：

场景1：用户问"分析项目结构"
❌ 错误：用多个bash命令逐个查看
✅ 正确：
   1. 用 project_analysis action="summary" 获取项目概览
   2. 如需详细结构，用 project_analysis action="structure"
   3. 基于结果给出总结

场景2：用户说"查找所有测试文件"
❌ 错误：用 bash_tool 执行 find
✅ 正确：
   1. 用 glob_file_search pattern="test_*.py"
   2. 或 glob_file_search pattern="**/*test*.py" recursive=true

场景3：用户问"这个项目用了哪些库"
❌ 错误：手动查看requirements.txt
✅ 正确：
   1. 用 project_analysis action="dependencies"
   2. 用 file_tool read "requirements.txt" 作为补充

场景4：用户说"把所有的old_api改成new_api"
❌ 错误：用 bash sed 命令
✅ 正确：
   1. 先用 grep_tool 搜索"old_api"定位所有位置
   2. 用 search_replace_tool 逐个文件替换，或replace_all=true批量替换

场景5：用户问"当前目录有什么"
❌ 错误：用 bash_tool 执行 ls
✅ 正确：
   1. 用 list_dir 列出当前目录
   2. 结果更清晰，自动分类文件和目录

场景6：用户说"检查是否有大文件"
❌ 错误：用 bash du 和 find 命令
✅ 正确：
   1. 用 project_analysis action="large_files"
   2. 可选设置 large_file_threshold=5（MB）

⚠️ 重要提醒：
• 专用工具 > bash命令（更快、更安全、输出更清晰）
• 一次性工具 > 多次调用（如用project_analysis代替多个ls/find）
• 每次都要先用工具获取实际数据
• 基于工具执行结果回答
• 不要凭空推测或给通用建议

🚀 工作模式：
看到用户请求 → 选择最合适的专用工具 → 调用工具 → 分析结果 → 给出答案

现在开始，主动、积极地使用工具！优先选择专用工具！"""
    
    def __init__(
        self,
        llm_client: K2Client,
        tools: List[BaseTool] = None,
        agent_id: str = "main",
        max_iterations: int = 20
    ):
        self.llm_client = llm_client
        self.agent_id = agent_id
        self.max_iterations = max_iterations
        
        # 工具注册
        self.tools: Dict[str, BaseTool] = {}
        if tools:
            for tool in tools:
                self.register_tool(tool)
        
        # 对话历史
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"Agent {self.agent_id} 初始化完成，注册了 {len(self.tools)} 个工具")
    
    def register_tool(self, tool: BaseTool):
        """注册工具"""
        self.tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")
    
    def get_tools_definition(self) -> List[Dict[str, Any]]:
        """获取所有工具的定义（OpenAI格式）"""
        return [tool.to_openai_tool() for tool in self.tools.values()]
    
    def run(self, task: str, reset_history: bool = False) -> Dict[str, Any]:
        """
        执行任务
        
        Args:
            task: 任务描述
            reset_history: 是否重置对话历史（默认False，保持多轮对话）
            
        Returns:
            执行结果
        """
        logger.info(f"[{self.agent_id}] 开始执行任务: {task}")
        
        # 如果需要重置或第一次运行，初始化对话
        if reset_history or not self.conversation_history:
            self.conversation_history = [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT
                }
            ]
        
        # 添加当前用户消息
        self.conversation_history.append({
            "role": "user",
            "content": task
        })
        
        # 执行推理循环
        for iteration in range(self.max_iterations):
            logger.info(f"[{self.agent_id}] 推理迭代 {iteration + 1}/{self.max_iterations}")
            
            try:
                # 调用LLM
                response = self.llm_client.chat(
                    messages=self.conversation_history,
                    tools=self.get_tools_definition()
                )
                
                # 如果有工具调用，添加assistant消息
                if response["tool_calls"]:
                    # 使用包含tool_calls的完整assistant消息
                    self.conversation_history.append(response["assistant_message"])
                else:
                    # 如果没有工具调用，只添加文本内容
                    if response["content"]:
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": response["content"]
                        })
                
                # 如果没有工具调用，说明任务完成
                if not response["tool_calls"]:
                    logger.info(f"[{self.agent_id}] 任务完成")
                    return {
                        "success": True,
                        "result": response["content"],
                        "iterations": iteration + 1,
                        "agent_id": self.agent_id
                    }
                
                # 执行工具调用
                tool_results = []
                for tool_call in response["tool_calls"]:
                    tool_call_id = tool_call["id"]
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]
                    
                    logger.info(f"[{self.agent_id}] 调用工具: {tool_name}")
                    logger.debug(f"工具参数: {tool_args}")
                    
                    if tool_name not in self.tools:
                        result = {
                            "success": False,
                            "error": f"工具 {tool_name} 不存在"
                        }
                    else:
                        tool = self.tools[tool_name]
                        result = tool.execute(**tool_args)
                    
                    tool_results.append({
                        "tool": tool_name,
                        "result": result
                    })
                    
                    # 将工具执行结果添加到对话历史（使用新版 tool 格式）
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
                    
                    # 🔧 智能错误重试机制
                    if not result.get("success", True):
                        error_msg = result.get("error", "未知错误")
                        suggestion = self._get_error_suggestion(tool_name, error_msg, tool_args)
                        
                        if suggestion:
                            logger.info(f"[{self.agent_id}] 工具调用失败，提供建议: {suggestion}")
                            # 添加系统建议到对话历史
                            self.conversation_history.append({
                                "role": "system",
                                "content": f"⚠️ 工具 {tool_name} 执行失败。\n错误: {error_msg}\n💡 建议: {suggestion}"
                            })
                
                logger.info(f"[{self.agent_id}] 完成 {len(tool_results)} 个工具调用")
                
            except Exception as e:
                logger.error(f"[{self.agent_id}] 执行出错: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "iterations": iteration + 1,
                    "agent_id": self.agent_id
                }
        
        # 达到最大迭代次数
        logger.warning(f"[{self.agent_id}] 达到最大迭代次数 {self.max_iterations}")
        return {
            "success": False,
            "error": f"达到最大迭代次数 {self.max_iterations}",
            "iterations": self.max_iterations,
            "agent_id": self.agent_id
        }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history
    
    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = []
        logger.info(f"[{self.agent_id}] 对话历史已重置")
    
    def _get_error_suggestion(
        self,
        tool_name: str,
        error_msg: str,
        tool_args: Dict[str, Any]
    ) -> Optional[str]:
        """
        根据工具执行错误，提供智能建议
        
        Args:
            tool_name: 工具名称
            error_msg: 错误消息
            tool_args: 工具参数
            
        Returns:
            建议信息（如果有）
        """
        error_lower = error_msg.lower()
        
        # 文件/路径相关错误
        if "不存在" in error_msg or "not found" in error_lower or "no such file" in error_lower:
            if tool_name in ["file_tool", "search_replace_tool"]:
                return (
                    f"文件或路径不存在。建议：\n"
                    f"1. 使用 list_dir 查看当前目录内容\n"
                    f"2. 使用 glob_file_search 搜索文件名\n"
                    f"3. 检查路径拼写是否正确"
                )
            elif tool_name == "grep_tool":
                return (
                    f"搜索路径不存在。建议：\n"
                    f"1. 使用 list_dir 确认目录结构\n"
                    f"2. 检查路径是否正确"
                )
        
        # 权限错误
        if "权限" in error_msg or "permission" in error_lower:
            return (
                f"权限不足。建议：\n"
                f"1. 检查文件/目录权限\n"
                f"2. 尝试访问其他路径\n"
                f"3. 使用 bash_tool 执行 'ls -l' 查看权限"
            )
        
        # 路径安全错误
        if "不安全" in error_msg or "security" in error_lower or "路径遍历" in error_msg:
            return (
                f"路径安全检查失败。建议：\n"
                f"1. 使用工作目录内的相对路径\n"
                f"2. 避免使用 ../ 或绝对路径\n"
                f"3. 检查路径是否在允许范围内"
            )
        
        # 搜索无结果
        if "未找到" in error_msg or "no matches" in error_lower or "找到 0" in error_msg:
            if tool_name == "grep_tool":
                pattern = tool_args.get("pattern", "")
                return (
                    f"未找到匹配项。建议：\n"
                    f"1. 尝试更宽泛的搜索模式\n"
                    f"2. 检查是否需要 case_insensitive=true\n"
                    f"3. 使用 list_dir 或 glob_file_search 确认文件存在\n"
                    f"4. 当前搜索: '{pattern}'，可能需要调整"
                )
            elif tool_name == "glob_file_search":
                pattern = tool_args.get("pattern", "")
                return (
                    f"未找到匹配文件。建议：\n"
                    f"1. 检查文件名模式是否正确: '{pattern}'\n"
                    f"2. 尝试 recursive=true 递归搜索\n"
                    f"3. 使用 list_dir 查看目录内容"
                )
        
        # 字符串替换失败
        if tool_name == "search_replace_tool":
            if "不存在" in error_msg or "未找到" in error_msg:
                old_string = tool_args.get("old_string", "")
                return (
                    f"未找到要替换的字符串。建议：\n"
                    f"1. 使用 file_tool read 查看文件内容\n"
                    f"2. 使用 grep_tool 搜索确认字符串存在\n"
                    f"3. 检查字符串是否完全匹配（包括空格、换行）\n"
                    f"4. 当前搜索: '{old_string[:50]}...'"
                )
            elif "不唯一" in error_msg or "多次匹配" in error_msg:
                return (
                    f"找到多个匹配项。建议：\n"
                    f"1. 增加更多上下文使匹配唯一\n"
                    f"2. 或使用 replace_all=true 替换所有匹配"
                )
        
        # 工具不存在
        if "不存在" in error_msg and "工具" in error_msg:
            return (
                f"工具不存在。可用工具：\n"
                f"search_replace_tool, grep_tool, list_dir, glob_file_search, "
                f"project_analysis, file_tool, bash_tool, task_tool"
            )
        
        # bash命令失败
        if tool_name == "bash_tool":
            if "command not found" in error_lower:
                return (
                    f"命令不存在。建议：\n"
                    f"1. 检查命令拼写\n"
                    f"2. 尝试使用专用工具代替bash命令\n"
                    f"3. 使用 'which <command>' 检查命令是否安装"
                )
            elif "超时" in error_msg or "timeout" in error_lower:
                return (
                    f"命令执行超时。建议：\n"
                    f"1. 简化命令或减少处理数据量\n"
                    f"2. 使用专用工具（如 list_dir, glob_file_search）\n"
                    f"3. 分解为多个小任务"
                )
        
        # 项目分析错误
        if tool_name == "project_analysis":
            action = tool_args.get("action", "")
            return (
                f"项目分析失败。建议：\n"
                f"1. 检查路径是否为目录\n"
                f"2. 尝试其他 action: summary, structure, statistics, dependencies\n"
                f"3. 当前 action: {action}"
            )
        
        # 通用建议
        return (
            f"工具执行失败。通用建议：\n"
            f"1. 检查参数是否正确\n"
            f"2. 尝试使用其他工具完成任务\n"
            f"3. 查看错误信息了解具体原因"
        )
    
    def run_stream(self, task: str, reset_history: bool = False) -> Iterator[Dict[str, Any]]:
        """
        流式执行任务
        
        Args:
            task: 任务描述
            reset_history: 是否重置对话历史
            
        Yields:
            执行过程中的各种事件
        """
        logger.info(f"[{self.agent_id}] 开始流式执行任务: {task}")
        
        # 如果需要重置或第一次运行，初始化对话
        if reset_history or not self.conversation_history:
            self.conversation_history = [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT
                }
            ]
        
        # 添加当前用户消息
        self.conversation_history.append({
            "role": "user",
            "content": task
        })
        
        # 执行推理循环
        for iteration in range(self.max_iterations):
            logger.info(f"[{self.agent_id}] 推理迭代 {iteration + 1}/{self.max_iterations}")
            
            yield {
                "type": "iteration",
                "iteration": iteration + 1,
                "max_iterations": self.max_iterations
            }
            
            try:
                # 调用LLM（流式）
                final_response = None
                
                for chunk in self.llm_client.chat_stream(
                    messages=self.conversation_history,
                    tools=self.get_tools_definition()
                ):
                    if chunk.get("type") == "content":
                        # 流式输出文本内容
                        yield {
                            "type": "content",
                            "content": chunk["content"]
                        }
                    elif chunk.get("type") == "tool_call_delta":
                        # 工具调用进度
                        yield {
                            "type": "tool_call_delta",
                            "tool_calls": chunk["tool_calls"]
                        }
                    elif chunk.get("type") == "final":
                        # 最终结果
                        final_response = chunk
                    elif chunk.get("type") == "error":
                        # 错误
                        yield {
                            "type": "error",
                            "error": chunk["error"]
                        }
                        return
                
                if not final_response:
                    yield {
                        "type": "error",
                        "error": "未收到完整响应"
                    }
                    return
                
                # 处理响应
                if final_response.get("tool_calls"):
                    # 使用包含tool_calls的完整assistant消息
                    self.conversation_history.append(final_response["assistant_message"])
                else:
                    # 如果没有工具调用，只添加文本内容
                    if final_response.get("content"):
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": final_response["content"]
                        })
                
                # 如果没有工具调用，说明任务完成
                if not final_response.get("tool_calls"):
                    logger.info(f"[{self.agent_id}] 任务完成")
                    yield {
                        "type": "complete",
                        "result": final_response["content"],
                        "iterations": iteration + 1
                    }
                    return
                
                # 执行工具调用
                tool_results = []
                for tool_call in final_response["tool_calls"]:
                    tool_call_id = tool_call["id"]
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]
                    
                    logger.info(f"[{self.agent_id}] 调用工具: {tool_name}")
                    logger.debug(f"工具参数: {tool_args}")
                    
                    yield {
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "tool_args": tool_args
                    }
                    
                    if tool_name not in self.tools:
                        result = {
                            "success": False,
                            "error": f"工具 {tool_name} 不存在"
                        }
                    else:
                        tool = self.tools[tool_name]
                        result = tool.execute(**tool_args)
                    
                    tool_results.append({
                        "tool": tool_name,
                        "result": result
                    })
                    
                    yield {
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "result": result
                    }
                    
                    # 将工具执行结果添加到对话历史
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
                
                logger.info(f"[{self.agent_id}] 完成 {len(tool_results)} 个工具调用")
                
            except Exception as e:
                logger.error(f"[{self.agent_id}] 执行出错: {e}")
                yield {
                    "type": "error",
                    "error": str(e),
                    "iterations": iteration + 1
                }
                return
        
        # 达到最大迭代次数
        logger.warning(f"[{self.agent_id}] 达到最大迭代次数 {self.max_iterations}")
        yield {
            "type": "max_iterations",
            "error": f"达到最大迭代次数 {self.max_iterations}",
            "iterations": self.max_iterations
        }


class AgentManager:
    """
    Agent管理器
    
    负责创建和管理sub-agent，支持并行执行任务
    """
    
    def __init__(self, llm_client: K2Client):
        self.llm_client = llm_client
        self.sub_agents: Dict[str, Agent] = {}
        self._agent_counter = 0
    
    def create_main_agent(self) -> Agent:
        """创建主Agent"""
        # 创建共享的安全文件处理器
        safe_handler = SafeFileHandler()
        
        # 创建工具（按优先级排序）
        search_replace_tool = SearchReplaceTool(safe_handler=safe_handler)  # 精确代码编辑
        grep_tool = GrepTool(safe_handler=safe_handler)  # 代码搜索
        list_dir_tool = ListDirTool(safe_handler=safe_handler)  # 目录列表
        glob_search_tool = GlobSearchTool(safe_handler=safe_handler)  # 文件名搜索
        project_analysis_tool = ProjectAnalysisTool(safe_handler=safe_handler)  # 项目分析
        file_tool = FileTool(safe_handler=safe_handler)  # 文件操作
        bash_tool = BashTool()  # Bash命令工具
        task_tool = TaskTool(agent_manager=self)  # 任务管理工具
        
        agent = Agent(
            llm_client=self.llm_client,
            tools=[
                search_replace_tool,
                grep_tool,
                list_dir_tool,
                glob_search_tool,
                project_analysis_tool,
                file_tool,
                bash_tool,
                task_tool
            ],
            agent_id="main",
            max_iterations=30  # 可以为主 Agent 单独设置更大的值
        )
        
        return agent
    
    def create_sub_agent(self, agent_id: str = None) -> Agent:
        """创建sub-agent"""
        if agent_id is None:
            self._agent_counter += 1
            agent_id = f"sub-agent-{self._agent_counter}"
        
        # 创建安全文件处理器
        safe_handler = SafeFileHandler()
        
        # sub-agent 也拥有核心工具
        search_replace_tool = SearchReplaceTool(safe_handler=safe_handler)
        grep_tool = GrepTool(safe_handler=safe_handler)
        list_dir_tool = ListDirTool(safe_handler=safe_handler)
        glob_search_tool = GlobSearchTool(safe_handler=safe_handler)
        project_analysis_tool = ProjectAnalysisTool(safe_handler=safe_handler)
        file_tool = FileTool(safe_handler=safe_handler)
        bash_tool = BashTool()
        
        agent = Agent(
            llm_client=self.llm_client,
            tools=[
                search_replace_tool, grep_tool, list_dir_tool,
                glob_search_tool, project_analysis_tool,
                file_tool, bash_tool
            ],
            agent_id=agent_id,
            max_iterations=5  # sub-agent的迭代次数可以少一些
        )
        
        self.sub_agents[agent_id] = agent
        logger.info(f"创建sub-agent: {agent_id}")
        
        return agent
    
    def execute_task_with_sub_agent_sync(self, task: str, agent_id: str = None) -> Dict[str, Any]:
        """
        使用 sub-agent 执行任务（同步版本）
        
        Args:
            task: 任务描述
            agent_id: 指定的 agent ID（可选）
            
        Returns:
            执行结果
        """
        # 创建 sub-agent
        agent = self.create_sub_agent(agent_id)
        
        logger.info(f"Sub-agent {agent.agent_id} 开始执行任务: {task[:50]}...")
        
        try:
            # 执行任务
            result = agent.run(task, reset_history=True)
            
            logger.info(f"Sub-agent {agent.agent_id} 任务完成")
            
            return {
                "success": result.get("success", False),
                "agent_id": agent.agent_id,
                "result": result.get("result"),
                "iterations": result.get("iterations"),
                "error": result.get("error")
            }
        
        except Exception as e:
            logger.error(f"Sub-agent {agent.agent_id} 执行失败: {e}")
            return {
                "success": False,
                "agent_id": agent.agent_id,
                "error": str(e),
                "result": None
            }
    
    async def execute_task_with_sub_agent(self, task: str, agent_id: str = None) -> Dict[str, Any]:
        """
        使用 sub-agent 执行任务（异步版本）
        
        Args:
            task: 任务描述
            agent_id: 指定的 agent ID（可选）
            
        Returns:
            执行结果
        """
        # 创建 sub-agent
        agent = self.create_sub_agent(agent_id)
        
        logger.info(f"Sub-agent {agent.agent_id} 开始异步执行任务: {task[:50]}...")
        
        try:
            # 在线程池中执行同步的 run 方法
            import concurrent.futures
            
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    lambda: agent.run(task, reset_history=True)
                )
            
            logger.info(f"Sub-agent {agent.agent_id} 异步任务完成")
            
            return {
                "success": result.get("success", False),
                "agent_id": agent.agent_id,
                "result": result.get("result"),
                "iterations": result.get("iterations"),
                "error": result.get("error")
            }
        
        except Exception as e:
            logger.error(f"Sub-agent {agent.agent_id} 异步执行失败: {e}")
            return {
                "success": False,
                "agent_id": agent.agent_id,
                "error": str(e),
                "result": None
            }
    
    def get_all_agents(self) -> Dict[str, Agent]:
        """获取所有 sub-agent"""
        return self.sub_agents
    
    def cleanup_sub_agents(self) -> int:
        """
        清理已完成的 sub-agent
        
        Returns:
            清理的 agent 数量
        """
        count = len(self.sub_agents)
        self.sub_agents.clear()
        logger.info(f"清理了 {count} 个 sub-agent")
        return count
    
    def get_sub_agent_count(self) -> int:
        """获取当前 sub-agent 数量"""
        return len(self.sub_agents)

