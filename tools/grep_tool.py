"""代码搜索工具 - 在文件中搜索文本和模式"""
import os
import re
import subprocess
from typing import Dict, Any, List, Optional
from loguru import logger
from .base import BaseTool, ToolParameter
from .safe_file_handler import SafeFileHandler


class GrepTool(BaseTool):
    """
    代码搜索工具
    
    功能：
    1. 在文件或目录中搜索文本模式
    2. 支持正则表达式
    3. 支持文件类型过滤
    4. 显示匹配行和上下文
    5. 优先使用 ripgrep (rg)，回退到 grep
    """
    
    def __init__(self, safe_handler: Optional[SafeFileHandler] = None):
        """
        初始化搜索工具
        
        Args:
            safe_handler: 安全文件处理器
        """
        self.safe_handler = safe_handler or SafeFileHandler()
        
        # 检测可用的搜索工具
        self.search_cmd = self._detect_search_tool()
        logger.info(f"GrepTool 使用搜索工具: {self.search_cmd}")
    
    def _detect_search_tool(self) -> str:
        """检测可用的搜索工具"""
        # 尝试 ripgrep
        try:
            subprocess.run(
                ["rg", "--version"],
                capture_output=True,
                timeout=1
            )
            return "rg"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # 回退到 grep
        try:
            subprocess.run(
                ["grep", "--version"],
                capture_output=True,
                timeout=1
            )
            return "grep"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # 如果都不可用，使用 Python 实现
        return "python"
    
    @property
    def name(self) -> str:
        return "grep_tool"
    
    @property
    def description(self) -> str:
        return """代码搜索工具。在文件或目录中搜索文本模式。
        
        特性：
        - 支持正则表达式搜索
        - 支持文件类型过滤（如只搜索.py文件）
        - 显示匹配行号和上下文
        - 高性能（使用 ripgrep 或 grep）
        - 自动排除 .git, __pycache__, .venv 等目录
        
        使用场景：
        - 查找函数定义
        - 搜索变量使用位置
        - 查找 TODO 注释
        - 搜索错误信息
        - 代码重构前的影响分析"""
    
    @property
    def parameters(self) -> Dict[str, ToolParameter]:
        return {
            "pattern": ToolParameter(
                type="string",
                description="搜索模式（支持正则表达式）",
                required=True
            ),
            "path": ToolParameter(
                type="string",
                description="搜索路径（文件或目录，默认为当前目录）",
                required=False
            ),
            "file_type": ToolParameter(
                type="string",
                description="文件类型过滤（如 'py', 'js', 'md'）",
                required=False
            ),
            "case_insensitive": ToolParameter(
                type="boolean",
                description="是否忽略大小写（默认False）",
                required=False
            ),
            "context_lines": ToolParameter(
                type="integer",
                description="显示上下文行数（默认2）",
                required=False
            ),
            "max_results": ToolParameter(
                type="integer",
                description="最大返回结果数（默认50）",
                required=False
            )
        }
    
    def execute(
        self,
        pattern: str,
        path: str = ".",
        file_type: str = None,
        case_insensitive: bool = False,
        context_lines: int = 2,
        max_results: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行搜索
        
        Args:
            pattern: 搜索模式
            path: 搜索路径
            file_type: 文件类型
            case_insensitive: 是否忽略大小写
            context_lines: 上下文行数
            max_results: 最大结果数
            
        Returns:
            搜索结果
        """
        logger.info(f"搜索: pattern='{pattern}', path='{path}'")
        
        try:
            # 验证路径
            abs_path = self.safe_handler.validate_path(path)
            
            if not os.path.exists(abs_path):
                return {
                    "success": False,
                    "error": f"路径不存在: {path}",
                    "path": path
                }
            
            # 根据可用工具选择搜索方法
            if self.search_cmd == "rg":
                results = self._search_with_rg(
                    pattern, abs_path, file_type,
                    case_insensitive, context_lines, max_results
                )
            elif self.search_cmd == "grep":
                results = self._search_with_grep(
                    pattern, abs_path, file_type,
                    case_insensitive, context_lines, max_results
                )
            else:
                results = self._search_with_python(
                    pattern, abs_path, file_type,
                    case_insensitive, context_lines, max_results
                )
            
            return results
        
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "pattern": pattern,
                "path": path
            }
    
    def _search_with_rg(
        self,
        pattern: str,
        path: str,
        file_type: Optional[str],
        case_insensitive: bool,
        context_lines: int,
        max_results: int
    ) -> Dict[str, Any]:
        """使用 ripgrep 搜索"""
        cmd = ["rg", pattern, path]
        
        # 添加选项
        cmd.extend(["--line-number"])  # 显示行号
        cmd.extend(["-C", str(context_lines)])  # 上下文行数
        cmd.extend(["--max-count", str(max_results)])  # 每个文件最大匹配数
        
        if case_insensitive:
            cmd.append("--ignore-case")
        
        if file_type:
            cmd.extend(["--type", file_type])
        
        # 自动排除常见目录
        cmd.extend([
            "--glob", "!.git/",
            "--glob", "!__pycache__/",
            "--glob", "!.venv/",
            "--glob", "!node_modules/",
            "--glob", "!*.pyc"
        ])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # rg 返回码：0=找到，1=未找到，2=错误
            if result.returncode == 2:
                return {
                    "success": False,
                    "error": f"ripgrep 错误: {result.stderr}",
                    "pattern": pattern
                }
            
            matches = self._parse_rg_output(result.stdout)
            
            return {
                "success": True,
                "pattern": pattern,
                "path": self.safe_handler.get_relative_path(path),
                "match_count": len(matches),
                "matches": matches[:max_results],
                "truncated": len(matches) > max_results,
                "tool": "ripgrep"
            }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "搜索超时（10秒）",
                "pattern": pattern
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"ripgrep 执行失败: {str(e)}",
                "pattern": pattern
            }
    
    def _search_with_grep(
        self,
        pattern: str,
        path: str,
        file_type: Optional[str],
        case_insensitive: bool,
        context_lines: int,
        max_results: int
    ) -> Dict[str, Any]:
        """使用 grep 搜索"""
        cmd = ["grep", "-rn"]  # 递归，显示行号
        
        cmd.extend(["-C", str(context_lines)])  # 上下文
        
        if case_insensitive:
            cmd.append("-i")
        
        # 排除常见目录
        cmd.extend([
            "--exclude-dir=.git",
            "--exclude-dir=__pycache__",
            "--exclude-dir=.venv",
            "--exclude-dir=node_modules"
        ])
        
        if file_type:
            cmd.extend(["--include", f"*.{file_type}"])
        
        cmd.extend([pattern, path])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            matches = self._parse_grep_output(result.stdout)
            
            return {
                "success": True,
                "pattern": pattern,
                "path": self.safe_handler.get_relative_path(path),
                "match_count": len(matches),
                "matches": matches[:max_results],
                "truncated": len(matches) > max_results,
                "tool": "grep"
            }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "搜索超时（10秒）",
                "pattern": pattern
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"grep 执行失败: {str(e)}",
                "pattern": pattern
            }
    
    def _search_with_python(
        self,
        pattern: str,
        path: str,
        file_type: Optional[str],
        case_insensitive: bool,
        context_lines: int,
        max_results: int
    ) -> Dict[str, Any]:
        """使用 Python 实现搜索"""
        try:
            # 编译正则表达式
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern, flags)
        except re.error as e:
            return {
                "success": False,
                "error": f"无效的正则表达式: {str(e)}",
                "pattern": pattern
            }
        
        matches = []
        
        # 排除目录
        exclude_dirs = {".git", "__pycache__", ".venv", "node_modules", ".tox"}
        
        # 遍历文件
        if os.path.isfile(path):
            files = [path]
        else:
            files = []
            for root, dirs, filenames in os.walk(path):
                # 排除目录
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                for filename in filenames:
                    # 文件类型过滤
                    if file_type and not filename.endswith(f".{file_type}"):
                        continue
                    
                    # 排除 .pyc 等
                    if filename.endswith(('.pyc', '.pyo', '.so', '.dll')):
                        continue
                    
                    files.append(os.path.join(root, filename))
        
        # 搜索文件
        for file_path in files:
            if len(matches) >= max_results:
                break
            
            try:
                # 检查是否为文本文件
                if not self.safe_handler.is_text_file(file_path):
                    continue
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                # 搜索每一行
                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        # 获取上下文
                        start = max(0, line_num - context_lines - 1)
                        end = min(len(lines), line_num + context_lines)
                        
                        context = []
                        for i in range(start, end):
                            context.append({
                                "line_number": i + 1,
                                "content": lines[i].rstrip('\n'),
                                "is_match": i + 1 == line_num
                            })
                        
                        rel_path = self.safe_handler.get_relative_path(file_path)
                        
                        matches.append({
                            "file": rel_path,
                            "line_number": line_num,
                            "line": line.rstrip('\n'),
                            "context": context
                        })
                        
                        if len(matches) >= max_results:
                            break
            
            except Exception as e:
                logger.debug(f"跳过文件 {file_path}: {e}")
                continue
        
        return {
            "success": True,
            "pattern": pattern,
            "path": self.safe_handler.get_relative_path(path),
            "match_count": len(matches),
            "matches": matches,
            "truncated": False,
            "tool": "python"
        }
    
    def _parse_rg_output(self, output: str) -> List[Dict[str, Any]]:
        """解析 ripgrep 输出"""
        matches = []
        current_file = None
        current_match = None
        
        for line in output.split('\n'):
            if not line:
                continue
            
            # 文件名:行号:内容
            parts = line.split(':', 2)
            if len(parts) >= 3:
                file_path = parts[0]
                line_number = parts[1]
                content = parts[2] if len(parts) > 2 else ""
                
                if file_path != current_file or (current_match and 
                    abs(int(line_number) - current_match["line_number"]) > 5):
                    # 新匹配
                    current_file = file_path
                    current_match = {
                        "file": self.safe_handler.get_relative_path(file_path),
                        "line_number": int(line_number),
                        "line": content,
                        "context": []
                    }
                    matches.append(current_match)
                
                # 添加上下文
                if current_match:
                    current_match["context"].append({
                        "line_number": int(line_number),
                        "content": content,
                        "is_match": True  # ripgrep 不区分匹配行和上下文行
                    })
        
        return matches
    
    def _parse_grep_output(self, output: str) -> List[Dict[str, Any]]:
        """解析 grep 输出"""
        # grep 输出格式类似 ripgrep
        return self._parse_rg_output(output)

