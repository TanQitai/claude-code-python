"""精确代码编辑工具 - 支持字符串搜索和替换"""
from typing import Dict, Any, Optional
from loguru import logger
from .base import BaseTool, ToolParameter
from .safe_file_handler import SafeFileHandler


class SearchReplaceTool(BaseTool):
    """
    精确代码编辑工具
    
    功能：
    1. 精确字符串搜索和替换
    2. 支持全局替换或单次替换
    3. 保留代码格式和缩进
    4. 安全的文件操作
    """
    
    def __init__(self, safe_handler: Optional[SafeFileHandler] = None):
        """
        初始化搜索替换工具
        
        Args:
            safe_handler: 安全文件处理器（可选，默认创建新的）
        """
        self.safe_handler = safe_handler or SafeFileHandler()
    
    @property
    def name(self) -> str:
        return "search_replace_tool"
    
    @property
    def description(self) -> str:
        return """精确代码编辑工具。用于在文件中搜索并替换文本。
        
        特性：
        - 精确字符串匹配（不是正则表达式）
        - 支持单次替换或全局替换
        - 保留所有空格、缩进和换行
        - 自动备份原文件
        - 安全路径验证
        
        使用场景：
        - 修改变量名、函数名
        - 更新配置值
        - 修复代码错误
        - 重构代码片段
        
        注意：old_string 必须在文件中精确存在，包括所有空格和缩进"""
    
    @property
    def parameters(self) -> Dict[str, ToolParameter]:
        return {
            "file_path": ToolParameter(
                type="string",
                description="要编辑的文件路径（相对或绝对路径）",
                required=True
            ),
            "old_string": ToolParameter(
                type="string",
                description="要查找的字符串（必须精确匹配，包括空格和缩进）",
                required=True
            ),
            "new_string": ToolParameter(
                type="string",
                description="替换后的字符串",
                required=True
            ),
            "replace_all": ToolParameter(
                type="boolean",
                description="是否替换所有匹配项（默认False，只替换第一个）",
                required=False
            )
        }
    
    def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行搜索替换
        
        Args:
            file_path: 文件路径
            old_string: 要查找的字符串
            new_string: 替换后的字符串
            replace_all: 是否替换所有匹配项
            
        Returns:
            执行结果
        """
        logger.info(f"搜索替换: {file_path}")
        logger.debug(f"查找: {old_string[:50]}... -> 替换: {new_string[:50]}...")
        
        try:
            # 验证参数
            if not old_string:
                return {
                    "success": False,
                    "error": "old_string 不能为空"
                }
            
            if old_string == new_string:
                return {
                    "success": False,
                    "error": "old_string 和 new_string 相同，无需替换"
                }
            
            # 读取文件
            try:
                content = self.safe_handler.safe_read(file_path)
            except FileNotFoundError as e:
                return {
                    "success": False,
                    "error": f"文件不存在: {file_path}",
                    "file_path": file_path
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"读取文件失败: {str(e)}",
                    "file_path": file_path
                }
            
            # 检查字符串是否存在
            if old_string not in content:
                return {
                    "success": False,
                    "error": "在文件中未找到指定的字符串",
                    "file_path": file_path,
                    "old_string": old_string[:100],
                    "suggestion": "请确保 old_string 完全匹配，包括空格、缩进和换行符"
                }
            
            # 统计匹配次数
            match_count = content.count(old_string)
            
            # 检查唯一性（如果不是全局替换）
            if not replace_all and match_count > 1:
                # 获取匹配位置的上下文
                contexts = self._get_match_contexts(content, old_string, max_contexts=3)
                
                return {
                    "success": False,
                    "error": f"字符串在文件中出现 {match_count} 次，不唯一",
                    "file_path": file_path,
                    "match_count": match_count,
                    "contexts": contexts,
                    "suggestion": "请提供更多上下文使字符串唯一，或使用 replace_all=true 替换所有匹配项"
                }
            
            # 执行替换
            if replace_all:
                new_content = content.replace(old_string, new_string)
                replaced_count = match_count
            else:
                # 只替换第一个
                new_content = content.replace(old_string, new_string, 1)
                replaced_count = 1
            
            # 写入文件
            self.safe_handler.safe_write(file_path, new_content)
            
            # 获取相对路径
            rel_path = self.safe_handler.get_relative_path(file_path)
            
            logger.info(f"替换成功: {file_path}, 替换了 {replaced_count} 处")
            
            return {
                "success": True,
                "file_path": rel_path,
                "replaced_count": replaced_count,
                "total_matches": match_count,
                "replace_all": replace_all,
                "old_length": len(old_string),
                "new_length": len(new_string),
                "message": f"成功替换 {replaced_count} 处"
            }
        
        except Exception as e:
            logger.error(f"搜索替换失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    def _get_match_contexts(
        self,
        content: str,
        search_string: str,
        max_contexts: int = 3,
        context_lines: int = 2
    ) -> list:
        """
        获取匹配位置的上下文
        
        Args:
            content: 文件内容
            search_string: 搜索字符串
            max_contexts: 最多返回几个匹配位置
            context_lines: 上下文行数
            
        Returns:
            匹配位置的上下文列表
        """
        lines = content.split('\n')
        contexts = []
        
        # 查找所有匹配行
        match_line_numbers = []
        for i, line in enumerate(lines):
            if search_string in line:
                match_line_numbers.append(i)
        
        # 获取上下文
        for line_num in match_line_numbers[:max_contexts]:
            start = max(0, line_num - context_lines)
            end = min(len(lines), line_num + context_lines + 1)
            
            context_lines_list = []
            for i in range(start, end):
                marker = ">>> " if i == line_num else "    "
                context_lines_list.append(f"{marker}{i+1}: {lines[i]}")
            
            contexts.append({
                "line_number": line_num + 1,
                "context": "\n".join(context_lines_list)
            })
        
        return contexts

