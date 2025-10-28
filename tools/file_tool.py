"""文件操作工具 - 专门用于文件的创建、读取、写入、删除等操作"""
import os
from typing import Dict, Any, Optional
from loguru import logger
from .base import BaseTool, ToolParameter
from .safe_file_handler import SafeFileHandler, SecurityError


class FileTool(BaseTool):
    """
    文件操作工具
    
    提供更友好的文件操作接口，使 LLM 更容易理解和使用
    集成安全文件处理器，防止路径遍历等安全问题
    """
    
    def __init__(self, safe_handler: Optional[SafeFileHandler] = None):
        """
        初始化文件工具
        
        Args:
            safe_handler: 安全文件处理器（可选，默认创建新的）
        """
        self.safe_handler = safe_handler or SafeFileHandler()
    
    @property
    def name(self) -> str:
        return "file_tool"
        
    
    @property
    def description(self) -> str:
        return """文件操作工具。用于创建、读取、写入、删除文件。
        这是最推荐用于文件操作的工具，比 bash_tool 更直接。
        
        支持的操作：
        - create: 创建新文件并写入内容
        - read: 读取文件内容
        - append: 追加内容到文件
        - delete: 删除文件
        - exists: 检查文件是否存在"""
    
    @property
    def parameters(self) -> Dict[str, ToolParameter]:
        return {
            "action": ToolParameter(
                type="string",
                description="操作类型",
                enum=["create", "read", "append", "delete", "exists"],
                required=True
            ),
            "file_path": ToolParameter(
                type="string",
                description="文件路径（相对或绝对路径）",
                required=True
            ),
            "content": ToolParameter(
                type="string",
                description="文件内容（用于 create 和 append 操作）",
                required=False
            )
        }
    
    def execute(self, action: str, file_path: str, content: str = None, **kwargs) -> Dict[str, Any]:
        """
        执行文件操作
        
        Args:
            action: 操作类型 (create, read, append, delete, exists)
            file_path: 文件路径
            content: 文件内容
        """
        logger.info(f"FileTool 执行操作: {action} - {file_path}")
        
        try:
            if action == "create":
                return self._create_file(file_path, content)
            elif action == "read":
                return self._read_file(file_path)
            elif action == "append":
                return self._append_file(file_path, content)
            elif action == "delete":
                return self._delete_file(file_path)
            elif action == "exists":
                return self._check_exists(file_path)
            else:
                return {
                    "success": False,
                    "error": f"未知的操作类型: {action}"
                }
        
        except SecurityError as e:
            logger.error(f"文件操作安全错误: {e}")
            return {
                "success": False,
                "error": f"安全错误: {str(e)}",
                "action": action,
                "file_path": file_path
            }
        
        except Exception as e:
            logger.error(f"文件操作失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "action": action,
                "file_path": file_path
            }
    
    def _create_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """创建文件"""
        if content is None:
            return {
                "success": False,
                "error": "create 操作需要提供 content 参数"
            }
        
        # 使用安全处理器写入文件
        self.safe_handler.safe_write(file_path, content)
        
        # 获取相对路径
        rel_path = self.safe_handler.get_relative_path(file_path)
        
        logger.info(f"文件创建成功: {rel_path}")
        
        return {
            "success": True,
            "action": "create",
            "file_path": rel_path,
            "size": len(content),
            "message": f"文件已创建: {rel_path}"
        }
    
    def _read_file(self, file_path: str) -> Dict[str, Any]:
        """读取文件"""
        # 使用安全处理器读取文件
        content = self.safe_handler.safe_read(file_path)
        
        # 获取相对路径
        rel_path = self.safe_handler.get_relative_path(file_path)
        
        logger.info(f"文件读取成功: {rel_path}")
        
        return {
            "success": True,
            "action": "read",
            "file_path": rel_path,
            "content": content,
            "size": len(content)
        }
    
    def _append_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """追加内容到文件"""
        if content is None:
            return {
                "success": False,
                "error": "append 操作需要提供 content 参数"
            }
        
        # 先读取现有内容
        try:
            existing_content = self.safe_handler.safe_read(file_path)
        except FileNotFoundError:
            existing_content = ""
        
        # 追加新内容
        new_content = existing_content + content
        self.safe_handler.safe_write(file_path, new_content)
        
        # 获取相对路径
        rel_path = self.safe_handler.get_relative_path(file_path)
        
        logger.info(f"内容已追加到文件: {rel_path}")
        
        return {
            "success": True,
            "action": "append",
            "file_path": rel_path,
            "appended_size": len(content),
            "message": f"内容已追加到: {rel_path}"
        }
    
    def _delete_file(self, file_path: str) -> Dict[str, Any]:
        """删除文件"""
        # 验证路径并获取绝对路径
        abs_path = self.safe_handler.validate_path(file_path, check_exists=True)
        
        # 删除文件
        os.remove(abs_path)
        
        # 获取相对路径
        rel_path = self.safe_handler.get_relative_path(file_path)
        
        logger.info(f"文件已删除: {rel_path}")
        
        return {
            "success": True,
            "action": "delete",
            "file_path": rel_path,
            "message": f"文件已删除: {rel_path}"
        }
    
    def _check_exists(self, file_path: str) -> Dict[str, Any]:
        """检查文件是否存在"""
        # 验证路径
        try:
            abs_path = self.safe_handler.validate_path(file_path, check_exists=False)
            exists = os.path.exists(abs_path)
        except SecurityError:
            # 如果路径不安全，返回不存在
            exists = False
        
        # 获取相对路径
        rel_path = self.safe_handler.get_relative_path(file_path) if exists else file_path
        
        return {
            "success": True,
            "action": "exists",
            "file_path": rel_path,
            "exists": exists,
            "message": f"文件{'存在' if exists else '不存在'}: {rel_path}"
        }

