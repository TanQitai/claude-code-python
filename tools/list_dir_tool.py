"""目录列表工具 - 智能列出目录内容"""
from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger
from .base import BaseTool, ToolParameter
from .safe_file_handler import SafeFileHandler, SecurityError


class ListDirTool(BaseTool):
    """
    目录列表工具
    
    功能：
    1. 列出指定目录的文件和子目录
    2. 支持递归列出
    3. 自动排除常见无用目录
    4. 显示文件大小和类型
    5. 集成安全路径验证
    """
    
    # 默认排除的目录
    DEFAULT_IGNORE_DIRS = {
        '.git', '.svn', '.hg',  # 版本控制
        '__pycache__', '.pytest_cache', '.mypy_cache',  # Python缓存
        'node_modules', 'bower_components',  # JS依赖
        '.venv', 'venv', 'env', 'virtualenv',  # Python虚拟环境
        'dist', 'build', '.egg-info',  # 构建产物
        '.idea', '.vscode', '.vs',  # IDE配置
        '.DS_Store', 'Thumbs.db',  # 系统文件
    }
    
    # 默认排除的文件模式
    DEFAULT_IGNORE_FILES = {
        '.pyc', '.pyo', '.pyd',  # Python编译文件
        '.so', '.dll', '.dylib',  # 二进制库
        '.log',  # 日志文件
    }
    
    def __init__(self, safe_handler: Optional[SafeFileHandler] = None):
        self.safe_handler = safe_handler or SafeFileHandler()
    
    @property
    def name(self) -> str:
        return "list_dir"
    
    @property
    def description(self) -> str:
        return "列出目录内容，支持递归列出，自动排除无用文件。自动排除.git, __pycache__, .venv等常见无用目录。"
    
    @property
    def parameters(self) -> Dict[str, ToolParameter]:
        return {
            "path": ToolParameter(
                type="string",
                description="要列出的目录路径（默认为当前目录）",
                required=False
            ),
            "recursive": ToolParameter(
                type="boolean",
                description="是否递归列出子目录（默认False）",
                required=False
            ),
            "max_depth": ToolParameter(
                type="integer",
                description="递归最大深度（默认3，防止输出过多）",
                required=False
            ),
            "show_hidden": ToolParameter(
                type="boolean",
                description="是否显示隐藏文件（以.开头，默认False）",
                required=False
            ),
            "file_types": ToolParameter(
                type="array",
                description="只显示指定类型的文件（如['.py', '.js']，为空则全部显示）",
                required=False
            )
        }
    
    def execute(
        self,
        path: str = ".",
        recursive: bool = False,
        max_depth: int = 3,
        show_hidden: bool = False,
        file_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        列出目录内容
        
        Args:
            path: 目录路径
            recursive: 是否递归
            max_depth: 递归最大深度
            show_hidden: 是否显示隐藏文件
            file_types: 文件类型过滤（如['.py', '.js']）
            
        Returns:
            包含目录和文件列表的字典
        """
        try:
            # 安全验证路径
            try:
                validated_path = self.safe_handler.validate_path(path)
            except SecurityError as e:
                logger.warning(f"路径安全检查失败: {e}")
                return {
                    "success": False,
                    "error": f"路径不安全: {str(e)}"
                }
            
            path_obj = Path(validated_path)
            
            # 检查路径是否存在
            if not path_obj.exists():
                return {
                    "success": False,
                    "error": f"路径不存在: {path}"
                }
            
            # 检查是否为目录
            if not path_obj.is_dir():
                return {
                    "success": False,
                    "error": f"路径不是目录: {path}"
                }
            
            logger.info(f"列出目录: {path} (recursive={recursive}, max_depth={max_depth})")
            
            if recursive:
                result = self._list_recursive(
                    path_obj, 
                    max_depth=max_depth,
                    show_hidden=show_hidden,
                    file_types=file_types
                )
            else:
                result = self._list_single(
                    path_obj,
                    show_hidden=show_hidden,
                    file_types=file_types
                )
            
            return {
                "success": True,
                "path": str(path),
                "total_dirs": result["total_dirs"],
                "total_files": result["total_files"],
                "directories": result["directories"],
                "files": result["files"],
                "message": f"找到 {result['total_dirs']} 个目录，{result['total_files']} 个文件"
            }
            
        except PermissionError as e:
            logger.error(f"权限不足: {e}")
            return {
                "success": False,
                "error": f"权限不足，无法访问: {path}"
            }
        except Exception as e:
            logger.error(f"列出目录失败: {e}")
            return {
                "success": False,
                "error": f"列出目录失败: {str(e)}"
            }
    
    def _should_ignore(self, path: Path, show_hidden: bool = False) -> bool:
        """判断是否应该忽略该路径"""
        name = path.name
        
        # 隐藏文件/目录
        if not show_hidden and name.startswith('.'):
            return True
        
        # 忽略的目录
        if path.is_dir() and name in self.DEFAULT_IGNORE_DIRS:
            return True
        
        # 忽略的文件后缀
        if path.is_file() and path.suffix in self.DEFAULT_IGNORE_FILES:
            return True
        
        return False
    
    def _matches_file_type(self, path: Path, file_types: Optional[List[str]]) -> bool:
        """检查文件是否匹配指定类型"""
        if not file_types:
            return True
        if path.is_dir():
            return True
        return path.suffix in file_types
    
    def _get_file_info(self, path: Path) -> Dict[str, Any]:
        """获取文件/目录信息"""
        try:
            stat = path.stat()
            size = stat.st_size
            
            # 格式化文件大小
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f}KB"
            elif size < 1024 * 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.1f}MB"
            else:
                size_str = f"{size / (1024 * 1024 * 1024):.1f}GB"
            
            return {
                "name": path.name,
                "path": str(path),
                "type": "directory" if path.is_dir() else "file",
                "size": size,
                "size_str": size_str,
                "extension": path.suffix if path.is_file() else None
            }
        except Exception as e:
            logger.warning(f"获取文件信息失败 {path}: {e}")
            return {
                "name": path.name,
                "path": str(path),
                "type": "directory" if path.is_dir() else "file",
                "error": str(e)
            }
    
    def _list_single(
        self,
        path: Path,
        show_hidden: bool = False,
        file_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """列出单个目录（非递归）"""
        directories = []
        files = []
        
        try:
            for item in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                # 检查是否应该忽略
                if self._should_ignore(item, show_hidden):
                    continue
                
                # 检查文件类型
                if not self._matches_file_type(item, file_types):
                    continue
                
                info = self._get_file_info(item)
                
                if item.is_dir():
                    directories.append(info)
                else:
                    files.append(info)
        
        except PermissionError:
            logger.warning(f"无权限访问目录: {path}")
        
        return {
            "directories": directories,
            "files": files,
            "total_dirs": len(directories),
            "total_files": len(files)
        }
    
    def _list_recursive(
        self,
        path: Path,
        max_depth: int = 3,
        current_depth: int = 0,
        show_hidden: bool = False,
        file_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """递归列出目录"""
        all_directories = []
        all_files = []
        
        # 检查深度限制
        if current_depth >= max_depth:
            return {
                "directories": [],
                "files": [],
                "total_dirs": 0,
                "total_files": 0
            }
        
        try:
            for item in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                # 检查是否应该忽略
                if self._should_ignore(item, show_hidden):
                    continue
                
                # 检查文件类型
                if not self._matches_file_type(item, file_types):
                    continue
                
                info = self._get_file_info(item)
                info["depth"] = current_depth
                
                if item.is_dir():
                    all_directories.append(info)
                    
                    # 递归处理子目录
                    sub_result = self._list_recursive(
                        item,
                        max_depth=max_depth,
                        current_depth=current_depth + 1,
                        show_hidden=show_hidden,
                        file_types=file_types
                    )
                    all_directories.extend(sub_result["directories"])
                    all_files.extend(sub_result["files"])
                else:
                    all_files.append(info)
        
        except PermissionError:
            logger.warning(f"无权限访问目录: {path}")
        
        return {
            "directories": all_directories,
            "files": all_files,
            "total_dirs": len(all_directories),
            "total_files": len(all_files)
        }

