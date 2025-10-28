"""文件名搜索工具 - 使用glob模式搜索文件"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import fnmatch
from loguru import logger
from .base import BaseTool, ToolParameter
from .safe_file_handler import SafeFileHandler, SecurityError


class GlobSearchTool(BaseTool):
    """
    文件名搜索工具（Glob模式）
    
    功能：
    1. 使用通配符搜索文件（*.py, test_*.py等）
    2. 支持递归搜索
    3. 自动排除常见无用目录
    4. 比bash find命令更友好
    """
    
    # 默认排除的目录（与ListDirTool保持一致）
    DEFAULT_IGNORE_DIRS = {
        '.git', '.svn', '.hg',
        '__pycache__', '.pytest_cache', '.mypy_cache',
        'node_modules', 'bower_components',
        '.venv', 'venv', 'env', 'virtualenv',
        'dist', 'build', '.egg-info',
        '.idea', '.vscode', '.vs',
    }
    
    def __init__(self, safe_handler: Optional[SafeFileHandler] = None):
        self.safe_handler = safe_handler or SafeFileHandler()
    
    @property
    def name(self) -> str:
        return "glob_file_search"
    
    @property
    def description(self) -> str:
        return "使用glob模式搜索文件名，支持通配符（*、?、[]等）。自动排除.git, __pycache__, .venv等目录。"
    
    @property
    def parameters(self) -> Dict[str, ToolParameter]:
        return {
            "pattern": ToolParameter(
                type="string",
                description="搜索模式，支持通配符。例如：'*.py', '**/*.js', 'test_*.py'",
                required=True
            ),
            "path": ToolParameter(
                type="string",
                description="搜索路径（默认为当前目录）",
                required=False
            ),
            "recursive": ToolParameter(
                type="boolean",
                description="是否递归搜索子目录（默认True）",
                required=False
            ),
            "max_results": ToolParameter(
                type="integer",
                description="最大返回结果数（默认100，防止输出过多）",
                required=False
            ),
            "case_sensitive": ToolParameter(
                type="boolean",
                description="是否区分大小写（默认True）",
                required=False
            )
        }
    
    def execute(
        self,
        pattern: str,
        path: str = ".",
        recursive: bool = True,
        max_results: int = 100,
        case_sensitive: bool = True
    ) -> Dict[str, Any]:
        """
        搜索文件名
        
        Args:
            pattern: glob模式（如'*.py', '**/*.js', 'test_*.py'）
            path: 搜索路径
            recursive: 是否递归搜索
            max_results: 最大返回结果数
            case_sensitive: 是否区分大小写
            
        Returns:
            包含匹配文件列表的字典
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
            
            logger.info(f"搜索文件: pattern='{pattern}', path='{path}', recursive={recursive}")
            
            # 执行搜索
            matches = self._search_files(
                path_obj,
                pattern,
                recursive=recursive,
                max_results=max_results,
                case_sensitive=case_sensitive
            )
            
            # 获取文件详细信息
            results = []
            for match_path in matches:
                try:
                    stat = match_path.stat()
                    size = stat.st_size
                    
                    # 格式化大小
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f}MB"
                    
                    # 计算相对路径
                    try:
                        rel_path = match_path.relative_to(path_obj)
                    except ValueError:
                        rel_path = match_path
                    
                    results.append({
                        "path": str(match_path),
                        "relative_path": str(rel_path),
                        "name": match_path.name,
                        "size": size,
                        "size_str": size_str,
                        "extension": match_path.suffix,
                        "is_file": match_path.is_file(),
                        "is_dir": match_path.is_dir()
                    })
                except Exception as e:
                    logger.warning(f"获取文件信息失败 {match_path}: {e}")
                    results.append({
                        "path": str(match_path),
                        "name": match_path.name,
                        "error": str(e)
                    })
            
            # 按名称排序
            results.sort(key=lambda x: x.get("name", "").lower())
            
            truncated = len(matches) > max_results
            
            return {
                "success": True,
                "pattern": pattern,
                "search_path": str(path),
                "total_matches": len(results),
                "truncated": truncated,
                "matches": results,
                "message": f"找到 {len(results)} 个匹配项" + 
                          (f"（已截断，实际更多）" if truncated else "")
            }
            
        except Exception as e:
            logger.error(f"搜索文件失败: {e}")
            return {
                "success": False,
                "error": f"搜索失败: {str(e)}"
            }
    
    def _should_ignore_dir(self, dir_path: Path) -> bool:
        """判断是否应该忽略该目录"""
        return dir_path.name in self.DEFAULT_IGNORE_DIRS
    
    def _search_files(
        self,
        base_path: Path,
        pattern: str,
        recursive: bool = True,
        max_results: int = 100,
        case_sensitive: bool = True
    ) -> List[Path]:
        """
        搜索匹配的文件
        
        Args:
            base_path: 基础路径
            pattern: 搜索模式
            recursive: 是否递归
            max_results: 最大结果数
            case_sensitive: 是否区分大小写
            
        Returns:
            匹配的文件路径列表
        """
        matches = []
        
        # 如果pattern包含路径分隔符，使用rglob
        if '/' in pattern or '**' in pattern:
            # 处理 **/ 前缀
            if pattern.startswith('**/'):
                pattern = pattern[3:]
                recursive = True
            
            try:
                if recursive:
                    search_pattern = f"**/{pattern}"
                else:
                    search_pattern = pattern
                
                for match in base_path.glob(search_pattern):
                    # 检查路径中是否包含应忽略的目录
                    if any(self._should_ignore_dir(Path(p)) for p in match.parts):
                        continue
                    
                    matches.append(match)
                    
                    if len(matches) >= max_results:
                        break
            except Exception as e:
                logger.warning(f"Glob搜索失败: {e}")
        else:
            # 简单模式，手动遍历
            matches = self._manual_search(
                base_path,
                pattern,
                recursive=recursive,
                max_results=max_results,
                case_sensitive=case_sensitive
            )
        
        return matches
    
    def _manual_search(
        self,
        base_path: Path,
        pattern: str,
        recursive: bool = True,
        max_results: int = 100,
        case_sensitive: bool = True
    ) -> List[Path]:
        """
        手动递归搜索（当glob不适用时）
        
        Args:
            base_path: 基础路径
            pattern: 搜索模式
            recursive: 是否递归
            max_results: 最大结果数
            case_sensitive: 是否区分大小写
            
        Returns:
            匹配的文件路径列表
        """
        matches = []
        
        def search_dir(current_path: Path, depth: int = 0):
            # 防止递归过深
            if depth > 10:
                return
            
            try:
                for item in current_path.iterdir():
                    # 检查是否达到最大结果数
                    if len(matches) >= max_results:
                        return
                    
                    # 忽略特定目录
                    if item.is_dir() and self._should_ignore_dir(item):
                        continue
                    
                    # 匹配文件名
                    if self._matches_pattern(item.name, pattern, case_sensitive):
                        matches.append(item)
                    
                    # 递归搜索子目录
                    if recursive and item.is_dir():
                        search_dir(item, depth + 1)
                        
            except PermissionError:
                logger.warning(f"无权限访问目录: {current_path}")
            except Exception as e:
                logger.warning(f"搜索目录失败 {current_path}: {e}")
        
        search_dir(base_path)
        return matches
    
    def _matches_pattern(
        self,
        filename: str,
        pattern: str,
        case_sensitive: bool = True
    ) -> bool:
        """
        检查文件名是否匹配模式
        
        Args:
            filename: 文件名
            pattern: 匹配模式
            case_sensitive: 是否区分大小写
            
        Returns:
            是否匹配
        """
        if not case_sensitive:
            filename = filename.lower()
            pattern = pattern.lower()
        
        return fnmatch.fnmatch(filename, pattern)

