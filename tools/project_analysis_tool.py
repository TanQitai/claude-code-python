"""项目分析工具 - 快速分析项目结构和统计信息"""
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from collections import defaultdict
import json
from loguru import logger
from .base import BaseTool, ToolParameter
from .safe_file_handler import SafeFileHandler, SecurityError


class ProjectAnalysisTool(BaseTool):
    """
    项目分析工具
    
    功能：
    1. 项目结构概览
    2. 代码统计（文件数、行数等）
    3. 文件类型分布
    4. 依赖关系分析（Python import）
    5. 大文件检测
    """
    
    # 忽略的目录
    IGNORE_DIRS = {
        '.git', '.svn', '.hg',
        '__pycache__', '.pytest_cache', '.mypy_cache',
        'node_modules', 'bower_components',
        '.venv', 'venv', 'env', 'virtualenv',
        'dist', 'build', '.egg-info',
        '.idea', '.vscode', '.vs',
    }
    
    # 代码文件扩展名
    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx',
        '.java', '.c', '.cpp', '.h', '.hpp',
        '.go', '.rs', '.rb', '.php',
        '.cs', '.swift', '.kt', '.scala'
    }
    
    # 配置文件扩展名
    CONFIG_EXTENSIONS = {
        '.json', '.yaml', '.yml', '.toml',
        '.ini', '.cfg', '.conf', '.xml'
    }
    
    # 文档文件扩展名
    DOC_EXTENSIONS = {
        '.md', '.txt', '.rst', '.adoc'
    }
    
    def __init__(self, safe_handler: Optional[SafeFileHandler] = None):
        self.safe_handler = safe_handler or SafeFileHandler()
    
    @property
    def name(self) -> str:
        return "project_analysis"
    
    @property
    def description(self) -> str:
        return """分析项目结构、统计代码信息、检测大文件等。
        
        分析类型(action):
        - summary: 项目概览（推荐，包含最重要信息）
        - structure: 目录结构树
        - statistics: 详细统计信息
        - dependencies: Python依赖分析
        - large_files: 大文件检测"""
    
    @property
    def parameters(self) -> Dict[str, ToolParameter]:
        return {
            "path": ToolParameter(
                type="string",
                description="项目路径（默认为当前目录）",
                required=False
            ),
            "action": ToolParameter(
                type="string",
                description="分析类型：summary(概览), structure(目录树), statistics(统计), dependencies(依赖), large_files(大文件)",
                enum=["summary", "structure", "statistics", "dependencies", "large_files"],
                required=False
            ),
            "max_depth": ToolParameter(
                type="integer",
                description="目录树最大深度（默认3）",
                required=False
            ),
            "large_file_threshold": ToolParameter(
                type="integer",
                description="大文件阈值，单位MB（默认1MB）",
                required=False
            )
        }
    
    def execute(
        self,
        path: str = ".",
        action: str = "summary",
        max_depth: int = 3,
        large_file_threshold: int = 1
    ) -> Dict[str, Any]:
        """
        执行项目分析
        
        Args:
            path: 项目路径
            action: 分析类型
            max_depth: 目录树最大深度
            large_file_threshold: 大文件阈值（MB）
            
        Returns:
            分析结果
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
            
            if not path_obj.exists():
                return {
                    "success": False,
                    "error": f"路径不存在: {path}"
                }
            
            if not path_obj.is_dir():
                return {
                    "success": False,
                    "error": f"路径不是目录: {path}"
                }
            
            logger.info(f"分析项目: {path}, action={action}")
            
            # 根据action执行不同的分析
            if action == "summary":
                result = self._analyze_summary(path_obj)
            elif action == "structure":
                result = self._analyze_structure(path_obj, max_depth)
            elif action == "statistics":
                result = self._analyze_statistics(path_obj)
            elif action == "dependencies":
                result = self._analyze_dependencies(path_obj)
            elif action == "large_files":
                result = self._analyze_large_files(path_obj, large_file_threshold)
            else:
                return {
                    "success": False,
                    "error": f"未知的分析类型: {action}"
                }
            
            result["success"] = True
            result["path"] = str(path)
            result["action"] = action
            
            return result
            
        except Exception as e:
            logger.error(f"项目分析失败: {e}")
            return {
                "success": False,
                "error": f"分析失败: {str(e)}"
            }
    
    def _scan_project(self, path: Path) -> Dict[str, Any]:
        """
        扫描项目，收集基础信息
        
        Returns:
            包含所有文件信息的字典
        """
        files_by_ext = defaultdict(list)
        total_size = 0
        total_lines = 0
        total_files = 0
        total_dirs = 0
        
        def scan_dir(current_path: Path, depth: int = 0):
            nonlocal total_size, total_files, total_dirs
            
            if depth > 10:  # 防止过深
                return
            
            try:
                for item in current_path.iterdir():
                    if item.is_dir():
                        if item.name in self.IGNORE_DIRS:
                            continue
                        total_dirs += 1
                        scan_dir(item, depth + 1)
                    else:
                        total_files += 1
                        try:
                            size = item.stat().st_size
                            total_size += size
                            
                            files_by_ext[item.suffix].append({
                                "path": str(item),
                                "size": size,
                                "name": item.name
                            })
                        except Exception as e:
                            logger.warning(f"获取文件信息失败 {item}: {e}")
            except PermissionError:
                logger.warning(f"无权限访问: {current_path}")
        
        scan_dir(path)
        
        # 统计代码行数（只统计代码文件）
        for ext, file_list in files_by_ext.items():
            if ext in self.CODE_EXTENSIONS:
                for file_info in file_list:
                    try:
                        lines = self._count_lines(Path(file_info["path"]))
                        file_info["lines"] = lines
                        total_lines += lines
                    except:
                        pass
        
        return {
            "files_by_ext": dict(files_by_ext),
            "total_size": total_size,
            "total_lines": total_lines,
            "total_files": total_files,
            "total_dirs": total_dirs
        }
    
    def _count_lines(self, file_path: Path) -> int:
        """统计文件行数"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def _format_size(self, size: int) -> str:
        """格式化文件大小"""
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f}KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f}MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.2f}GB"
    
    def _analyze_summary(self, path: Path) -> Dict[str, Any]:
        """项目概览"""
        scan_result = self._scan_project(path)
        
        # 按文件类型分组统计
        code_files = 0
        config_files = 0
        doc_files = 0
        other_files = 0
        
        for ext, files in scan_result["files_by_ext"].items():
            count = len(files)
            if ext in self.CODE_EXTENSIONS:
                code_files += count
            elif ext in self.CONFIG_EXTENSIONS:
                config_files += count
            elif ext in self.DOC_EXTENSIONS:
                doc_files += count
            else:
                other_files += count
        
        # Top 5 文件类型
        top_extensions = sorted(
            scan_result["files_by_ext"].items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
        
        top_ext_summary = [
            {
                "extension": ext if ext else "(无扩展名)",
                "count": len(files),
                "total_size": self._format_size(sum(f["size"] for f in files))
            }
            for ext, files in top_extensions
        ]
        
        return {
            "project_name": path.name,
            "total_files": scan_result["total_files"],
            "total_directories": scan_result["total_dirs"],
            "total_size": self._format_size(scan_result["total_size"]),
            "total_code_lines": scan_result["total_lines"],
            "file_categories": {
                "code_files": code_files,
                "config_files": config_files,
                "documentation": doc_files,
                "other": other_files
            },
            "top_file_types": top_ext_summary,
            "summary": f"项目包含 {scan_result['total_files']} 个文件，"
                      f"{code_files} 个代码文件，"
                      f"共 {scan_result['total_lines']} 行代码，"
                      f"总大小 {self._format_size(scan_result['total_size'])}"
        }
    
    def _analyze_structure(self, path: Path, max_depth: int = 3) -> Dict[str, Any]:
        """分析目录结构"""
        structure = []
        
        def build_tree(current_path: Path, depth: int = 0, prefix: str = ""):
            if depth >= max_depth:
                return
            
            try:
                items = sorted(current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                
                for i, item in enumerate(items):
                    if item.name in self.IGNORE_DIRS:
                        continue
                    
                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    
                    if item.is_dir():
                        structure.append({
                            "line": f"{prefix}{current_prefix}{item.name}/",
                            "type": "directory",
                            "name": item.name,
                            "depth": depth
                        })
                        
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        build_tree(item, depth + 1, next_prefix)
                    else:
                        size_str = self._format_size(item.stat().st_size)
                        structure.append({
                            "line": f"{prefix}{current_prefix}{item.name} ({size_str})",
                            "type": "file",
                            "name": item.name,
                            "depth": depth,
                            "size": size_str
                        })
            except PermissionError:
                structure.append({
                    "line": f"{prefix}[权限不足]",
                    "type": "error",
                    "depth": depth
                })
        
        build_tree(path)
        
        # 生成树形文本
        tree_text = f"{path.name}/\n" + "\n".join(item["line"] for item in structure)
        
        return {
            "directory_tree": tree_text,
            "structure_data": structure,
            "max_depth": max_depth
        }
    
    def _analyze_statistics(self, path: Path) -> Dict[str, Any]:
        """详细统计信息"""
        scan_result = self._scan_project(path)
        
        # 按扩展名统计
        ext_stats = []
        for ext, files in sorted(scan_result["files_by_ext"].items(), 
                                 key=lambda x: len(x[1]), reverse=True):
            total_size = sum(f["size"] for f in files)
            total_lines = sum(f.get("lines", 0) for f in files)
            
            ext_stats.append({
                "extension": ext if ext else "(无扩展名)",
                "file_count": len(files),
                "total_size": self._format_size(total_size),
                "total_lines": total_lines if total_lines > 0 else None,
                "avg_size": self._format_size(total_size // len(files)) if files else "0B"
            })
        
        return {
            "total_files": scan_result["total_files"],
            "total_directories": scan_result["total_dirs"],
            "total_size": self._format_size(scan_result["total_size"]),
            "total_code_lines": scan_result["total_lines"],
            "statistics_by_extension": ext_stats
        }
    
    def _analyze_dependencies(self, path: Path) -> Dict[str, Any]:
        """分析Python依赖关系"""
        imports = defaultdict(set)
        files_analyzed = 0
        
        def extract_imports(file_path: Path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('import '):
                            module = line.split()[1].split('.')[0]
                            imports[str(file_path)].add(module)
                        elif line.startswith('from '):
                            parts = line.split()
                            if len(parts) >= 2:
                                module = parts[1].split('.')[0]
                                imports[str(file_path)].add(module)
            except:
                pass
        
        # 扫描Python文件
        for item in path.rglob("*.py"):
            if any(ignored in item.parts for ignored in self.IGNORE_DIRS):
                continue
            extract_imports(item)
            files_analyzed += 1
        
        # 统计最常用的模块
        module_count = defaultdict(int)
        for file_imports in imports.values():
            for module in file_imports:
                module_count[module] += 1
        
        top_modules = sorted(module_count.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "files_analyzed": files_analyzed,
            "total_imports": sum(len(v) for v in imports.values()),
            "unique_modules": len(module_count),
            "top_modules": [
                {"module": mod, "used_in_files": count}
                for mod, count in top_modules
            ],
            "message": f"分析了 {files_analyzed} 个Python文件，"
                      f"发现 {len(module_count)} 个不同的导入模块"
        }
    
    def _analyze_large_files(
        self,
        path: Path,
        threshold_mb: int = 1
    ) -> Dict[str, Any]:
        """检测大文件"""
        threshold_bytes = threshold_mb * 1024 * 1024
        large_files = []
        
        def scan_files(current_path: Path):
            try:
                for item in current_path.iterdir():
                    if item.is_dir():
                        if item.name not in self.IGNORE_DIRS:
                            scan_files(item)
                    else:
                        try:
                            size = item.stat().st_size
                            if size >= threshold_bytes:
                                large_files.append({
                                    "path": str(item),
                                    "name": item.name,
                                    "size": size,
                                    "size_str": self._format_size(size),
                                    "extension": item.suffix
                                })
                        except:
                            pass
            except PermissionError:
                pass
        
        scan_files(path)
        
        # 按大小排序
        large_files.sort(key=lambda x: x["size"], reverse=True)
        
        return {
            "threshold": f"{threshold_mb}MB",
            "large_files_count": len(large_files),
            "large_files": large_files[:20],  # 最多返回20个
            "total_size": self._format_size(sum(f["size"] for f in large_files)),
            "message": f"发现 {len(large_files)} 个大于 {threshold_mb}MB 的文件"
        }

