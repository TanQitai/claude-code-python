"""文件安全管理器 - 防止路径遍历和危险操作"""
import os
from typing import Optional
from pathlib import Path
from loguru import logger


class SecurityError(Exception):
    """安全错误异常"""
    pass


class SafeFileHandler:
    """
    安全文件处理器
    
    功能：
    1. 防止路径遍历攻击
    2. 限制文件大小
    3. 文件类型白名单
    4. 工作目录限制
    """
    
    # 危险目录黑名单
    DANGEROUS_PATHS = [
        "/etc",
        "/sys",
        "/proc",
        "/dev",
        "/boot",
        "/root",
        "/var/log",
    ]
    
    # 危险文件扩展名
    DANGEROUS_EXTENSIONS = [
        ".exe",
        ".dll",
        ".so",
        ".dylib",
    ]
    
    # 允许的文本文件扩展名（用于编辑）
    ALLOWED_TEXT_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".c", ".cpp", ".h", ".hpp",
        ".go", ".rs", ".rb", ".php",
        ".html", ".css", ".scss", ".sass",
        ".json", ".yaml", ".yml", ".toml", ".ini",
        ".xml", ".md", ".txt", ".sh", ".bash",
        ".sql", ".env", ".gitignore",
        ".conf", ".cfg", ".log",
    }
    
    def __init__(
        self,
        workspace_root: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        enforce_workspace: bool = True
    ):
        """
        初始化安全文件处理器
        
        Args:
            workspace_root: 工作空间根目录（默认为当前目录）
            max_file_size: 最大文件大小（字节）
            enforce_workspace: 是否强制限制在工作空间内
        """
        if workspace_root is None:
            workspace_root = os.getcwd()
        
        self.workspace_root = os.path.abspath(workspace_root)
        self.max_file_size = max_file_size
        self.enforce_workspace = enforce_workspace
        
        logger.info(f"SafeFileHandler 初始化: workspace={self.workspace_root}")
    
    def validate_path(self, file_path: str, check_exists: bool = False) -> str:
        """
        验证文件路径安全性
        
        Args:
            file_path: 文件路径
            check_exists: 是否检查文件必须存在
            
        Returns:
            规范化后的绝对路径
            
        Raises:
            SecurityError: 路径不安全
        """
        # 转换为绝对路径
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.workspace_root, file_path)
        
        abs_path = os.path.abspath(file_path)
        
        # 检查是否在工作空间内
        if self.enforce_workspace:
            if not abs_path.startswith(self.workspace_root):
                raise SecurityError(
                    f"路径遍历检测: 文件必须在工作空间内\n"
                    f"工作空间: {self.workspace_root}\n"
                    f"请求路径: {abs_path}"
                )
        
        # 检查危险路径
        for dangerous_path in self.DANGEROUS_PATHS:
            if abs_path.startswith(dangerous_path):
                raise SecurityError(f"危险路径: {abs_path} 位于受保护目录 {dangerous_path}")
        
        # 检查危险扩展名
        ext = os.path.splitext(abs_path)[1].lower()
        if ext in self.DANGEROUS_EXTENSIONS:
            raise SecurityError(f"危险文件类型: {ext}")
        
        # 检查文件是否存在（如果需要）
        if check_exists and not os.path.exists(abs_path):
            raise FileNotFoundError(f"文件不存在: {abs_path}")
        
        logger.debug(f"路径验证通过: {abs_path}")
        return abs_path
    
    def validate_file_size(self, file_path: str) -> bool:
        """
        验证文件大小
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否在限制内
            
        Raises:
            SecurityError: 文件过大
        """
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                raise SecurityError(
                    f"文件过大: {file_size} 字节 > {self.max_file_size} 字节 "
                    f"({file_size / 1024 / 1024:.2f}MB > {self.max_file_size / 1024 / 1024:.2f}MB)"
                )
        return True
    
    def is_text_file(self, file_path: str) -> bool:
        """
        检查是否为文本文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否为文本文件
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        # 检查扩展名白名单
        if ext in self.ALLOWED_TEXT_EXTENSIONS:
            return True
        
        # 没有扩展名的文件也可能是文本文件（如 Dockerfile, Makefile）
        if not ext:
            basename = os.path.basename(file_path)
            if basename in ["Dockerfile", "Makefile", "README", "LICENSE"]:
                return True
        
        return False
    
    def safe_read(self, file_path: str, encoding: str = "utf-8") -> str:
        """
        安全读取文件
        
        Args:
            file_path: 文件路径
            encoding: 文件编码
            
        Returns:
            文件内容
        """
        abs_path = self.validate_path(file_path, check_exists=True)
        self.validate_file_size(abs_path)
        
        logger.info(f"安全读取文件: {abs_path}")
        
        with open(abs_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        return content
    
    def safe_write(self, file_path: str, content: str, encoding: str = "utf-8") -> None:
        """
        安全写入文件
        
        Args:
            file_path: 文件路径
            content: 文件内容
            encoding: 文件编码
        """
        abs_path = self.validate_path(file_path, check_exists=False)
        
        # 检查内容大小
        content_size = len(content.encode(encoding))
        if content_size > self.max_file_size:
            raise SecurityError(
                f"内容过大: {content_size} 字节 > {self.max_file_size} 字节"
            )
        
        # 确保目录存在
        directory = os.path.dirname(abs_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"创建目录: {directory}")
        
        # 写入文件前备份（如果文件存在）
        if os.path.exists(abs_path):
            backup_path = abs_path + ".backup"
            import shutil
            shutil.copy2(abs_path, backup_path)
            logger.debug(f"备份文件: {backup_path}")
        
        logger.info(f"安全写入文件: {abs_path}")
        
        with open(abs_path, 'w', encoding=encoding) as f:
            f.write(content)
    
    def get_relative_path(self, file_path: str) -> str:
        """
        获取相对于工作空间的路径
        
        Args:
            file_path: 文件路径
            
        Returns:
            相对路径
        """
        abs_path = self.validate_path(file_path)
        try:
            return os.path.relpath(abs_path, self.workspace_root)
        except ValueError:
            # 不同驱动器，返回绝对路径
            return abs_path

