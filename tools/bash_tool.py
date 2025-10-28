"""Bash命令执行工具"""
import subprocess
import asyncio
from typing import Dict, Any
from loguru import logger
from .base import BaseTool, ToolParameter
import config


class BashTool(BaseTool):
    """执行Bash命令的工具"""
    
    # 危险命令黑名单
    DANGEROUS_COMMANDS = [
        "rm -rf /",
        "mkfs",
        "dd if=/dev/zero",
        ":(){:|:&};:",  # fork炸弹
        "chmod -R 777 /",
        "wget http",  # 可以根据需要调整
    ]
    
    @property
    def name(self) -> str:
        return "bash_tool"
    
    @property
    def description(self) -> str:
        return "执行bash命令。可以运行系统命令、脚本等。返回命令的输出结果。"
    
    @property
    def parameters(self) -> Dict[str, ToolParameter]:
        return {
            "command": ToolParameter(
                type="string",
                description="要执行的bash命令",
                required=True
            ),
            "timeout": ToolParameter(
                type="integer",
                description=f"命令执行超时时间（秒），默认{config.BASH_TIMEOUT}秒",
                required=False
            )
        }
    
    def _is_safe_command(self, command: str) -> bool:
        """检查命令是否安全"""
        command_lower = command.lower().strip()
        
        for dangerous in self.DANGEROUS_COMMANDS:
            if dangerous in command_lower:
                return False
        
        return True
    
    def execute(self, command: str, timeout: int = None, **kwargs) -> Dict[str, Any]:
        """
        执行bash命令
        
        Args:
            command: 要执行的命令
            timeout: 超时时间（秒）
            
        Returns:
            执行结果
        """
        if timeout is None:
            timeout = config.BASH_TIMEOUT
        
        logger.info(f"执行bash命令: {command}")
        
        # 安全检查
        if not self._is_safe_command(command):
            logger.warning(f"检测到危险命令，拒绝执行: {command}")
            return {
                "success": False,
                "error": "检测到危险命令，拒绝执行",
                "command": command
            }
        
        try:
            # 执行命令
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=None  # 可以根据需要设置工作目录
            )
            
            success = result.returncode == 0
            
            # 限制输出长度，避免超过 token 限制
            stdout = result.stdout
            stderr = result.stderr
            
            MAX_OUTPUT_LENGTH = 4000  # 限制输出字符数
            if len(stdout) > MAX_OUTPUT_LENGTH:
                lines = stdout.split('\n')
                if len(lines) > 100:
                    # 如果行数太多，只保留前50行和后50行
                    stdout = '\n'.join(lines[:50]) + f"\n\n... (省略 {len(lines)-100} 行) ...\n\n" + '\n'.join(lines[-50:])
                else:
                    stdout = stdout[:MAX_OUTPUT_LENGTH] + f"\n\n... (输出太长，已截断，共 {len(result.stdout)} 字符) ..."
            
            response = {
                "success": success,
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "command": command
            }
            
            if success:
                logger.info(f"命令执行成功: {command}")
            else:
                logger.warning(f"命令执行失败 (返回码{result.returncode}): {command}")
            
            return response
            
        except subprocess.TimeoutExpired:
            logger.error(f"命令执行超时 ({timeout}秒): {command}")
            return {
                "success": False,
                "error": f"命令执行超时（{timeout}秒）",
                "command": command
            }
        
        except Exception as e:
            logger.error(f"命令执行异常: {e}")
            return {
                "success": False,
                "error": str(e),
                "command": command
            }
    
    async def execute_async(self, command: str, timeout: int = None) -> Dict[str, Any]:
        """异步执行命令"""
        if timeout is None:
            timeout = config.BASH_TIMEOUT
        
        logger.info(f"异步执行bash命令: {command}")
        
        if not self._is_safe_command(command):
            logger.warning(f"检测到危险命令，拒绝执行: {command}")
            return {
                "success": False,
                "error": "检测到危险命令，拒绝执行",
                "command": command
            }
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                success = process.returncode == 0
                
                return {
                    "success": success,
                    "returncode": process.returncode,
                    "stdout": stdout.decode() if stdout else "",
                    "stderr": stderr.decode() if stderr else "",
                    "command": command
                }
                
            except asyncio.TimeoutError:
                process.kill()
                logger.error(f"命令执行超时 ({timeout}秒): {command}")
                return {
                    "success": False,
                    "error": f"命令执行超时（{timeout}秒）",
                    "command": command
                }
        
        except Exception as e:
            logger.error(f"命令执行异常: {e}")
            return {
                "success": False,
                "error": str(e),
                "command": command
            }

