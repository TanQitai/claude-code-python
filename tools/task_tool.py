"""任务管理工具 - 支持创建sub-agent和并行执行任务"""
import asyncio
import uuid
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
from .base import BaseTool, ToolParameter
try:
    from .. import config
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import config


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"      # 等待执行
    RUNNING = "running"      # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    CANCELLED = "cancelled"  # 已取消


@dataclass
class Task:
    """任务数据结构"""
    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    sub_agent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "sub_agent_id": self.sub_agent_id
        }


class TaskTool(BaseTool):
    """
    任务管理工具
    
    功能：
    1. 创建任务
    2. 查询任务状态
    3. 执行并行任务
    4. 管理sub-agent
    """
    
    def __init__(self, agent_manager=None):
        self.tasks: Dict[str, Task] = {}
        self.agent_manager = agent_manager  # 用于创建sub-agent
        self._lock = asyncio.Lock()
    
    @property
    def name(self) -> str:
        return "task_tool"
    
    @property
    def description(self) -> str:
        return """任务管理工具。可以创建任务、查询任务状态、执行并行任务。
        支持将复杂任务分解为多个子任务，并创建sub-agent并行执行。"""
    
    @property
    def parameters(self) -> Dict[str, ToolParameter]:
        return {
            "action": ToolParameter(
                type="string",
                description="操作类型",
                enum=["create", "get_status", "execute_parallel", "cancel"],
                required=True
            ),
            "task_ids": ToolParameter(
                type="array",
                description="任务ID列表（用于get_status, execute_parallel, cancel）",
                required=False
            ),
            "descriptions": ToolParameter(
                type="array",
                description="任务描述列表（用于create创建多个任务）",
                required=False
            )
        }
    
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        执行任务工具操作
        
        Args:
            action: 操作类型 (create, get_status, execute_parallel, cancel)
            **kwargs: 其他参数
        """
        logger.info(f"TaskTool执行操作: {action}")
        
        if action == "create":
            return self._create_tasks(kwargs.get("descriptions", []))
        
        elif action == "get_status":
            return self._get_status(kwargs.get("task_ids", []))
        
        elif action == "execute_parallel":
            # 同步版本 - 使用 asyncio.run() 执行异步并行任务
            task_ids = kwargs.get("task_ids", [])
            max_parallel = kwargs.get("max_parallel", None)
            
            logger.info(f"同步模式下执行并行任务: {len(task_ids)} 个任务")
            
            # 在同步上下文中运行异步代码
            try:
                import asyncio
                result = asyncio.run(
                    self.execute_parallel_async(task_ids, max_parallel)
                )
                return result
            except Exception as e:
                logger.error(f"并行执行失败: {e}")
                return {
                    "success": False,
                    "error": f"并行执行失败: {str(e)}",
                    "task_ids": task_ids
                }
        
        elif action == "cancel":
            return self._cancel_tasks(kwargs.get("task_ids", []))
        
        else:
            return {
                "success": False,
                "error": f"未知的操作类型: {action}"
            }
    
    def _create_tasks(self, descriptions: List[str]) -> Dict[str, Any]:
        """创建任务"""
        if not descriptions:
            return {
                "success": False,
                "error": "任务描述列表不能为空"
            }
        
        created_tasks = []
        
        for desc in descriptions:
            task_id = str(uuid.uuid4())[:8]
            
            # 处理不同格式的任务描述
            # LLM 可能返回字符串或字典
            if isinstance(desc, dict):
                # 如果是字典，提取 task 或 description 字段
                task_desc = desc.get('task') or desc.get('description') or str(desc)
            else:
                task_desc = str(desc)
            
            task = Task(
                task_id=task_id,
                description=task_desc,
                status=TaskStatus.PENDING
            )
            self.tasks[task_id] = task
            created_tasks.append(task.to_dict())
            logger.info(f"创建任务 {task_id}: {task_desc[:60]}...")
        
        return {
            "success": True,
            "tasks": created_tasks,
            "message": f"成功创建 {len(created_tasks)} 个任务"
        }
    
    def _get_status(self, task_ids: List[str]) -> Dict[str, Any]:
        """查询任务状态"""
        if not task_ids:
            # 返回所有任务状态
            task_ids = list(self.tasks.keys())
        
        statuses = []
        not_found = []
        
        for task_id in task_ids:
            if task_id in self.tasks:
                statuses.append(self.tasks[task_id].to_dict())
            else:
                not_found.append(task_id)
        
        result = {
            "success": True,
            "tasks": statuses
        }
        
        if not_found:
            result["not_found"] = not_found
        
        return result
    
    def _cancel_tasks(self, task_ids: List[str]) -> Dict[str, Any]:
        """取消任务"""
        cancelled = []
        not_found = []
        
        for task_id in task_ids:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
                    cancelled.append(task_id)
                    logger.info(f"取消任务: {task_id}")
            else:
                not_found.append(task_id)
        
        result = {
            "success": True,
            "cancelled": cancelled
        }
        
        if not_found:
            result["not_found"] = not_found
        
        return result
    
    async def execute_parallel_async(
        self,
        task_ids: List[str],
        max_parallel: int = None
    ) -> Dict[str, Any]:
        """
        并行执行多个任务（异步版本）
        
        Args:
            task_ids: 要执行的任务ID列表
            max_parallel: 最大并行数
        """
        if not task_ids:
            return {
                "success": False,
                "error": "任务ID列表不能为空"
            }
        
        if max_parallel is None:
            max_parallel = config.MAX_PARALLEL_TASKS
        
        logger.info(f"开始并行执行 {len(task_ids)} 个任务，最大并行数: {max_parallel}")
        
        # 检查任务是否存在
        valid_tasks = []
        not_found = []
        
        for task_id in task_ids:
            if task_id in self.tasks:
                valid_tasks.append(self.tasks[task_id])
            else:
                not_found.append(task_id)
        
        if not valid_tasks:
            return {
                "success": False,
                "error": "没有找到有效的任务",
                "not_found": not_found
            }
        
        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def execute_single_task(task: Task):
            """执行单个任务"""
            async with semaphore:
                try:
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()
                    logger.info(f"开始执行任务 {task.task_id}: {task.description}")
                    
                    # 创建sub-agent执行任务
                    if self.agent_manager:
                        result = await self.agent_manager.execute_task_with_sub_agent(
                            task.description
                        )
                        task.result = result
                        task.status = TaskStatus.COMPLETED
                    else:
                        # 如果没有agent_manager，模拟执行
                        await asyncio.sleep(1)  # 模拟执行时间
                        task.result = f"任务 {task.description} 执行完成（模拟）"
                        task.status = TaskStatus.COMPLETED
                    
                    task.completed_at = datetime.now()
                    logger.info(f"任务 {task.task_id} 执行成功")
                    
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.now()
                    logger.error(f"任务 {task.task_id} 执行失败: {e}")
        
        # 并行执行所有任务
        await asyncio.gather(
            *[execute_single_task(task) for task in valid_tasks],
            return_exceptions=True
        )
        
        # 收集结果
        results = [task.to_dict() for task in valid_tasks]
        
        success_count = sum(1 for t in valid_tasks if t.status == TaskStatus.COMPLETED)
        failed_count = sum(1 for t in valid_tasks if t.status == TaskStatus.FAILED)
        
        return {
            "success": True,
            "total": len(valid_tasks),
            "success_count": success_count,
            "failed_count": failed_count,
            "results": results,
            "not_found": not_found if not_found else None
        }
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """获取所有任务"""
        return [task.to_dict() for task in self.tasks.values()]
    
    def clear_completed_tasks(self) -> int:
        """清理已完成的任务"""
        to_remove = [
            task_id for task_id, task in self.tasks.items()
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        ]
        
        for task_id in to_remove:
            del self.tasks[task_id]
        
        logger.info(f"清理了 {len(to_remove)} 个已完成的任务")
        return len(to_remove)

