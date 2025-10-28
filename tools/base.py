"""工具基类"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel


class ToolParameter(BaseModel):
    """工具参数定义"""
    type: str
    description: str
    enum: List[str] = None
    required: bool = True


class BaseTool(ABC):
    """工具基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, ToolParameter]:
        """工具参数定义"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行工具
        
        Returns:
            包含success和result/error的字典
        """
        pass
    
    def to_openai_tool(self) -> Dict[str, Any]:
        """转换为OpenAI工具格式"""
        properties = {}
        required = []
        
        for param_name, param_def in self.parameters.items():
            prop = {
                "type": param_def.type,
                "description": param_def.description
            }
            if param_def.enum:
                prop["enum"] = param_def.enum
            
            properties[param_name] = prop
            
            if param_def.required:
                required.append(param_name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

