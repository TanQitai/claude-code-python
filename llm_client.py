"""K2 LLM API客户端封装"""
import json
from typing import List, Dict, Any, Optional, AsyncIterator, Iterator
from openai import OpenAI
from loguru import logger
import config


class K2Client:
    """K2 API客户端，兼容OpenAI接口"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or config.K2_API_KEY
        self.base_url = base_url or config.K2_BASE_URL
        self.model = model or config.K2_MODEL
        
        if not self.api_key:
            raise ValueError("K2_API_KEY未设置，请在.env文件中配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        logger.info(f"K2Client初始化完成: model={self.model}, base_url={self.base_url}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        调用LLM进行对话
        
        Args:
            messages: 对话历史
            tools: 可用工具列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            LLM响应
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            logger.debug(f"调用LLM API: {len(messages)}条消息, {len(tools) if tools else 0}个工具")
            
            response = self.client.chat.completions.create(**kwargs)
            
            result = {
                "content": response.choices[0].message.content,
                "tool_calls": None,
                "finish_reason": response.choices[0].finish_reason
            }
            
            # 处理工具调用
            if response.choices[0].message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                    }
                    for tc in response.choices[0].message.tool_calls
                ]
                logger.info(f"LLM请求调用 {len(result['tool_calls'])} 个工具")
                
                # 保存assistant消息（包含tool_calls）
                result["assistant_message"] = {
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in response.choices[0].message.tool_calls
                    ]
                }
            
            return result
            
        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            raise
    
    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """异步版本的chat方法"""
        # 这里可以使用异步HTTP客户端，简化起见先用同步版本
        return self.chat(messages, tools, temperature, max_tokens)
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Iterator[Dict[str, Any]]:
        """
        流式调用LLM进行对话
        
        Args:
            messages: 对话历史
            tools: 可用工具列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Yields:
            流式响应块
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True  # 启用流式输出
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            logger.debug(f"流式调用LLM API: {len(messages)}条消息, {len(tools) if tools else 0}个工具")
            
            stream = self.client.chat.completions.create(**kwargs)
            
            # 用于累积内容
            accumulated_content = ""
            tool_calls_data = []
            finish_reason = None
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason
                
                # 处理文本内容
                if delta.content:
                    accumulated_content += delta.content
                    yield {
                        "type": "content",
                        "content": delta.content,
                        "accumulated": accumulated_content
                    }
                
                # 处理工具调用
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        # 确保工具调用列表足够长
                        while len(tool_calls_data) <= tc_delta.index:
                            tool_calls_data.append({
                                "id": "",
                                "name": "",
                                "arguments": ""
                            })
                        
                        # 累积工具调用数据
                        if tc_delta.id:
                            tool_calls_data[tc_delta.index]["id"] = tc_delta.id
                        if tc_delta.function.name:
                            tool_calls_data[tc_delta.index]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_data[tc_delta.index]["arguments"] += tc_delta.function.arguments
                        
                        yield {
                            "type": "tool_call_delta",
                            "index": tc_delta.index,
                            "tool_calls": tool_calls_data
                        }
            
            # 最后返回完整结果
            result = {
                "type": "final",
                "content": accumulated_content,
                "tool_calls": None,
                "finish_reason": finish_reason
            }
            
            # 解析工具调用
            if tool_calls_data:
                result["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "name": tc["name"],
                        "arguments": json.loads(tc["arguments"])
                    }
                    for tc in tool_calls_data
                ]
                
                # 保存assistant消息（包含tool_calls）
                result["assistant_message"] = {
                    "role": "assistant",
                    "content": accumulated_content,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"]
                            }
                        }
                        for tc in tool_calls_data
                    ]
                }
                
                logger.info(f"LLM请求调用 {len(result['tool_calls'])} 个工具")
            
            yield result
            
        except Exception as e:
            logger.error(f"LLM 流式API调用失败: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }
            raise

