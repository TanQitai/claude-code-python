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
        max_tokens: int = 8000  # 从2000增加到8000，支持更大的代码文件
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
                parsed_tool_calls = []
                for tc in response.choices[0].message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as e:
                        logger.error(f"工具 {tc.function.name} 参数解析失败: {e}")
                        logger.debug(f"原始参数: {tc.function.arguments[:200]}...")
                        # 尝试修复
                        try:
                            cleaned = tc.function.arguments.replace('\n', '\\n').replace('\r', '\\r')
                            arguments = json.loads(cleaned)
                            logger.info("参数修复成功")
                        except:
                            logger.warning("参数无法解析，使用空字典")
                            arguments = {}
                    
                    parsed_tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": arguments
                    })
                
                result["tool_calls"] = parsed_tool_calls
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
        max_tokens: int = 8000  # 从2000增加到8000，支持更大的代码文件
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
            
            # ⚠️ 检查是否因为token限制被截断
            if finish_reason == "length":
                logger.error("⚠️ 响应因max_tokens限制被截断！")
                logger.error(f"当前max_tokens={max_tokens}，建议增加到8000或更高")
                if tool_calls_data:
                    logger.error("工具调用可能不完整，建议使用非流式模式或减少代码量")
            
            # 解析工具调用
            if tool_calls_data:
                parsed_tool_calls = []
                for tc in tool_calls_data:
                    arguments = None
                    original_args = tc["arguments"]
                    
                    # ✅ 先检查JSON是否完整（基础验证）
                    if original_args:
                        open_braces = original_args.count('{')
                        close_braces = original_args.count('}')
                        open_brackets = original_args.count('[')
                        close_brackets = original_args.count(']')
                        
                        if open_braces != close_braces or open_brackets != close_brackets:
                            logger.error(f"⚠️ 工具 {tc['name']} 的参数JSON不平衡！")
                            logger.error(f"括号统计: {{ {open_braces} vs }} {close_braces}, [ {open_brackets} vs ] {close_brackets}")
                            logger.error("这通常是因为max_tokens过小导致响应被截断")
                            
                            # 保存到文件以便调试
                            debug_file = f"logs/tool_args_debug_{tc['name']}.txt"
                            try:
                                import os
                                os.makedirs("logs", exist_ok=True)
                                with open(debug_file, 'w', encoding='utf-8') as f:
                                    f.write(original_args)
                                logger.info(f"不完整的参数已保存到: {debug_file}")
                            except:
                                pass
                            
                            raise ValueError(
                                f"工具 {tc['name']} 的参数被截断（JSON不完整）。\n"
                                f"原因：max_tokens={max_tokens} 过小，导致大代码文件无法完整传输。\n"
                                f"解决方案：\n"
                                f"1. 增加max_tokens（推荐8000+）\n"
                                f"2. 将大文件拆分为多个小文件\n"
                                f"3. 使用bash命令创建文件，而不是file_tool\n"
                                f"不完整的参数已保存到: {debug_file}"
                            )
                    
                    # 尝试多种方法解析 JSON
                    try:
                        # 方法1: 直接解析
                        arguments = json.loads(original_args)
                        logger.debug(f"工具 {tc['name']} 参数解析成功")
                    except json.JSONDecodeError as e:
                        logger.warning(f"工具 {tc['name']} 参数解析失败: {e}")
                        
                        # 保存原始参数到文件以便调试
                        debug_file = f"logs/tool_args_debug_{tc['name']}.txt"
                        try:
                            import os
                            os.makedirs("logs", exist_ok=True)
                            with open(debug_file, 'w', encoding='utf-8') as f:
                                f.write(original_args)
                            logger.info(f"原始参数已保存到: {debug_file}")
                        except:
                            pass
                        
                        logger.debug(f"原始参数前500字符: {original_args[:500]}")
                        
                        # 方法2: 尝试基本修复
                        try:
                            # 修复常见问题：未转义的换行符
                            fixed = original_args
                            # 只修复明显的问题，保持原始结构
                            if '\\n' not in fixed and '\n' in fixed:
                                logger.info("尝试修复未转义的换行符...")
                                # 这里不做替换，因为可能破坏已经转义的内容
                            arguments = json.loads(fixed)
                            logger.info("✅ 参数修复成功（方法2）")
                        except:
                            # 方法3: 等待完整参数
                            logger.error(f"❌ 参数无法解析，这可能是 API 流式返回不完整")
                            logger.error(f"建议：1) 使用非流式模式 2) 减少单次传输的代码量")
                            
                            # 不使用空字典，而是抛出错误让上层处理
                            raise ValueError(
                                f"工具 {tc['name']} 的参数无法解析。\n"
                                f"这通常是因为参数包含大量代码，导致 JSON 解析失败。\n"
                                f"原始参数已保存到: {debug_file}\n"
                                f"建议：将代码拆分为多个较小的文件，或使用 bash 命令创建文件。"
                            )
                    
                    if arguments is not None:
                        parsed_tool_calls.append({
                            "id": tc["id"],
                            "name": tc["name"],
                            "arguments": arguments
                        })
                
                result["tool_calls"] = parsed_tool_calls
                
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

