#!/usr/bin/env python3
"""
交互式 Agent CLI - 真正的 Agentic Tool System

用户可以输入任意自然语言指令，Agent 自动分析并调用相应工具完成任务。
"""
import sys
import os
import json
from datetime import datetime
from llm_client import K2Client
from agent import AgentManager
import config
from loguru import logger


class AgentCLI:
    """交互式 Agent 命令行界面"""
    
    def __init__(self):
        """初始化 Agent CLI"""
        # 配置日志
        logger.remove()  # 移除默认handler
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            level="INFO"
        )
        
        # 初始化 LLM 和 Agent
        try:
            self.llm_client = K2Client()
            self.agent_manager = AgentManager(self.llm_client)
            self.agent = self.agent_manager.create_main_agent()
            logger.info("Agent 系统初始化成功")
        except Exception as e:
            print(f"\n❌ 初始化失败: {e}")
            print("\n请检查:")
            print("  1. .env 文件是否配置正确")
            print("  2. API Key 是否有效")
            print("  3. 账户是否有余额")
            sys.exit(1)
        
        # 对话记录相关
        self.conversation_log = []
        self.session_start_time = datetime.now()
        self.log_dir = "conversation_logs"
        self._ensure_log_dir()
        self.current_log_file = self._create_log_file()
        
        print(f"📝 对话记录将保存到: {self.current_log_file}")
    
    def _ensure_log_dir(self):
        """确保日志目录存在"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            logger.info(f"创建对话记录目录: {self.log_dir}")
    
    def _create_log_file(self):
        """创建日志文件"""
        timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.jsonl"
        filepath = os.path.join(self.log_dir, filename)
        
        # 写入会话开始信息
        session_info = {
            "type": "session_start",
            "timestamp": self.session_start_time.isoformat(),
            "model": config.K2_MODEL,
            "base_url": config.K2_BASE_URL
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json.dumps(session_info, ensure_ascii=False) + '\n')
        
        logger.info(f"创建对话记录文件: {filepath}")
        return filepath
    
    def _log_interaction(self, interaction_type, content, **kwargs):
        """记录交互内容"""
        log_entry = {
            "type": interaction_type,
            "timestamp": datetime.now().isoformat(),
            "content": content,
            **kwargs
        }
        
        # 添加到内存记录
        self.conversation_log.append(log_entry)
        
        # 实时追加到文件
        try:
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"保存对话记录失败: {e}")
    
    def save_conversation_summary(self):
        """保存对话摘要（可读格式）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.log_dir, f"summary_{timestamp}.txt")
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                # 写入标题
                f.write("="*70 + "\n")
                f.write("Claude Code Python - 对话记录\n")
                f.write("="*70 + "\n\n")
                
                # 写入会话信息
                f.write(f"会话开始时间: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"会话结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型: {config.K2_MODEL}\n")
                f.write(f"对话轮数: {len([l for l in self.conversation_log if l['type'] == 'user_input'])}\n")
                f.write("\n" + "="*70 + "\n\n")
                
                # 写入对话内容
                conversation_count = 0
                for entry in self.conversation_log:
                    if entry['type'] == 'user_input':
                        conversation_count += 1
                        f.write(f"\n{'='*70}\n")
                        f.write(f"对话 #{conversation_count}\n")
                        f.write(f"{'='*70}\n\n")
                        f.write(f"时间: {entry['timestamp']}\n")
                        f.write(f"👤 用户: {entry['content']}\n\n")
                    
                    elif entry['type'] == 'agent_response':
                        f.write(f"🤖 Agent: {entry['content']}\n")
                        if entry.get('iterations'):
                            f.write(f"   推理轮数: {entry['iterations']}\n")
                    
                    elif entry['type'] == 'tool_call':
                        f.write(f"   🔧 工具调用: {entry.get('tool_name')}\n")
                        if entry.get('tool_args'):
                            f.write(f"      参数: {json.dumps(entry['tool_args'], ensure_ascii=False, indent=2)}\n")
                    
                    elif entry['type'] == 'tool_result':
                        success = entry.get('success', False)
                        status = "✅ 成功" if success else "❌ 失败"
                        f.write(f"   {status}: {entry.get('message', '')}\n")
                    
                    elif entry['type'] == 'error':
                        f.write(f"   ❌ 错误: {entry.get('error')}\n")
                
                # 写入结束标记
                f.write(f"\n{'='*70}\n")
                f.write("会话结束\n")
                f.write("="*70 + "\n")
            
            print(f"\n✅ 对话摘要已保存到: {summary_file}")
            return summary_file
        
        except Exception as e:
            logger.error(f"保存对话摘要失败: {e}")
            return None
    
    def print_welcome(self):
        """打印欢迎信息"""
        print("\n" + "="*70)
        print("🤖 Claude Code Python - 交互式 Agent 系统")
        print("="*70)
        print()
        print("💡 使用说明:")
        print("   • 输入任意自然语言指令，Agent 会自动分析并执行")
        print("   • 支持文件操作、系统命令、任务管理等")
        print("   • 支持多轮对话，Agent 会记住上下文")
        print("   • 对话记录会自动保存")
        print()
        print("📋 命令:")
        print("   • 'help'  - 查看使用示例")
        print("   • 'save'  - 保存对话摘要")
        print("   • 'reset' - 清空对话历史")
        print("   • 'clear' - 清空屏幕")
        print("   • 'quit'  - 退出程序")
        print()
        print("🔧 当前配置:")
        print(f"   • Model: {config.K2_MODEL}")
        print(f"   • Base URL: {config.K2_BASE_URL}")
        print()
        print("="*70)
        print()
    
    def print_help(self):
        """打印帮助信息"""
        print("\n" + "="*70)
        print("📚 示例指令")
        print("="*70)
        print()
        print("🐍 Python相关:")
        print("   • 创建一个Python文件，实现冒泡排序算法")
        print("   • 写一个脚本，读取CSV文件并统计行数")
        print("   • 生成一个简单的Flask Web应用")
        print()
        print("📁 文件操作:")
        print("   • 在当前目录创建一个名为 test 的文件夹")
        print("   • 删除 temp 目录下的所有 .log 文件")
        print("   • 列出所有Python文件并统计代码行数")
        print()
        print("🔍 系统分析:")
        print("   • 检查系统资源使用情况")
        print("   • 分析当前项目的代码结构")
        print("   • 查找所有大于100MB的文件")
        print()
        print("⚡ 并行任务:")
        print("   • 同时检查Python版本、磁盘空间和内存使用")
        print("   • 并行处理多个数据文件")
        print()
        print("🛠️ 复杂任务:")
        print("   • 帮我创建一个项目，包含README和示例代码")
        print("   • 自动化部署检查：测试、构建、生成报告")
        print()
        print("📝 对话记录:")
        print("   • 所有对话会自动保存到 conversation_logs/ 目录")
        print("   • 输入 'save' 可生成可读的对话摘要")
        print("   • 退出时会自动保存完整记录")
        print()
        print("="*70)
        print()
    
    def run(self):
        """运行交互式 Agent"""
        self.print_welcome()
        
        conversation_count = 0
        
        while True:
            try:
                # 获取用户输入
                user_input = input("👤 你: ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 再见！正在保存对话记录...")
                    self._log_interaction("session_end", "用户退出")
                    self.save_conversation_summary()
                    break
                
                if user_input.lower() in ['help', 'h', '?']:
                    self.print_help()
                    continue
                
                if user_input.lower() == 'save':
                    print("\n💾 保存对话摘要中...")
                    self.save_conversation_summary()
                    continue
                
                if user_input.lower() == 'clear':
                    print("\033c", end="")  # 清屏
                    self.agent.reset_conversation()  # 重置对话历史
                    self._log_interaction("reset", "清空屏幕并重置对话")
                    self.print_welcome()
                    print("💡 对话历史已清空，开始新对话\n")
                    continue
                
                if user_input.lower() in ['reset', 'new']:
                    self.agent.reset_conversation()
                    self._log_interaction("reset", "重置对话历史")
                    print("\n✅ 对话历史已重置，开始新对话\n")
                    continue
                
                # 记录用户输入
                self._log_interaction("user_input", user_input)
                
                # 执行任务（流式）
                print()
                print("🤔 Agent 思考中...")
                print()
                
                # 流式执行任务
                result_text = ""
                current_iteration = 0
                success = False
                
                try:
                    for event in self.agent.run_stream(user_input):
                        event_type = event.get("type")
                        
                        if event_type == "iteration":
                            current_iteration = event["iteration"]
                            print(f"\n🔄 推理轮 {current_iteration}/{event['max_iterations']}")
                            print("-" * 70)
                        
                        elif event_type == "content":
                            # 流式输出 LLM 的文本内容
                            print(event["content"], end="", flush=True)
                            result_text += event["content"]
                        
                        elif event_type == "tool_call":
                            # 显示工具调用
                            print(f"\n\n🔧 调用工具: {event['tool_name']}")
                            # 简化参数显示
                            args = event['tool_args']
                            if len(str(args)) > 100:
                                print(f"   参数: {str(args)[:100]}...")
                            else:
                                print(f"   参数: {args}")
                            
                            # 记录工具调用
                            self._log_interaction("tool_call", 
                                                event['tool_name'],
                                                tool_name=event['tool_name'],
                                                tool_args=event['tool_args'])
                        
                        elif event_type == "tool_result":
                            # 显示工具结果
                            result = event["result"]
                            success_flag = result.get("success", False)
                            if success_flag:
                                print(f"   ✅ 成功: {result.get('message', '执行成功')}")
                            else:
                                print(f"   ❌ 失败: {result.get('error', '未知错误')}")
                            
                            # 记录工具结果
                            message = result.get('message', result.get('error', ''))
                            self._log_interaction("tool_result",
                                                message,
                                                success=success_flag,
                                                message=message)
                        
                        elif event_type == "complete":
                            # 任务完成
                            success = True
                            if not result_text:
                                result_text = event.get("result", "")
                        
                        elif event_type == "error":
                            # 错误
                            error_msg = event.get('error', '未知错误')
                            print(f"\n\n❌ 错误: {error_msg}")
                            self._log_interaction("error", error_msg, error=error_msg)
                            break
                        
                        elif event_type == "max_iterations":
                            # 达到最大迭代次数
                            print(f"\n\n⚠️  达到最大迭代次数 ({event['iterations']})")
                            self._log_interaction("max_iterations", 
                                                f"达到最大迭代次数 {event['iterations']}", 
                                                iterations=event['iterations'])
                            break
                    
                    # 记录 Agent 响应
                    if result_text:
                        self._log_interaction("agent_response", 
                                            result_text,
                                            iterations=current_iteration,
                                            success=success)
                    
                    # 显示最终结果
                    print("\n")
                    print("="*70)
                    
                    if success:
                        print("✅ 任务完成")
                        print("="*70)
                        if result_text and result_text.strip():
                            print()
                            print("🤖 Agent:")
                            print(result_text)
                    else:
                        print("⚠️  任务未完全完成")
                        print("="*70)
                    
                    print()
                    print(f"📊 推理轮数: {current_iteration}")
                    
                except Exception as e:
                    print(f"\n\n❌ 执行出错: {e}")
                    logger.exception("流式执行出错")
                
                print()
                print("="*70)
                print()
                
                conversation_count += 1
                
            except KeyboardInterrupt:
                print("\n\n⚠️  用户中断")
                print("输入 'quit' 退出，或继续输入指令")
                print()
            
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                print()
                logger.exception("Agent 执行出错")


def main():
    """主函数"""
    # 检查配置
    if not config.K2_API_KEY:
        print("\n❌ 错误: 未配置 K2_API_KEY")
        print("\n请先配置 API Key:")
        print("  1. 编辑 .env 文件")
        print("  2. 或运行: bash setup_kimi_api.sh")
        print()
        sys.exit(1)
    
    # 运行 Agent CLI
    cli = AgentCLI()
    cli.run()


if __name__ == "__main__":
    main()

