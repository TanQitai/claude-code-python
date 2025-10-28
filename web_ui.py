#!/usr/bin/env python3
"""
Claude Code Python - Web UI
简约高效的 Web 界面
"""
import sys
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from loguru import logger
from llm_client import K2Client
from agent import AgentManager
import config
import threading

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
logger.add("logs/web_ui_{time}.log", rotation="1 day", level="DEBUG")

# 创建 Flask 应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'claude-code-python-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局 Agent 实例
llm_client = None
agent_manager = None
agent = None

# 用户会话管理（简单实现，生产环境需要更完善的会话管理）
user_agents = {}


def init_agent():
    """初始化 Agent"""
    global llm_client, agent_manager, agent
    try:
        llm_client = K2Client()
        agent_manager = AgentManager(llm_client)
        agent = agent_manager.create_main_agent()
        logger.info("Agent 系统初始化成功")
        return True
    except Exception as e:
        logger.error(f"Agent 初始化失败: {e}")
        return False


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """系统状态"""
    return jsonify({
        'status': 'ok' if agent else 'error',
        'model': config.K2_MODEL,
        'base_url': config.K2_BASE_URL
    })


@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    logger.info(f"客户端连接: {request.sid}")
    emit('connected', {'message': '已连接到 Agent 系统'})


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开"""
    logger.info(f"客户端断开: {request.sid}")
    # 清理会话
    if request.sid in user_agents:
        del user_agents[request.sid]


@socketio.on('execute_task')
def handle_execute_task(data):
    """执行任务"""
    task = data.get('task', '').strip()
    reset_history = data.get('reset_history', False)
    
    if not task:
        emit('error', {'message': '请输入任务内容'})
        return
    
    logger.info(f"[{request.sid}] 收到任务: {task}")
    
    # 保存当前会话ID
    current_sid = request.sid
    
    # 获取或创建用户的 Agent
    if current_sid not in user_agents or reset_history:
        user_agents[current_sid] = agent_manager.create_main_agent()
        if reset_history:
            socketio.emit('info', {'message': '✨ 已重置对话历史'}, room=current_sid)
    
    user_agent = user_agents[current_sid]
    
    # 使用 SocketIO 的后台任务功能
    def run_task():
        try:
            socketio.emit('task_start', {'task': task}, room=current_sid)
            
            current_iteration = 0
            result_text = ""
            success = False
            
            # 流式执行任务
            for event in user_agent.run_stream(task, reset_history=False):
                event_type = event.get("type")
                
                if event_type == "iteration":
                    current_iteration = event["iteration"]
                    socketio.emit('iteration', {
                        'iteration': current_iteration,
                        'max_iterations': event['max_iterations']
                    }, room=current_sid)
                
                elif event_type == "content":
                    # 流式输出 LLM 的文本内容
                    socketio.emit('content', {'content': event["content"]}, room=current_sid)
                    result_text += event["content"]
                
                elif event_type == "tool_call":
                    # 工具调用
                    socketio.emit('tool_call', {
                        'tool_name': event['tool_name'],
                        'tool_args': event['tool_args']
                    }, room=current_sid)
                
                elif event_type == "tool_result":
                    # 工具结果
                    result = event["result"]
                    socketio.emit('tool_result', {
                        'tool_name': event['tool_name'],
                        'success': result.get('success', True),
                        'message': result.get('message', result.get('error', ''))
                    }, room=current_sid)
                
                elif event_type == "complete":
                    # 任务完成
                    success = True
                    if not result_text:
                        result_text = event.get("result", "")
                    socketio.emit('task_complete', {
                        'result': result_text,
                        'iterations': event['iterations']
                    }, room=current_sid)
                
                elif event_type == "error":
                    # 错误
                    socketio.emit('error', {
                        'message': event.get('error', '未知错误')
                    }, room=current_sid)
                    return
                
                elif event_type == "max_iterations":
                    # 达到最大迭代次数
                    socketio.emit('max_iterations', {
                        'iterations': event['iterations']
                    }, room=current_sid)
                    return
            
            if not success:
                socketio.emit('task_incomplete', {
                    'message': '任务未完全完成',
                    'iterations': current_iteration
                }, room=current_sid)
        
        except Exception as e:
            logger.exception(f"任务执行出错: {e}")
            socketio.emit('error', {'message': f'执行出错: {str(e)}'}, room=current_sid)
    
    # 使用 SocketIO 的后台任务启动器
    socketio.start_background_task(run_task)


@socketio.on('reset_conversation')
def handle_reset():
    """重置对话"""
    current_sid = request.sid
    if current_sid in user_agents:
        user_agents[current_sid].reset_conversation()
        emit('info', {'message': '✨ 对话历史已重置'})
        logger.info(f"[{current_sid}] 重置对话")
    else:
        emit('info', {'message': '✨ 已创建新对话'})


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🚀 Claude Code Python - Web UI")
    print("="*70)
    
    # 初始化 Agent
    print("\n正在初始化 Agent 系统...")
    if not init_agent():
        print("\n❌ 初始化失败！")
        print("\n请检查:")
        print("  1. .env 文件是否配置正确")
        print("  2. API Key 是否有效")
        print("  3. 账户是否有余额")
        sys.exit(1)
    
    print("✅ Agent 系统初始化成功")
    print(f"\n📋 配置信息:")
    print(f"  • Model: {config.K2_MODEL}")
    print(f"  • Base URL: {config.K2_BASE_URL}")
    
    print("\n" + "="*70)
    print("🌐 Web UI 启动中...")
    print("="*70)
    print("\n📍 访问地址: http://localhost:5000")
    print("\n💡 提示: 按 Ctrl+C 停止服务\n")
    
    # 启动服务器
    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 服务器错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

