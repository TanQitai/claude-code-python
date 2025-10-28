// Claude Code Python - Web UI 前端逻辑

class AgentUI {
    constructor() {
        this.socket = null;
        this.isExecuting = false;
        this.currentMessage = null;
        
        // DOM 元素
        this.elements = {
            messages: document.getElementById('messages'),
            taskInput: document.getElementById('taskInput'),
            sendBtn: document.getElementById('sendBtn'),
            resetBtn: document.getElementById('resetBtn'),
            status: document.getElementById('status')
        };
        
        this.init();
    }
    
    init() {
        this.setupSocket();
        this.setupEventListeners();
        this.setupKeyboardShortcuts();
    }
    
    // WebSocket 设置
    setupSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            this.updateStatus('connected', '已连接');
            this.addSystemMessage('✅ 已连接到 Agent 系统');
        });
        
        this.socket.on('disconnect', () => {
            this.updateStatus('error', '连接断开');
            this.addSystemMessage('⚠️ 与服务器断开连接');
        });
        
        this.socket.on('connected', (data) => {
            console.log('Connected:', data);
        });
        
        this.socket.on('task_start', (data) => {
            this.isExecuting = true;
            this.updateButtonState();
            this.addUserMessage(data.task);
            this.currentMessage = this.createAssistantMessage();
        });
        
        this.socket.on('iteration', (data) => {
            this.addIterationIndicator(data.iteration, data.max_iterations);
        });
        
        this.socket.on('content', (data) => {
            if (this.currentMessage) {
                this.appendToMessage(this.currentMessage, data.content);
            }
        });
        
        this.socket.on('tool_call', (data) => {
            this.addToolCall(data.tool_name, data.tool_args);
        });
        
        this.socket.on('tool_result', (data) => {
            this.addToolResult(data.tool_name, data.success, data.message);
        });
        
        this.socket.on('task_complete', (data) => {
            this.isExecuting = false;
            this.updateButtonState();
            this.addSystemMessage(`✅ 任务完成 (${data.iterations} 轮推理)`);
            this.currentMessage = null;
        });
        
        this.socket.on('task_incomplete', (data) => {
            this.isExecuting = false;
            this.updateButtonState();
            this.addSystemMessage(`⚠️ ${data.message} (${data.iterations} 轮推理)`);
            this.currentMessage = null;
        });
        
        this.socket.on('max_iterations', (data) => {
            this.isExecuting = false;
            this.updateButtonState();
            this.addSystemMessage(`⚠️ 达到最大迭代次数 (${data.iterations})`);
            this.currentMessage = null;
        });
        
        this.socket.on('error', (data) => {
            this.isExecuting = false;
            this.updateButtonState();
            this.addSystemMessage(`❌ 错误: ${data.message}`, 'error');
            this.currentMessage = null;
        });
        
        this.socket.on('info', (data) => {
            this.addSystemMessage(data.message);
        });
    }
    
    // 事件监听
    setupEventListeners() {
        // 发送按钮
        this.elements.sendBtn.addEventListener('click', () => {
            this.sendTask();
        });
        
        // 重置按钮
        this.elements.resetBtn.addEventListener('click', () => {
            this.resetConversation();
        });
        
        // 输入框自动调整高度
        this.elements.taskInput.addEventListener('input', (e) => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
        });
    }
    
    // 键盘快捷键
    setupKeyboardShortcuts() {
        this.elements.taskInput.addEventListener('keydown', (e) => {
            // Enter 发送 (不按 Shift)
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendTask();
            }
            // Esc 清空
            else if (e.key === 'Escape') {
                this.elements.taskInput.value = '';
                this.elements.taskInput.style.height = 'auto';
            }
        });
    }
    
    // 发送任务
    sendTask() {
        const task = this.elements.taskInput.value.trim();
        
        if (!task) {
            return;
        }
        
        if (this.isExecuting) {
            this.addSystemMessage('⚠️ 正在执行任务，请稍候...', 'warning');
            return;
        }
        
        // 发送到后端
        this.socket.emit('execute_task', { task });
        
        // 清空输入
        this.elements.taskInput.value = '';
        this.elements.taskInput.style.height = 'auto';
    }
    
    // 重置对话
    resetConversation() {
        if (this.isExecuting) {
            if (!confirm('正在执行任务，确定要重置吗？')) {
                return;
            }
        }
        
        this.socket.emit('reset_conversation');
        
        // 清空消息（保留欢迎消息）
        const welcomeMsg = this.elements.messages.firstElementChild;
        this.elements.messages.innerHTML = '';
        if (welcomeMsg) {
            this.elements.messages.appendChild(welcomeMsg);
        }
    }
    
    // UI 更新方法
    updateStatus(status, text) {
        this.elements.status.className = `status-indicator ${status}`;
        this.elements.status.querySelector('.status-text').textContent = text;
    }
    
    updateButtonState() {
        this.elements.sendBtn.disabled = this.isExecuting;
        this.elements.taskInput.disabled = this.isExecuting;
        
        if (this.isExecuting) {
            this.elements.sendBtn.innerHTML = `
                <span class="loading">执行中</span>
            `;
        } else {
            this.elements.sendBtn.innerHTML = `
                <svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M15.854.146a.5.5 0 0 1 .11.54l-5.819 14.547a.75.75 0 0 1-1.329.124l-3.178-4.995L.643 7.184a.75.75 0 0 1 .124-1.33L15.314.037a.5.5 0 0 1 .54.11ZM6.636 10.07l2.761 4.338L14.13 2.576 6.636 10.07Zm6.787-8.201L1.591 6.602l4.339 2.76 7.494-7.493Z"/>
                </svg>
                发送
            `;
        }
    }
    
    // 消息创建方法
    addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.innerHTML = `
            <div class="message-content">
                ${this.escapeHtml(text)}
            </div>
        `;
        this.elements.messages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    createAssistantMessage() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message';
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="content"></div>
            </div>
        `;
        this.elements.messages.appendChild(messageDiv);
        this.scrollToBottom();
        return messageDiv.querySelector('.content');
    }
    
    appendToMessage(element, text) {
        element.textContent += text;
        this.scrollToBottom();
    }
    
    addSystemMessage(text, type = 'info') {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system-message';
        messageDiv.innerHTML = `
            <div class="message-content">
                ${this.escapeHtml(text)}
            </div>
        `;
        this.elements.messages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    addIterationIndicator(iteration, maxIterations) {
        const indicator = document.createElement('div');
        indicator.className = 'iteration-indicator';
        indicator.innerHTML = `
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                <path d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/>
                <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/>
            </svg>
            <span>推理轮 ${iteration}/${maxIterations}</span>
        `;
        this.elements.messages.appendChild(indicator);
        this.scrollToBottom();
    }
    
    addToolCall(toolName, toolArgs) {
        const toolDiv = document.createElement('div');
        toolDiv.className = 'tool-call';
        
        // 简化参数显示
        let argsText = JSON.stringify(toolArgs, null, 2);
        if (argsText.length > 100) {
            argsText = argsText.substring(0, 100) + '...';
        }
        
        toolDiv.innerHTML = `
            <div class="tool-icon">🔧</div>
            <div class="tool-info">
                <div class="tool-name">${this.escapeHtml(toolName)}</div>
                <div class="tool-args">${this.escapeHtml(argsText)}</div>
            </div>
        `;
        this.elements.messages.appendChild(toolDiv);
        this.scrollToBottom();
    }
    
    addToolResult(toolName, success, message) {
        const resultDiv = document.createElement('div');
        resultDiv.className = `tool-result ${success ? 'success' : 'error'}`;
        resultDiv.innerHTML = `
            <span>${success ? '✅' : '❌'}</span>
            <span>${this.escapeHtml(message)}</span>
        `;
        this.elements.messages.appendChild(resultDiv);
        this.scrollToBottom();
    }
    
    // 工具方法
    scrollToBottom() {
        this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    window.agentUI = new AgentUI();
});

