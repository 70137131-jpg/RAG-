class ChatBot {
    constructor() {
        this.chatArea = document.getElementById('chatArea');
        this.chatMessages = document.getElementById('chatMessages');
        this.chatForm = document.getElementById('chatForm');
        this.questionInput = document.getElementById('questionInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.composerWrapper = document.getElementById('composerWrapper');
        this.greeting = document.getElementById('greeting');
        
        // New elements
        this.settingsBtn = document.getElementById('settingsBtn');
        this.settingsDropdown = document.getElementById('settingsDropdown');
        this.topKSelect = document.getElementById('topKSelect');
        this.attachBtn = document.getElementById('attachBtn');
        this.modelName = document.getElementById('modelName');
        this.docCount = document.getElementById('docCount');
        this.sessionHistoryList = document.getElementById('sessionHistoryList');
        this.chatLogList = document.getElementById('chatLogList');
        
        // UI Buttons
        this.shareBtn = document.getElementById('shareBtn');
        this.userProfileBtn = document.getElementById('userProfileBtn');
        this.chatsNavBtn = document.getElementById('chatsNavBtn');
        this.chatTitleBtn = document.getElementById('chatTitleBtn');
        
        this.isLoading = false;
        this.hasMessages = false;
        this.currentSessionId = null;
        this.currentConversation = [];
        this.sessionStorageKey = 'rag_chat_sessions';
        this.sessions = this.readStoredSessions();

        // Configure marked.js for markdown parsing
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                breaks: true,
                gfm: true,
                highlight: function(code, lang) {
                    if (typeof hljs !== 'undefined') {
                        const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                        return hljs.highlight(code, { language }).value;
                    }
                    return code;
                }
            });
        }

        this.initEventListeners();
        this.initResizers();
        this.autoResizeTextarea();
        this.loadSystemStats();
        this.loadChatHistory();
    }

    initResizers() {
        const leftResizer = document.getElementById('leftResizer');
        const leftSidebar = document.getElementById('leftSidebar');
        const rightResizer = document.getElementById('rightResizer');
        const rightSidebar = document.getElementById('rightSidebar');

        let isResizingLeft = false;
        let isResizingRight = false;

        if (leftResizer && leftSidebar) {
            leftResizer.addEventListener('mousedown', (e) => {
                isResizingLeft = true;
                leftResizer.classList.add('resizing');
                document.body.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';
            });
        }

        if (rightResizer && rightSidebar) {
            rightResizer.addEventListener('mousedown', (e) => {
                isResizingRight = true;
                rightResizer.classList.add('resizing');
                document.body.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';
            });
        }

        document.addEventListener('mousemove', (e) => {
            if (!isResizingLeft && !isResizingRight) return;
            e.preventDefault(); // Prevent text selection

            if (isResizingLeft) {
                const newWidth = e.clientX;
                if (newWidth >= 200 && newWidth <= 600) {
                    leftSidebar.style.width = `${newWidth}px`;
                }
            } else if (isResizingRight) {
                const newWidth = document.body.clientWidth - e.clientX;
                if (newWidth >= 200 && newWidth <= 600) {
                    rightSidebar.style.width = `${newWidth}px`;
                }
            }
        });

        document.addEventListener('mouseup', () => {
            if (isResizingLeft) {
                isResizingLeft = false;
                if(leftResizer) leftResizer.classList.remove('resizing');
            }
            if (isResizingRight) {
                isResizingRight = false;
                if(rightResizer) rightResizer.classList.remove('resizing');
            }
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        });
    }

    initEventListeners() {
        if (this.chatForm) {
            this.chatForm.addEventListener('submit', (event) => {
                event.preventDefault();
                this.sendMessage();
            });
        }

        if (this.questionInput) {
            this.questionInput.addEventListener('input', () => {
                this.updateSendState();
                this.autoResizeTextarea();
            });

            this.questionInput.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    if (!this.sendBtn.disabled && !this.isLoading) {
                        this.sendMessage();
                    }
                }
            });
        }

        if (this.clearBtn) {
            this.clearBtn.addEventListener('click', () => this.startNewChat());
        }

        if (this.settingsBtn && this.settingsDropdown) {
            this.settingsBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.settingsDropdown.classList.toggle('active');
            });

            document.addEventListener('click', (e) => {
                if (!this.settingsDropdown.contains(e.target) && !this.settingsBtn.contains(e.target)) {
                    this.settingsDropdown.classList.remove('active');
                }
            });
        }

        if (this.attachBtn) {
            this.attachBtn.addEventListener('click', () => {
                alert('Document upload is not currently supported in this demo environment.');
            });
        }

        if (this.shareBtn) {
            this.shareBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(window.location.href);
                alert('URL copied to clipboard!');
            });
        }

        if (this.userProfileBtn) {
            this.userProfileBtn.addEventListener('click', () => {
                alert('User profile settings coming soon.');
            });
        }

        if (this.chatsNavBtn) {
            this.chatsNavBtn.addEventListener('click', (e) => {
                e.preventDefault();
                const historyEl = document.getElementById('sidebarHistory');
                if(historyEl) historyEl.scrollIntoView({ behavior: 'smooth' });
            });
        }

        if (this.chatTitleBtn) {
            this.chatTitleBtn.addEventListener('click', () => {
                alert('Chat configuration coming soon.');
            });
        }

        this.indexDataBtn = document.getElementById('indexDataBtn');
        this.indexLoader = document.getElementById('indexLoader');

        if (this.indexDataBtn) {
            this.indexDataBtn.addEventListener('click', async () => {
                this.indexDataBtn.style.display = 'none';
                if (this.indexLoader) this.indexLoader.style.display = 'flex';
                
                try {
                    const response = await fetch('/api/index-data', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    const data = await response.json();
                    if (response.ok && data.success) {
                        alert(`Successfully indexed ${data.indexed_count} documents!`);
                        this.loadSystemStats();
                    } else {
                        alert(`Failed to index data: ${data.detail || data.error}`);
                    }
                } catch (err) {
                    alert('Error connecting to the server while indexing.');
                } finally {
                    this.indexDataBtn.style.display = 'flex';
                    if (this.indexLoader) this.indexLoader.style.display = 'none';
                }
            });
        }
    }

    updateSendState() {
        if (!this.sendBtn || !this.questionInput) return;
        this.sendBtn.disabled = this.questionInput.value.trim().length === 0 || this.isLoading;
    }

    autoResizeTextarea() {
        if (!this.questionInput) return;
        this.questionInput.style.height = 'auto';
        this.questionInput.style.height = `${Math.min(this.questionInput.scrollHeight, 200)}px`;
    }

    switchToChatView() {
        if (this.hasMessages) return;
        this.hasMessages = true;
        
        if (this.greeting) this.greeting.style.display = 'none';
        if (this.composerWrapper) this.composerWrapper.classList.remove('centered');
        if (this.chatMessages) this.chatMessages.style.display = 'flex';
    }

    switchToWelcomeView() {
        this.hasMessages = false;
        
        if (this.greeting) this.greeting.style.display = 'block';
        if (this.composerWrapper) this.composerWrapper.classList.add('centered');
        if (this.chatMessages) {
            this.chatMessages.style.display = 'none';
            this.chatMessages.innerHTML = '';
        }
    }

    async loadSystemStats() {
        try {
            const response = await fetch('/api/stats');
            if (response.ok) {
                const data = await response.json();
                if (data.success && data.stats) {
                    if (this.modelName) this.modelName.textContent = data.stats.llm_model || 'LLM';
                    if (this.docCount) this.docCount.textContent = `${data.stats.total_documents} docs indexed`;
                    const latencyStat = document.getElementById('latencyStat');
                    if (latencyStat) latencyStat.textContent = `${data.stats.last_response_time_ms || '--'} ms`;
                }
            } else if (response.status === 401) {
                const healthRes = await fetch('/health');
                if (healthRes.ok) {
                    const healthData = await healthRes.json();
                    if (this.docCount) this.docCount.textContent = `${healthData.documents_indexed || 0} docs indexed`;
                    if (this.modelName) this.modelName.textContent = 'Gemini Pro'; 
                }
            }
        } catch (error) {
            console.error('Error fetching stats:', error);
        }
    }

    async loadChatHistory() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();

            if (data.success) {
                this.currentSessionId = data.session_id;
                this.renderConversation(data.history || []);
                if (data.history && data.history.length > 0) {
                    this.upsertSession(this.currentSessionId, this.getSessionTitle(data.history));
                } else {
                    this.renderSessionSidebar();
                }
                this.loadSessionLogs();
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    readStoredSessions() {
        try {
            const stored = JSON.parse(localStorage.getItem(this.sessionStorageKey) || '[]');
            return Array.isArray(stored) ? stored.filter((session) => session && session.id) : [];
        } catch (error) {
            return [];
        }
    }

    saveStoredSessions() {
        localStorage.setItem(this.sessionStorageKey, JSON.stringify(this.sessions.slice(0, 50)));
    }

    getSessionTitle(history) {
        const firstQuestion = history.find((item) => item.question)?.question || 'New chat';
        return firstQuestion.length > 48 ? `${firstQuestion.slice(0, 45)}...` : firstQuestion;
    }

    upsertSession(sessionId, title) {
        if (!sessionId) return;
        const existing = this.sessions.find((session) => session.id === sessionId);
        if (existing) {
            existing.title = existing.title === 'New chat' ? title : existing.title;
            existing.updatedAt = new Date().toISOString();
        } else {
            this.sessions.unshift({
                id: sessionId,
                title,
                updatedAt: new Date().toISOString(),
            });
        }
        this.sessions.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
        this.saveStoredSessions();
        this.renderSessionSidebar();
    }

    renderSessionSidebar() {
        if (!this.sessionHistoryList) return;

        this.sessionHistoryList.innerHTML = '';
        if (this.sessions.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'empty-history';
            empty.textContent = 'No saved chats yet';
            this.sessionHistoryList.appendChild(empty);
            return;
        }

        this.sessions.forEach((session) => {
            const a = document.createElement('a');
            a.href = '#';
            a.className = `history-item${session.id === this.currentSessionId ? ' active' : ''}`;
            a.textContent = session.title || 'Untitled chat';
            a.title = session.title || 'Untitled chat';

            a.addEventListener('click', (e) => {
                e.preventDefault();
                this.loadSession(session.id);
            });

            this.sessionHistoryList.appendChild(a);
        });
    }

    renderConversation(history) {
        this.currentConversation = history || [];
        if (!this.currentConversation.length) {
            this.switchToWelcomeView();
            return;
        }

        if (this.chatMessages) {
            this.chatMessages.innerHTML = '';
        }
        this.switchToChatView();
        this.currentConversation.forEach((item) => {
            this.addMessage(item.question, 'user', null, false);
            this.addMessage(item.answer, 'bot', item.sources || null, false);
        });
    }

    async loadSession(sessionId) {
        if (!sessionId || sessionId === this.currentSessionId) return;

        try {
            await fetch('/api/switch-chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            });

            const response = await fetch(`/api/history?session_id=${encodeURIComponent(sessionId)}`);
            const data = await response.json();
            if (data.success) {
                this.currentSessionId = data.session_id;
                this.renderConversation(data.history || []);
                if (data.history && data.history.length > 0) {
                    this.upsertSession(this.currentSessionId, this.getSessionTitle(data.history));
                } else {
                    this.renderSessionSidebar();
                }
                this.loadSessionLogs();
            }
        } catch (error) {
            console.error('Error loading saved chat:', error);
        }
    }

    async loadSessionLogs() {
        if (!this.chatLogList) return;

        try {
            const response = await fetch('/api/session-chat-logs?limit=25');
            const data = await response.json();
            if (data.success) {
                this.renderSessionLogs(data.logs || []);
            }
        } catch (error) {
            console.error('Error loading chat logs:', error);
        }
    }

    renderSessionLogs(logs) {
        if (!this.chatLogList) return;

        this.chatLogList.innerHTML = '';
        if (!logs.length) {
            const empty = document.createElement('div');
            empty.className = 'empty-log';
            empty.textContent = 'No logs yet';
            this.chatLogList.appendChild(empty);
            return;
        }

        logs.forEach((log) => {
            const item = document.createElement('div');
            item.className = 'log-item';

            const question = document.createElement('div');
            question.className = 'log-question';
            question.textContent = log.question || 'Untitled question';

            const meta = document.createElement('div');
            meta.className = 'log-meta';
            const created = log.created_at ? new Date(log.created_at).toLocaleString() : 'Unknown time';
            const latency = log.response_time_ms ? `${Math.round(log.response_time_ms)} ms` : '-- ms';
            meta.textContent = `${created} · ${latency}`;

            item.appendChild(question);
            item.appendChild(meta);
            this.chatLogList.appendChild(item);
        });
    }

    async startNewChat() {
        try {
            if (this.currentSessionId && this.currentConversation.length > 0) {
                this.upsertSession(this.currentSessionId, this.getSessionTitle(this.currentConversation));
            }

            const response = await fetch('/api/new-chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (data.success) {
                this.currentSessionId = data.session_id;
                this.currentConversation = [];
                this.switchToWelcomeView();
                this.renderSessionSidebar();
                this.renderSessionLogs([]);
                if (this.questionInput) {
                    this.questionInput.value = '';
                    this.autoResizeTextarea();
                    this.updateSendState();
                    this.questionInput.focus();
                }
            }
        } catch (error) {
            console.error('Error starting new chat:', error);
        }
    }

    addHistoryToSidebar(question) {
        if (!this.currentSessionId) return;
        const title = this.currentConversation.length > 0
            ? this.getSessionTitle(this.currentConversation)
            : question;
        this.upsertSession(this.currentSessionId, title);
    }

    async sendMessage() {
        const question = this.questionInput.value.trim();
        if (!question || this.isLoading) return;

        const topK = this.topKSelect ? parseInt(this.topKSelect.value, 10) : 3;
        
        this.isLoading = true;
        this.updateSendState();
        this.switchToChatView();
        
        this.addMessage(question, 'user');
        
        this.questionInput.value = '';
        this.autoResizeTextarea();
        this.showTyping();

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, top_k: topK })
            });

            const data = await response.json();
            this.hideTyping();

            if (data.success) {
                this.addMessage(data.answer, 'bot', data.sources);
                this.currentSessionId = data.session_id || this.currentSessionId;
                this.currentConversation.push({
                    question,
                    answer: data.answer,
                    sources: data.sources || [],
                    timestamp: data.timestamp,
                    response_time_ms: data.response_time_ms,
                });
                this.addHistoryToSidebar(question);
                this.loadSessionLogs();
                const latencyStat = document.getElementById('latencyStat');
                if (latencyStat && data.response_time_ms) {
                    latencyStat.textContent = `${data.response_time_ms} ms`;
                }
                this.loadSystemStats(); // Update other stats
            } else {
                const errorMessage = data.error || data.detail || 'Unknown error';
                this.addMessage(`Sorry, I encountered an error: ${errorMessage}`, 'bot');
            }
        } catch (error) {
            this.hideTyping();
            this.addMessage('Sorry, there was a network error. Please check if the server is running.', 'bot');
            console.error('Error:', error);
        } finally {
            this.isLoading = false;
            this.updateSendState();
            this.questionInput.focus();
        }
    }

    addMessage(text, type, sources = null, animate = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        if (animate) messageDiv.style.animation = 'fadeIn 0.2s ease';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (type === 'bot') {
            contentDiv.innerHTML = typeof marked !== 'undefined' ? marked.parse(text) : `<p>${text}</p>`;
            
            if (sources && sources.length > 0) {
                contentDiv.appendChild(this.createSourcesDrawer(sources));
            }
        } else {
            contentDiv.textContent = text;
        }

        messageDiv.appendChild(contentDiv);

        if (this.chatMessages) {
            this.chatMessages.appendChild(messageDiv);
        }

        this.scrollToBottom();
    }

    createSourcesDrawer(sources) {
        const panel = document.createElement('div');
        // Added 'expanded' class by default
        panel.className = 'sources-panel expanded';

        const toggle = document.createElement('button');
        toggle.className = 'sources-toggle';
        toggle.innerHTML = `
            <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none"><polyline points="9 18 15 12 9 6"></polyline></svg>
            Sources (${sources.length})
        `;
        
        const content = document.createElement('div');
        content.className = 'sources-content';

        sources.forEach((source, idx) => {
            const item = document.createElement('div');
            item.className = 'source-item';
            
            const similarityScore = source.similarity ? ` · ${source.similarity}% match` : '';
            item.innerHTML = `
                <h4>${source.id || `Source ${idx + 1}`}${similarityScore}</h4>
                <p>${source.text}</p>
            `;
            content.appendChild(item);
        });

        toggle.addEventListener('click', () => {
            panel.classList.toggle('expanded');
        });

        panel.appendChild(toggle);
        panel.appendChild(content);

        return panel;
    }

    showTyping() {
        if (!this.typingIndicator) return;
        this.typingIndicator.style.display = 'flex';
        this.chatMessages.appendChild(this.typingIndicator);
        this.scrollToBottom();
    }

    hideTyping() {
        if (!this.typingIndicator) return;
        this.typingIndicator.style.display = 'none';
    }

    scrollToBottom() {
        setTimeout(() => {
            if (this.chatArea) {
                this.chatArea.scrollTop = this.chatArea.scrollHeight;
            }
        }, 50);
    }

}

document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new ChatBot();
});
