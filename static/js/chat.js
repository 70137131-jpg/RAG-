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

        this.settingsBtn = document.getElementById('settingsBtn');
        this.settingsDropdown = document.getElementById('settingsDropdown');
        this.topKSelect = document.getElementById('topKSelect');
        this.modelName = document.getElementById('modelName');
        this.docCount = document.getElementById('docCount');
        this.sessionHistoryList = document.getElementById('sessionHistoryList');
        this.chatLogList = document.getElementById('chatLogList');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.adminTokenInput = document.getElementById('adminTokenInput');
        this.saveAdminTokenBtn = document.getElementById('saveAdminTokenBtn');
        this.adminTokenHint = document.getElementById('adminTokenHint');
        this.indexDataBtn = document.getElementById('indexDataBtn');
        this.indexLoader = document.getElementById('indexLoader');

        this.shareBtn = document.getElementById('shareBtn');
        this.menuBtn = document.getElementById('menuBtn');
        this.leftSidebar = document.getElementById('leftSidebar');
        this.sidebarBackdrop = document.getElementById('sidebarBackdrop');
        this.searchToggleBtn = document.getElementById('searchToggleBtn');
        this.sessionSearch = document.getElementById('sessionSearch');
        this.sessionSearchInput = document.getElementById('sessionSearchInput');
        this.toastContainer = document.getElementById('toastContainer');

        this.isLoading = false;
        this.hasMessages = false;
        this.MAX_QUESTION_LENGTH = 5000; // Keep in sync with backend QueryRequest validator
        this.currentSessionId = null;
        this.currentConversation = [];
        this.sessionStorageKey = 'rag_chat_sessions';
        this.sessions = this.readStoredSessions();
        this.adminToken = '';
        this.sessionFilter = '';
        this.historyLoadController = null;

        // Markdown parsing; syntax highlighting is applied post-render with
        // hljs.highlightElement (marked's `highlight` option is deprecated).
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                breaks: true,
                gfm: true
            });
        }

        this.initEventListeners();
        this.initResizers();
        this.autoResizeTextarea();
        this.updateAdminControls();
        this.loadSystemStats();
        this.loadChatHistory();
    }

    // ---------- Toast notifications ----------

    showToast(message, type = 'info', duration = 3500) {
        if (!this.toastContainer) {
            console.log(`[${type}] ${message}`);
            return;
        }
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        this.toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.classList.add('hiding');
            setTimeout(() => toast.remove(), 350);
        }, duration);
    }

    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (error) {
            // Fallback for non-secure contexts or denied permissions
            try {
                const helper = document.createElement('textarea');
                helper.value = text;
                helper.style.position = 'fixed';
                helper.style.opacity = '0';
                document.body.appendChild(helper);
                helper.select();
                const ok = document.execCommand('copy');
                helper.remove();
                return ok;
            } catch (fallbackError) {
                return false;
            }
        }
    }

    // ---------- Shared helpers ----------

    getAdminHeaders() {
        const headers = { 'Content-Type': 'application/json' };
        if (this.adminToken) {
            headers.Authorization = `Bearer ${this.adminToken}`;
        }
        return headers;
    }

    formatErrorDetail(detail) {
        // FastAPI validation errors (422) return detail as an array of objects
        if (Array.isArray(detail)) {
            return detail.map((item) => item.msg || JSON.stringify(item)).join('; ');
        }
        return detail;
    }

    async parseJsonResponse(response) {
        const text = await response.text();
        if (!text) return {};
        try {
            return JSON.parse(text);
        } catch (error) {
            return { error: text };
        }
    }

    formatTime(isoString) {
        if (!isoString) return '';
        const date = new Date(isoString);
        if (Number.isNaN(date.getTime())) return '';
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    setSystemStatus(state, text) {
        if (!this.statusIndicator) return;
        const dot = this.statusIndicator.querySelector('.status-dot');
        if (dot) {
            dot.className = `status-dot ${state}`;
        }
        if (this.statusText) {
            this.statusText.textContent = text;
        }
    }

    updateAdminControls() {
        const hasToken = Boolean(this.adminToken);
        if (this.indexDataBtn) {
            this.indexDataBtn.disabled = !hasToken || this.isLoading;
            this.indexDataBtn.title = hasToken
                ? 'Index the configured dataset'
                : 'Save an admin token to enable indexing';
        }
        if (this.adminTokenHint) {
            this.adminTokenHint.textContent = hasToken
                ? 'Admin token active for this page session. Refreshing the page clears it.'
                : 'Public chat works without a token. Admin actions stay locked until a valid token is saved.';
        }
    }

    // ---------- Layout: resizers and mobile drawer ----------

    initResizers() {
        const leftResizer = document.getElementById('leftResizer');
        const leftSidebar = this.leftSidebar;
        const rightResizer = document.getElementById('rightResizer');
        const rightSidebar = document.getElementById('rightSidebar');

        let isResizingLeft = false;
        let isResizingRight = false;

        if (leftResizer && leftSidebar) {
            leftResizer.addEventListener('mousedown', () => {
                isResizingLeft = true;
                leftResizer.classList.add('resizing');
                document.body.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';
            });
        }

        if (rightResizer && rightSidebar) {
            rightResizer.addEventListener('mousedown', () => {
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
                if (leftResizer) leftResizer.classList.remove('resizing');
            }
            if (isResizingRight) {
                isResizingRight = false;
                if (rightResizer) rightResizer.classList.remove('resizing');
            }
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        });
    }

    openMobileSidebar() {
        if (!this.leftSidebar) return;
        this.leftSidebar.classList.add('mobile-open');
        if (this.sidebarBackdrop) this.sidebarBackdrop.hidden = false;
        if (this.menuBtn) this.menuBtn.setAttribute('aria-expanded', 'true');
    }

    closeMobileSidebar() {
        if (!this.leftSidebar) return;
        this.leftSidebar.classList.remove('mobile-open');
        if (this.sidebarBackdrop) this.sidebarBackdrop.hidden = true;
        if (this.menuBtn) this.menuBtn.setAttribute('aria-expanded', 'false');
    }

    isMobileSidebarOpen() {
        return Boolean(this.leftSidebar && this.leftSidebar.classList.contains('mobile-open'));
    }

    // ---------- Event wiring ----------

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
                this.toggleSettingsDropdown();
            });

            document.addEventListener('click', (e) => {
                if (!this.settingsDropdown.contains(e.target) && !this.settingsBtn.contains(e.target)) {
                    this.toggleSettingsDropdown(false);
                }
            });
        }

        // One Escape handler for every dismissible surface
        document.addEventListener('keydown', (event) => {
            if (event.key !== 'Escape') return;
            if (this.settingsDropdown && this.settingsDropdown.classList.contains('active')) {
                this.toggleSettingsDropdown(false);
                if (this.settingsBtn) this.settingsBtn.focus();
            } else if (this.isMobileSidebarOpen()) {
                this.closeMobileSidebar();
                if (this.menuBtn) this.menuBtn.focus();
            }
        });

        if (this.menuBtn) {
            this.menuBtn.addEventListener('click', () => {
                if (this.isMobileSidebarOpen()) {
                    this.closeMobileSidebar();
                } else {
                    this.openMobileSidebar();
                }
            });
        }

        if (this.sidebarBackdrop) {
            this.sidebarBackdrop.addEventListener('click', () => this.closeMobileSidebar());
        }

        if (this.searchToggleBtn && this.sessionSearch && this.sessionSearchInput) {
            this.searchToggleBtn.addEventListener('click', () => {
                const showing = this.sessionSearch.hidden;
                this.sessionSearch.hidden = !showing;
                this.searchToggleBtn.setAttribute('aria-expanded', String(showing));
                if (showing) {
                    this.sessionSearchInput.focus();
                } else {
                    this.sessionSearchInput.value = '';
                    this.sessionFilter = '';
                    this.renderSessionSidebar();
                }
            });

            this.sessionSearchInput.addEventListener('input', () => {
                this.sessionFilter = this.sessionSearchInput.value;
                this.renderSessionSidebar();
            });
        }

        if (this.shareBtn) {
            this.shareBtn.addEventListener('click', async () => {
                const copied = await this.copyToClipboard(window.location.href);
                if (copied) {
                    this.showToast('Link copied to clipboard', 'success');
                } else {
                    this.showToast('Could not copy the link', 'error');
                }
            });
        }

        if (this.saveAdminTokenBtn && this.adminTokenInput) {
            this.saveAdminTokenBtn.addEventListener('click', () => {
                this.adminToken = this.adminTokenInput.value.trim();
                this.updateAdminControls();
                this.loadSystemStats();
                if (this.adminToken) {
                    this.showToast('Admin token saved for this session', 'success');
                }
            });
        }

        if (this.indexDataBtn) {
            this.indexDataBtn.addEventListener('click', () => this.indexDataset());
        }
    }

    toggleSettingsDropdown(force) {
        if (!this.settingsDropdown || !this.settingsBtn) return;
        const shouldOpen = typeof force === 'boolean'
            ? force
            : !this.settingsDropdown.classList.contains('active');
        this.settingsDropdown.classList.toggle('active', shouldOpen);
        this.settingsBtn.setAttribute('aria-expanded', String(shouldOpen));
    }

    async indexDataset() {
        if (!this.adminToken) {
            this.showToast('Enter the admin token before indexing data', 'error');
            return;
        }
        this.indexDataBtn.style.display = 'none';
        if (this.indexLoader) this.indexLoader.style.display = 'flex';

        try {
            const response = await fetch('/api/index-data', {
                method: 'POST',
                headers: this.getAdminHeaders()
            });
            const data = await this.parseJsonResponse(response);
            if (response.ok && data.success) {
                this.showToast(`Successfully indexed ${data.indexed_count} documents`, 'success', 5000);
                this.loadSystemStats();
            } else if (response.status === 401 || response.status === 403) {
                this.showToast('Indexing failed: the admin token was rejected', 'error');
            } else {
                const message = data.error || this.formatErrorDetail(data.detail) || 'Unknown error';
                this.showToast(`Failed to index data: ${message}`, 'error', 5000);
            }
        } catch (err) {
            this.showToast('Error connecting to the server while indexing', 'error');
        } finally {
            this.indexDataBtn.style.display = 'flex';
            if (this.indexLoader) this.indexLoader.style.display = 'none';
            this.updateAdminControls();
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

    // ---------- Stats and health ----------

    async loadSystemStats() {
        try {
            if (this.adminToken) {
                const response = await fetch('/api/stats', {
                    headers: this.getAdminHeaders()
                });
                const data = await this.parseJsonResponse(response);
                if (response.ok && data.success && data.stats) {
                    this.setSystemStatus('green', 'Ready');
                    this.updateStatsPanel(data.stats);
                    return;
                }
                if (response.status === 401 || response.status === 403) {
                    this.setSystemStatus('yellow', 'Admin token rejected');
                    if (this.adminTokenHint) {
                        this.adminTokenHint.textContent = 'The token was rejected. Check it and save again.';
                    }
                }
            }

            const healthRes = await fetch('/health');
            const healthData = await this.parseJsonResponse(healthRes);
            if (healthRes.ok) {
                this.setSystemStatus('green', 'Ready');
                if (this.docCount) this.docCount.textContent = `${healthData.documents_indexed || 0} docs indexed`;
            } else {
                this.setSystemStatus('yellow', 'Degraded');
                if (this.docCount) this.docCount.textContent = `${healthData.documents_indexed || 0} docs indexed`;
            }
        } catch (error) {
            this.setSystemStatus('red', 'Offline');
            console.error('Error fetching stats:', error);
        }
    }

    updateStatsPanel(stats) {
        if (!stats) return;
        if (this.modelName) this.modelName.textContent = stats.llm_model || 'LLM';
        if (this.docCount) this.docCount.textContent = `${stats.total_documents} docs indexed`;
        const latencyStat = document.getElementById('latencyStat');
        if (latencyStat) {
            latencyStat.textContent = `${stats.last_response_time_ms || '--'} ms`;
        }
    }

    // ---------- Session persistence ----------

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

    removeSession(sessionId) {
        this.sessions = this.sessions.filter((session) => session.id !== sessionId);
        this.saveStoredSessions();
        this.renderSessionSidebar();
    }

    deleteSession(sessionId) {
        const wasCurrent = sessionId === this.currentSessionId;
        this.removeSession(sessionId);
        this.showToast('Chat removed from sidebar', 'info');
        if (wasCurrent) {
            this.startNewChat();
        }
    }

    renderSessionSidebar() {
        if (!this.sessionHistoryList) return;

        this.sessionHistoryList.innerHTML = '';

        const query = this.sessionFilter.trim().toLowerCase();
        const visibleSessions = query
            ? this.sessions.filter((session) => (session.title || '').toLowerCase().includes(query))
            : this.sessions;

        if (visibleSessions.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'empty-history';
            empty.textContent = query ? 'No chats match your search' : 'No saved chats yet';
            this.sessionHistoryList.appendChild(empty);
            return;
        }

        visibleSessions.forEach((session) => {
            const row = document.createElement('div');
            row.className = `history-item-row${session.id === this.currentSessionId ? ' active' : ''}`;
            row.setAttribute('role', 'listitem');

            const item = document.createElement('button');
            item.type = 'button';
            item.className = 'history-item';
            item.textContent = session.title || 'Untitled chat';
            item.title = session.title || 'Untitled chat';
            item.addEventListener('click', () => {
                this.loadSession(session.id);
                this.closeMobileSidebar();
            });

            const deleteBtn = document.createElement('button');
            deleteBtn.type = 'button';
            deleteBtn.className = 'history-delete-btn';
            deleteBtn.setAttribute('aria-label', `Delete chat: ${session.title || 'Untitled chat'}`);
            deleteBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>';
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteSession(session.id);
            });

            row.appendChild(item);
            row.appendChild(deleteBtn);
            this.sessionHistoryList.appendChild(row);
        });
    }

    // ---------- Conversation loading ----------

    beginHistoryLoad() {
        if (this.historyLoadController) {
            this.historyLoadController.abort();
        }
        this.historyLoadController = new AbortController();
        return this.historyLoadController.signal;
    }

    applySessionData(data) {
        this.currentSessionId = data.session_id;
        this.renderConversation(data.history || []);
        if (data.history && data.history.length > 0) {
            this.upsertSession(this.currentSessionId, this.getSessionTitle(data.history));
        } else {
            this.renderSessionSidebar();
        }
        this.loadSessionLogs();
    }

    async loadChatHistory() {
        const signal = this.beginHistoryLoad();
        try {
            const response = await fetch('/api/history', { signal });
            const data = await this.parseJsonResponse(response);

            if (response.ok && data.success) {
                this.applySessionData(data);
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error('Error loading chat history:', error);
            }
        }
    }

    async loadSession(sessionId) {
        if (!sessionId || sessionId === this.currentSessionId) return;

        const signal = this.beginHistoryLoad();
        try {
            // Fetch first so expired sessions are detected before switching the cookie
            const response = await fetch(`/api/history?session_id=${encodeURIComponent(sessionId)}`, { signal });
            const data = await this.parseJsonResponse(response);
            if (!response.ok || !data.success) {
                this.showToast('Could not load that chat', 'error');
                return;
            }

            if (!data.history || data.history.length === 0) {
                this.removeSession(sessionId);
                this.showToast('That chat has expired and is no longer available', 'error');
                return;
            }

            await fetch('/api/switch-chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId }),
                signal
            });

            this.applySessionData(data);
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error('Error loading saved chat:', error);
                this.showToast('Could not load that chat', 'error');
            }
        }
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
            this.addMessage(item.answer, 'bot', item.sources || null, false, item.timestamp);
        });
    }

    async loadSessionLogs() {
        if (!this.chatLogList) return;

        try {
            const response = await fetch('/api/session-chat-logs?limit=25');
            const data = await this.parseJsonResponse(response);
            if (response.ok && data.success) {
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

        const newestFirstLogs = [...logs].sort((a, b) => {
            const aTime = a.created_at ? new Date(a.created_at).getTime() : 0;
            const bTime = b.created_at ? new Date(b.created_at).getTime() : 0;
            return bTime - aTime;
        });

        newestFirstLogs.forEach((log) => {
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
            const data = await this.parseJsonResponse(response);
            if (response.ok && data.success) {
                this.currentSessionId = data.session_id;
                this.currentConversation = [];
                this.switchToWelcomeView();
                this.renderSessionSidebar();
                this.renderSessionLogs([]);
                this.closeMobileSidebar();
                if (this.questionInput) {
                    this.questionInput.value = '';
                    this.autoResizeTextarea();
                    this.updateSendState();
                    this.questionInput.focus();
                }
            }
        } catch (error) {
            console.error('Error starting new chat:', error);
            this.showToast('Could not start a new chat', 'error');
        }
    }

    addHistoryToSidebar(question) {
        if (!this.currentSessionId) return;
        const title = this.currentConversation.length > 0
            ? this.getSessionTitle(this.currentConversation)
            : question;
        this.upsertSession(this.currentSessionId, title);
    }

    // ---------- Messages ----------

    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    sanitizeHtml(html) {
        if (typeof DOMPurify !== 'undefined') {
            return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
        }
        // Fallback: strip all HTML tags if DOMPurify is not loaded
        const div = document.createElement('div');
        div.textContent = html.replace(/<[^>]*>/g, '');
        return div.innerHTML;
    }

    async sendMessage() {
        const question = this.questionInput.value.trim();
        if (!question || this.isLoading) return;

        if (question.length > this.MAX_QUESTION_LENGTH) {
            this.showToast(`Question is too long. Maximum ${this.MAX_QUESTION_LENGTH} characters allowed.`, 'error');
            return;
        }

        const topK = this.topKSelect ? parseInt(this.topKSelect.value, 10) : 3;
        const sessionAtSend = this.currentSessionId;

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

            const data = await this.parseJsonResponse(response);
            this.hideTyping();

            if (response.ok && data.success) {
                // If the user switched chats while the request was in flight,
                // don't paint the answer into the wrong transcript.
                if (sessionAtSend && this.currentSessionId && this.currentSessionId !== sessionAtSend
                    && data.session_id !== this.currentSessionId) {
                    this.showToast('The answer was saved to your previous chat', 'info');
                    return;
                }

                this.addMessage(data.answer, 'bot', data.sources, true, data.timestamp);
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
                const errorMessage = data.error || this.formatErrorDetail(data.detail) || `Request failed with status ${response.status}`;
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

    addMessage(text, type, sources = null, animate = true, timestamp = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        if (animate) messageDiv.style.animation = 'fadeIn 0.2s ease';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (type === 'bot') {
            const rawHtml = typeof marked !== 'undefined' ? marked.parse(text) : `<p>${this.escapeHtml(text)}</p>`;
            contentDiv.innerHTML = this.sanitizeHtml(rawHtml);

            if (typeof hljs !== 'undefined') {
                contentDiv.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            }

            if (sources && sources.length > 0) {
                contentDiv.appendChild(this.createSourcesDrawer(sources));
            }
        } else {
            contentDiv.textContent = text;
        }

        messageDiv.appendChild(contentDiv);

        if (type === 'bot') {
            messageDiv.appendChild(this.createMessageMeta(text, timestamp));
        }

        if (this.chatMessages) {
            this.chatMessages.appendChild(messageDiv);
        }

        this.scrollToBottom();
    }

    createMessageMeta(rawText, timestamp) {
        const meta = document.createElement('div');
        meta.className = 'message-meta';

        const time = this.formatTime(timestamp);
        if (time) {
            const timeEl = document.createElement('span');
            timeEl.textContent = time;
            meta.appendChild(timeEl);
        }

        const copyBtn = document.createElement('button');
        copyBtn.type = 'button';
        copyBtn.className = 'copy-btn';
        copyBtn.setAttribute('aria-label', 'Copy answer');
        copyBtn.innerHTML = '<svg viewBox="0 0 24 24" width="13" height="13" stroke="currentColor" stroke-width="2" fill="none" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy';
        copyBtn.addEventListener('click', async () => {
            const copied = await this.copyToClipboard(rawText);
            if (copied) {
                this.showToast('Answer copied to clipboard', 'success', 2000);
            } else {
                this.showToast('Could not copy the answer', 'error');
            }
        });
        meta.appendChild(copyBtn);

        return meta;
    }

    createSourcesDrawer(sources) {
        const panel = document.createElement('div');
        // Collapsed by default; long transcripts stay scannable
        panel.className = 'sources-panel';

        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'sources-toggle';
        toggle.setAttribute('aria-expanded', 'false');
        toggle.innerHTML = `
            <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none" aria-hidden="true"><polyline points="9 18 15 12 9 6"></polyline></svg>
            Sources (${sources.length})
        `;

        const content = document.createElement('div');
        content.className = 'sources-content';

        sources.forEach((source, idx) => {
            const item = document.createElement('div');
            item.className = 'source-item';

            const similarityScore = source.similarity ? ` · ${this.escapeHtml(String(source.similarity))}% match` : '';
            const safeId = this.escapeHtml(source.id || `Source ${idx + 1}`);
            const safeText = this.escapeHtml(source.text || '');
            item.innerHTML = `
                <h4>${safeId}${similarityScore}</h4>
                <p>${safeText}</p>
            `;
            content.appendChild(item);
        });

        toggle.addEventListener('click', () => {
            const expanded = panel.classList.toggle('expanded');
            toggle.setAttribute('aria-expanded', String(expanded));
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
