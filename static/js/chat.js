// RAG Chatbot — Gemini-style Chat Interface

class ChatBot {
    constructor() {
        this.chatView = document.getElementById('chatView');
        this.welcomeView = document.getElementById('welcomeView');
        this.chatMessages = document.getElementById('chatMessages');
        this.chatForm = document.getElementById('chatForm');
        this.questionInput = document.getElementById('questionInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.statsBtn = document.getElementById('statsBtn');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.charCount = document.getElementById('charCount');
        this.topKSelect = document.getElementById('topK');
        this.statsModal = document.getElementById('statsModal');
        this.suggestionChips = document.getElementById('suggestionChips');

        // Dashboard
        this.dashboardBtn = document.getElementById('dashboardBtn');
        this.dashboardPanel = document.getElementById('dashboardPanel');
        this.dashboardClose = document.getElementById('dashboardClose');
        this.dashboardOverlay = document.getElementById('dashboardOverlay');
        this.dashboardContent = document.getElementById('dashboardContent');

        this.isLoading = false;
        this.hasMessages = false;

        this.initEventListeners();
        this.autoResizeTextarea();
        this.loadChatHistory();
    }

    initEventListeners() {
        // Form submission
        if (this.chatForm) {
            this.chatForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.sendMessage();
            });
        }

        // Input validation and auto-resize
        if (this.questionInput) {
            this.questionInput.addEventListener('input', () => {
                const value = this.questionInput.value.trim();
                this.sendBtn.disabled = value.length === 0 || this.isLoading;
                if (this.charCount) {
                    this.charCount.textContent = this.questionInput.value.length;
                }
                this.autoResizeTextarea();
            });

            // Enter key to send (Shift+Enter for new line)
            this.questionInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (!this.sendBtn.disabled && !this.isLoading) {
                        this.sendMessage();
                    }
                }
            });
        }

        // Clear chat (new chat)
        if (this.clearBtn) {
            this.clearBtn.addEventListener('click', () => {
                this.clearChat();
            });
        }

        // Stats modal
        if (this.statsBtn) {
            this.statsBtn.addEventListener('click', () => {
                this.showStats();
            });
        }

        // Close modal
        if (this.statsModal) {
            const modalClose = this.statsModal.querySelector('.modal-close');
            if (modalClose) {
                modalClose.addEventListener('click', () => {
                    this.statsModal.classList.remove('active');
                });
            }
            this.statsModal.addEventListener('click', (e) => {
                if (e.target === this.statsModal) {
                    this.statsModal.classList.remove('active');
                }
            });
        }

        // Suggestion chips & example buttons
        document.addEventListener('click', (e) => {
            const chip = e.target.closest('.chip') || e.target.closest('.example-btn');
            if (chip) {
                const question = chip.dataset.question;
                if (question && this.questionInput) {
                    this.questionInput.value = question;
                    this.sendBtn.disabled = false;
                    if (this.charCount) this.charCount.textContent = question.length;
                    this.autoResizeTextarea();
                    this.sendMessage();
                }
            }
        });

        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (this.statsModal && this.statsModal.classList.contains('active')) {
                    this.statsModal.classList.remove('active');
                }
                this.closeDashboard();
            }
        });

        // Dashboard panel
        if (this.dashboardBtn) {
            this.dashboardBtn.addEventListener('click', () => this.showDashboard());
        }
        if (this.dashboardClose) {
            this.dashboardClose.addEventListener('click', () => this.closeDashboard());
        }
        if (this.dashboardOverlay) {
            this.dashboardOverlay.addEventListener('click', () => this.closeDashboard());
        }
    }

    autoResizeTextarea() {
        if (this.questionInput) {
            this.questionInput.style.height = 'auto';
            this.questionInput.style.height = Math.min(this.questionInput.scrollHeight, 120) + 'px';
        }
    }

    switchToChatView() {
        if (!this.hasMessages) {
            this.hasMessages = true;
            if (this.welcomeView) this.welcomeView.style.display = 'none';
            if (this.chatView) this.chatView.style.display = 'flex';
            if (this.suggestionChips) this.suggestionChips.style.display = 'none';
        }
    }

    switchToWelcomeView() {
        this.hasMessages = false;
        if (this.welcomeView) this.welcomeView.style.display = 'flex';
        if (this.chatView) this.chatView.style.display = 'none';
        if (this.suggestionChips) this.suggestionChips.style.display = 'flex';
    }

    async loadChatHistory() {
        try {
            const response = await fetch('/api/history', {
                headers: { 'Authorization': 'Bearer super-secret-student-key' }
            });
            const data = await response.json();

            if (data.success && data.history && data.history.length > 0) {
                this.switchToChatView();
                data.history.forEach(item => {
                    this.addMessage(item.question, 'user', null, false);
                    this.addMessage(item.answer, 'bot', null, false);
                });
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    async sendMessage() {
        const question = this.questionInput.value.trim();
        if (!question || this.isLoading) return;

        const topK = this.topKSelect ? parseInt(this.topKSelect.value) : 3;

        this.isLoading = true;
        this.sendBtn.disabled = true;

        // Switch to chat view
        this.switchToChatView();

        // Add user message
        this.addMessage(question, 'user');

        // Clear input
        this.questionInput.value = '';
        if (this.charCount) this.charCount.textContent = '0';
        this.autoResizeTextarea();

        // Show typing
        this.showTyping();

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer super-secret-student-key'
                },
                body: JSON.stringify({ question, top_k: topK })
            });

            const data = await response.json();
            this.hideTyping();

            if (data.success) {
                this.addMessage(data.answer, 'bot', data.sources);
            } else {
                const errMsg = data.error || data.detail || 'Unknown error';
                this.addMessage(`Sorry, I encountered an error: ${errMsg}`, 'bot');
            }
        } catch (error) {
            this.hideTyping();
            this.addMessage('Sorry, there was a network error. Please check if the server is running.', 'bot');
            console.error('Error:', error);
        } finally {
            this.isLoading = false;
            this.sendBtn.disabled = this.questionInput.value.trim().length === 0;
        }
    }

    addMessage(text, type, sources = null, animate = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        if (animate) messageDiv.style.animation = 'fadeIn 0.3s ease';

        // Avatar
        const avatarDiv = document.createElement('div');
        avatarDiv.className = `message-avatar ${type}-avatar`;

        if (type === 'bot') {
            avatarDiv.innerHTML = '<span class="sparkle-icon small">✦</span>';
        } else {
            avatarDiv.textContent = 'A';
        }

        // Content
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';

        const textParagraph = document.createElement('p');
        textParagraph.textContent = text;
        textDiv.appendChild(textParagraph);

        // Sources
        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources';

            const sourcesHeader = document.createElement('div');
            sourcesHeader.className = 'sources-header';
            sourcesHeader.innerHTML = `
                <span class="sources-title">Sources (${sources.length})</span>
                <button class="sources-toggle" onclick="this.parentElement.parentElement.classList.toggle('collapsed')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="6 9 12 15 18 9"></polyline>
                    </svg>
                </button>
            `;
            sourcesDiv.appendChild(sourcesHeader);

            const sourcesList = document.createElement('div');
            sourcesList.className = 'sources-list';

            sources.forEach((source, index) => {
                const sourceItem = document.createElement('div');
                sourceItem.className = 'source-item';
                const similarityClass = source.similarity >= 80 ? 'high' : source.similarity >= 60 ? 'medium' : 'low';
                sourceItem.innerHTML = `
                    <div class="source-header">
                        <span class="source-id">Source ${index + 1}</span>
                        <span class="source-similarity ${similarityClass}">${source.similarity}% match</span>
                    </div>
                    <div class="source-text">${this.escapeHtml(source.text)}</div>
                `;
                sourcesList.appendChild(sourceItem);
            });

            sourcesDiv.appendChild(sourcesList);
            textDiv.appendChild(sourcesDiv);
        }

        contentDiv.appendChild(textDiv);
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);

        if (this.chatMessages) {
            this.chatMessages.appendChild(messageDiv);
        }

        this.scrollToBottom();
    }

    showTyping() {
        if (this.typingIndicator) {
            this.typingIndicator.style.display = 'flex';
            this.scrollToBottom();
        }
    }

    hideTyping() {
        if (this.typingIndicator) {
            this.typingIndicator.style.display = 'none';
        }
    }

    scrollToBottom() {
        setTimeout(() => {
            if (this.chatView) {
                this.chatView.scrollTop = this.chatView.scrollHeight;
            }
        }, 50);
    }

    async clearChat() {
        try {
            const response = await fetch('/api/clear', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer super-secret-student-key'
                }
            });

            const data = await response.json();
            if (data.success) {
                if (this.chatMessages) this.chatMessages.innerHTML = '';
                this.switchToWelcomeView();
            }
        } catch (error) {
            console.error('Error clearing chat:', error);
        }
    }

    async showStats() {
        if (!this.statsModal) return;

        this.statsModal.classList.add('active');
        const statsContent = document.getElementById('statsContent');
        if (statsContent) {
            statsContent.innerHTML = '<div class="loading">Loading statistics...</div>';
        }

        try {
            const response = await fetch('/api/stats', {
                headers: { 'Authorization': 'Bearer super-secret-student-key' }
            });
            const data = await response.json();

            if (data.success && statsContent) {
                const stats = data.stats;
                statsContent.innerHTML = `
                    <div class="stat-item">
                        <span class="stat-label">Documents Indexed</span>
                        <span class="stat-value">${stats.total_documents ? stats.total_documents.toLocaleString() : 'N/A'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Embedding Model</span>
                        <span class="stat-value">${stats.embedding_model || 'N/A'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">LLM Model</span>
                        <span class="stat-value">${stats.llm_model || 'N/A'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Collection</span>
                        <span class="stat-value">${stats.collection_name || 'N/A'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Total Sessions</span>
                        <span class="stat-value">${stats.total_sessions || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Total Queries</span>
                        <span class="stat-value">${stats.total_queries || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Avg Response Time</span>
                        <span class="stat-value">${stats.avg_response_time_ms ? stats.avg_response_time_ms + 'ms' : 'N/A'}</span>
                    </div>
                `;
            } else if (statsContent) {
                statsContent.innerHTML = `<div class="loading">Failed to load: ${data.error || 'Unknown error'}</div>`;
            }
        } catch (error) {
            console.error('Error loading stats:', error);
            const statsContent = document.getElementById('statsContent');
            if (statsContent) {
                statsContent.innerHTML = '<div class="loading">Error loading statistics.</div>';
            }
        }
    }

    closeDashboard() {
        if (this.dashboardPanel) this.dashboardPanel.classList.remove('active');
        if (this.dashboardOverlay) this.dashboardOverlay.classList.remove('active');
    }

    async showDashboard() {
        if (!this.dashboardPanel) return;

        this.dashboardPanel.classList.add('active');
        if (this.dashboardOverlay) this.dashboardOverlay.classList.add('active');

        if (this.dashboardContent) {
            this.dashboardContent.innerHTML = '<div class="loading">Loading vitals...</div>';
        }

        try {
            const response = await fetch('/api/stats', {
                headers: { 'Authorization': 'Bearer super-secret-student-key' }
            });
            const data = await response.json();

            if (data.success && this.dashboardContent) {
                const s = data.stats;
                this.dashboardContent.innerHTML = `
                    <div class="vitals-section-title">Status</div>
                    <div class="vitals-grid">
                        <div class="vital-card full-width">
                            <div class="vital-label">Server</div>
                            <div class="status-row">
                                <span class="status-dot"></span>
                                <span class="vital-value green small">Online — ${s.uptime_minutes || 0} min uptime</span>
                            </div>
                        </div>
                    </div>

                    <div class="vitals-section-title">Data</div>
                    <div class="vitals-grid">
                        <div class="vital-card">
                            <div class="vital-icon">📄</div>
                            <div class="vital-label">Documents</div>
                            <div class="vital-value accent">${s.total_documents ? s.total_documents.toLocaleString() : 'N/A'}</div>
                        </div>
                        <div class="vital-card">
                            <div class="vital-icon">📦</div>
                            <div class="vital-label">Collection</div>
                            <div class="vital-value small">${s.collection_name || 'N/A'}</div>
                        </div>
                        <div class="vital-card full-width">
                            <div class="vital-icon">🗃️</div>
                            <div class="vital-label">Dataset</div>
                            <div class="vital-value small">${s.dataset_name || 'N/A'}</div>
                        </div>
                    </div>

                    <div class="vitals-section-title">Models</div>
                    <div class="vitals-grid">
                        <div class="vital-card">
                            <div class="vital-icon">🧠</div>
                            <div class="vital-label">LLM Model</div>
                            <div class="vital-value small pink">${s.llm_model || 'N/A'}</div>
                        </div>
                        <div class="vital-card">
                            <div class="vital-icon">🔗</div>
                            <div class="vital-label">Embeddings</div>
                            <div class="vital-value small">${s.embedding_model || 'N/A'}</div>
                        </div>
                    </div>

                    <div class="vitals-section-title">Performance</div>
                    <div class="vitals-grid">
                        <div class="vital-card">
                            <div class="vital-icon">💬</div>
                            <div class="vital-label">Total Queries</div>
                            <div class="vital-value">${s.total_queries || 0}</div>
                        </div>
                        <div class="vital-card">
                            <div class="vital-icon">👥</div>
                            <div class="vital-label">Sessions</div>
                            <div class="vital-value">${s.total_sessions || 0}</div>
                        </div>
                        <div class="vital-card">
                            <div class="vital-icon">⚡</div>
                            <div class="vital-label">Avg Latency</div>
                            <div class="vital-value amber">${s.avg_response_time_ms ? s.avg_response_time_ms + 'ms' : '—'}</div>
                        </div>
                        <div class="vital-card">
                            <div class="vital-icon">🏁</div>
                            <div class="vital-label">Last Latency</div>
                            <div class="vital-value amber">${s.last_response_time_ms ? s.last_response_time_ms + 'ms' : '—'}</div>
                        </div>
                    </div>
                `;
            } else if (this.dashboardContent) {
                this.dashboardContent.innerHTML = `<div class="loading">Error: ${data.error || 'Failed to load'}</div>`;
            }
        } catch (error) {
            console.error('Error loading dashboard:', error);
            if (this.dashboardContent) {
                this.dashboardContent.innerHTML = '<div class="loading">Error loading vitals. Is the server running?</div>';
            }
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new ChatBot();
});
