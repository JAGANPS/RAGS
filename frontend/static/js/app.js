/**
 * Multi-RAG System - Frontend JavaScript
 * Handles UI interactions, API calls, and real-time updates
 */

// ============== State Management ==============
const state = {
    selectedRAG: 'auto',
    selectedCategory: 'all',
    isProcessing: false,
    messages: [],
    latencyHistory: {
        standard: [],
        vectorless: [],
        agentic: []
    },
    currentSources: [],
    compareMode: false
};

// ============== DOM Elements ==============
const elements = {
    queryInput: null,
    sendBtn: null,
    messagesContainer: null,
    ragOptions: null,
    categoryBtns: null,
    latencyBars: null,
    sourcesList: null,
    uploadModal: null,
    compareBtn: null,
    statsBtn: null
};

// ============== Initialization ==============
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    setupEventListeners();
    loadInitialStats();
    addWelcomeMessage();
});

function initializeElements() {
    elements.queryInput = document.getElementById('queryInput');
    elements.sendBtn = document.getElementById('sendBtn');
    elements.messagesContainer = document.getElementById('messagesContainer');
    elements.ragOptions = document.querySelectorAll('.rag-option');
    elements.categoryBtns = document.querySelectorAll('.category-btn');
    elements.sourcesList = document.getElementById('sourcesList');
    elements.uploadModal = document.getElementById('uploadModal');
    elements.compareBtn = document.getElementById('compareBtn');
}

function setupEventListeners() {
    // Query input
    elements.queryInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleQuery();
        }
    });

    elements.queryInput?.addEventListener('input', autoResizeTextarea);

    // Send button
    elements.sendBtn?.addEventListener('click', handleQuery);

    // RAG selector
    elements.ragOptions?.forEach(option => {
        option.addEventListener('click', () => selectRAG(option.dataset.rag));
    });

    // Category filter
    elements.categoryBtns?.forEach(btn => {
        btn.addEventListener('click', () => selectCategory(btn.dataset.category));
    });

    // Compare button
    elements.compareBtn?.addEventListener('click', toggleCompareMode);

    // Upload modal
    document.getElementById('uploadBtn')?.addEventListener('click', openUploadModal);
    document.getElementById('closeUploadModal')?.addEventListener('click', closeUploadModal);

    // Drag and drop
    const uploadZone = document.getElementById('uploadZone');
    if (uploadZone) {
        uploadZone.addEventListener('dragover', handleDragOver);
        uploadZone.addEventListener('dragleave', handleDragLeave);
        uploadZone.addEventListener('drop', handleDrop);
        uploadZone.addEventListener('click', () => document.getElementById('fileInput')?.click());
    }

    document.getElementById('fileInput')?.addEventListener('change', handleFileSelect);
}

// ============== RAG Selection ==============
function selectRAG(ragType) {
    state.selectedRAG = ragType;
    elements.ragOptions?.forEach(option => {
        option.classList.toggle('active', option.dataset.rag === ragType);
    });
    // Reset latency display when changing RAG type
    resetLatencyDisplay();
}

function resetLatencyDisplay() {
    ['standard', 'vectorless', 'agentic'].forEach(type => {
        const bar = document.querySelector(`.latency-bar.${type}`);
        if (bar) bar.style.height = '20px';
        const value = document.getElementById(`latency-${type}`);
        if (value) value.textContent = '--';
    });
}

function selectCategory(category) {
    state.selectedCategory = category;
    elements.categoryBtns?.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.category === category);
    });
}

// ============== Query Handling ==============
async function handleQuery() {
    const query = elements.queryInput?.value.trim();
    if (!query || state.isProcessing) return;

    state.isProcessing = true;
    updateSendButton(true);

    // Add user message
    addMessage('user', query);
    elements.queryInput.value = '';
    autoResizeTextarea({ target: elements.queryInput });

    // Show processing indicator
    const loadingId = showProcessingIndicator();

    try {
        if (state.compareMode) {
            await handleCompareQuery(query, loadingId);
        } else {
            await handleSingleQuery(query, loadingId);
        }
    } catch (error) {
        removeProcessingIndicator(loadingId);
        addMessage('assistant', `Error: ${error.message}`, { error: true });
    } finally {
        state.isProcessing = false;
        updateSendButton(false);
    }
}

async function handleSingleQuery(query, loadingId) {
    const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: query,
            rag_type: state.selectedRAG,
            category: state.selectedCategory,
            top_k: 5,
            enable_reflection: true
        })
    });

    removeProcessingIndicator(loadingId);

    if (!response.ok) {
        throw new Error('Query failed');
    }

    const data = await response.json();

    // Add assistant message
    addMessage('assistant', data.answer, {
        ragType: data.rag_type,
        latency: data.latency,
        sources: data.sources
    });

    // Update metrics
    updateLatencyDisplay(data.rag_type, data.latency);
    updateSourcesList(data.sources);

    // Store in history
    state.latencyHistory[data.rag_type]?.push(data.latency.total_ms);
}

async function handleCompareQuery(query, loadingId) {
    const response = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: query,
            category: state.selectedCategory,
            top_k: 5
        })
    });

    removeProcessingIndicator(loadingId);

    if (!response.ok) {
        throw new Error('Compare failed');
    }

    const data = await response.json();

    // Add comparison message
    addCompareMessage(data);
}

// ============== Message Display ==============
function addMessage(type, content, metadata = {}) {
    const message = document.createElement('div');
    message.className = `message ${type}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = type === 'user' ? 'U' : 'AI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.innerHTML = formatMessageContent(content);

    contentDiv.appendChild(textDiv);

    if (type === 'assistant' && !metadata.error) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';

        if (metadata.ragType) {
            const ragBadge = document.createElement('span');
            ragBadge.className = 'rag-badge';
            ragBadge.textContent = formatRAGType(metadata.ragType);
            metaDiv.appendChild(ragBadge);
        }

        if (metadata.latency) {
            const latencySpan = document.createElement('span');
            latencySpan.textContent = `${metadata.latency.total_ms.toFixed(0)}ms`;
            metaDiv.appendChild(latencySpan);
        }

        if (metadata.sources?.length > 0) {
            const sourcesSpan = document.createElement('span');
            sourcesSpan.textContent = `${metadata.sources.length} sources`;
            metaDiv.appendChild(sourcesSpan);
        }

        contentDiv.appendChild(metaDiv);
    }

    message.appendChild(avatar);
    message.appendChild(contentDiv);

    elements.messagesContainer?.appendChild(message);
    scrollToBottom();

    state.messages.push({ type, content, metadata, timestamp: new Date() });
}

function addCompareMessage(data) {
    const message = document.createElement('div');
    message.className = 'message assistant compare-message';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'AI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.style.maxWidth = '90%';

    // Header
    const header = document.createElement('div');
    header.innerHTML = `<strong>Comparison Results</strong><br><small>${data.recommendation}</small>`;
    contentDiv.appendChild(header);

    // Results grid
    const grid = document.createElement('div');
    grid.className = 'compare-results';

    Object.entries(data.results).forEach(([name, result]) => {
        const card = document.createElement('div');
        card.className = 'compare-card';
        if (data.recommendation.toLowerCase().includes(name)) {
            card.classList.add('recommended');
        }

        card.innerHTML = `
            <div class="compare-card-header">
                <span class="compare-card-title">${formatRAGType(name)}</span>
                ${card.classList.contains('recommended') ? '<span class="compare-badge">Best</span>' : ''}
            </div>
            <div class="stat-card">
                <div class="stat-value">${result.latency.total_ms.toFixed(0)}ms</div>
                <div class="stat-label">Latency</div>
            </div>
            <p style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.75rem; max-height: 80px; overflow: hidden;">
                ${result.answer.substring(0, 150)}...
            </p>
        `;

        grid.appendChild(card);
    });

    contentDiv.appendChild(grid);

    message.appendChild(avatar);
    message.appendChild(contentDiv);

    elements.messagesContainer?.appendChild(message);
    scrollToBottom();
}

function formatMessageContent(content) {
    // Simple markdown-like formatting
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

function formatRAGType(type) {
    const types = {
        'standard': 'Standard RAG',
        'standard_rag': 'Standard RAG',
        'vectorless': 'Vectorless RAG',
        'vectorless_rag': 'Vectorless RAG',
        'agentic': 'Agentic RAG',
        'agentic_rag': 'Agentic RAG',
        'auto': 'Auto'
    };
    return types[type] || type;
}

// ============== Processing Indicators ==============
function showProcessingIndicator() {
    const id = Date.now();
    const indicator = document.createElement('div');
    indicator.id = `processing-${id}`;
    indicator.className = 'message assistant';
    indicator.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="loader-orbit">
                <div class="center"></div>
                <div class="orbit"></div>
                <div class="orbit"></div>
                <div class="orbit"></div>
            </div>
            <div class="processing-steps">
                <div class="processing-step active" id="step-routing-${id}">
                    <div class="step-icon">1</div>
                    <span class="step-text">Analyzing query...</span>
                </div>
                <div class="processing-step" id="step-retrieval-${id}">
                    <div class="step-icon">2</div>
                    <span class="step-text">Retrieving documents...</span>
                </div>
                <div class="processing-step" id="step-generation-${id}">
                    <div class="step-icon">3</div>
                    <span class="step-text">Generating response...</span>
                </div>
            </div>
        </div>
    `;

    elements.messagesContainer?.appendChild(indicator);
    scrollToBottom();

    // Animate steps
    setTimeout(() => {
        document.getElementById(`step-routing-${id}`)?.classList.add('completed');
        document.getElementById(`step-routing-${id}`)?.classList.remove('active');
        document.getElementById(`step-retrieval-${id}`)?.classList.add('active');
    }, 800);

    setTimeout(() => {
        document.getElementById(`step-retrieval-${id}`)?.classList.add('completed');
        document.getElementById(`step-retrieval-${id}`)?.classList.remove('active');
        document.getElementById(`step-generation-${id}`)?.classList.add('active');
    }, 1600);

    return id;
}

function removeProcessingIndicator(id) {
    document.getElementById(`processing-${id}`)?.remove();
}

// ============== Metrics Display ==============
function updateLatencyDisplay(ragType, latency) {
    const normalizedType = ragType.replace('_rag', '');

    // Update bar height
    const maxLatency = 5000; // 5 seconds max for visualization
    const barHeight = Math.min((latency.total_ms / maxLatency) * 150, 150);

    const bar = document.querySelector(`.latency-bar.${normalizedType}`);
    if (bar) {
        bar.style.height = `${barHeight}px`;
    }

    // Update value display
    const valueEl = document.getElementById(`latency-${normalizedType}`);
    if (valueEl) {
        valueEl.textContent = `${latency.total_ms.toFixed(0)}ms`;
    }
}

function updateSourcesList(sources) {
    if (!elements.sourcesList) return;

    state.currentSources = sources;
    elements.sourcesList.innerHTML = '';

    sources.forEach((source, index) => {
        const item = document.createElement('div');
        item.className = 'source-item';
        item.innerHTML = `
            <div class="source-header">
                <span class="source-file">${source.source_file}</span>
                <span class="source-score">${(source.score * 100).toFixed(1)}%</span>
            </div>
            <p class="source-text">${source.text}</p>
            <span class="source-category">${source.category}</span>
        `;
        elements.sourcesList.appendChild(item);
    });
}

// ============== File Upload ==============
function openUploadModal() {
    elements.uploadModal?.classList.add('active');
}

function closeUploadModal() {
    elements.uploadModal?.classList.remove('active');
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFiles(files);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        uploadFiles(files);
    }
}

async function uploadFiles(files) {
    const category = document.getElementById('uploadCategory')?.value || 'sam';
    const formData = new FormData();

    Array.from(files).forEach(file => {
        formData.append('files', file);
    });
    formData.append('category', category);

    const uploadStatus = document.getElementById('uploadStatus');
    if (uploadStatus) {
        uploadStatus.innerHTML = `
            <div class="loader-dna">
                <div class="strand"></div>
                <div class="strand"></div>
                <div class="strand"></div>
                <div class="strand"></div>
                <div class="strand"></div>
            </div>
            <p>Uploading ${files.length} file(s)...</p>
        `;
    }

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (uploadStatus) {
            if (data.success) {
                uploadStatus.innerHTML = `
                    <p style="color: var(--secondary);">Successfully uploaded ${data.documents_processed} document(s)!</p>
                `;
            } else {
                uploadStatus.innerHTML = `
                    <p style="color: var(--danger);">Upload completed with errors:</p>
                    <ul>${data.errors.map(e => `<li>${e}</li>`).join('')}</ul>
                `;
            }
        }

        setTimeout(closeUploadModal, 2000);

    } catch (error) {
        if (uploadStatus) {
            uploadStatus.innerHTML = `<p style="color: var(--danger);">Upload failed: ${error.message}</p>`;
        }
    }
}

// ============== Compare Mode ==============
function toggleCompareMode() {
    state.compareMode = !state.compareMode;
    elements.compareBtn?.classList.toggle('active', state.compareMode);

    const label = elements.compareBtn?.querySelector('span');
    if (label) {
        label.textContent = state.compareMode ? 'Compare: ON' : 'Compare';
    }
}

// ============== Stats Loading ==============
async function loadInitialStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        // Update stats display
        const docsCount = document.getElementById('docsCount');
        if (docsCount) {
            docsCount.textContent = data.total_documents;
        }

    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// ============== Utility Functions ==============
function autoResizeTextarea(e) {
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

function updateSendButton(isLoading) {
    if (elements.sendBtn) {
        elements.sendBtn.disabled = isLoading;
        elements.sendBtn.innerHTML = isLoading
            ? '<div class="typing-indicator"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>'
            : '&#10148;';
    }
}

function scrollToBottom() {
    if (elements.messagesContainer) {
        elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
    }
}

function addWelcomeMessage() {
    addMessage('assistant', `Welcome to the Multi-RAG System!

I can help you query information about **Software Asset Management (SAM)**, **IT Asset Management (ITAM)**, **IT Value (ITV)**, and **Site Reliability Engineering (SRE)**.

Choose a RAG type on the left:
- **Standard RAG**: Fast vector-based semantic search
- **Vectorless RAG**: Reasoning-based hierarchical navigation
- **Agentic RAG**: Intelligent multi-step query processing

Or leave it on **Auto** and I'll choose the best approach for your query.

Upload your documents using the upload button, then ask me anything!`);
}

// ============== Keyboard Shortcuts ==============
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to send
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        handleQuery();
    }

    // Escape to close modals
    if (e.key === 'Escape') {
        closeUploadModal();
    }
});
