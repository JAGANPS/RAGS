/**
 * Tabs functionality, Graph visualization, and API documentation
 */

// ============== Tab Navigation ==============
class TabManager {
    constructor() {
        this.tabs = document.querySelectorAll('.tab-btn');
        this.contents = document.querySelectorAll('.tab-content');
        this.init();
    }

    init() {
        this.tabs.forEach(tab => {
            tab.addEventListener('click', () => this.switchTab(tab.dataset.tab));
        });
    }

    switchTab(tabId) {
        // Update tab buttons
        this.tabs.forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabId);
        });

        // Update tab contents - match "tab-{tabId}" pattern
        const contentId = `tab-${tabId}`;
        this.contents.forEach(content => {
            content.classList.toggle('active', content.id === contentId);
        });

        // Trigger tab-specific initialization
        this.onTabSwitch(tabId);
    }

    onTabSwitch(tabId) {
        switch(tabId) {
            case 'graph-view':
                graphManager.initialize();
                break;
            case 'vectorless-rag':
                treeManager.loadTree();
                break;
            case 'standard-rag':
                statsManager.loadStats('standard');
                break;
            case 'agentic-rag':
                workflowManager.initialize();
                break;
        }
    }
}

// ============== Graph Visualization ==============
class GraphManager {
    constructor() {
        this.network = null;
        this.nodes = null;
        this.edges = null;
        this.container = null;
        this.initialized = false;
    }

    async initialize() {
        this.container = document.getElementById('knowledge-graph');
        if (!this.container) return;

        // Clear existing network for refresh
        if (this.network) {
            this.network.destroy();
            this.network = null;
        }
        this.initialized = false;

        // Show loading state
        this.container.innerHTML = '<div class="loading-graph">Loading graph data...</div>';

        try {
            const response = await fetch('/api/graph');
            const data = await response.json();

            if (data.nodes && data.nodes.length > 0) {
                this.renderGraph(data);
            } else {
                this.container.innerHTML = `
                    <div class="empty-graph">
                        <span class="icon">📊</span>
                        <p>No documents indexed yet</p>
                        <p class="hint">Upload documents to see the knowledge graph</p>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Failed to load graph:', error);
            this.container.innerHTML = `
                <div class="error-graph">
                    <span class="icon">⚠️</span>
                    <p>Failed to load graph data</p>
                    <button class="btn btn-secondary" onclick="graphManager.initialize()">Retry</button>
                </div>
            `;
        }
    }

    renderGraph(data) {
        // Create vis.js datasets
        this.nodes = new vis.DataSet(data.nodes.map(node => ({
            id: node.id,
            label: this.truncateLabel(node.label),
            title: node.title || node.label,
            group: node.type,
            size: this.getNodeSize(node.type),
            font: { color: '#e2e8f0', size: 12 }
        })));

        this.edges = new vis.DataSet(data.edges.map(edge => ({
            from: edge.from,
            to: edge.to,
            label: edge.label || '',
            arrows: 'to',
            color: { color: 'rgba(99, 102, 241, 0.5)', highlight: '#6366f1' },
            font: { color: '#94a3b8', size: 10 }
        })));

        // Graph options
        const options = {
            nodes: {
                shape: 'dot',
                borderWidth: 2,
                shadow: true
            },
            edges: {
                width: 1,
                smooth: {
                    type: 'cubicBezier',
                    forceDirection: 'horizontal'
                }
            },
            groups: {
                document: { color: { background: '#6366f1', border: '#818cf8' } },
                section: { color: { background: '#10b981', border: '#34d399' } },
                chunk: { color: { background: '#f59e0b', border: '#fbbf24' } },
                query: { color: { background: '#ef4444', border: '#f87171' } },
                category: { color: { background: '#8b5cf6', border: '#a78bfa' } }
            },
            physics: {
                enabled: true,
                barnesHut: {
                    gravitationalConstant: -2000,
                    centralGravity: 0.3,
                    springLength: 95,
                    springConstant: 0.04
                },
                stabilization: { iterations: 150 }
            },
            interaction: {
                hover: true,
                tooltipDelay: 200,
                hideEdgesOnDrag: true
            }
        };

        // Create network
        this.network = new vis.Network(this.container, {
            nodes: this.nodes,
            edges: this.edges
        }, options);

        // Event handlers
        this.network.on('click', (params) => this.onNodeClick(params));
        this.network.on('hoverNode', (params) => this.onNodeHover(params));

        this.initialized = true;
    }

    truncateLabel(label) {
        return label.length > 20 ? label.substring(0, 17) + '...' : label;
    }

    getNodeSize(type) {
        const sizes = { document: 25, category: 20, section: 15, chunk: 10, query: 20 };
        return sizes[type] || 15;
    }

    onNodeClick(params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            const node = this.nodes.get(nodeId);
            this.showNodeInfo(node, params.pointer.DOM);
        } else {
            this.hideNodeInfo();
        }
    }

    onNodeHover(params) {
        document.body.style.cursor = 'pointer';
    }

    showNodeInfo(node, position) {
        const infoPanel = document.getElementById('node-info');
        if (!infoPanel) return;

        infoPanel.innerHTML = `
            <h4>${node.title || node.label}</h4>
            <p><strong>Type:</strong> ${node.group}</p>
            <p><strong>ID:</strong> ${node.id}</p>
        `;
        infoPanel.style.left = `${position.x + 10}px`;
        infoPanel.style.top = `${position.y + 10}px`;
        infoPanel.classList.add('visible');
    }

    hideNodeInfo() {
        const infoPanel = document.getElementById('node-info');
        if (infoPanel) {
            infoPanel.classList.remove('visible');
        }
    }

    // Graph controls
    zoomIn() {
        if (this.network) {
            const scale = this.network.getScale() * 1.2;
            this.network.moveTo({ scale });
        }
    }

    zoomOut() {
        if (this.network) {
            const scale = this.network.getScale() / 1.2;
            this.network.moveTo({ scale });
        }
    }

    fitGraph() {
        if (this.network) {
            this.network.fit({ animation: true });
        }
    }

    resetLayout() {
        if (this.network) {
            this.network.stabilize(100);
        }
    }
}

// ============== Tree View for Vectorless RAG ==============
class TreeManager {
    constructor() {
        this.treeContainer = null;
        this.loaded = false;
    }

    async loadTree() {
        this.treeContainer = document.getElementById('vectorless-tree');
        if (!this.treeContainer || this.loaded) return;

        this.treeContainer.innerHTML = '<div class="loading-tree">Loading document tree...</div>';

        try {
            const response = await fetch('/api/tree');
            const data = await response.json();

            if (data.tree && data.tree.length > 0) {
                this.renderTree(data.tree);
                this.loaded = true;
            } else {
                this.treeContainer.innerHTML = `
                    <div class="empty-tree">
                        <p>No documents indexed yet</p>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Failed to load tree:', error);
            this.treeContainer.innerHTML = `
                <div class="error-tree">
                    <p>Failed to load document tree</p>
                </div>
            `;
        }
    }

    renderTree(nodes, parent = null) {
        const container = parent || this.treeContainer;
        container.innerHTML = '';

        nodes.forEach(node => {
            const nodeEl = this.createTreeNode(node);
            container.appendChild(nodeEl);
        });
    }

    createTreeNode(node) {
        const wrapper = document.createElement('div');
        wrapper.className = 'tree-node';

        const item = document.createElement('div');
        item.className = 'tree-item';
        item.innerHTML = `
            <span class="tree-icon">${this.getIcon(node.type)}</span>
            <span class="tree-label">${node.title}</span>
            ${node.children ? `<span class="tree-meta">${node.children.length} items</span>` : ''}
        `;

        item.addEventListener('click', () => {
            item.classList.toggle('selected');
            if (node.children) {
                wrapper.classList.toggle('expanded');
            }
        });

        wrapper.appendChild(item);

        if (node.children && node.children.length > 0) {
            const childContainer = document.createElement('div');
            childContainer.className = 'tree-children';
            node.children.forEach(child => {
                childContainer.appendChild(this.createTreeNode(child));
            });
            wrapper.appendChild(childContainer);
        }

        return wrapper;
    }

    getIcon(type) {
        const icons = {
            document: '📄',
            section: '📁',
            chunk: '📝',
            category: '🏷️'
        };
        return icons[type] || '📋';
    }
}

// ============== Stats Manager ==============
class StatsManager {
    async loadStats(ragType) {
        const container = document.getElementById(`${ragType}-stats`);
        if (!container) return;

        try {
            const response = await fetch('/api/stats');
            const data = await response.json();

            this.renderStats(container, data, ragType);
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }

    renderStats(container, data, ragType) {
        const stats = data[ragType] || data;

        container.innerHTML = `
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-value">${stats.documents || 0}</span>
                    <span class="stat-label">Documents</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${stats.chunks || 0}</span>
                    <span class="stat-label">Chunks</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${stats.queries || 0}</span>
                    <span class="stat-label">Queries</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${stats.avg_latency || '-'}ms</span>
                    <span class="stat-label">Avg Latency</span>
                </div>
            </div>
        `;
    }
}

// ============== Workflow Visualization ==============
class WorkflowManager {
    constructor() {
        this.steps = [];
        this.currentStep = -1;
    }

    initialize() {
        // Highlight the agent flow nodes on hover
        const nodes = document.querySelectorAll('.agent-node');
        nodes.forEach(node => {
            node.addEventListener('mouseenter', () => this.highlightNode(node));
            node.addEventListener('mouseleave', () => this.resetNodes());
        });
    }

    highlightNode(node) {
        const nodes = document.querySelectorAll('.agent-node');
        nodes.forEach(n => n.style.opacity = '0.5');
        node.style.opacity = '1';
    }

    resetNodes() {
        const nodes = document.querySelectorAll('.agent-node');
        nodes.forEach(n => n.style.opacity = '1');
    }

    // Simulate workflow for demo purposes
    async simulateWorkflow(query) {
        const steps = document.querySelectorAll('.workflow-step');

        for (let i = 0; i < steps.length; i++) {
            steps.forEach(s => s.classList.remove('active', 'completed'));

            for (let j = 0; j < i; j++) {
                steps[j].classList.add('completed');
            }
            steps[i].classList.add('active');

            await this.delay(800);
        }

        steps.forEach(s => {
            s.classList.remove('active');
            s.classList.add('completed');
        });
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// ============== API Documentation ==============
class ApiDocsManager {
    constructor() {
        this.init();
    }

    init() {
        // Toggle endpoint details
        document.querySelectorAll('.api-endpoint-header').forEach(header => {
            header.addEventListener('click', () => {
                const endpoint = header.closest('.api-endpoint');
                endpoint.classList.toggle('expanded');
            });
        });

        // Try-it buttons
        document.querySelectorAll('.try-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.tryEndpoint(e));
        });
    }

    async tryEndpoint(event) {
        const btn = event.target;
        const endpoint = btn.closest('.api-endpoint');
        const method = endpoint.querySelector('.api-method').textContent;
        const path = endpoint.querySelector('.api-path').textContent;

        btn.disabled = true;
        btn.textContent = 'Loading...';

        try {
            let response;
            if (method === 'GET') {
                response = await fetch(path);
            } else {
                // For POST endpoints, use sample data
                const sampleData = this.getSampleData(path);
                response = await fetch(path, {
                    method: method,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(sampleData)
                });
            }

            const data = await response.json();
            this.showResponse(endpoint, data);
        } catch (error) {
            this.showResponse(endpoint, { error: error.message });
        } finally {
            btn.disabled = false;
            btn.textContent = 'Try It';
        }
    }

    getSampleData(path) {
        const samples = {
            '/api/query': {
                query: 'What is software asset management?',
                rag_type: 'standard',
                top_k: 3
            },
            '/api/compare': {
                query: 'Explain ITAM best practices'
            }
        };
        return samples[path] || {};
    }

    showResponse(endpoint, data) {
        let responseEl = endpoint.querySelector('.api-response');
        if (!responseEl) {
            responseEl = document.createElement('div');
            responseEl.className = 'api-response';
            endpoint.querySelector('.api-endpoint-body').appendChild(responseEl);
        }

        responseEl.innerHTML = `
            <h5>Response</h5>
            <pre class="code-block">${JSON.stringify(data, null, 2)}</pre>
        `;
    }
}

// ============== Initialize Managers ==============
let tabManager, graphManager, treeManager, statsManager, workflowManager, apiDocsManager;

document.addEventListener('DOMContentLoaded', () => {
    tabManager = new TabManager();
    graphManager = new GraphManager();
    treeManager = new TreeManager();
    statsManager = new StatsManager();
    workflowManager = new WorkflowManager();
    apiDocsManager = new ApiDocsManager();

    // Export for global access (after initialization)
    window.graphManager = graphManager;
    window.tabManager = tabManager;

    console.log('Tab system initialized');
});

// ============== Global Helper Functions ==============

function toggleEndpoint(header) {
    const endpoint = header.closest('.api-endpoint');
    endpoint.classList.toggle('expanded');
}

async function tryEndpoint(path, method) {
    const modal = document.getElementById('apiResponseModal');
    const content = document.getElementById('apiResponseContent');

    modal.style.display = 'flex';
    content.textContent = 'Loading...';

    try {
        let response;
        if (method === 'GET') {
            response = await fetch(path);
        } else {
            // Sample data for POST endpoints
            const sampleData = {
                '/api/query': {
                    query: 'What is software asset management?',
                    rag_type: 'standard',
                    top_k: 3
                },
                '/api/compare': {
                    query: 'Explain ITAM lifecycle management'
                }
            };

            response = await fetch(path, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(sampleData[path] || {})
            });
        }

        const data = await response.json();
        content.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        content.textContent = JSON.stringify({ error: error.message }, null, 2);
    }
}

function closeApiModal() {
    document.getElementById('apiResponseModal').style.display = 'none';
}

// Make functions globally accessible
window.toggleEndpoint = toggleEndpoint;
window.tryEndpoint = tryEndpoint;
window.closeApiModal = closeApiModal;
