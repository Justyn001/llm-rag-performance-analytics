/**
 * RAG Performance Analytics — Frontend Logic
 */

const API = '';  // Same origin

// --- State ---
let selectedModel = '';
let selectedLang = 'en';
let topK = 5;
let queryCount = 0;
let models = [];

// --- DOM Elements ---
const modelSelect = document.getElementById('model-select');
const topKSlider = document.getElementById('top-k-slider');
const topKValue = document.getElementById('top-k-value');
const queryForm = document.getElementById('query-form');
const questionInput = document.getElementById('question-input');
const submitBtn = document.getElementById('submit-btn');
const chatContainer = document.getElementById('chat-container');
const chatWelcome = document.getElementById('chat-welcome');
const chatMessages = document.getElementById('chat-messages');
const modelInfo = document.getElementById('model-info');
const modelProvider = document.getElementById('model-provider');
const modelType = document.getElementById('model-type');
const modelDesc = document.getElementById('model-desc');
const queriesCount = document.getElementById('queries-count');

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    setupEventListeners();
});

async function loadModels() {
    try {
        const resp = await fetch(`${API}/api/models`);
        const data = await resp.json();
        models = data.models;

        modelSelect.innerHTML = '';
        if (models.length === 0) {
            modelSelect.innerHTML = '<option value="">No models available</option>';
            return;
        }

        // Group by type
        const local = models.filter(m => m.is_local);
        const cloud = models.filter(m => !m.is_local);

        if (local.length) {
            const group = document.createElement('optgroup');
            group.label = '🖥️ Local (GPU)';
            local.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.name;
                opt.textContent = m.name;
                group.appendChild(opt);
            });
            modelSelect.appendChild(group);
        }

        if (cloud.length) {
            const group = document.createElement('optgroup');
            group.label = '☁️ Cloud (API)';
            cloud.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.name;
                opt.textContent = m.name;
                group.appendChild(opt);
            });
            modelSelect.appendChild(group);
        }

        selectedModel = models[0].name;
        modelSelect.value = selectedModel;
        updateModelInfo(models[0]);
    } catch (err) {
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
        console.error('Failed to load models:', err);
    }
}

function setupEventListeners() {
    // Model select
    modelSelect.addEventListener('change', (e) => {
        selectedModel = e.target.value;
        const model = models.find(m => m.name === selectedModel);
        if (model) updateModelInfo(model);
    });

    // Language toggle
    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedLang = btn.dataset.lang;
        });
    });

    // Top K slider
    topKSlider.addEventListener('input', (e) => {
        topK = parseInt(e.target.value);
        topKValue.textContent = topK;
    });

    // Form submit
    queryForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const question = questionInput.value.trim();
        if (question && selectedModel) {
            sendQuery(question);
        }
    });

    // Example questions
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const q = btn.dataset.q;
            questionInput.value = q;
            if (selectedModel) sendQuery(q);
        });
    });
}

function updateModelInfo(model) {
    modelInfo.style.display = 'block';
    modelProvider.textContent = `Provider: ${model.provider}`;
    modelType.textContent = model.is_local ? '🖥️ Local (GPU)' : '☁️ Cloud (API)';
    modelDesc.textContent = model.description;
}

async function sendQuery(question) {
    // Hide welcome, show chat
    chatWelcome.style.display = 'none';

    // Add question bubble
    addMessage('question', question);
    questionInput.value = '';
    submitBtn.disabled = true;

    // Add loading bubble
    const loadingId = addLoadingMessage();

    try {
        const resp = await fetch(`${API}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                model_name: selectedModel,
                language: selectedLang,
                top_k: topK,
            }),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Query failed');
        }

        const data = await resp.json();

        // Remove loading, add answer
        removeMessage(loadingId);
        addAnswerMessage(data);

        queryCount++;
        queriesCount.textContent = queryCount;
    } catch (err) {
        removeMessage(loadingId);
        addMessage('answer', `❌ Error: ${err.message}`, true);
    }

    submitBtn.disabled = false;
    questionInput.focus();
}

function addMessage(type, content) {
    const div = document.createElement('div');
    div.className = `message message-${type}`;
    div.innerHTML = `<div class="bubble">${escapeHtml(content)}</div>`;
    chatMessages.appendChild(div);
    scrollToBottom();
    return div;
}

function addLoadingMessage() {
    const id = 'loading-' + Date.now();
    const div = document.createElement('div');
    div.className = 'message message-answer';
    div.id = id;
    div.innerHTML = `
        <div class="bubble">
            <div class="loading-dots"><span></span><span></span><span></span></div>
        </div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function addAnswerMessage(data) {
    const ctxId = 'ctx-' + Date.now();
    const div = document.createElement('div');
    div.className = 'message message-answer';

    let contextsHtml = '';
    if (data.contexts && data.contexts.length) {
        contextsHtml = data.contexts.map(c => `
            <div class="context-chunk">
                <div class="context-chunk-title">${escapeHtml(c.title)}</div>
                <div>${escapeHtml(c.content.substring(0, 200))}...</div>
            </div>
        `).join('');
    }

    div.innerHTML = `
        <div class="bubble">${escapeHtml(data.answer)}</div>
        <div class="timing-bar">
            <span class="timing-badge model-badge">
                <span class="label">Model</span>
                <span class="value">${escapeHtml(data.model_name)}</span>
            </span>
            <span class="timing-badge">
                <span class="label">Retrieval</span>
                <span class="value">${data.retrieval_time_ms.toFixed(0)}ms</span>
            </span>
            <span class="timing-badge">
                <span class="label">Generation</span>
                <span class="value">${data.generation_time_ms.toFixed(0)}ms</span>
            </span>
            <span class="timing-badge">
                <span class="label">Total</span>
                <span class="value">${data.total_time_ms.toFixed(0)}ms</span>
            </span>
        </div>
        ${contextsHtml ? `
            <button class="context-toggle" onclick="document.getElementById('${ctxId}').classList.toggle('open')">
                📄 Show ${data.contexts.length} context chunks
            </button>
            <div class="context-panel" id="${ctxId}">${contextsHtml}</div>
        ` : ''}
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
