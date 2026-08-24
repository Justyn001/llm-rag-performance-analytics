/**
 * RAG Performance Analytics — Multi-Panel Frontend
 *
 * One question → sent to ALL panels simultaneously → compare side by side
 */

const API = '';

// --- State ---
let selectedLang = 'en';
let topK = 5;
let panelCount = 2;
let queryCount = 0;
let models = [];

// --- DOM ---
const topKSlider = document.getElementById('top-k-slider');
const topKValue = document.getElementById('top-k-value');
const queryForm = document.getElementById('query-form');
const questionInput = document.getElementById('question-input');
const submitBtn = document.getElementById('submit-btn');
const panelsContainer = document.getElementById('panels-container');
const queriesCountEl = document.getElementById('queries-count');
const exampleBar = document.getElementById('example-bar');

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    setupControls();
});

// ===== MODELS =====

async function loadModels() {
    try {
        const resp = await fetch(`${API}/api/models`);
        const data = await resp.json();
        models = data.models;
        buildPanels();
    } catch (err) {
        console.error('Failed to load models:', err);
        models = [];
        buildPanels();
    }
}

// ===== PANELS =====

function buildPanels() {
    panelsContainer.innerHTML = '';
    panelsContainer.dataset.count = panelCount;

    for (let i = 0; i < panelCount; i++) {
        const panel = createPanel(i);
        panelsContainer.appendChild(panel);
    }
}

function createPanel(index) {
    const panel = document.createElement('div');
    panel.className = 'panel';
    panel.dataset.index = index;

    // Build model select options grouped by type
    let options = '';
    const local = models.filter(m => m.is_local);
    const cloud = models.filter(m => !m.is_local);

    if (local.length) {
        options += '<optgroup label="🖥️ Local (GPU)">';
        local.forEach(m => {
            const selected = getDefaultModel(index) === m.name ? ' selected' : '';
            options += `<option value="${esc(m.name)}"${selected}>${esc(m.name)}</option>`;
        });
        options += '</optgroup>';
    }

    if (cloud.length) {
        options += '<optgroup label="☁️ Cloud (API)">';
        cloud.forEach(m => {
            const selected = getDefaultModel(index) === m.name ? ' selected' : '';
            options += `<option value="${esc(m.name)}"${selected}>${esc(m.name)}</option>`;
        });
        options += '</optgroup>';
    }

    if (!models.length) {
        options = '<option value="">No models</option>';
    }

    const model = models.find(m => m.name === getDefaultModel(index));
    const badgeClass = model && model.is_local ? 'badge-local' : 'badge-cloud';
    const badgeText = model && model.is_local ? 'LOCAL' : 'CLOUD';

    panel.innerHTML = `
        <div class="panel-header">
            <div class="panel-number">${index + 1}</div>
            <select class="panel-select" data-panel="${index}">${options}</select>
            <span class="panel-type-badge ${badgeClass}" data-badge="${index}">${badgeText}</span>
        </div>
        <div class="panel-chat" data-chat="${index}">
            <div class="panel-empty">Ask a question below...</div>
        </div>
    `;

    // Update badge on model change
    const select = panel.querySelector('.panel-select');
    select.addEventListener('change', () => {
        const m = models.find(x => x.name === select.value);
        const badge = panel.querySelector('.panel-type-badge');
        if (m) {
            badge.className = `panel-type-badge ${m.is_local ? 'badge-local' : 'badge-cloud'}`;
            badge.textContent = m.is_local ? 'LOCAL' : 'CLOUD';
        }
    });

    return panel;
}

function getDefaultModel(index) {
    if (!models.length) return '';
    // Distribute different models to different panels
    return models[index % models.length].name;
}

// ===== CONTROLS =====

function setupControls() {
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

    // Panel count
    document.querySelectorAll('.panel-count-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.panel-count-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            panelCount = parseInt(btn.dataset.count);
            buildPanels();
        });
    });

    // Form submit
    queryForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const q = questionInput.value.trim();
        if (q) sendToAllPanels(q);
    });

    // Example questions
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const q = btn.dataset.q;
            questionInput.value = q;
            sendToAllPanels(q);
        });
    });
}

// ===== QUERY =====

async function sendToAllPanels(question) {
    submitBtn.disabled = true;
    questionInput.value = '';
    exampleBar.style.display = 'none';

    // Send to each panel in parallel
    const promises = [];
    for (let i = 0; i < panelCount; i++) {
        const select = document.querySelector(`[data-panel="${i}"]`);
        const modelName = select ? select.value : '';
        if (modelName) {
            promises.push(sendToPanel(i, question, modelName));
        }
    }

    await Promise.allSettled(promises);

    queryCount++;
    queriesCountEl.textContent = queryCount;
    submitBtn.disabled = false;
    questionInput.focus();
}

async function sendToPanel(panelIndex, question, modelName) {
    const chat = document.querySelector(`[data-chat="${panelIndex}"]`);
    if (!chat) return;

    // Clear "empty" placeholder
    const empty = chat.querySelector('.panel-empty');
    if (empty) empty.remove();

    // Add question
    const qDiv = document.createElement('div');
    qDiv.className = 'p-msg p-question';
    qDiv.textContent = question;
    chat.appendChild(qDiv);

    // Add loading
    const loadDiv = document.createElement('div');
    loadDiv.className = 'p-msg';
    loadDiv.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';
    chat.appendChild(loadDiv);
    chat.scrollTop = chat.scrollHeight;

    try {
        const resp = await fetch(`${API}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                model_name: modelName,
                language: selectedLang,
                top_k: topK,
            }),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Query failed');
        }

        const data = await resp.json();
        loadDiv.remove();

        // Add answer with timing
        const ctxId = `ctx-${panelIndex}-${Date.now()}`;
        let ctxHtml = '';
        if (data.contexts && data.contexts.length) {
            ctxHtml = data.contexts.map(c => `
                <div class="p-ctx-chunk">
                    <div class="p-ctx-title">${esc(c.title)}</div>
                    <div>${esc(c.content.substring(0, 150))}...</div>
                </div>
            `).join('');
        }

        const aDiv = document.createElement('div');
        aDiv.className = 'p-msg';
        aDiv.innerHTML = `
            <div class="p-answer">${esc(data.answer)}</div>
            <div class="p-timing">
                <span class="p-badge">⏱ <span class="val">${data.retrieval_time_ms.toFixed(0)}ms</span> retrieval</span>
                <span class="p-badge">🤖 <span class="val">${data.generation_time_ms.toFixed(0)}ms</span> generation</span>
                <span class="p-badge">Σ <span class="val">${data.total_time_ms.toFixed(0)}ms</span> total</span>
            </div>
            ${ctxHtml ? `
                <button class="p-context-btn" onclick="document.getElementById('${ctxId}').classList.toggle('open')">
                    📄 ${data.contexts.length} chunks
                </button>
                <div class="p-context" id="${ctxId}">${ctxHtml}</div>
            ` : ''}
        `;
        chat.appendChild(aDiv);
    } catch (err) {
        loadDiv.remove();
        const eDiv = document.createElement('div');
        eDiv.className = 'p-msg';
        eDiv.innerHTML = `<div class="p-answer" style="border-color:var(--error);color:var(--error)">❌ ${esc(err.message)}</div>`;
        chat.appendChild(eDiv);
    }

    chat.scrollTop = chat.scrollHeight;
}

// ===== UTILS =====

function esc(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}
