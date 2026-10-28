/**
 * RAG Performance Analytics — Multi-Panel + Analytics + Auto-Test
 *
 * Chat tab: ask questions to 1-4 models simultaneously
 * Analytics tab: live charts and stats per panel
 * Auto-test: run pre-made question sets sequentially
 * RAG configs: switch chunk sizes and reranking
 * Datasets: switch between knowledge bases
 */

const API = '';

// --- State ---
let selectedLang = 'en';
let topK = 5;
let panelCount = 2;
let queryCount = 0;
let models = [];
let activeTab = 'chat';
let selectedRagConfig = 'basic-512';
let selectedDataset = 'wiki';

// Per-panel history
let panelHistory = {};

// Chart.js instances
let timingCharts = {};
let lengthCharts = {};

// Auto-test state
let autotestRunning = false;
let autotestAbort = false;

// --- DOM ---
const topKSlider = document.getElementById('top-k-slider');
const topKValue = document.getElementById('top-k-value');
const queryForm = document.getElementById('query-form');
const questionInput = document.getElementById('question-input');
const submitBtn = document.getElementById('submit-btn');
const chatView = document.getElementById('chat-view');
const analyticsView = document.getElementById('analytics-view');
const queriesCountEl = document.getElementById('queries-count');
const exampleBar = document.getElementById('example-bar');
const inputArea = document.getElementById('input-area');
const autotestBtn = document.getElementById('autotest-btn');
const autotestStopBtn = document.getElementById('autotest-stop-btn');
const autotestProgress = document.getElementById('autotest-progress');
const testSetSelect = document.getElementById('test-set-select');
const testCountSelect = document.getElementById('test-count-select');
const ragConfigSelect = document.getElementById('rag-config-select');
const datasetSelect = document.getElementById('dataset-select');
const reportBar = document.getElementById('report-bar');
const downloadReportBtn = document.getElementById('download-report-btn');

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    loadTestSets();
    loadRagConfigs();
    setupControls();
});

// ===== MODELS =====

async function loadModels() {
    try {
        const resp = await fetch(`${API}/api/models`);
        const data = await resp.json();
        models = data.models;
    } catch (err) {
        console.error('Failed to load models:', err);
        models = [];
    }
    buildPanels();
    buildAnalyticsPanels();
}

// ===== RAG CONFIGS =====

async function loadRagConfigs() {
    try {
        const resp = await fetch(`${API}/api/rag-configs`);
        const data = await resp.json();
        ragConfigSelect.innerHTML = '';
        data.configs.forEach(c => {
            const opt = document.createElement('option');
            opt.value = c.name;
            opt.textContent = c.display_name;
            opt.title = c.description;
            if (c.name === 'basic-512') opt.selected = true;
            ragConfigSelect.appendChild(opt);
        });
    } catch (err) {
        console.error('Failed to load RAG configs:', err);
    }
}

// ===== TEST SETS =====

async function loadTestSets() {
    try {
        const resp = await fetch(`${API}/api/test-sets`);
        const data = await resp.json();
        const sets = data.test_sets;
        testSetSelect.innerHTML = '';
        sets.forEach(ts => {
            const opt = document.createElement('option');
            opt.value = ts.name;
            opt.textContent = `${ts.display_name} (${ts.total_available})`;
            testSetSelect.appendChild(opt);
        });
        autotestBtn.disabled = sets.length === 0;
    } catch (err) {
        console.error('Failed to load test sets:', err);
        testSetSelect.innerHTML = '<option value="">No tests</option>';
    }
}

async function runAutoTest() {
    if (autotestRunning) return;

    const setName = testSetSelect.value;
    const count = parseInt(testCountSelect.value);
    if (!setName) return;

    let questions;
    try {
        const resp = await fetch(`${API}/api/test-sets/${setName}?count=${count}`);
        const data = await resp.json();
        questions = data.questions;
    } catch (err) {
        console.error('Failed to load test questions:', err);
        return;
    }

    if (!questions.length) return;

    autotestRunning = true;
    autotestAbort = false;
    autotestBtn.classList.add('hidden');
    autotestStopBtn.classList.remove('hidden');
    autotestProgress.classList.remove('hidden');
    submitBtn.disabled = true;
    questionInput.disabled = true;
    exampleBar.style.display = 'none';
    switchTab('chat');

    for (let i = 0; i < questions.length; i++) {
        if (autotestAbort) break;

        const q = questions[i].question;
        autotestProgress.textContent = `(${i + 1}/${questions.length})`;
        questionInput.value = q;

        await sendToAllPanels(q);

        if (i < questions.length - 1 && !autotestAbort) {
            await new Promise(r => setTimeout(r, 400));
        }
    }

    autotestRunning = false;
    autotestBtn.classList.remove('hidden');
    autotestStopBtn.classList.add('hidden');
    autotestProgress.classList.add('hidden');
    submitBtn.disabled = false;
    questionInput.disabled = false;
    questionInput.value = '';

    if (!autotestAbort) switchTab('analytics');
}

function stopAutoTest() {
    autotestAbort = true;
}

// ===== CHAT PANELS =====

function buildPanels() {
    chatView.innerHTML = '';
    chatView.dataset.count = panelCount;

    for (let i = 0; i < panelCount; i++) {
        chatView.appendChild(createPanel(i));
        if (!panelHistory[i]) {
            panelHistory[i] = { modelName: getDefaultModel(i), queries: [] };
        }
    }
}

function createPanel(index) {
    const panel = document.createElement('div');
    panel.className = 'panel';
    panel.dataset.index = index;

    let options = buildModelOptions(index);
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

    const select = panel.querySelector('.panel-select');
    select.addEventListener('change', () => {
        const m = models.find(x => x.name === select.value);
        const badge = panel.querySelector('.panel-type-badge');
        if (m) {
            badge.className = `panel-type-badge ${m.is_local ? 'badge-local' : 'badge-cloud'}`;
            badge.textContent = m.is_local ? 'LOCAL' : 'CLOUD';
        }
        if (panelHistory[index]) panelHistory[index].modelName = select.value;
    });

    return panel;
}

function buildModelOptions(index) {
    let options = '';
    const local = models.filter(m => m.is_local);
    const cloud = models.filter(m => !m.is_local);
    if (local.length) {
        options += '<optgroup label="🖥️ Local (GPU)">';
        local.forEach(m => {
            const sel = getDefaultModel(index) === m.name ? ' selected' : '';
            options += `<option value="${esc(m.name)}"${sel}>${esc(m.name)}</option>`;
        });
        options += '</optgroup>';
    }
    if (cloud.length) {
        options += '<optgroup label="☁️ Cloud (API)">';
        cloud.forEach(m => {
            const sel = getDefaultModel(index) === m.name ? ' selected' : '';
            options += `<option value="${esc(m.name)}"${sel}>${esc(m.name)}</option>`;
        });
        options += '</optgroup>';
    }
    if (!models.length) options = '<option value="">No models</option>';
    return options;
}

function getDefaultModel(index) {
    if (!models.length) return '';
    return models[index % models.length].name;
}

// ===== ANALYTICS PANELS =====

function buildAnalyticsPanels() {
    analyticsView.innerHTML = '';
    analyticsView.dataset.count = panelCount;

    Object.values(timingCharts).forEach(c => c.destroy());
    Object.values(lengthCharts).forEach(c => c.destroy());
    timingCharts = {};
    lengthCharts = {};

    for (let i = 0; i < panelCount; i++) {
        analyticsView.appendChild(createAnalyticsPanel(i));
    }
}

function createAnalyticsPanel(index) {
    const panel = document.createElement('div');
    panel.className = 'analytics-panel';

    const history = panelHistory[index];
    const modelName = history ? history.modelName : getDefaultModel(index);

    panel.innerHTML = `
        <div class="panel-header">
            <div class="panel-number">${index + 1}</div>
            <span style="flex:1;font-size:13px;font-weight:600;color:var(--text-primary)">${esc(modelName)}</span>
        </div>
        <div class="analytics-body" data-analytics="${index}">
            <div class="analytics-empty">
                <div class="analytics-empty-icon">📊</div>
                <div>No data yet — ask questions in Chat tab</div>
            </div>
        </div>
    `;

    return panel;
}

function renderAnalytics(index) {
    const body = document.querySelector(`[data-analytics="${index}"]`);
    if (!body) return;

    const history = panelHistory[index];
    if (!history || !history.queries.length) return;

    const queries = history.queries;
    const n = queries.length;

    const avgRetrieval = queries.reduce((s, q) => s + q.retrieval_ms, 0) / n;
    const avgGeneration = queries.reduce((s, q) => s + q.generation_ms, 0) / n;
    const avgTotal = queries.reduce((s, q) => s + q.total_ms, 0) / n;
    const avgLen = queries.reduce((s, q) => s + q.answer_len, 0) / n;
    const fastestGen = Math.min(...queries.map(q => q.generation_ms));
    const avgTokS = queries.filter(q => q.tokens_per_second > 0);
    const avgTokenSpeed = avgTokS.length ? avgTokS.reduce((s, q) => s + q.tokens_per_second, 0) / avgTokS.length : 0;
    const avgTokIn = queries.reduce((s, q) => s + q.tokens_in, 0) / n;
    const avgTokOut = queries.reduce((s, q) => s + q.tokens_out, 0) / n;

    // Human eval stats
    const rated = (field) => queries.filter(q => q[field] !== null);
    const pct = (field) => {
        const r = rated(field);
        if (!r.length) return null;
        return Math.round(r.filter(q => q[field] === true).length / r.length * 100);
    };
    const correctPct = pct('eval_correct');
    const faithfulPct = pct('eval_faithful');
    const contextPct = pct('eval_context');
    const ratedCount = rated('eval_correct').length;

    const evalHtml = ratedCount > 0 ? `
        <div class="stats-grid">
            <div class="stat-card ${correctPct !== null && correctPct >= 70 ? '' : 'warn'}">
                <div class="stat-value">${correctPct !== null ? correctPct + '%' : '—'}</div>
                <div class="stat-label">Correctness</div>
            </div>
            <div class="stat-card ${faithfulPct !== null && faithfulPct >= 70 ? 'accent' : 'warn'}">
                <div class="stat-value">${faithfulPct !== null ? faithfulPct + '%' : '—'}</div>
                <div class="stat-label">Faithfulness</div>
            </div>
            <div class="stat-card ${contextPct !== null && contextPct >= 70 ? '' : 'warn'}">
                <div class="stat-value">${contextPct !== null ? contextPct + '%' : '—'}</div>
                <div class="stat-label">Context Rel.</div>
            </div>
        </div>
        <div style="text-align:center;font-size:10px;color:var(--text-muted);margin-top:-8px">
            ${ratedCount}/${n} queries rated
        </div>
    ` : `
        <div style="text-align:center;padding:8px;font-size:11px;color:var(--text-muted);background:var(--bg-card);border-radius:var(--radius-sm);border:1px solid var(--border)">
            👤 No human evaluations yet — rate answers in Chat tab
        </div>
    `;

    body.innerHTML = `
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${avgRetrieval.toFixed(0)}<span style="font-size:12px">ms</span></div>
                <div class="stat-label">Avg Retrieval</div>
            </div>
            <div class="stat-card accent">
                <div class="stat-value">${avgGeneration.toFixed(0)}<span style="font-size:12px">ms</span></div>
                <div class="stat-label">Avg Generation</div>
            </div>
            <div class="stat-card warn">
                <div class="stat-value">${avgTotal.toFixed(0)}<span style="font-size:12px">ms</span></div>
                <div class="stat-label">Avg Total</div>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${n}</div>
                <div class="stat-label">Queries</div>
            </div>
            <div class="stat-card accent">
                <div class="stat-value">${avgTokenSpeed > 0 ? avgTokenSpeed.toFixed(1) : '—'}</div>
                <div class="stat-label">Avg tok/s</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${avgTokIn > 0 ? avgTokIn.toFixed(0) : '—'}<span style="font-size:10px">→</span>${avgTokOut > 0 ? avgTokOut.toFixed(0) : '—'}</div>
                <div class="stat-label">Avg Tokens In→Out</div>
            </div>
        </div>

        <div class="chart-title" style="margin-top:4px">👤 Human Evaluation</div>
        ${evalHtml}

        <div class="chart-box">
            <div class="chart-title">⏱️ Response Time per Query</div>
            <div class="chart-wrap"><canvas id="timing-chart-${index}"></canvas></div>
        </div>

        <div class="chart-box">
            <div class="chart-title">📝 Answer Length per Query</div>
            <div class="chart-wrap"><canvas id="length-chart-${index}"></canvas></div>
        </div>

        <div class="query-log">
            <div class="query-log-title">📋 Query History</div>
            ${queries.map((q, i) => `
                <div class="query-log-entry">
                    <div class="qle-question">Q${i + 1}: ${esc(q.question)}</div>
                    <div class="qle-answer">${esc(q.answer.substring(0, 120))}${q.answer.length > 120 ? '...' : ''}</div>
                    <div class="qle-meta">
                        retrieval: ${q.retrieval_ms.toFixed(0)}ms · generation: ${q.generation_ms.toFixed(0)}ms · total: ${q.total_ms.toFixed(0)}ms · ${q.answer_len} chars
                        ${q.tokens_per_second > 0 ? ` · ${q.tokens_per_second.toFixed(1)} tok/s` : ''}
                        ${q.eval_correct !== null ? ` · ${q.eval_correct ? '✅' : '❌'}correct` : ''}
                        ${q.eval_faithful !== null ? ` · ${q.eval_faithful ? '✅' : '❌'}faithful` : ''}
                        ${q.eval_context !== null ? ` · ${q.eval_context ? '✅' : '❌'}context` : ''}
                    </div>
                </div>
            `).join('')}
        </div>
    `;

    createTimingChart(index, queries);
    createLengthChart(index, queries);
}

function _chartTheme() {
    const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    return {
        text: dark ? '#a1a3a8' : '#495057',
        tick: dark ? '#6c6e75' : '#868e96',
        grid: dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
        colors: dark
            ? ['#4b8df8', '#2dd4bf', '#fbbf24', '#a78bfa']
            : ['#2563eb', '#0d9488', '#d97706', '#7c3aed'],
        retBg: dark ? 'rgba(75, 141, 248, 0.4)' : 'rgba(37, 99, 235, 0.35)',
    };
}

function createTimingChart(index, queries) {
    const ctx = document.getElementById(`timing-chart-${index}`);
    if (!ctx) return;
    if (timingCharts[index]) timingCharts[index].destroy();

    const labels = queries.map((_, i) => `Q${i + 1}`);
    const t = _chartTheme();
    const panelColor = t.colors[index % t.colors.length];

    timingCharts[index] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                { label: 'Retrieval', data: queries.map(q => q.retrieval_ms), backgroundColor: t.retBg, borderRadius: 3 },
                { label: 'Generation', data: queries.map(q => q.generation_ms), backgroundColor: panelColor + 'AA', borderRadius: 3 },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'top', labels: { color: t.text, font: { size: 10, family: 'Inter' }, boxWidth: 12 } } },
            scales: {
                x: { stacked: true, ticks: { color: t.tick, font: { size: 10 } }, grid: { display: false } },
                y: { stacked: true, ticks: { color: t.tick, font: { size: 10 }, callback: v => v + 'ms' }, grid: { color: t.grid } },
            },
        },
    });
}

function createLengthChart(index, queries) {
    const ctx = document.getElementById(`length-chart-${index}`);
    if (!ctx) return;
    if (lengthCharts[index]) lengthCharts[index].destroy();

    const labels = queries.map((_, i) => `Q${i + 1}`);
    const t = _chartTheme();
    const panelColor = t.colors[index % t.colors.length];

    lengthCharts[index] = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Answer length (chars)', data: queries.map(q => q.answer_len), backgroundColor: panelColor + '66', borderRadius: 3 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: t.text, font: { size: 10, family: 'Inter' }, boxWidth: 12 } } },
            scales: {
                x: { ticks: { color: t.tick, font: { size: 10 } }, grid: { display: false } },
                y: { ticks: { color: t.tick, font: { size: 10 } }, grid: { color: t.grid } },
            },
        },
    });
}

// ===== TAB SWITCHING =====

function switchTab(tab) {
    activeTab = tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`[data-tab="${tab}"]`).classList.add('active');

    if (tab === 'chat') {
        chatView.classList.remove('hidden');
        analyticsView.classList.add('hidden');
        inputArea.classList.remove('hidden');
        reportBar.classList.add('hidden');
    } else {
        chatView.classList.add('hidden');
        analyticsView.classList.remove('hidden');
        inputArea.classList.add('hidden');
        buildAnalyticsPanels();
        for (let i = 0; i < panelCount; i++) renderAnalytics(i);

        // Show report bar if there's any data
        const hasData = Object.values(panelHistory).some(h => h.queries && h.queries.length > 0);
        if (hasData) {
            reportBar.classList.remove('hidden');
        } else {
            reportBar.classList.add('hidden');
        }
    }
}

// ===== CONTROLS =====

function setupControls() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedLang = btn.dataset.lang;
        });
    });

    topKSlider.addEventListener('input', (e) => {
        topK = parseInt(e.target.value);
        topKValue.textContent = topK;
    });

    document.querySelectorAll('.panel-count-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.panel-count-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            panelCount = parseInt(btn.dataset.count);
            buildPanels();
            buildAnalyticsPanels();
        });
    });

    // RAG Config
    ragConfigSelect.addEventListener('change', () => {
        selectedRagConfig = ragConfigSelect.value;
    });

    // Dataset
    datasetSelect.addEventListener('change', () => {
        selectedDataset = datasetSelect.value;
    });

    queryForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const q = questionInput.value.trim();
        if (q) sendToAllPanels(q);
    });

    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            questionInput.value = btn.dataset.q;
            sendToAllPanels(btn.dataset.q);
        });
    });

    autotestBtn.addEventListener('click', () => runAutoTest());
    autotestStopBtn.addEventListener('click', () => stopAutoTest());
    downloadReportBtn.addEventListener('click', () => downloadReport());

    document.getElementById('reset-btn').addEventListener('click', () => {
        if (autotestRunning) stopAutoTest();
        panelHistory = {};
        queryCount = 0;
        queriesCountEl.textContent = 0;
        exampleBar.style.display = '';
        buildPanels();
        buildAnalyticsPanels();
        switchTab('chat');
    });
}

// ===== QUERY =====

async function sendToAllPanels(question) {
    submitBtn.disabled = true;
    questionInput.value = '';
    exampleBar.style.display = 'none';

    const promises = [];
    for (let i = 0; i < panelCount; i++) {
        const select = document.querySelector(`[data-panel="${i}"]`);
        const modelName = select ? select.value : '';
        if (modelName) promises.push(sendToPanel(i, question, modelName));
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

    const empty = chat.querySelector('.panel-empty');
    if (empty) empty.remove();

    const qDiv = document.createElement('div');
    qDiv.className = 'p-msg p-question';
    qDiv.textContent = question;
    chat.appendChild(qDiv);

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
                rag_config: selectedRagConfig,
                dataset: selectedDataset,
            }),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Query failed');
        }

        const data = await resp.json();
        loadDiv.remove();

        if (!panelHistory[panelIndex]) panelHistory[panelIndex] = { modelName, queries: [] };
        panelHistory[panelIndex].modelName = modelName;
        const queryIndex = panelHistory[panelIndex].queries.length;
        panelHistory[panelIndex].queries.push({
            question,
            answer: data.answer,
            retrieval_ms: data.retrieval_time_ms,
            generation_ms: data.generation_time_ms,
            total_ms: data.total_time_ms,
            answer_len: data.answer.length,
            contexts: data.contexts,
            rag_config: data.rag_config,
            dataset: data.dataset,
            tokens_in: data.tokens_in || 0,
            tokens_out: data.tokens_out || 0,
            tokens_per_second: data.tokens_per_second || 0,
            eval_correct: null,
            eval_faithful: null,
            eval_context: null,
        });

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

        const tokBadges = data.tokens_out > 0 ? `
                <span class="p-badge">📥 <span class="val">${data.tokens_in}</span> tok in</span>
                <span class="p-badge">📤 <span class="val">${data.tokens_out}</span> tok out</span>
                <span class="p-badge">⚡ <span class="val">${data.tokens_per_second.toFixed(1)}</span> tok/s</span>
            ` : '';

        const evalId = `eval-${panelIndex}-${queryIndex}`;

        const aDiv = document.createElement('div');
        aDiv.className = 'p-msg';
        aDiv.innerHTML = `
            <div class="p-answer">${esc(data.answer)}</div>
            <div class="p-timing">
                <span class="p-badge">⏱ <span class="val">${data.retrieval_time_ms.toFixed(0)}ms</span> retrieval</span>
                <span class="p-badge">🤖 <span class="val">${data.generation_time_ms.toFixed(0)}ms</span> generation</span>
                <span class="p-badge">Σ <span class="val">${data.total_time_ms.toFixed(0)}ms</span> total</span>
                ${tokBadges}
            </div>
            <div class="p-eval" id="${evalId}">
                <span class="eval-label">Evaluate:</span>
                <button class="eval-btn" data-metric="correct" data-panel="${panelIndex}" data-qi="${queryIndex}" title="Is the answer correct?">✅ Correct</button>
                <button class="eval-btn" data-metric="faithful" data-panel="${panelIndex}" data-qi="${queryIndex}" title="Does the answer use only the provided context?">📎 Faithful</button>
                <button class="eval-btn" data-metric="context" data-panel="${panelIndex}" data-qi="${queryIndex}" title="Are the retrieved chunks relevant?">🎯 Context OK</button>
            </div>
            ${ctxHtml ? `
                <button class="p-context-btn" onclick="document.getElementById('${ctxId}').classList.toggle('open')">
                    📄 ${data.contexts.length} chunks
                </button>
                <div class="p-context" id="${ctxId}">${ctxHtml}</div>
            ` : ''}
        `;
        chat.appendChild(aDiv);

        // Wire up eval buttons
        aDiv.querySelectorAll('.eval-btn').forEach(btn => {
            btn.addEventListener('click', () => toggleEval(btn));
        });
    } catch (err) {
        loadDiv.remove();
        const eDiv = document.createElement('div');
        eDiv.className = 'p-msg';
        eDiv.innerHTML = `<div class="p-answer" style="border-color:var(--error);color:var(--error)">❌ ${esc(err.message)}</div>`;
        chat.appendChild(eDiv);
    }

    chat.scrollTop = chat.scrollHeight;
}

// ===== HUMAN EVALUATION =====

function toggleEval(btn) {
    const panel = parseInt(btn.dataset.panel);
    const qi = parseInt(btn.dataset.qi);
    const metric = btn.dataset.metric; // 'correct', 'faithful', 'context'
    const field = `eval_${metric}`;

    const query = panelHistory[panel]?.queries[qi];
    if (!query) return;

    // Cycle: null → true → false → null
    if (query[field] === null) {
        query[field] = true;
        btn.classList.add('eval-positive');
        btn.classList.remove('eval-negative');
    } else if (query[field] === true) {
        query[field] = false;
        btn.classList.remove('eval-positive');
        btn.classList.add('eval-negative');
    } else {
        query[field] = null;
        btn.classList.remove('eval-positive', 'eval-negative');
    }
}

// ===== PDF REPORT =====

async function downloadReport() {
    const panels = [];
    for (let i = 0; i < panelCount; i++) {
        const h = panelHistory[i];
        if (!h || !h.queries.length) continue;
        panels.push({
            model_name: h.modelName,
            queries: h.queries.map(q => ({
                question: q.question,
                answer: q.answer,
                retrieval_ms: q.retrieval_ms,
                generation_ms: q.generation_ms,
                total_ms: q.total_ms,
                answer_len: q.answer_len,
                tokens_in: q.tokens_in || 0,
                tokens_out: q.tokens_out || 0,
                tokens_per_second: q.tokens_per_second || 0,
                eval_correct: q.eval_correct,
                eval_faithful: q.eval_faithful,
                eval_context: q.eval_context,
            })),
        });
    }

    if (!panels.length) {
        alert('No data to export — run some queries first.');
        return;
    }

    downloadReportBtn.disabled = true;
    downloadReportBtn.textContent = '⏳ Generating...';

    try {
        const resp = await fetch(`${API}/api/report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                panels,
                rag_config: selectedRagConfig,
                dataset: selectedDataset,
                language: selectedLang,
                top_k: topK,
            }),
        });

        if (!resp.ok) throw new Error('Report generation failed');

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `rag_report_${new Date().toISOString().slice(0, 16).replace(/[:\-T]/g, '')}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    } catch (err) {
        console.error('Report download failed:', err);
        alert('Failed to generate report: ' + err.message);
    } finally {
        downloadReportBtn.disabled = false;
        downloadReportBtn.textContent = '📥 Download Report';
    }
}

// ===== UTILS =====

function esc(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}
