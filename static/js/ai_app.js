/* ==========================================================================
   ai_app.js — AI Writing Pattern Analysis UI Logic
   Handles tab switching, AI dashboard rendering, chunk cards, detail panel,
   all AI charts, combined report, and progress stage tracking.
   ========================================================================== */

/* ── Shared state (read by both app.js and ai_app.js) ────────────────────── */
window.aiReportData = null;          // ai_analysis section of combined report
window.fullReportData = null;        // full combined report JSON

/* ── AI Pagination state ─────────────────────────────────────────────────── */
let aiCurrentPage = 1;
const AI_PAGE_SIZE = 9;
let aiAllChunks = [];
let aiFilteredChunks = [];

/* ── Chart references (destroyed on re-render) ───────────────────────────── */
let aiGaugeChart = null;
let aiDistChart = null;
let aiHistChart = null;
let aiFeatChart = null;
let combinedChart = null;

/* ══════════════════════════════════════════════════════════════════════════
   INIT
   ══════════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
    initTabNav();
    initAIChunkSearch();
    initAIDetailClose();
});

/* ══════════════════════════════════════════════════════════════════════════
   TAB NAVIGATION
   ══════════════════════════════════════════════════════════════════════════ */
function initTabNav() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            switchTab(target);
        });
    });
}

function switchTab(tabName) {
    // Update button states
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabName));

    // Show/hide panes
    ['plagiarism', 'ai-analysis', 'combined'].forEach(name => {
        const pane = document.getElementById(`tab-pane-${name}`);
        if (pane) pane.style.display = (name === tabName) ? '' : 'none';
    });

    // Lazy-render AI charts on first visit
    if (tabName === 'ai-analysis' && window.aiReportData) {
        renderAICharts(window.aiReportData);
    }
    if (tabName === 'combined' && window.fullReportData) {
        renderCombinedReport(window.fullReportData);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
   CALLED BY app.js AFTER REPORT LOADS (hook)
   ══════════════════════════════════════════════════════════════════════════ */
window.onReportLoaded = function(reportData) {
    window.fullReportData = reportData;
    window.aiReportData = reportData.ai_analysis || null;

    // Show tab nav
    const nav = document.getElementById('tab-nav');
    if (nav) nav.style.display = 'flex';

    if (window.aiReportData) {
        renderAIDashboard(window.aiReportData);
    }
    renderCombinedReport(reportData);
};

/* ══════════════════════════════════════════════════════════════════════════
   AI DASHBOARD RENDERER
   ══════════════════════════════════════════════════════════════════════════ */
function renderAIDashboard(ai) {
    renderAIKPIs(ai);
    renderAIChunkGrid(ai.chunk_results || []);
    renderAISummary(ai);
    // Charts rendered lazily when tab is clicked
}

/* ── KPI Cards ───────────────────────────────────────────────────────────── */
function renderAIKPIs(ai) {
    const score = ai.overall_ai_score || 0;
    const conf  = ai.average_confidence || 0;
    const high  = (ai.high_probability_chunks || 0) + (ai.very_high_probability_chunks || 0);
    const mod   = ai.moderate_probability_chunks || 0;

    setEl('ai-kpi-score', `${score}%`);
    setEl('ai-kpi-confidence', `${conf}%`);
    setEl('ai-kpi-high', high);
    setEl('ai-kpi-moderate', mod);

    const scoreEl = document.getElementById('ai-kpi-score');
    if (scoreEl) {
        scoreEl.className = 'kpi-value';
        if (score > 70) scoreEl.classList.add('text-glow-red');
        else if (score > 40) scoreEl.classList.add('text-glow-orange');
        else scoreEl.classList.add('text-glow-green');
    }

    const badgeEl = document.getElementById('ai-kpi-badge');
    if (badgeEl) {
        badgeEl.textContent = ai.overall_classification || 'N/A';
        badgeEl.className = `kpi-badge ${getAIBadgeClass(score)}`;
    }

    setEl('ai-disclaimer-text', ai.disclaimer || '');
}

/* ── Chunk Grid ──────────────────────────────────────────────────────────── */
function renderAIChunkGrid(chunks) {
    aiAllChunks = chunks;
    aiFilteredChunks = [...chunks];
    aiCurrentPage = 1;
    renderAIChunkPage();
}

function renderAIChunkPage() {
    const grid = document.getElementById('ai-chunk-grid');
    if (!grid) return;

    const total = aiFilteredChunks.length;
    const totalPages = Math.max(1, Math.ceil(total / AI_PAGE_SIZE));
    aiCurrentPage = Math.min(aiCurrentPage, totalPages);
    const start = (aiCurrentPage - 1) * AI_PAGE_SIZE;
    const page  = aiFilteredChunks.slice(start, start + AI_PAGE_SIZE);

    grid.innerHTML = '';
    page.forEach(cr => {
        const card = buildAIChunkCard(cr);
        grid.appendChild(card);
    });

    setEl('ai-page-indicator', `Page ${aiCurrentPage} of ${totalPages}`);
    const prevBtn = document.getElementById('ai-prev-page');
    const nextBtn = document.getElementById('ai-next-page');
    if (prevBtn) prevBtn.disabled = aiCurrentPage <= 1;
    if (nextBtn) nextBtn.disabled = aiCurrentPage >= totalPages;
}

function buildAIChunkCard(cr) {
    const prob = cr.ai_probability || 0;
    const colorClass = getAIColorClass(prob);
    const badgeClass = getAIBadgeClass(prob);
    const feats = (cr.features || []).slice(0, 3);

    const card = document.createElement('div');
    card.className = 'ai-chunk-card';
    card.dataset.chunkId = cr.chunk_id;

    card.innerHTML = `
        <div class="ai-chunk-bar ${colorClass}"></div>
        <div class="ai-chunk-header">
            <span class="ai-chunk-id">${escHtml(cr.chunk_id)}</span>
            <span class="ai-prob-badge ${badgeClass}">${prob}%</span>
        </div>
        <div class="ai-chunk-preview">${escHtml((cr.chunk_text || '').slice(0, 130))}…</div>
        <div class="ai-chunk-meta">
            ${feats.map(f => `<span class="ai-feat-tag">${escHtml(f)}</span>`).join('')}
        </div>`;

    card.addEventListener('click', () => openAIDetailPanel(cr));
    return card;
}

/* ── Detail Panel ────────────────────────────────────────────────────────── */
function openAIDetailPanel(cr) {
    const prob = cr.ai_probability || 0;
    const panel = document.getElementById('ai-detail-panel');
    if (!panel) return;

    // Highlight selected card
    document.querySelectorAll('.ai-chunk-card').forEach(c => c.classList.remove('selected'));
    const selectedCard = document.querySelector(`.ai-chunk-card[data-chunk-id="${cr.chunk_id}"]`);
    if (selectedCard) selectedCard.classList.add('selected');

    setEl('ai-detail-chunk-id', cr.chunk_id);
    setEl('ai-detail-prob', `${prob}%`);
    setEl('ai-detail-conf', `${cr.confidence || 0}%`);
    setEl('ai-detail-reason', cr.reason || '');
    setEl('ai-detail-text', cr.chunk_text || '');

    const probEl = document.getElementById('ai-detail-prob');
    if (probEl) {
        probEl.style.color = getAIHexColor(prob);
        probEl.style.textShadow = `0 0 12px ${getAIHexColor(prob)}66`;
    }

    const cls = document.getElementById('ai-detail-classification');
    if (cls) {
        cls.textContent = cr.classification || '';
        cls.className = `ai-class-badge ${getAIBadgeClass(prob)}`;
    }

    const featsEl = document.getElementById('ai-detail-features');
    if (featsEl) {
        featsEl.innerHTML = (cr.features || [])
            .map(f => `<span class="ai-feature-tag">${escHtml(f)}</span>`).join('');
    }

    panel.style.display = 'block';
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function initAIDetailClose() {
    const closeBtn = document.getElementById('ai-panel-close');
    if (closeBtn) closeBtn.addEventListener('click', () => {
        const panel = document.getElementById('ai-detail-panel');
        if (panel) panel.style.display = 'none';
        document.querySelectorAll('.ai-chunk-card').forEach(c => c.classList.remove('selected'));
    });
}

/* ── Search ──────────────────────────────────────────────────────────────── */
function initAIChunkSearch() {
    const input = document.getElementById('ai-chunk-search');
    if (!input) return;
    input.addEventListener('input', () => {
        const q = input.value.toLowerCase();
        aiFilteredChunks = aiAllChunks.filter(c =>
            (c.chunk_id || '').toLowerCase().includes(q) ||
            (c.chunk_text || '').toLowerCase().includes(q) ||
            (c.classification || '').toLowerCase().includes(q)
        );
        aiCurrentPage = 1;
        renderAIChunkPage();
    });

    document.getElementById('ai-prev-page')?.addEventListener('click', () => { aiCurrentPage--; renderAIChunkPage(); });
    document.getElementById('ai-next-page')?.addEventListener('click', () => { aiCurrentPage++; renderAIChunkPage(); });
}

/* ══════════════════════════════════════════════════════════════════════════
   AI CHARTS
   ══════════════════════════════════════════════════════════════════════════ */
function renderAICharts(ai) {
    const chunks = ai.chunk_results || [];
    const probs  = chunks.map(c => c.ai_probability || 0);

    renderAIGauge(ai.overall_ai_score || 0);
    renderAIDistPie(ai);
    renderAIHistogram(probs);
    renderAIFeaturesBar(ai.top_features || []);
}

function renderAIGauge(score) {
    const ctx = document.getElementById('ai-gauge-chart');
    if (!ctx) return;
    if (aiGaugeChart) aiGaugeChart.destroy();

    const remaining = 100 - score;
    const color = getAIHexColor(score);

    aiGaugeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [score, remaining],
                backgroundColor: [color, 'rgba(255,255,255,0.05)'],
                borderWidth: 0,
                circumference: 180,
                rotation: 270,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            cutout: '75%',
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false },
            },
        },
        plugins: [{
            id: 'gaugeText',
            afterDraw(chart) {
                const { ctx: c, chartArea: { left, right, top, bottom } } = chart;
                const cx = (left + right) / 2;
                const cy = (top + bottom) / 2 + 20;
                c.save();
                c.font = 'bold 2rem Outfit, sans-serif';
                c.fillStyle = color;
                c.textAlign = 'center';
                c.textBaseline = 'middle';
                c.fillText(`${score}%`, cx, cy);
                c.font = '0.8rem Inter, sans-serif';
                c.fillStyle = '#9ca3af';
                c.fillText('AI Pattern Score', cx, cy + 28);
                c.restore();
            }
        }]
    });
}

function renderAIDistPie(ai) {
    const ctx = document.getElementById('ai-dist-chart');
    if (!ctx) return;
    if (aiDistChart) aiDistChart.destroy();

    const labels = ['Very Low (≤20)', 'Low (21-40)', 'Moderate (41-60)', 'High (61-80)', 'Very High (>80)'];
    const values = [
        ai.very_low_probability_chunks || 0,
        ai.low_probability_chunks || 0,
        ai.moderate_probability_chunks || 0,
        ai.high_probability_chunks || 0,
        ai.very_high_probability_chunks || 0,
    ];
    const colors = ['#10b981', '#8BC34A', '#f59e0b', '#f97316', '#ef4444'];

    aiDistChart = new Chart(ctx, {
        type: 'pie',
        data: { labels, datasets: [{ data: values, backgroundColor: colors, borderWidth: 1, borderColor: 'rgba(0,0,0,0.3)' }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { color: '#9ca3af', font: { size: 11 } } }
            }
        }
    });
}

function renderAIHistogram(probs) {
    const ctx = document.getElementById('ai-histogram-chart');
    if (!ctx) return;
    if (aiHistChart) aiHistChart.destroy();

    const bins = [0, 0, 0, 0, 0];
    probs.forEach(p => {
        if (p <= 20) bins[0]++;
        else if (p <= 40) bins[1]++;
        else if (p <= 60) bins[2]++;
        else if (p <= 80) bins[3]++;
        else bins[4]++;
    });

    aiHistChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'],
            datasets: [{
                label: 'Chunks',
                data: bins,
                backgroundColor: ['#10b981','#8BC34A','#f59e0b','#f97316','#ef4444'],
                borderRadius: 6,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                y: { ticks: { color: '#9ca3af', stepSize: 1 }, grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true }
            }
        }
    });
}

function renderAIFeaturesBar(topFeatures) {
    const ctx = document.getElementById('ai-features-chart');
    if (!ctx) return;
    if (aiFeatChart) aiFeatChart.destroy();

    const top8 = topFeatures.slice(0, 8);
    const labels = top8.map(f => f.feature);
    const values = top8.map(f => f.count);

    aiFeatChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Occurrences',
                data: values,
                backgroundColor: 'rgba(139,92,246,0.6)',
                borderColor: '#8b5cf6',
                borderWidth: 1,
                borderRadius: 5,
                borderSkipped: false,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true },
                y: { ticks: { color: '#9ca3af', font: { size: 10 } }, grid: { display: false } }
            }
        }
    });
}

/* ── AI Summary ──────────────────────────────────────────────────────────── */
function renderAISummary(ai) {
    const container = document.getElementById('ai-summary-grid');
    if (!container) return;

    const hc = ai.highest_chunk || {};
    const lc = ai.lowest_chunk || {};

    const items = [
        { label: 'Overall AI Score',   value: `${ai.overall_ai_score || 0}%`,   sub: ai.overall_classification || '' },
        { label: 'Average Confidence', value: `${ai.average_confidence || 0}%`,  sub: 'Analysis reliability' },
        { label: 'Total Chunks',       value: ai.total_chunks || 0,              sub: 'Sections analysed' },
        { label: 'Very High Prob.',    value: ai.very_high_probability_chunks||0, sub: '> 80% AI probability' },
        { label: 'High Probability',   value: ai.high_probability_chunks || 0,   sub: '61–80% AI probability' },
        { label: 'Moderate',           value: ai.moderate_probability_chunks||0,  sub: '41–60% AI probability' },
        { label: 'Low Probability',    value: ai.low_probability_chunks || 0,    sub: '21–40% AI probability' },
        { label: 'Very Low',           value: ai.very_low_probability_chunks||0,  sub: '≤ 20% AI probability' },
        { label: 'Highest Section',    value: `${hc.ai_probability||0}%`,        sub: hc.chunk_id || 'N/A' },
        { label: 'Lowest Section',     value: `${lc.ai_probability||0}%`,        sub: lc.chunk_id || 'N/A' },
    ];

    container.innerHTML = items.map(it => `
        <div class="ai-summary-item">
            <div class="ai-summary-label">${escHtml(it.label)}</div>
            <div class="ai-summary-value">${escHtml(String(it.value))}</div>
            <div class="ai-summary-sub">${escHtml(String(it.sub))}</div>
        </div>`).join('');
}

/* ══════════════════════════════════════════════════════════════════════════
   COMBINED REPORT
   ══════════════════════════════════════════════════════════════════════════ */
function renderCombinedReport(report) {
    const ai   = report.ai_analysis || {};
    const plag = report;

    // Plagiarism KPIs
    renderCombinedKPIs('combined-plag-kpis', [
        { label: 'Overall Similarity',  value: `${plag.overall_similarity || 0}%` },
        { label: 'Exact Copy',          value: `${plag.overall_exact_copy || 0}%` },
        { label: 'Paraphrasing',        value: `${plag.overall_paraphrase || 0}%` },
        { label: 'Classification',      value: plag.overall_classification || 'N/A' },
        { label: 'Avg Confidence',      value: `${plag.average_confidence || 0}%` },
        { label: 'Flagged Chunks',      value: `${plag.flagged_chunks_count || 0} / ${(plag.chunks||[]).length}` },
    ]);

    // AI KPIs
    renderCombinedKPIs('combined-ai-kpis', [
        { label: 'AI Pattern Score',    value: `${ai.overall_ai_score || 0}%` },
        { label: 'Classification',      value: ai.overall_classification || 'N/A' },
        { label: 'Avg Confidence',      value: `${ai.average_confidence || 0}%` },
        { label: 'High Prob. Sections', value: (ai.high_probability_chunks||0) + (ai.very_high_probability_chunks||0) },
        { label: 'Moderate Sections',   value: ai.moderate_probability_chunks || 0 },
        { label: 'Total Chunks',        value: ai.total_chunks || 0 },
    ]);

    renderCombinedChart(plag, ai);
    renderCombinedRecommendations(plag, ai);
}

function renderCombinedKPIs(containerId, items) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = items.map(it => `
        <div class="combined-kpi-row">
            <span class="combined-kpi-label">${escHtml(it.label)}</span>
            <span class="combined-kpi-val">${escHtml(String(it.value))}</span>
        </div>`).join('');
}

function renderCombinedChart(plag, ai) {
    const ctx = document.getElementById('combined-chart');
    if (!ctx) return;
    if (combinedChart) combinedChart.destroy();

    const plagScore = plag.overall_similarity || 0;
    const aiScore   = ai.overall_ai_score || 0;
    const conf      = plag.average_confidence || 0;

    combinedChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Plagiarism %', 'AI Pattern %', 'Avg Confidence', 'Flagged Chunks', 'Exact Copy %'],
            datasets: [
                {
                    label: 'Document Scores',
                    data: [
                        plagScore,
                        aiScore,
                        conf,
                        Math.min(100, ((plag.flagged_chunks_count||0) / Math.max(1,(plag.chunks||[]).length)) * 100),
                        plag.overall_exact_copy || 0,
                    ],
                    backgroundColor: 'rgba(139,92,246,0.15)',
                    borderColor: '#8b5cf6',
                    borderWidth: 2,
                    pointBackgroundColor: '#8b5cf6',
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                r: {
                    min: 0, max: 100,
                    ticks: { stepSize: 25, color: '#9ca3af', backdropColor: 'transparent' },
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    pointLabels: { color: '#d1d5db', font: { size: 11 } },
                    angleLines: { color: 'rgba(255,255,255,0.06)' },
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function renderCombinedRecommendations(plag, ai) {
    const container = document.getElementById('combined-rec-content');
    if (!container) return;

    const plagScore = plag.overall_similarity || 0;
    const aiScore   = ai.overall_ai_score || 0;
    const recs = [];

    if (plagScore >= 60) {
        recs.push({ icon: '🚨', text: 'Critical: High plagiarism similarity detected. Immediate review and citation of all matched sources is required.' });
    } else if (plagScore >= 30) {
        recs.push({ icon: '⚠️', text: 'Warning: Moderate plagiarism similarity. Review paraphrased sections and ensure proper attribution.' });
    } else {
        recs.push({ icon: '✅', text: 'Plagiarism: The document appears largely original compared to retrieved literature.' });
    }

    if (aiScore >= 70) {
        recs.push({ icon: '🤖', text: 'Very high AI writing patterns detected. Highlighted sections show strong stylistic indicators commonly associated with AI-generated text. Manual review strongly recommended.' });
    } else if (aiScore >= 40) {
        recs.push({ icon: '💡', text: 'Moderate AI writing patterns present. Some sections may have been AI-assisted. Consider reviewing those sections for authenticity.' });
    } else {
        recs.push({ icon: '✅', text: 'AI Pattern: Writing style appears predominantly human-authored based on stylistic analysis.' });
    }

    if (plagScore >= 30 && aiScore >= 50) {
        recs.push({ icon: '🔎', text: 'Combined risk: Both plagiarism and AI pattern indicators are elevated. This document warrants thorough manual review.' });
    }

    container.innerHTML = recs.map(r => `
        <div class="rec-item">
            <span class="rec-icon">${r.icon}</span>
            <span>${escHtml(r.text)}</span>
        </div>`).join('');
}

/* ══════════════════════════════════════════════════════════════════════════
   PROGRESS STAGE TRACKER (called by app.js polling loop)
   ══════════════════════════════════════════════════════════════════════════ */
window.updateProgressStages = function(progress, message) {
    const msg = (message || '').toLowerCase();

    const stageMap = {
        'stage-extract': ['extract', 'text'],
        'stage-search':  ['search', 'scholar', 'query', 'keyword'],
        'stage-plag':    ['plagiarism', 'comparison', 'comparing', 'chunk', 'screening'],
        'stage-ai':      ['ai', 'pattern', 'writing'],
        'stage-report':  ['report', 'pdf', 'aggregat', 'generat'],
    };

    let activeStage = null;
    for (const [id, keywords] of Object.entries(stageMap)) {
        if (keywords.some(k => msg.includes(k))) {
            activeStage = id;
            break;
        }
    }

    Object.keys(stageMap).forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        const stageKeys = Object.keys(stageMap);
        const activeIdx = stageKeys.indexOf(activeStage);
        const thisIdx   = stageKeys.indexOf(id);

        el.classList.remove('active', 'done');
        if (id === activeStage) el.classList.add('active');
        else if (activeIdx > -1 && thisIdx < activeIdx) el.classList.add('done');
    });
};

/* ══════════════════════════════════════════════════════════════════════════
   HELPER UTILITIES
   ══════════════════════════════════════════════════════════════════════════ */
function getAIColorClass(prob) {
    if (prob <= 20) return 'ai-color-very-low';
    if (prob <= 40) return 'ai-color-low';
    if (prob <= 60) return 'ai-color-moderate';
    if (prob <= 80) return 'ai-color-high';
    return 'ai-color-very-high';
}

function getAIBadgeClass(prob) {
    if (prob <= 20) return 'ai-badge-very-low';
    if (prob <= 40) return 'ai-badge-low';
    if (prob <= 60) return 'ai-badge-moderate';
    if (prob <= 80) return 'ai-badge-high';
    return 'ai-badge-very-high';
}

function getAIHexColor(prob) {
    if (prob <= 20) return '#10b981';
    if (prob <= 40) return '#8BC34A';
    if (prob <= 60) return '#f59e0b';
    if (prob <= 80) return '#f97316';
    return '#ef4444';
}

function setEl(id, html) {
    const el = document.getElementById(id);
    if (el) el.textContent = html;
}

function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
