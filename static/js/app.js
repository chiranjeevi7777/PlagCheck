// State management variables
let uploadedFile = null;
let currentTaskId = null;
let pollInterval = null;
let reportId = null;
let reportData = null;

// Table state
let filteredChunks = [];
let currentPage = 1;
const rowsPerPage = 5;
let currentSortColumn = 'id';
let currentSortOrder = 'asc'; // 'asc' or 'desc'

// Chart instances
let distributionChartInstance = null;
let classificationChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    initUploadDropzones();
    initActionButtons();
    initTableListeners();
    
    // Auto-load report if report_id parameter is present in URL
    const urlParams = new URLSearchParams(window.location.search);
    const urlReportId = urlParams.get('report_id');
    if (urlReportId) {
        reportId = urlReportId;
        updateProgressBar(100, 'Loading report...');
        showSection('progress-section');
        loadReport(reportId);
    }
});

/* ==========================================================================
   File Selection & Drag/Drop Handling
   ========================================================================== */
function initUploadDropzones() {
    const dropzone = document.getElementById('file-dropzone');
    const fileInput = document.getElementById('file-input');
    const detailsEl = document.getElementById('file-details');
    const nameEl = detailsEl.querySelector('.file-name');
    const sizeEl = detailsEl.querySelector('.file-size');
    const removeBtn = document.getElementById('file-remove');

    // Click triggers input click
    dropzone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    // File selection event
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0], dropzone, nameEl, sizeEl);
        }
    });

    // Drag events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelection(files[0], dropzone, nameEl, sizeEl);
        }
    });

    // Remove button
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        uploadedFile = null;
        fileInput.value = '';
        dropzone.classList.remove('has-file');
        validateStartButton();
    });
}

function handleFileSelection(file, dropzone, nameEl, sizeEl) {
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (ext !== '.pdf' && ext !== '.docx') {
        alert(`Unsupported file format for ${file.name}. Only PDF and DOCX files are allowed.`);
        return;
    }
    
    // Check file size (50MB)
    if (file.size > 50 * 1024 * 1024) {
        alert(`File ${file.name} exceeds the 50MB limit.`);
        return;
    }

    uploadedFile = file;
    nameEl.textContent = file.name;
    sizeEl.textContent = formatBytes(file.size);
    dropzone.classList.add('has-file');
    validateStartButton();
}

function validateStartButton() {
    const startBtn = document.getElementById('start-btn');
    if (uploadedFile) {
        startBtn.removeAttribute('disabled');
    } else {
        startBtn.setAttribute('disabled', 'true');
    }
}

function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/* ==========================================================================
   Pipeline Flow: Uploading -> Analyzing -> Polling
   ========================================================================== */
function initActionButtons() {
    const startBtn = document.getElementById('start-btn');
    const newCompareBtn = document.getElementById('new-compare-btn');
    const pdfBtn = document.getElementById('pdf-btn');

    startBtn.addEventListener('click', startPipeline);
    newCompareBtn.addEventListener('click', resetPipeline);
    pdfBtn.addEventListener('click', () => {
        if (reportId) {
            window.open(`/report/pdf?report_id=${reportId}`, '_blank');
        }
    });
}

async function startPipeline() {
    if (!uploadedFile) return;

    // Transition to progress section
    showSection('progress-section');
    updateProgressBar(0, 'Preparing upload...');

    try {
        // 1. Upload files
        const formData = new FormData();
        formData.append('file', uploadedFile);
        
        const queryVal = document.getElementById('query-input').value.trim();
        if (queryVal) {
            formData.append('search_query', queryVal);
        }

        updateProgressBar(5, 'Uploading document to server...');
        const uploadResponse = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            const err = await uploadResponse.json();
            throw new Error(err.detail || 'Failed to upload file.');
        }

        const uploadData = await uploadResponse.json();
        updateProgressBar(12, 'File uploaded. Queueing analysis...');

        // 2. Start analysis
        const analyzeResponse = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: uploadData.file_path,
                filename: uploadData.filename,
                search_query: uploadData.search_query
            })
        });

        if (!analyzeResponse.ok) {
            const err = await analyzeResponse.json();
            throw new Error(err.detail || 'Failed to initialize analysis.');
        }

        const analyzeData = await analyzeResponse.json();
        currentTaskId = analyzeData.task_id;

        // 3. Start polling progress
        startPolling(currentTaskId);

    } catch (error) {
        handlePipelineFailure(error.message);
    }
}

function startPolling(taskId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/analyze/status/${taskId}`);
            if (!response.ok) throw new Error('Failed to retrieve task status.');

            const task = await response.json();

            if (task.status === 'processing' || task.status === 'queued') {
                updateProgressBar(task.progress, task.message);
                if (typeof window.updateProgressStages === 'function') {
                    window.updateProgressStages(task.progress, task.message);
                }
            } else if (task.status === 'completed') {
                clearInterval(pollInterval);
                updateProgressBar(100, 'Analysis complete! Loading dashboard...');
                reportId = task.report_id;
                loadReport(reportId);
            } else if (task.status === 'failed') {
                clearInterval(pollInterval);
                throw new Error(task.error || 'The analysis failed on the server.');
            }
        } catch (error) {
            clearInterval(pollInterval);
            handlePipelineFailure(error.message);
        }
    }, 1200);
}

function updateProgressBar(percent, message) {
    const fill = document.getElementById('progress-fill');
    const text = document.getElementById('progress-text');
    const msg = document.getElementById('progress-msg');
    
    fill.style.width = `${percent}%`;
    text.textContent = `${percent}%`;
    msg.textContent = message;
}

function handlePipelineFailure(errorMsg) {
    showSection('upload-section');
    alert(`Error: ${errorMsg}`);
}

async function loadReport(id) {
    try {
        const response = await fetch(`/report?report_id=${id}`);
        if (!response.ok) throw new Error('Could not fetch report details.');

        reportData = await response.json();
        renderDashboard(reportData);
        showSection('dashboard-section');
        // Notify AI module
        if (typeof window.onReportLoaded === 'function') {
            window.onReportLoaded(reportData);
        }
    } catch (err) {
        alert(`Error loading report details: ${err.message}`);
        resetPipeline();
    }
}

function showSection(sectionId) {
    ['upload-section', 'progress-section', 'dashboard-section'].forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.classList.remove('active', 'active-flex');
    });
    const target = document.getElementById(sectionId);
    if (!target) return;
    if (sectionId === 'dashboard-section') {
        target.classList.add('active');
    } else {
        target.classList.add('active');
    }
}

function resetPipeline() {
    uploadedFile = null;
    currentTaskId = null;
    reportId = null;
    reportData = null;
    if (pollInterval) clearInterval(pollInterval);

    // Reset upload boxes
    document.getElementById('file-dropzone').classList.remove('has-file');
    document.getElementById('file-input').value = '';
    document.getElementById('query-input').value = '';

    // Hide tab nav
    const tabNav = document.getElementById('tab-nav');
    if (tabNav) tabNav.style.display = 'none';

    validateStartButton();
    showSection('upload-section');
}

/* ==========================================================================
   Dashboard Renderer & Charts
   ========================================================================== */
function renderDashboard(data) {
    // 1. Populate metadata
    document.getElementById('meta-query-val').textContent = data.metadata.query || 'Auto-extracted';
    document.getElementById('meta-susp-file').textContent = data.metadata.suspected_filename;
    document.getElementById('meta-susp-words').textContent = data.metadata.suspected_word_count.toLocaleString();
    const chunksEl = document.getElementById('meta-chunks');
    if (chunksEl) chunksEl.textContent = data.metadata.suspected_chunk_count || 0;

    // 2. Populate numerical KPIs
    const simEl = document.getElementById('kpi-similarity');
    const badgeEl = document.getElementById('kpi-similarity-badge');
    
    simEl.textContent = `${data.overall_similarity}%`;
    document.getElementById('kpi-exact').textContent = `${data.overall_exact_copy}%`;
    document.getElementById('kpi-paraphrase').textContent = `${data.overall_paraphrase}%`;
    document.getElementById('kpi-confidence').textContent = `${data.average_confidence}%`;

    // Reset indicator classes
    simEl.className = 'kpi-value';
    badgeEl.className = 'kpi-badge';

    // Set colors based on similarity
    if (data.overall_similarity >= 60) {
        simEl.classList.add('text-glow-red');
        badgeEl.textContent = 'High Similarity';
        badgeEl.classList.add('badge-red');
    } else if (data.overall_similarity >= 30) {
        simEl.classList.add('text-glow-orange');
        badgeEl.textContent = 'Moderate';
        badgeEl.classList.add('badge-orange');
    } else {
        simEl.classList.add('text-glow-green');
        badgeEl.textContent = 'Clean / Low';
        badgeEl.classList.add('badge-green');
    }

    // 3. Render charts
    renderCharts(data);

    // 4. Initialize Table
    filteredChunks = [...data.chunks];
    currentPage = 1;
    sortChunks('id', 'asc'); // Initial sorting by ID
    renderChunksTable();
}

function renderCharts(data) {
    const distributionCtx = document.getElementById('distribution-chart').getContext('2d');
    const classificationCtx = document.getElementById('classification-chart').getContext('2d');

    // Destroy existing instances to prevent overlays
    if (distributionChartInstance) distributionChartInstance.destroy();
    if (classificationChartInstance) classificationChartInstance.destroy();

    // Chart 1: Bar chart showing similarity of chunks
    const chunkLabels = data.chunks.map((c, i) => `Chunk ${i + 1}`);
    const chunkSimilarities = data.chunks.map(c => c.semantic_similarity);
    const chunkColors = data.chunks.map(c => {
        if (c.semantic_similarity >= 60) return '#ef4444'; // red
        if (c.semantic_similarity >= 30) return '#f59e0b'; // orange
        return '#10b981'; // green
    });

    distributionChartInstance = new Chart(distributionCtx, {
        type: 'bar',
        data: {
            labels: chunkLabels,
            datasets: [{
                label: 'Semantic Similarity Score',
                data: chunkSimilarities,
                backgroundColor: chunkColors,
                borderWidth: 0,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
            },
            scales: {
                y: { min: 0, max: 100, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af' } },
                x: { grid: { display: false }, ticks: { color: '#9ca3af' } }
            }
        }
    });

    // Chart 2: Classification breakdown doughnut chart
    const labels = Object.keys(data.classification_counts);
    const counts = Object.values(data.classification_counts);
    const presetColors = {
        'Original': '#10b981',
        'Minor Similarity': '#34d399',
        'Light Rewrite': '#60a5fa',
        'Heavy Rewrite': '#818cf8',
        'Heavy Paraphrasing': '#f59e0b',
        'Near Duplicate': '#f87171',
        'Exact Copy': '#ef4444'
    };
    const pieColors = labels.map(l => presetColors[l] || '#8b5cf6');

    classificationChartInstance = new Chart(classificationCtx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: counts,
                backgroundColor: pieColors,
                borderWidth: 1,
                borderColor: '#1f2937'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#9ca3af', font: { family: 'Inter', size: 11 } }
                }
            }
        }
    });
}

/* ==========================================================================
   Table Operations: Searching, Sorting, Pagination
   ========================================================================== */
function initTableListeners() {
    const searchInput = document.getElementById('chunk-search');
    const prevBtn = document.getElementById('prev-page');
    const nextBtn = document.getElementById('next-page');

    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        filteredChunks = reportData.chunks.filter(c => 
            c.suspected_chunk_id.toLowerCase().includes(query) ||
            c.classification.toLowerCase().includes(query) ||
            c.suspected_text.toLowerCase().includes(query)
        );
        currentPage = 1;
        renderChunksTable();
    });

    prevBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderChunksTable();
        }
    });

    nextBtn.addEventListener('click', () => {
        const maxPages = Math.ceil(filteredChunks.length / rowsPerPage);
        if (currentPage < maxPages) {
            currentPage++;
            renderChunksTable();
        }
    });

    // Sort handlers
    const headers = document.querySelectorAll('#chunks-table th.sortable');
    headers.forEach(header => {
        header.addEventListener('click', () => {
            const col = header.getAttribute('data-sort');
            const order = (currentSortColumn === col && currentSortOrder === 'asc') ? 'desc' : 'asc';
            sortChunks(col, order);
            renderChunksTable();
        });
    });
}

function sortChunks(column, order) {
    currentSortColumn = column;
    currentSortOrder = order;

    filteredChunks.sort((a, b) => {
        let valA = a[column];
        let valB = b[column];

        // Parse custom IDs for natural sorting (e.g. chunk_10 vs chunk_2)
        if (column === 'id') {
            valA = parseInt(a.suspected_chunk_id.split('_')[1]) || 0;
            valB = parseInt(b.suspected_chunk_id.split('_')[1]) || 0;
        }

        if (valA < valB) return order === 'asc' ? -1 : 1;
        if (valA > valB) return order === 'asc' ? 1 : -1;
        return 0;
    });

    // Update table header arrows indicator
    const headers = document.querySelectorAll('#chunks-table th.sortable');
    headers.forEach(h => {
        const col = h.getAttribute('data-sort');
        const arrow = (currentSortColumn === col) ? (currentSortOrder === 'asc' ? ' ▲' : ' ▼') : ' ↕';
        h.textContent = h.textContent.split(' ')[0] + arrow;
    });
}

function renderChunksTable() {
    const tbody = document.getElementById('chunks-table-body');
    tbody.innerHTML = '';

    if (filteredChunks.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-center" style="color: var(--text-muted); font-style: italic;">No matching chunks found.</td></tr>`;
        updatePaginationUI();
        return;
    }

    const startIndex = (currentPage - 1) * rowsPerPage;
    const endIndex = Math.min(startIndex + rowsPerPage, filteredChunks.length);
    const pageChunks = filteredChunks.slice(startIndex, endIndex);

    pageChunks.forEach((chunk, index) => {
        const row = document.createElement('tr');
        row.setAttribute('data-id', chunk.suspected_chunk_id);
        
        // Define statuses based on score
        let statusBadge = '';
        if (chunk.semantic_similarity >= 60) {
            statusBadge = '<span class="status-indicator-pill status-flagged"><span class="status-dot"></span>High</span>';
        } else if (chunk.semantic_similarity >= 30) {
            statusBadge = '<span class="status-indicator-pill status-warning"><span class="status-dot"></span>Medium</span>';
        } else {
            statusBadge = '<span class="status-indicator-pill status-clean"><span class="status-dot"></span>Original</span>';
        }

        // Clean label for chunk
        const chunkNum = parseInt(chunk.suspected_chunk_id.split('_')[1]) + 1;

        row.innerHTML = `
            <td>Chunk ${chunkNum}</td>
            <td style="font-weight: 700; color: ${chunk.semantic_similarity >= 60 ? 'var(--danger)' : chunk.semantic_similarity >= 30 ? 'var(--warning)' : 'var(--success)'}">${chunk.semantic_similarity}%</td>
            <td>${chunk.confidence}%</td>
            <td>${chunk.classification}</td>
            <td>${statusBadge}</td>
        `;

        row.addEventListener('click', () => selectChunkRow(row, chunk));
        tbody.appendChild(row);
    });

    updatePaginationUI();

    // Auto-select first chunk on render if table has rows
    if (pageChunks.length > 0) {
        const firstRow = tbody.querySelector('tr');
        selectChunkRow(firstRow, pageChunks[0]);
    }
}

function updatePaginationUI() {
    const prevBtn = document.getElementById('prev-page');
    const nextBtn = document.getElementById('next-page');
    const indicator = document.getElementById('page-indicator');

    const maxPages = Math.ceil(filteredChunks.length / rowsPerPage) || 1;
    indicator.textContent = `Page ${currentPage} of ${maxPages}`;

    prevBtn.disabled = currentPage === 1;
    nextBtn.disabled = currentPage === maxPages;
}

function selectChunkRow(rowElement, chunk) {
    // Clear previous selections
    const selected = document.querySelectorAll('#chunks-table-body tr.selected-row');
    selected.forEach(r => r.classList.remove('selected-row'));

    // Highlight row
    rowElement.classList.add('selected-row');

    // Populate comparison highlights viewer
    populateHighlightViewer(chunk);
}

/* ==========================================================================
   Side by Side Highlight Highlighting Engine
   ========================================================================== */
function escapeHtml(text) {
    if (!text) return '';
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, function(m) { return map[m]; });
}

function populateHighlightViewer(chunk) {
    const metadataEl = document.getElementById('viewer-metadata');
    const suspectedPane = document.getElementById('pane-suspected');
    const originalPane = document.getElementById('pane-original');
    const reasonBox = document.getElementById('pane-reason-box');
    const reasonContent = document.getElementById('pane-reason-content');

    const chunkNum = parseInt(chunk.suspected_chunk_id.split('_')[1]) + 1;
    
    let paperText = '';
    if (chunk.original_title && chunk.original_title !== 'N/A') {
        paperText = `<br>Matched Reference Paper: <strong>${escapeHtml(chunk.original_title)}</strong>`;
        if (chunk.original_authors && chunk.original_authors !== 'N/A') {
            paperText += ` by <em>${escapeHtml(chunk.original_authors)}</em>`;
        }
        if (chunk.original_url) {
            paperText += ` (<a href="${escapeHtml(chunk.original_url)}" target="_blank" style="color: #60a5fa; text-decoration: underline;">View Paper</a>)`;
        }
    } else {
        paperText = ' | Matched: <strong>No Direct Match</strong>';
    }

    metadataEl.innerHTML = `<strong>Suspected Chunk ${chunkNum}</strong> | Similarity: <strong>${chunk.semantic_similarity}%</strong> | Confidence: <strong>${chunk.confidence}%</strong>${paperText}`;

    // Highlight rendering
    let suspHtml = escapeHtml(chunk.suspected_text);
    let origHtml = escapeHtml(chunk.original_text);

    if (chunk.sentence_matches && chunk.sentence_matches.length > 0) {
        // Sort matches by target length descending to avoid nested tag injection conflicts
        const sortedMatches = [...chunk.sentence_matches].sort((a, b) => b.suspected_sentence.length - a.suspected_sentence.length);

        sortedMatches.forEach(match => {
            const suspTarget = escapeHtml(match.suspected_sentence);
            const origTarget = escapeHtml(match.original_sentence);

            const highlightClass = match.match_type === 'exact_copy' ? 'highlight-exact' : 'highlight-paraphrase';

            if (suspTarget && suspHtml.includes(suspTarget)) {
                suspHtml = suspHtml.replace(suspTarget, `<span class="${highlightClass}">${suspTarget}</span>`);
            }
            if (origTarget && origHtml.includes(origTarget)) {
                origHtml = origHtml.replace(origTarget, `<span class="${highlightClass}">${origTarget}</span>`);
            }
        });
    }

    suspectedPane.innerHTML = `<p>${suspHtml}</p>`;
    originalPane.innerHTML = `<p>${origHtml}</p>`;

    // Heuristics explanation
    reasonContent.textContent = chunk.reason;
    reasonBox.style.display = 'block';
}
