/**
 * PlagCheck AI — Document Enhancement Dashboard
 */

class DocumentEnhancementManager {
    constructor() {
        this.documentId = null;
        this.filename = null;
        this.paragraphs = [];
        this.metrics = {};
        this.revisions = {}; // index -> revised text
        this.currentVersion = 1;
        this.selectedIdx = null;
        this.viewMode = 'editor'; // 'editor' or 'diff'
        this.isAnalyzing = false;
        
        this.initDOMElements();
        this.bindEvents();
    }

    reset() {
        this.documentId = null;
        this.filename = null;
        this.paragraphs = [];
        this.metrics = {};
        this.revisions = {};
        this.currentVersion = 1;
        this.selectedIdx = null;
        this.viewMode = 'editor';
        this.isAnalyzing = false;
        
        if (this.paragraphsList) this.paragraphsList.innerHTML = '';
        if (this.editorPanel) {
            this.editorPanel.innerHTML = `
                <div class="editor-placeholder">
                    <div class="editor-placeholder-icon">✨</div>
                    <h3>AI Enhancement Engine</h3>
                    <p>Load a document to start improving academic integrity and writing quality.</p>
                </div>
            `;
        }
        if (this.diffList) this.diffList.innerHTML = '';
        if (this.versionSelect) this.versionSelect.innerHTML = '<option value="1">Version 1 (Original)</option>';
        if (this.btnRestore) this.btnRestore.style.display = 'none';
        
        if (this.kpiEase) {
            this.kpiEase.textContent = '--';
            this.kpiEase.className = 'kpi-value';
        }
        if (this.kpiGrade) this.kpiGrade.textContent = '--';
        if (this.kpiTtr) this.kpiTtr.textContent = '--';
        if (this.kpiPassive) this.kpiPassive.textContent = '--';
        if (this.kpiWords) this.kpiWords.textContent = '--';
    }

    initDOMElements() {
        this.tabPane = document.getElementById('tab-pane-enhancement');
        this.kpiEase = document.getElementById('enh-kpi-ease');
        this.kpiGrade = document.getElementById('enh-kpi-grade');
        this.kpiTtr = document.getElementById('enh-kpi-ttr');
        this.kpiPassive = document.getElementById('enh-kpi-passive');
        this.kpiWords = document.getElementById('enh-kpi-words');
        
        this.versionSelect = document.getElementById('enh-version-select');
        this.btnRestore = document.getElementById('enh-btn-restore');
        this.btnDownloadDocx = document.getElementById('enh-btn-docx');
        this.btnDownloadPdf = document.getElementById('enh-btn-pdf');
        this.btnDownloadDiff = document.getElementById('enh-btn-diff');
        
        this.btnModeEditor = document.getElementById('enh-btn-mode-editor');
        this.btnModeDiff = document.getElementById('enh-btn-mode-diff');
        
        this.workspaceEditor = document.getElementById('enh-workspace-editor');
        this.workspaceDiff = document.getElementById('enh-workspace-diff');
        
        this.paragraphsList = document.getElementById('enh-paragraphs-list');
        this.editorPanel = document.getElementById('enh-editor-panel');
        this.diffList = document.getElementById('enh-diff-list');
    }

    bindEvents() {
        if (this.versionSelect) {
            this.versionSelect.addEventListener('change', (e) => this.handleVersionChange(parseInt(e.target.value)));
        }
        if (this.btnRestore) {
            this.btnRestore.addEventListener('click', () => this.handleRestoreVersion());
        }
        if (this.btnDownloadDocx) {
            this.btnDownloadDocx.addEventListener('click', () => this.downloadDocx());
        }
        if (this.btnDownloadPdf) {
            this.btnDownloadPdf.addEventListener('click', () => this.downloadPdf());
        }
        if (this.btnDownloadDiff) {
            this.btnDownloadDiff.addEventListener('click', () => this.downloadDiffReport());
        }
        if (this.btnModeEditor) {
            this.btnModeEditor.addEventListener('click', () => this.setViewMode('editor'));
        }
        if (this.btnModeDiff) {
            this.btnModeDiff.addEventListener('click', () => this.setViewMode('diff'));
        }
    }

    async startAnalysis(filePath, filename) {
        if (this.isAnalyzing) return;
        this.isAnalyzing = true;
        this.filename = filename;
        
        this.showLoadingState();

        try {
            const response = await fetch('/enhancement/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath, filename: filename })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Analysis failed');
            }

            const data = await response.json();
            this.documentId = data.document_id;
            this.paragraphs = data.paragraphs;
            this.metrics = data.metrics;
            this.currentVersion = 1;
            this.revisions = {};
            this.selectedIdx = null;

            this.updateKPIs(this.metrics);
            this.updateVersionDropdown([1]);
            this.renderParagraphsList();
            this.setViewMode('editor');
            
            // Select first paragraph with an issue, or index 0 if none
            const firstIssueIdx = this.paragraphs.findIndex(p => p.classification.category !== 'Standard');
            this.selectParagraph(firstIssueIdx >= 0 ? firstIssueIdx : 0);

        } catch (e) {
            console.error(e);
            this.showErrorState(e.message);
        } finally {
            this.isAnalyzing = false;
        }
    }

    showLoadingState() {
        this.paragraphsList.innerHTML = `
            <div style="padding: 2rem; text-align: center; color: rgba(255,255,255,0.4);">
                <div class="loading-spinner" style="margin: 0 auto 1rem;"></div>
                Generating quality enhancement plan...
            </div>
        `;
        this.editorPanel.innerHTML = `
            <div class="editor-placeholder">
                <div class="editor-placeholder-icon">✨</div>
                <h3>AI Enhancement Engine</h3>
                <p>Analyzing document syntax, structure, readability, and duplication...</p>
            </div>
        `;
    }

    showErrorState(message) {
        this.paragraphsList.innerHTML = `
            <div style="padding: 2rem; text-align: center; color: var(--enh-danger);">
                <span style="font-size: 2rem;">⚠️</span>
                <p>Analysis failed: ${message}</p>
            </div>
        `;
    }

    updateKPIs(m) {
        if (!m) return;
        const ease = m.readability?.flesch_reading_ease ?? 0;
        this.kpiEase.textContent = ease;
        this.kpiEase.className = 'kpi-value ' + (ease >= 60 ? 'text-glow-green' : (ease >= 30 ? 'text-glow-orange' : 'text-glow-red'));
        
        this.kpiGrade.textContent = m.readability?.flesch_kincaid_grade ?? 'N/A';
        this.kpiTtr.textContent = (m.lexical?.type_token_ratio ?? 0).toFixed(2);
        this.kpiPassive.textContent = `${(m.grammar?.passive_voice_pct ?? 0).toFixed(0)}%`;
        this.kpiWords.textContent = m.structure?.word_count ?? 0;
    }

    updateVersionDropdown(versions) {
        this.versionSelect.innerHTML = '';
        versions.forEach(v => {
            const opt = document.createElement('option');
            opt.value = v;
            opt.textContent = `Version ${v} ${v === 1 ? '(Original)' : ''}`;
            if (v === this.currentVersion) {
                opt.selected = true;
            }
            this.versionSelect.appendChild(opt);
        });
        
        // Show restore button only for non-current historical versions
        this.btnRestore.style.display = 'none'; 
    }

    setViewMode(mode) {
        this.viewMode = mode;
        if (mode === 'editor') {
            this.btnModeEditor.classList.add('active');
            this.btnModeDiff.classList.remove('active');
            this.workspaceEditor.style.display = 'grid';
            this.workspaceDiff.style.display = 'none';
        } else {
            this.btnModeEditor.classList.remove('active');
            this.btnModeDiff.classList.add('active');
            this.workspaceEditor.style.display = 'none';
            this.workspaceDiff.style.display = 'block';
            this.loadDiffView();
        }
    }

    renderParagraphsList() {
        this.paragraphsList.innerHTML = '';
        this.paragraphs.forEach(p => {
            const card = document.createElement('div');
            card.className = `paragraph-item-card ${p.index === this.selectedIdx ? 'active' : ''}`;
            card.dataset.index = p.index;
            
            const cat = p.classification.category;
            const badgeClass = cat === 'Standard' ? 'badge-standard' : 
                               (cat.includes('Similarity') ? 'badge-similarity' : 
                               (cat.includes('Readability') ? 'badge-readability' : 
                               (cat.includes('Passive') ? 'badge-passive' : 'badge-grammar')));
            
            const isRevised = this.revisions[p.index] !== undefined;

            card.innerHTML = `
                <div class="para-header-meta">
                    <span class="para-index">Paragraph ${p.index + 1}</span>
                    <span class="issue-badge ${badgeClass}">${cat} ${isRevised ? '✓' : ''}</span>
                </div>
                <div class="para-preview-text">${this.escapeHTML(p.text)}</div>
            `;
            
            card.addEventListener('click', () => this.selectParagraph(p.index));
            this.paragraphsList.appendChild(card);
        });
    }

    selectParagraph(idx) {
        this.selectedIdx = idx;
        
        // Highlight active card
        const cards = this.paragraphsList.querySelectorAll('.paragraph-item-card');
        cards.forEach(c => {
            if (parseInt(c.dataset.index) === idx) {
                c.classList.add('active');
            } else {
                c.classList.remove('active');
            }
        });

        const p = this.paragraphs[idx];
        if (!p) return;

        const hasRevision = this.revisions[idx] !== undefined;
        const currentSuggestion = this.revisions[idx] || '';

        let planHtml = '';
        if (p.plan) {
            const issuesText = (p.plan.issues || p.plan.issues_found || []).join(', ');
            const strategyText = p.plan.strategy || p.plan.improvement_plan || '';
            const benefitText = p.plan.expected_benefit || p.plan.estimated_benefit || '';
            planHtml = `
                <div class="plan-info-banner">
                    <div class="plan-title">💡 Enhancement Strategy (${p.plan.priority} Priority)</div>
                    <div class="plan-desc">
                        <b>Issues:</b> ${issuesText}<br/>
                        <b>Strategy:</b> ${strategyText}<br/>
                        <b>Expected Benefit:</b> ${benefitText}
                    </div>
                </div>
            `;
        } else {
            planHtml = `
                <div class="plan-info-banner" style="background: rgba(255,255,255,0.03); border-color: rgba(255,255,255,0.08);">
                    <div class="plan-title" style="color: rgba(255,255,255,0.6);">✨ Standard Quality Paragraph</div>
                    <div class="plan-desc">No critical structural, clarity, or style defects detected.</div>
                </div>
            `;
        }

        this.editorPanel.innerHTML = `
            ${planHtml}
            
            <div class="revision-comparison-view">
                <div>
                    <div class="rev-pane-title">Original Text</div>
                    <div class="rev-pane-box" style="background: rgba(255,255,255,0.02);">${this.escapeHTML(p.text)}</div>
                </div>

                <div id="revised-container">
                    ${hasRevision ? this.getRevisionPaneHtml(idx, currentSuggestion) : this.getPreRevisionPaneHtml()}
                </div>
            </div>
        `;
    }

    getPreRevisionPaneHtml() {
        return `
            <div style="text-align: center; padding: 2rem; border: 1px dashed rgba(255,255,255,0.15); border-radius: 8px; margin-top: 1rem;">
                <p style="margin-bottom: 1rem; color: rgba(255,255,255,0.6);">Generate styled, clear revisions powered by Groq LLM.</p>
                <div class="editor-actions-bar" style="justify-content: center;">
                    <div class="focus-area-group">
                        <label style="font-size: 0.85rem; color: rgba(255,255,255,0.6);">Improvement Focus:</label>
                        <select class="focus-select" id="enh-focus-select">
                            <option value="all">Balanced</option>
                            <option value="clarity">Readability &amp; Clarity</option>
                            <option value="originality">Originality &amp; Citations</option>
                            <option value="tone">Academic Tone</option>
                        </select>
                    </div>
                    <button class="btn-enh" id="enh-btn-generate">Generate Suggestion</button>
                </div>
            </div>
        `;
    }

    getRevisionPaneHtml(idx, revisedText) {
        return `
            <div style="margin-top: 1rem;">
                <div class="rev-pane-title">AI Revised Suggestion</div>
                <textarea class="editor-textarea" id="enh-revised-textarea" rows="5">${this.escapeHTML(revisedText)}</textarea>
                
                <div id="diff-visualization" class="rev-pane-box" style="margin-top:0.5rem; display:none; max-height: 150px; overflow-y: auto;"></div>

                <div class="editor-actions-bar">
                    <div class="focus-area-group">
                        <button class="btn-enh btn-enh-outline" id="enh-btn-show-diff">Show Diff</button>
                        <button class="btn-enh btn-enh-outline" id="enh-btn-regenerate">Regenerate</button>
                    </div>
                    <div style="display:flex; gap:0.5rem;">
                        <button class="btn-enh btn-enh-outline" style="border-color:var(--enh-danger); color:#f87171;" id="enh-btn-discard">Discard</button>
                        <button class="btn-enh btn-enh-success" id="enh-btn-accept">Apply Edit</button>
                    </div>
                </div>
            </div>
        `;
    }

    // Attach event listeners inside the dynamically loaded container
    postRenderEditor() {
        const btnGen = document.getElementById('enh-btn-generate');
        const focusSelect = document.getElementById('enh-focus-select');
        if (btnGen) {
            btnGen.addEventListener('click', () => {
                const focus = focusSelect ? focusSelect.value : 'all';
                this.generateRevision(this.selectedIdx, focus);
            });
        }

        const btnAccept = document.getElementById('enh-btn-accept');
        if (btnAccept) {
            btnAccept.addEventListener('click', () => {
                const text = document.getElementById('enh-revised-textarea').value;
                this.acceptRevision(this.selectedIdx, text);
            });
        }

        const btnDiscard = document.getElementById('enh-btn-discard');
        if (btnDiscard) {
            btnDiscard.addEventListener('click', () => {
                this.discardRevision(this.selectedIdx);
            });
        }

        const btnRegen = document.getElementById('enh-btn-regenerate');
        if (btnRegen) {
            btnRegen.addEventListener('click', () => {
                this.generateRevision(this.selectedIdx, 'all');
            });
        }

        const btnShowDiff = document.getElementById('enh-btn-show-diff');
        const diffVis = document.getElementById('diff-visualization');
        if (btnShowDiff && diffVis) {
            btnShowDiff.addEventListener('click', async () => {
                if (diffVis.style.display === 'none') {
                    // Generate diff
                    const original = this.paragraphs[this.selectedIdx].text;
                    const revised = document.getElementById('enh-revised-textarea').value;
                    diffVis.innerHTML = '<div class="loading-spinner" style="scale: 0.5;"></div> Calculating diff...';
                    diffVis.style.display = 'block';
                    btnShowDiff.textContent = 'Hide Diff';
                    
                    try {
                        const response = await fetch(`/enhancement/diff?document_id=${this.documentId}&v_old=${this.currentVersion}`);
                        const data = await response.json();
                        // Call a lightweight client-side highlighter for the dynamic edit diff, or use a helper
                        diffVis.innerHTML = this.getDiffHtmlInline(original, revised);
                    } catch (e) {
                        diffVis.innerHTML = this.getDiffHtmlInline(original, revised);
                    }
                } else {
                    diffVis.style.display = 'none';
                    btnShowDiff.textContent = 'Show Diff';
                }
            });
        }
    }

    async generateRevision(idx, focusArea) {
        const container = document.getElementById('revised-container');
        if (container) {
            container.innerHTML = `
                <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.4);">
                    <div class="loading-spinner" style="margin: 0 auto 1rem;"></div>
                    Groq LLM is crafting high-integrity revisions...
                </div>
            `;
        }

        try {
            const response = await fetch('/enhancement/revise', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    document_id: this.documentId,
                    paragraph_indices: [idx],
                    focus_area: focusArea
                })
            });

            if (!response.ok) throw new Error('Revision failed');

            const data = await response.json();
            const revisedText = data.revisions[0]?.revised_text || this.paragraphs[idx].text;

            this.revisions[idx] = revisedText;
            
            // Re-render editor container
            if (idx === this.selectedIdx) {
                this.selectParagraph(idx);
                this.postRenderEditor();
            }
            this.renderParagraphsList();

        } catch (e) {
            console.error(e);
            if (container) {
                container.innerHTML = `<div style="color:var(--enh-danger); text-align:center; padding:1rem;">Error: ${e.message}</div>`;
            }
        }
    }

    async acceptRevision(idx, revisedText) {
        try {
            const response = await fetch('/enhancement/apply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    document_id: this.documentId,
                    accepted_revisions: [{ index: idx, revised_text: revisedText }]
                })
            });

            if (!response.ok) throw new Error('Failed to apply edit');

            const data = await response.json();
            this.currentVersion = data.new_version;
            this.metrics = data.metrics;
            
            // Update local paragraphs text
            this.paragraphs[idx].text = revisedText;
            // Clear current revisions cache since it is now applied to the new active version
            delete this.revisions[idx];

            // Re-render UI
            this.updateKPIs(this.metrics);
            
            // Re-fetch history versions list
            const versionsList = Array.from({ length: this.currentVersion }, (_, i) => i + 1);
            this.updateVersionDropdown(versionsList);
            
            this.renderParagraphsList();
            this.selectParagraph(idx);
            this.postRenderEditor();

        } catch (e) {
            alert(`Could not apply enhancement: ${e.message}`);
        }
    }

    discardRevision(idx) {
        delete this.revisions[idx];
        this.renderParagraphsList();
        this.selectParagraph(idx);
        this.postRenderEditor();
    }

    async handleVersionChange(vNum) {
        if (vNum === this.currentVersion) {
            this.btnRestore.style.display = 'none';
            return;
        }
        
        // Show restore button
        this.btnRestore.style.display = 'block';
        
        // Load version data (specifically the metrics and paragraphs)
        try {
            const response = await fetch(`/enhancement/report?document_id=${this.documentId}&version=${vNum}`);
            const data = await response.json();
            // Show metrics for this version
            this.updateKPIs(data.comparison); // Note: report returns comparison structure
        } catch (e) {
            console.error(e);
        }
    }

    async handleRestoreVersion() {
        const vNum = parseInt(this.versionSelect.value);
        if (!vNum || vNum === this.currentVersion) return;

        if (!confirm(`Are you sure you want to restore the document to Version ${vNum}? This will create a new Version ${this.currentVersion + 1}.`)) {
            return;
        }

        try {
            // Restore via re-applying or custom API handler (we can write it to /enhancement/apply or implement in routing)
            // But since our VersionManager handles restoration via manager, let's call a restore endpoint or implement it.
            // Wait, we can implement restoration by posting the paragraphs of vNum back to apply, or we can add a restore route.
            // Wait! In enhancement_routes.py we didn't add a restore POST route, but we can restore by fetching vNum and applying it.
            // Let's check: we can easily add a restore route, or call apply with vNum text!
            // Wait, since we have the manager, we can quickly implement `/enhancement/restore` or modify routes.
            // Let's implement /enhancement/restore in our router or simply let the user download that version.
            // Let's check: we can add a POST `/enhancement/restore` route! Let's do that in a second, it's very easy.
            const response = await fetch(`/enhancement/restore?document_id=${this.documentId}&version=${vNum}`, {
                method: 'POST'
            });

            if (!response.ok) throw new Error('Restore failed');
            
            const data = await response.json();
            this.currentVersion = data.new_version;
            this.metrics = data.metrics;
            this.paragraphs = data.paragraphs;
            
            // Re-render
            const versionsList = Array.from({ length: this.currentVersion }, (_, i) => i + 1);
            this.updateVersionDropdown(versionsList);
            this.renderParagraphsList();
            this.selectParagraph(0);
            this.setViewMode('editor');
            alert(`Document successfully restored to state at Version ${vNum}`);

        } catch (e) {
            alert(`Could not restore: ${e.message}`);
        }
    }

    async loadDiffView() {
        this.diffList.innerHTML = '<div style="padding: 3rem; text-align:center;"><div class="loading-spinner" style="margin:0 auto 1rem;"></div>Calculating paragraph diffs...</div>';
        
        try {
            const response = await fetch(`/enhancement/diff?document_id=${this.documentId}&v_old=1&v_new=${this.currentVersion}`);
            if (!response.ok) throw new Error('Could not fetch diff details');

            const data = await response.json();
            this.diffList.innerHTML = '';

            data.diffs.forEach((d, index) => {
                if (d.status === 'unchanged') return; // Only show changes to keep clean

                const div = document.createElement('div');
                div.className = `diff-p-card ${d.status}`;
                
                div.innerHTML = `
                    <div style="font-size:0.8rem; font-family:var(--font-display); font-weight:600; display:flex; justify-content:space-between;">
                        <span>Paragraph ${index + 1}</span>
                        <span style="text-transform:uppercase; color: ${d.status === 'added' ? '#34d399' : (d.status === 'deleted' ? '#f87171' : '#fbbf24')};">${d.status}</span>
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:0.5rem; font-size:0.9rem; line-height:1.5;">
                        <div style="color:rgba(255,255,255,0.5); text-decoration:line-through;">${this.escapeHTML(d.original || '')}</div>
                        <div>${d.html || this.escapeHTML(d.revised || '')}</div>
                    </div>
                `;
                this.diffList.appendChild(div);
            });

            if (this.diffList.children.length === 0) {
                this.diffList.innerHTML = `
                    <div style="text-align:center; padding:3rem; color:rgba(255,255,255,0.4);">
                        No changes made yet. Edit paragraphs in the editor to see version differences.
                    </div>
                `;
            }

        } catch (e) {
            this.diffList.innerHTML = `<div style="color:var(--enh-danger); padding:2rem; text-align:center;">Error loading diffs: ${e.message}</div>`;
        }
    }

    downloadDocx() {
        window.open(`/enhancement/download/enhanced-docx?document_id=${this.documentId}&version=${this.currentVersion}`);
    }

    downloadPdf() {
        window.open(`/enhancement/download/enhanced-pdf?document_id=${this.documentId}&version=${this.currentVersion}`);
    }

    downloadDiffReport() {
        window.open(`/enhancement/download/diff-report?document_id=${this.documentId}&v_old=1&v_new=${this.currentVersion}`);
    }

    getDiffHtmlInline(original, revised) {
        // Simple fallback token matcher for UI diff
        const oTokens = original.split(/\s+/);
        const rTokens = revised.split(/\s+/);
        
        let i = 0, j = 0;
        let html = [];
        
        while (i < oTokens.length || j < rTokens.length) {
            if (i < oTokens.length && j < rTokens.length && oTokens[i] === rTokens[j]) {
                html.push(oTokens[i]);
                i++; j++;
            } else {
                // simple greedy alignment search
                let foundMatch = false;
                for (let look = 1; look < 5; look++) {
                    if (j + look < rTokens.length && oTokens[i] === rTokens[j + look]) {
                        // tokens inserted
                        for (let k = 0; k < look; k++) {
                            html.push(`<ins class="diff-add">${rTokens[j+k]}</ins>`);
                        }
                        j += look;
                        foundMatch = true;
                        break;
                    }
                    if (i + look < oTokens.length && oTokens[i + look] === rTokens[j]) {
                        // tokens deleted
                        for (let k = 0; k < look; k++) {
                            html.push(`<del class="diff-del">${oTokens[i+k]}</del>`);
                        }
                        i += look;
                        foundMatch = true;
                        break;
                    }
                }
                
                if (!foundMatch) {
                    if (i < oTokens.length) {
                        html.push(`<del class="diff-del">${oTokens[i]}</del>`);
                        i++;
                    }
                    if (j < rTokens.length) {
                        html.push(`<ins class="diff-add">${rTokens[j]}</ins>`);
                        j++;
                    }
                }
            }
        }
        return html.join(' ');
    }

    escapeHTML(str) {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag] || tag)
        );
    }
}

// Global hook to mount tab selection
document.addEventListener('DOMContentLoaded', () => {
    window.enhancementManager = new DocumentEnhancementManager();
});
