/**
 * DocVis — Intelligent Document Vision System
 * Frontend Application & Interactive Canvas Engine
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- Application State ---
    const state = {
        currentSourceType: 'sample', // 'sample' or 'file'
        currentSampleId: 'invoice',
        currentFile: null,
        mode: 'enhanced_color',
        cannyLow: 50,
        cannyHigh: 150,
        activeStage: 'enhanced_result',
        
        // Image & Canvas state
        loadedImage: new Image(),
        imageLoaded: false,
        imageOriginalWidth: 0,
        imageOriginalHeight: 0,
        
        // 4 corner points in original image scale [TL, TR, BR, BL]
        cornersOriginal: [],
        cornersAutoDetected: true,
        activeCornerIndex: -1,
        isDraggingCorner: false,
        
        // Response data from backend CV pipeline
        lastResponse: null,
        
        // Comparison slider state
        sliderPercent: 50,
        isDraggingSlider: false
    };

    // --- DOM Elements ---
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const samplePills = document.getElementById('samplePills');
    const modeRadios = document.querySelectorAll('input[name="enhancementMode"]');
    const cannyLowInput = document.getElementById('cannyLow');
    const cannyHighInput = document.getElementById('cannyHigh');
    const cannyValDisplay = document.getElementById('cannyValDisplay');
    const processBtn = document.getElementById('processBtn');
    const applyCornersBtn = document.getElementById('applyCornersBtn');
    const resetBtn = document.getElementById('resetBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const toggleDebugBtn = document.getElementById('toggleDebugBtn');
    
    const canvas = document.getElementById('interactiveCanvas');
    const ctx = canvas.getContext('2d');
    const canvasContainer = document.getElementById('canvasContainer');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const detectionBadge = document.getElementById('detectionBadge');
    const manualNotice = document.getElementById('manualNotice');
    
    const stageTabs = document.querySelectorAll('.stage-tab');
    const stageImagePreview = document.getElementById('stageImagePreview');
    const stageCaption = document.getElementById('stageCaption');
    
    const viewInteractiveBtn = document.getElementById('viewInteractiveBtn');
    const viewComparisonBtn = document.getElementById('viewComparisonBtn');
    const comparisonContainer = document.getElementById('comparisonContainer');
    const compBeforeImg = document.getElementById('compBeforeImg');
    const compAfterImg = document.getElementById('compAfterImg');
    const compOverlay = document.getElementById('compOverlay');
    const compHandle = document.getElementById('compHandle');
    
    const metricInputRes = document.getElementById('metricInputRes');
    const metricOutputRes = document.getElementById('metricOutputRes');
    const metricTime = document.getElementById('metricTime');
    const metricArea = document.getElementById('metricArea');
    const cornerCoordsSummary = document.getElementById('cornerCoordsSummary');
    
    const debugDrawer = document.getElementById('debugDrawer');
    const drawerHeader = document.getElementById('drawerHeader');
    const matrixDisplay = document.getElementById('matrixDisplay');
    const timingBars = document.getElementById('timingBars');
    const contoursList = document.getElementById('contoursList');

    // --- Initialize ---
    initEventListeners();
    loadSampleDocument('invoice');

    function initEventListeners() {
        // Dropzone & File Input
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFileSelect(e.target.files[0]);
            }
        });

        // Sample Pills
        samplePills.addEventListener('click', (e) => {
            const btn = e.target.closest('.sample-btn');
            if (btn) {
                document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                loadSampleDocument(btn.dataset.sample);
            }
        });

        // Enhancement Mode Radios -> Immediately re-process & update output
        modeRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                state.mode = e.target.value;
                document.querySelectorAll('.radio-card').forEach(card => card.classList.remove('active'));
                e.target.closest('.radio-card').classList.add('active');
                const cornersToSend = (!state.cornersAutoDetected && state.cornersOriginal.length === 4) ? state.cornersOriginal : null;
                processDocument(cornersToSend);
            });
        });

        // Canny Sliders -> Enforce low < high
        cannyLowInput.addEventListener('input', () => {
            if (parseInt(cannyLowInput.value) >= parseInt(cannyHighInput.value)) {
                cannyHighInput.value = Math.min(255, parseInt(cannyLowInput.value) + 10);
            }
            updateCannyDisplay();
        });
        cannyHighInput.addEventListener('input', () => {
            if (parseInt(cannyHighInput.value) <= parseInt(cannyLowInput.value)) {
                cannyLowInput.value = Math.max(10, parseInt(cannyHighInput.value) - 10);
            }
            updateCannyDisplay();
        });
        cannyLowInput.addEventListener('change', () => processDocument(getEffectiveCorners()));
        cannyHighInput.addEventListener('change', () => processDocument(getEffectiveCorners()));

        // Buttons
        processBtn.addEventListener('click', () => processDocument(null)); // Fresh auto detect
        if (applyCornersBtn) {
            applyCornersBtn.addEventListener('click', () => {
                processDocument(state.cornersOriginal);
                applyCornersBtn.classList.add('hidden');
            });
        }
        resetBtn.addEventListener('click', () => resetCorners());
        downloadBtn.addEventListener('click', downloadResult);
        toggleDebugBtn.addEventListener('click', () => {
            debugDrawer.classList.toggle('collapsed');
            toggleDebugBtn.classList.toggle('active');
        });
        drawerHeader.addEventListener('click', () => debugDrawer.classList.toggle('collapsed'));

        // View Toggles
        viewInteractiveBtn.addEventListener('click', () => {
            viewInteractiveBtn.classList.add('active');
            viewComparisonBtn.classList.remove('active');
            canvasContainer.classList.remove('hidden');
            comparisonContainer.classList.add('hidden');
        });
        viewComparisonBtn.addEventListener('click', () => {
            viewComparisonBtn.classList.add('active');
            viewInteractiveBtn.classList.remove('active');
            canvasContainer.classList.add('hidden');
            comparisonContainer.classList.remove('hidden');
            updateComparisonSlider();
        });

        // Stage Tabs
        stageTabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                const target = e.currentTarget;
                stageTabs.forEach(t => t.classList.remove('active'));
                target.classList.add('active');
                state.activeStage = target.dataset.stage;
                updateStagePreview();
            });
        });

        // Canvas Dragging Handles
        canvas.addEventListener('mousedown', onCanvasMouseDown);
        canvas.addEventListener('mousemove', onCanvasMouseMove);
        window.addEventListener('mouseup', onCanvasMouseUp);
        
        canvas.addEventListener('touchstart', onCanvasTouchStart, { passive: false });
        canvas.addEventListener('touchmove', onCanvasTouchMove, { passive: false });
        window.addEventListener('touchend', onCanvasMouseUp);

        // Comparison Slider Dragging
        compHandle.addEventListener('mousedown', () => state.isDraggingSlider = true);
        window.addEventListener('mousemove', onSliderMouseMove);
        window.addEventListener('mouseup', () => state.isDraggingSlider = false);
        
        compHandle.addEventListener('touchstart', () => state.isDraggingSlider = true);
        window.addEventListener('touchmove', onSliderTouchMove);
        window.addEventListener('touchend', () => state.isDraggingSlider = false);

        // Responsive Resize
        window.addEventListener('resize', () => {
            if (state.imageLoaded) redrawCanvas();
        });
    }

    function getEffectiveCorners() {
        return (!state.cornersAutoDetected && state.cornersOriginal.length === 4) ? state.cornersOriginal : null;
    }

    function updateCannyDisplay() {
        state.cannyLow = parseInt(cannyLowInput.value, 10);
        state.cannyHigh = parseInt(cannyHighInput.value, 10);
        cannyValDisplay.textContent = `${state.cannyLow} / ${state.cannyHigh}`;
    }

    function handleFileSelect(file) {
        state.currentSourceType = 'file';
        state.currentFile = file;
        document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
        
        const reader = new FileReader();
        reader.onload = (e) => {
            state.loadedImage.onload = () => {
                state.imageLoaded = true;
                state.imageOriginalWidth = state.loadedImage.naturalWidth;
                state.imageOriginalHeight = state.loadedImage.naturalHeight;
                processDocument(null);
            };
            state.loadedImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    function loadSampleDocument(sampleId) {
        state.currentSourceType = 'sample';
        state.currentSampleId = sampleId;
        
        showLoading(true);
        fetch(`/api/scan-sample/${sampleId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({
                mode: state.mode,
                canny_low: state.cannyLow,
                canny_high: state.cannyHigh
            })
        })
        .then(res => res.json())
        .then(data => {
            showLoading(false);
            handleResponseData(data);
            
            state.loadedImage.onload = () => {
                state.imageLoaded = true;
                state.imageOriginalWidth = state.loadedImage.naturalWidth;
                state.imageOriginalHeight = state.loadedImage.naturalHeight;
                redrawCanvas();
            };
            state.loadedImage.src = data.stages.original;
        })
        .catch(err => {
            showLoading(false);
            console.error("Failed to load sample:", err);
        });
    }

    function processDocument(customCorners = null) {
        showLoading(true);
        const formData = new FormData();
        formData.append('mode', state.mode);
        formData.append('canny_low', state.cannyLow);
        formData.append('canny_high', state.cannyHigh);

        if (customCorners && customCorners.length === 4) {
            formData.append('manual_corners', JSON.stringify(customCorners));
        }

        let endpoint = '/api/scan';
        if (state.currentSourceType === 'file' && state.currentFile) {
            formData.append('file', state.currentFile);
        } else {
            endpoint = `/api/scan-sample/${state.currentSampleId}`;
        }

        fetch(endpoint, {
            method: 'POST',
            body: formData
        })
        .then(res => {
            if (!res.ok) throw new Error("Processing failed");
            return res.json();
        })
        .then(data => {
            showLoading(false);
            handleResponseData(data);
        })
        .catch(err => {
            showLoading(false);
            alert("Computer vision processing error: " + err.message);
        });
    }

    function handleResponseData(data) {
        state.lastResponse = data;
        
        // Update Corners & Auto Status
        state.cornersOriginal = data.cv_analysis.corners_original;
        state.cornersAutoDetected = data.metrics.is_auto_detected;
        
        const confPct = data.metrics.confidence_pct || 0;
        
        // Update Status Badges & Notices
        if (data.metrics.corner_selection_mode === 'auto') {
            detectionBadge.textContent = `Auto Detected (${confPct}%)`;
            detectionBadge.className = 'badge badge-success';
            manualNotice.classList.add('hidden');
        } else if (data.metrics.corner_selection_mode === 'manual') {
            detectionBadge.textContent = 'Manual Corners Applied';
            detectionBadge.className = 'badge badge-warning';
            manualNotice.classList.remove('hidden');
        } else {
            detectionBadge.textContent = `Uncertain Detection (${confPct}%) — Adjust Manually`;
            detectionBadge.className = 'badge badge-warning';
            manualNotice.classList.remove('hidden');
        }

        // Update Processing Statistics
        const origRes = data.metrics.original_resolution;
        const outRes = data.metrics.output_resolution;
        metricInputRes.textContent = `${origRes.width} x ${origRes.height} px`;
        metricOutputRes.textContent = `${outRes.width} x ${outRes.height} px`;
        metricTime.textContent = `${data.metrics.processing_time_ms} ms`;
        metricArea.textContent = `${data.metrics.document_area_percent}%`;

        // Update Corners Text Summary
        if (state.cornersOriginal.length === 4) {
            const [tl, tr, br, bl] = state.cornersOriginal;
            cornerCoordsSummary.textContent = `TL: (${Math.round(tl[0])},${Math.round(tl[1])}) | TR: (${Math.round(tr[0])},${Math.round(tr[1])}) | BR: (${Math.round(br[0])},${Math.round(br[1])}) | BL: (${Math.round(bl[0])},${Math.round(bl[1])})`;
        }

        // Update Stage Preview Image & Comparison View Images
        updateStagePreview();
        compBeforeImg.src = data.stages.original;
        compAfterImg.src = data.stages.enhanced_result;

        // Populate CV Debug Drawer Data
        renderCVDebugData(data);

        // Redraw Canvas Overlay
        redrawCanvas();
    }

    function renderCVDebugData(data) {
        // 1. Homography Matrix H
        const H = data.cv_analysis.homography_matrix;
        if (H && H.length === 3) {
            const formatted = H.map(row => 
                `[ ${row.map(val => (val >= 0 ? ' ' : '') + val.toFixed(4).padStart(10)).join(', ')} ]`
            ).join('\n');
            matrixDisplay.textContent = formatted;
        }

        // 2. Timing Breakdown
        const tb = data.metrics.timings_breakdown_ms;
        if (tb) {
            timingBars.innerHTML = `
                <div class="timing-row"><span>Preprocessing:</span><span>${tb.preprocessing_ms} ms</span></div>
                <div class="timing-row"><span>Edge & Detection:</span><span>${tb.edge_and_detection_ms} ms</span></div>
                <div class="timing-row"><span>Perspective Warp:</span><span>${tb.perspective_ms} ms</span></div>
                <div class="timing-row"><span>Enhancement (${state.mode}):</span><span>${tb.enhancement_ms} ms</span></div>
                <div class="timing-row" style="font-weight:600;color:#3b82f6;"><span>Total Execution:</span><span>${tb.total_ms} ms</span></div>
            `;
        }

        // 3. Top Contours Evaluated List
        const contours = data.cv_analysis.contours_evaluated || [];
        if (contours.length > 0) {
            contoursList.innerHTML = contours.map(c => `
                <div class="contour-item ${c.selected ? 'selected' : ''}">
                    <span>Contour #${c.index + 1} (${c.pass_source})</span>
                    <span>${c.area_percentage}% area | Conf ${c.confidence_pct}% ${c.selected ? '[SELECTED]' : ''}</span>
                </div>
            `).join('');
        } else {
            contoursList.innerHTML = '<div class="contour-item"><span>Manual / Fallback corners active</span></div>';
        }
    }

    function updateStagePreview() {
        if (!state.lastResponse) return;
        const b64 = state.lastResponse.stages[state.activeStage];
        if (b64) {
            stageImagePreview.src = b64;
            const names = {
                'original': 'Stage 1: Resized Original Input Photograph',
                'grayscale': 'Stage 2: Single Channel Grayscale Image',
                'edge_map': 'Stage 3: Canny & Morphological Edge Map',
                'detected_contour': 'Stage 4: Quadrilateral & Corner Localization',
                'perspective_corrected': 'Stage 5: Homography Perspective Unwarped Document',
                'enhanced_result': `Stage 6: Final Scan (${state.mode.toUpperCase()})`
            };
            stageCaption.textContent = names[state.activeStage] || state.activeStage;
        }
    }

    // --- Interactive Canvas Handle Editor ---
    function redrawCanvas() {
        if (!state.imageLoaded || !state.loadedImage.src) return;

        const containerW = canvasContainer.clientWidth || 600;
        const maxH = window.innerHeight - 170;

        const imgW = state.loadedImage.naturalWidth || 800;
        const imgH = state.loadedImage.naturalHeight || 600;
        const scale = Math.min(containerW / imgW, maxH / imgH);

        canvas.width = Math.round(imgW * scale);
        canvas.height = Math.round(imgH * scale);

        // Draw base image
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(state.loadedImage, 0, 0, canvas.width, canvas.height);

        if (state.cornersOriginal.length !== 4) return;

        // Convert corners from Original scale -> Canvas scale
        const canvasCorners = state.cornersOriginal.map(([x, y]) => [
            x * scale,
            y * scale
        ]);

        // Draw Polygon overlay
        ctx.beginPath();
        ctx.moveTo(canvasCorners[0][0], canvasCorners[0][1]);
        for (let i = 1; i < 4; i++) {
            ctx.lineTo(canvasCorners[i][0], canvasCorners[i][1]);
        }
        ctx.closePath();

        const strokeColor = state.cornersAutoDetected ? '#10b981' : '#f59e0b';
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.fillStyle = state.cornersAutoDetected ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)';
        ctx.fill();

        // Draw Corner Handles
        const labels = ["TL", "TR", "BR", "BL"];
        const handleColors = ['#ef4444', '#10b981', '#3b82f6', '#eab308'];

        canvasCorners.forEach(([cx, cy], i) => {
            ctx.beginPath();
            ctx.arc(cx, cy, 12, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.fill();

            ctx.beginPath();
            ctx.arc(cx, cy, 8, 0, 2 * Math.PI);
            ctx.fillStyle = handleColors[i];
            ctx.fill();

            ctx.font = '600 12px "Fira Code", monospace';
            ctx.fillStyle = '#ffffff';
            ctx.shadowColor = '#000000';
            ctx.shadowBlur = 4;
            ctx.fillText(`${labels[i]}`, cx + 14, cy + 4);
            ctx.shadowBlur = 0;
        });
    }

    function getCanvasCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return [clientX - rect.left, clientY - rect.top];
    }

    function findClosestCornerIndex(cx, cy) {
        if (state.cornersOriginal.length !== 4) return -1;
        const scale = canvas.width / state.loadedImage.naturalWidth;
        
        for (let i = 0; i < 4; i++) {
            const [ox, oy] = state.cornersOriginal[i];
            const handleX = ox * scale;
            const handleY = oy * scale;
            const dist = Math.hypot(cx - handleX, cy - handleY);
            if (dist <= 28) return i; // hit radius 28px
        }
        return -1;
    }

    function onCanvasMouseDown(e) {
        const [cx, cy] = getCanvasCoords(e);
        const idx = findClosestCornerIndex(cx, cy);
        if (idx !== -1) {
            state.activeCornerIndex = idx;
            state.isDraggingCorner = true;
            state.cornersAutoDetected = false;
            if (applyCornersBtn) applyCornersBtn.classList.remove('hidden');
        }
    }

    function onCanvasMouseMove(e) {
        const [cx, cy] = getCanvasCoords(e);

        if (state.isDraggingCorner && state.activeCornerIndex !== -1) {
            const scale = canvas.width / state.loadedImage.naturalWidth;
            const origX = Math.max(0, Math.min(state.imageOriginalWidth, cx / scale));
            const origY = Math.max(0, Math.min(state.imageOriginalHeight, cy / scale));

            state.cornersOriginal[state.activeCornerIndex] = [origX, origY];
            redrawCanvas();
        } else {
            const hoverIdx = findClosestCornerIndex(cx, cy);
            canvas.style.cursor = hoverIdx !== -1 ? 'grab' : 'crosshair';
        }
    }

    function onCanvasMouseUp() {
        if (state.isDraggingCorner) {
            state.isDraggingCorner = false;
            state.activeCornerIndex = -1;
            // Instantly re-process perspective & enhancement upon releasing handle drag
            processDocument(state.cornersOriginal);
        }
    }

    function onCanvasTouchStart(e) {
        e.preventDefault();
        onCanvasMouseDown(e);
    }

    function onCanvasTouchMove(e) {
        e.preventDefault();
        onCanvasMouseMove(e);
    }

    function resetCorners() {
        if (applyCornersBtn) applyCornersBtn.classList.add('hidden');
        processDocument(null); // Fresh auto detection
    }

    // --- Before / After Split Comparison Slider ---
    function onSliderMouseMove(e) {
        if (!state.isDraggingSlider) return;
        const rect = comparisonContainer.getBoundingClientRect();
        const offsetX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
        state.sliderPercent = (offsetX / rect.width) * 100;
        updateComparisonSlider();
    }

    function onSliderTouchMove(e) {
        if (!state.isDraggingSlider || !e.touches[0]) return;
        const rect = comparisonContainer.getBoundingClientRect();
        const offsetX = Math.max(0, Math.min(rect.width, e.touches[0].clientX - rect.left));
        state.sliderPercent = (offsetX / rect.width) * 100;
        updateComparisonSlider();
    }

    function updateComparisonSlider() {
        compOverlay.style.width = `${state.sliderPercent}%`;
        compHandle.style.left = `${state.sliderPercent}%`;
    }

    function downloadResult() {
        if (!state.lastResponse) return;
        const b64 = state.lastResponse.stages.enhanced_result;
        const a = document.createElement('a');
        a.href = b64;
        a.download = `DocVis_Scan_${state.mode}_${Date.now()}.jpg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    function showLoading(show) {
        if (show) loadingOverlay.classList.remove('hidden');
        else loadingOverlay.classList.add('hidden');
    }
});
