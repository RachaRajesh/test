(function () {
  'use strict';

  /* ========================================================================
   *  API base — configurable, persisted in localStorage.
   *  Users can set their own backend URL (for Cloudflare Tunnel, ngrok,
   *  or a different port) via the server config in the topbar.
   * ====================================================================== */
  const LS_KEY = 'pixelai_api_base';
  const DEFAULT_API_BASE = 'http://localhost:8000';
  let API_BASE = (localStorage.getItem(LS_KEY) || DEFAULT_API_BASE).replace(/\/+$/, '');

  /* ── DOM refs ─────────────────────────────────────────────── */
  const baseCanvas    = document.getElementById('base-canvas');
  const overlayCanvas = document.getElementById('overlay-canvas');
  const baseCtx       = baseCanvas.getContext('2d');
  const dropZone      = document.getElementById('drop-zone');
  const fileInput     = document.getElementById('file-input');
  const canvasInfo    = document.getElementById('canvas-size-display');
  const cropOverlay   = document.getElementById('crop-overlay');
  const cropRect      = document.getElementById('crop-rect');
  const historyList   = document.getElementById('history-list');
  const toastEl       = document.getElementById('toast');
  const canvasContainer = document.getElementById('canvas-container');

  /* ── State ───────────────────────────────────────────────── */
  let imageLoaded = false;
  // baselineImage = the image as it was before the current adjustment session.
  // Adjustments re-render from this baseline. AI tool outputs UPDATE this baseline.
  let baselineImage = null;
  let imageW = 0, imageH = 0;
  let currentTool = 'select';
  let zoom = 1;

  // Crop state
  let cropDragging = false, cropActive = false;
  let cropStartX = 0, cropStartY = 0, cropEndX = 0, cropEndY = 0, cropRatio = 'free';

  // Adjustment sliders
  let adjBrightness = 0, adjContrast = 0, adjSaturation = 0, adjBlur = 0;

  // Undo/redo history
  let history = [], historyIdx = -1;
  const HISTORY_MAX = 20;

  // Before/After slider overlay
  let beforeAfterCanvas = null;

  function getPos(e) {
    const rect = overlayCanvas.getBoundingClientRect();
    return { x: (e.clientX - rect.left) / zoom, y: (e.clientY - rect.top) / zoom };
  }

  function init() {
    bindServerConfig();
    bindUpload();
    bindToolbar();
    bindCrop();
    bindAdjustments();
    bindAIMenu();
    bindKeyboard();
    bindCollapsibles();

    document.getElementById('save-btn').addEventListener('click', saveImage);
    document.getElementById('new-btn').addEventListener('click', newCanvas);
    document.getElementById('reset-adj-btn').addEventListener('click', resetAllToOriginal);
    document.getElementById('undo-btn').addEventListener('click', undo);
    document.getElementById('redo-btn').addEventListener('click', redo);

    // Wire the AI blur radius number display
    const blurSlider = document.getElementById('ai-blur-radius');
    const blurVal    = document.getElementById('ai-blur-radius-val');
    blurSlider.addEventListener('input', () => blurVal.textContent = blurSlider.value);

    // Probe server on boot
    updateServerStatusIndicator();
  }

  /* ====================================================================
   *  SERVER URL CONFIG
   * ================================================================== */
  function bindServerConfig() {
    const input  = document.getElementById('server-url-input');
    const button = document.getElementById('server-url-save');
    input.value = API_BASE;

    const save = async () => {
      let val = input.value.trim().replace(/\/+$/, '');
      if (!val) val = DEFAULT_API_BASE;
      if (!/^https?:\/\//i.test(val)) val = 'http://' + val;
      API_BASE = val;
      localStorage.setItem(LS_KEY, API_BASE);
      input.value = API_BASE;
      showToast('Checking server…', 'info');
      const ok = await checkServer();
      if (ok) showToast('Connected to ' + API_BASE, 'success');
      else    showToast('Could not reach ' + API_BASE, 'error');
      updateServerStatusIndicator();
    };

    button.addEventListener('click', save);
    input.addEventListener('keydown', e => { if (e.key === 'Enter') save(); });
  }

  async function updateServerStatusIndicator() {
    const el = document.getElementById('server-status-indicator');
    if (!el) return;
    el.textContent = 'Server: checking…';
    const ok = await checkServer();
    if (ok) {
      try {
        const res = await fetch(API_BASE + '/health', { signal: AbortSignal.timeout(3000) });
        if (res.ok) {
          const d = await res.json();
          el.textContent = d.cuda_available
            ? `GPU: ${d.device}`
            : 'Server online (CPU)';
        } else {
          el.textContent = 'Server online';
        }
      } catch {
        el.textContent = 'Server online';
      }
    } else {
      el.textContent = 'Server: offline';
    }
  }

  /* ====================================================================
   *  TOOLBAR + UPLOAD
   * ================================================================== */
  function bindToolbar() {
    document.querySelectorAll('.tool-btn[data-tool]').forEach(btn => {
      btn.addEventListener('click', () => setTool(btn.dataset.tool));
    });
  }

  function setTool(tool) {
    currentTool = tool;
    document.querySelectorAll('.tool-btn[data-tool]').forEach(b =>
      b.classList.toggle('active', b.dataset.tool === tool));
    if (tool === 'crop') {
      cropOverlay.classList.remove('hidden');
      overlayCanvas.style.cursor = 'crosshair';
      resetCropState();
    } else {
      cropOverlay.classList.add('hidden');
      overlayCanvas.style.cursor = 'default';
    }
  }

  function bindUpload() {
    fileInput.addEventListener('change', e => {
      if (e.target.files[0]) loadFile(e.target.files[0]);
    });
    const wrapper = document.getElementById('canvas-wrapper');
    wrapper.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    wrapper.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    wrapper.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f && f.type.startsWith('image/')) loadFile(f);
    });
  }

  function loadFile(file) {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setupCanvases(img.width, img.height);
      baseCtx.drawImage(img, 0, 0);
      baselineImage = baseCtx.getImageData(0, 0, imageW, imageH);
      imageLoaded = true;
      dropZone.classList.add('hidden');
      URL.revokeObjectURL(url);
      canvasInfo.textContent = img.width + ' × ' + img.height;
      resetAdjustments(/* suppressApply */ true);
      history = []; historyIdx = -1;
      pushHistory('Upload');
      showToast('Image loaded ✓', 'success');
      removeBeforeAfter();
    };
    img.onerror = () => showToast('Failed to load image', 'error');
    img.src = url;
  }

  function setupCanvases(w, h, preserveDisplay = false) {
    // When preserveDisplay is true, keep the on-screen pixel size the canvas
    // currently occupies and recompute zoom so the new image fills the same
    // visible area. This prevents AI tools from making the canvas visually
    // jump (especially upscale, where bitmap dimensions change drastically).
    let prevDispW = null, prevDispH = null;
    if (preserveDisplay && imageLoaded) {
      prevDispW = imageW * zoom;
      prevDispH = imageH * zoom;
    }

    imageW = w; imageH = h;

    if (preserveDisplay && prevDispW !== null) {
      // Fit the new bitmap into the same displayed box
      zoom = Math.min(prevDispW / w, prevDispH / h);
    } else {
      // Auto-fit zoom to viewport (normal behavior on upload/crop)
      const wrapper = document.getElementById('canvas-wrapper');
      const maxW = wrapper.clientWidth  - 48;
      const maxH = wrapper.clientHeight - 48;
      zoom = Math.min(1, maxW / w, maxH / h);
    }
    zoom = Math.round(zoom * 1000) / 1000;
    document.getElementById('zoom-display').textContent = Math.round(zoom * 100) + '%';

    [baseCanvas, overlayCanvas].forEach((c, i) => {
      c.width = w; c.height = h;
      c.style.display  = 'block';
      c.style.position = 'absolute';
      c.style.top = '0'; c.style.left = '0';
      c.style.width  = (w * zoom) + 'px';
      c.style.height = (h * zoom) + 'px';
      c.style.zIndex = String(i + 1);
    });
    overlayCanvas.style.zIndex = '10';
    canvasContainer.style.width    = (w * zoom) + 'px';
    canvasContainer.style.height   = (h * zoom) + 'px';
    canvasContainer.style.position = 'relative';
  }

  /* ====================================================================
   *  BEFORE / AFTER SLIDER
   * ================================================================== */
  function showBeforeAfter(beforeData, afterData) {
    removeBeforeAfter();

    const baseW = beforeData.width;
    const baseH = beforeData.height;
    const dispW = Math.round(baseW * zoom);
    const dispH = Math.round(baseH * zoom);

    function makeScaled(data) {
      const tmp = document.createElement('canvas');
      tmp.width = data.width; tmp.height = data.height;
      tmp.getContext('2d').putImageData(data, 0, 0);

      const out = document.createElement('canvas');
      out.width = dispW; out.height = dispH;
      const ctx = out.getContext('2d');
      ctx.clearRect(0, 0, dispW, dispH);
      ctx.drawImage(tmp, 0, 0, dispW, dispH);
      return out;
    }

    const bCanvas = makeScaled(beforeData);
    const aCanvas = makeScaled(afterData);

    const wrap = document.createElement('div');
    wrap.id = 'ba-wrap';
    wrap.style.cssText = `position:absolute;top:0;left:0;width:${dispW}px;height:${dispH}px;z-index:20;overflow:hidden;user-select:none;`;

    aCanvas.style.cssText = `position:absolute;top:0;left:0;width:${dispW}px;height:${dispH}px;`;
    wrap.appendChild(aCanvas);

    const beforeWrap = document.createElement('div');
    beforeWrap.id = 'ba-before-wrap';
    beforeWrap.style.cssText = 'position:absolute;top:0;left:0;width:50%;height:100%;overflow:hidden;';
    bCanvas.style.cssText = `position:absolute;top:0;left:0;width:${dispW}px;height:${dispH}px;`;
    beforeWrap.appendChild(bCanvas);
    wrap.appendChild(beforeWrap);

    const divider = document.createElement('div');
    divider.id = 'ba-divider';
    divider.style.cssText = 'position:absolute;top:0;left:50%;width:3px;height:100%;background:#fff;box-shadow:0 0 8px rgba(0,0,0,.5);transform:translateX(-50%);cursor:ew-resize;z-index:30;';

    const handle = document.createElement('div');
    handle.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:36px;height:36px;background:#fff;border-radius:50%;box-shadow:0 2px 12px rgba(0,0,0,.3);display:flex;align-items:center;justify-content:center;font-size:14px;color:#555;';
    handle.innerHTML = '◁▷';
    divider.appendChild(handle);

    const lblBefore = document.createElement('div');
    lblBefore.textContent = 'BEFORE';
    lblBefore.style.cssText = 'position:absolute;top:12px;left:12px;background:rgba(0,0,0,.6);color:#fff;font-size:11px;font-weight:700;padding:4px 10px;border-radius:20px;letter-spacing:1px;pointer-events:none;';
    const lblAfter = document.createElement('div');
    lblAfter.textContent = 'AFTER';
    lblAfter.style.cssText = 'position:absolute;top:12px;right:12px;background:rgba(124,58,237,.8);color:#fff;font-size:11px;font-weight:700;padding:4px 10px;border-radius:20px;letter-spacing:1px;pointer-events:none;';

    wrap.appendChild(divider);
    wrap.appendChild(lblBefore);
    wrap.appendChild(lblAfter);

    const closeBtn = document.createElement('button');
    closeBtn.textContent = '✕ Close comparison';
    closeBtn.style.cssText = 'position:absolute;bottom:14px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.7);color:#fff;border:none;padding:6px 16px;border-radius:20px;font-size:12px;cursor:pointer;z-index:40;';
    closeBtn.addEventListener('click', removeBeforeAfter);
    wrap.appendChild(closeBtn);

    let dragging = false;
    divider.addEventListener('mousedown', e => { dragging = true; e.preventDefault(); });
    document.addEventListener('mouseup', () => dragging = false);
    document.addEventListener('mousemove', e => {
      if (!dragging) return;
      const rect = wrap.getBoundingClientRect();
      const pct = Math.min(100, Math.max(0, (e.clientX - rect.left) / rect.width * 100));
      divider.style.left     = pct + '%';
      beforeWrap.style.width = pct + '%';
    });

    canvasContainer.appendChild(wrap);
    beforeAfterCanvas = wrap;
    showToast('Drag the slider to compare before/after', 'info');
  }

  function removeBeforeAfter() {
    const el = document.getElementById('ba-wrap');
    if (el) el.remove();
    beforeAfterCanvas = null;
  }

  /* ====================================================================
   *  CROP TOOL
   *  Fix: aspect ratio was using `s/rw*rh` which broke for non-square.
   *  Now we use proper width/height calculation from rw:rh.
   * ================================================================== */
  function bindCrop() {
    overlayCanvas.addEventListener('mousedown', e => {
      if (!imageLoaded || currentTool !== 'crop') return;
      const {x, y} = getPos(e);
      cropDragging = true; cropActive = true;
      cropStartX = x; cropStartY = y; cropEndX = x; cropEndY = y;
      updateCropVisual();
    });
    overlayCanvas.addEventListener('mousemove', e => {
      if (!cropDragging) return;
      const {x, y} = getPos(e);
      if (cropRatio === 'free') {
        cropEndX = x; cropEndY = y;
      } else {
        const [rw, rh] = cropRatio.split(':').map(Number);
        const dx = x - cropStartX;
        const dy = y - cropStartY;
        // Use whichever axis the user dragged more along, keep the ratio strictly.
        const absDx = Math.abs(dx);
        const absDy = Math.abs(dy);
        let w, h;
        if (absDx / rw >= absDy / rh) {
          w = absDx;
          h = w * rh / rw;
        } else {
          h = absDy;
          w = h * rw / rh;
        }
        cropEndX = cropStartX + (dx < 0 ? -w : w);
        cropEndY = cropStartY + (dy < 0 ? -h : h);
      }
      updateCropVisual();
    });
    overlayCanvas.addEventListener('mouseup', () => { cropDragging = false; });
    document.getElementById('crop-apply-btn').addEventListener('click', applyCrop);
    document.getElementById('crop-cancel-btn').addEventListener('click', () => {
      cropOverlay.classList.add('hidden');
      resetCropState();
      setTool('select');
    });
    document.querySelectorAll('.ratio-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.ratio-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        cropRatio = btn.dataset.ratio;
      });
    });
  }

  function updateCropVisual() {
    const x = Math.min(cropStartX, cropEndX);
    const y = Math.min(cropStartY, cropEndY);
    const w = Math.abs(cropEndX - cropStartX);
    const h = Math.abs(cropEndY - cropStartY);
    cropRect.style.left   = x * zoom + 'px';
    cropRect.style.top    = y * zoom + 'px';
    cropRect.style.width  = w * zoom + 'px';
    cropRect.style.height = h * zoom + 'px';
    const hs = cropOverlay.querySelectorAll('.crop-handle');
    if (hs.length === 4) {
      hs[0].style.cssText = `left:${x * zoom - 5}px;top:${y * zoom - 5}px;`;
      hs[1].style.cssText = `left:${x * zoom + w * zoom - 5}px;top:${y * zoom - 5}px;`;
      hs[2].style.cssText = `left:${x * zoom - 5}px;top:${y * zoom + h * zoom - 5}px;`;
      hs[3].style.cssText = `left:${x * zoom + w * zoom - 5}px;top:${y * zoom + h * zoom - 5}px;`;
    }
  }

  function applyCrop() {
    if (!cropActive || !imageLoaded) return;
    // Clamp to image bounds
    const x = Math.max(0, Math.round(Math.min(cropStartX, cropEndX)));
    const y = Math.max(0, Math.round(Math.min(cropStartY, cropEndY)));
    const w = Math.min(imageW - x, Math.round(Math.abs(cropEndX - cropStartX)));
    const h = Math.min(imageH - y, Math.round(Math.abs(cropEndY - cropStartY)));
    if (w < 5 || h < 5) { showToast('Select a larger area', 'error'); return; }
    const cropped = baseCtx.getImageData(x, y, w, h);
    setupCanvases(w, h);
    baseCtx.putImageData(cropped, 0, 0);
    baselineImage = baseCtx.getImageData(0, 0, w, h);
    canvasInfo.textContent = w + ' × ' + h;
    cropOverlay.classList.add('hidden');
    resetCropState();
    setTool('select');
    resetAdjustments(/* suppressApply */ true);
    pushHistory('Crop');
    showToast('Cropped to ' + w + '×' + h, 'success');
    removeBeforeAfter();
  }

  function resetCropState() {
    cropStartX = cropStartY = cropEndX = cropEndY = 0;
    cropActive = cropDragging = false;
    cropRect.style.width = cropRect.style.height = '0';
  }

  /* ====================================================================
   *  ADJUSTMENTS
   *  Adjustments re-apply from baselineImage, which is updated whenever
   *  a new image is loaded, cropped, or AI-tool-modified. This way sliders
   *  don't wipe AI results.
   * ================================================================== */
  function bindAdjustments() {
    [
      ['adj-brightness', 'adj-brightness-val', v => { adjBrightness = v; applyAdj(); }],
      ['adj-contrast',   'adj-contrast-val',   v => { adjContrast   = v; applyAdj(); }],
      ['adj-saturation', 'adj-saturation-val', v => { adjSaturation = v; applyAdj(); }],
      ['adj-blur',       'adj-blur-val',       v => { adjBlur       = v; applyAdj(); }],
    ].forEach(([id, vid, cb]) => {
      const s = document.getElementById(id);
      const v = document.getElementById(vid);
      s.addEventListener('input', () => { v.textContent = s.value; cb(+s.value); });
    });
  }

  function applyAdj() {
    if (!imageLoaded || !baselineImage) return;
    const src = baselineImage.data;
    const data = new Uint8ClampedArray(src);
    const bright = adjBrightness * 2.55;
    const con    = (adjContrast   + 100) / 100;
    const sat    = (adjSaturation + 100) / 100;
    for (let i = 0; i < data.length; i += 4) {
      let r = data[i] + bright, g = data[i+1] + bright, b = data[i+2] + bright;
      r = (r - 128) * con + 128;
      g = (g - 128) * con + 128;
      b = (b - 128) * con + 128;
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      r = gray + sat * (r - gray);
      g = gray + sat * (g - gray);
      b = gray + sat * (b - gray);
      data[i]   = Math.min(255, Math.max(0, r));
      data[i+1] = Math.min(255, Math.max(0, g));
      data[i+2] = Math.min(255, Math.max(0, b));
      // alpha passes through unchanged
    }
    baseCtx.putImageData(new ImageData(data, baselineImage.width, baselineImage.height), 0, 0);
    baseCanvas.style.filter = adjBlur > 0 ? `blur(${adjBlur * 0.5}px)` : '';
  }

  function resetAdjustments(suppressApply = false) {
    adjBrightness = adjContrast = adjSaturation = adjBlur = 0;
    ['adj-brightness', 'adj-contrast', 'adj-saturation', 'adj-blur'].forEach(id => {
      const el = document.getElementById(id); if (el) el.value = 0;
    });
    ['adj-brightness-val', 'adj-contrast-val', 'adj-saturation-val', 'adj-blur-val'].forEach(id => {
      const el = document.getElementById(id); if (el) el.textContent = 0;
    });
    baseCanvas.style.filter = '';
    if (!suppressApply && baselineImage && imageLoaded) {
      baseCtx.putImageData(baselineImage, 0, 0);
    }
  }

  // Full reset: jump to the first history entry (original upload) and clear
  // all adjustment sliders. This wipes any AI operations performed since upload.
  function resetAllToOriginal() {
    if (!imageLoaded || history.length === 0) {
      showToast('No image loaded', 'error');
      return;
    }
    if (historyIdx === 0 && adjBrightness === 0 && adjContrast === 0
        && adjSaturation === 0 && adjBlur === 0) {
      showToast('Already at original', 'info');
      return;
    }
    // Jump history pointer to the original upload
    historyIdx = 0;
    restore(0);  // restore already calls resetAdjustments(suppressApply=true)
    showToast('Reset to original ✓', 'success');
  }

  /* ====================================================================
   *  HISTORY (undo/redo)
   * ================================================================== */
  function pushHistory(label) {
    history = history.slice(0, historyIdx + 1);
    history.push({
      label,
      data: baseCtx.getImageData(0, 0, imageW, imageH),
      w: imageW, h: imageH,
    });
    if (history.length > HISTORY_MAX) history.shift();
    historyIdx = history.length - 1;
    renderHistory();
  }

  function undo() {
    if (historyIdx <= 0) { showToast('Nothing to undo', 'error'); return; }
    restore(--historyIdx);
  }
  function redo() {
    if (historyIdx >= history.length - 1) { showToast('Nothing to redo', 'error'); return; }
    restore(++historyIdx);
  }
  function restore(idx) {
    const s = history[idx]; if (!s) return;
    if (s.w !== imageW || s.h !== imageH) {
      setupCanvases(s.w, s.h);
    }
    baseCtx.putImageData(s.data, 0, 0);
    baselineImage = s.data;
    resetAdjustments(/* suppressApply */ true);
    removeBeforeAfter();
    renderHistory();
  }
  function renderHistory() {
    historyList.innerHTML = '';
    history.forEach((h, i) => {
      const li = document.createElement('li');
      li.className = 'history-item' + (i === historyIdx ? ' active' : '');
      li.textContent = h.label;
      li.addEventListener('click', () => { historyIdx = i; restore(i); });
      historyList.appendChild(li);
    });
    historyList.scrollTop = historyList.scrollHeight;
  }

  /* ====================================================================
   *  SAVE / NEW
   * ================================================================== */
  function saveImage() {
    if (!imageLoaded) { showToast('No image to save', 'error'); return; }
    const tmp = document.createElement('canvas');
    tmp.width = imageW; tmp.height = imageH;
    tmp.getContext('2d').drawImage(baseCanvas, 0, 0);
    const a = document.createElement('a');
    a.download = 'group3-edit.png';
    a.href = tmp.toDataURL('image/png');
    a.click();
    showToast('Saved ✓', 'success');
  }

  function newCanvas() {
    if (imageLoaded && !confirm('Start new canvas? Unsaved changes will be lost.')) return;
    baseCtx.clearRect(0, 0, imageW, imageH);
    [baseCanvas, overlayCanvas].forEach(c => c.style.display = 'none');
    dropZone.classList.remove('hidden');
    imageLoaded = false;
    baselineImage = null;
    history = []; historyIdx = -1;
    renderHistory();
    canvasInfo.textContent = 'No image';
    baseCanvas.style.filter = '';
    resetAdjustments(/* suppressApply */ true);
    removeBeforeAfter();
  }

  /* ====================================================================
   *  AI TOOLS
   * ================================================================== */
  function bindAIMenu() {
    const btn  = document.getElementById('ai-tools-btn');
    const menu = document.getElementById('ai-menu');
    const chev = document.getElementById('ai-chevron');

    btn.addEventListener('click', e => {
      e.stopPropagation();
      const open = !menu.classList.contains('hidden');
      menu.classList.toggle('hidden', open);
      chev.style.transform = open ? '' : 'rotate(180deg)';
    });
    document.addEventListener('click', () => {
      menu.classList.add('hidden');
      chev.style.transform = '';
    });

    // Any interactive element INSIDE the menu (like the upscale select)
    // should not close the menu when clicked.
    menu.addEventListener('click', e => e.stopPropagation());

    document.querySelectorAll('.ai-menu-item[data-tool]').forEach(item => {
      item.addEventListener('click', e => {
        // Clicks on the select inside the upscale row should not trigger the tool.
        if (e.target.tagName === 'SELECT' || e.target.tagName === 'OPTION') return;
        e.stopPropagation();
        menu.classList.add('hidden');
        chev.style.transform = '';
        if (!imageLoaded) { showToast('Upload an image first', 'error'); return; }
        runAITool(item.dataset.tool);
      });
    });
  }

  function getCanvasBlob() {
    return new Promise(resolve => {
      const tmp = document.createElement('canvas');
      tmp.width = imageW; tmp.height = imageH;
      tmp.getContext('2d').drawImage(baseCanvas, 0, 0);
      tmp.toBlob(resolve, 'image/png');
    });
  }

  async function checkServer() {
    try {
      const res = await fetch(API_BASE + '/', { signal: AbortSignal.timeout(4000) });
      return res.ok;
    } catch {
      return false;
    }
  }

  async function runAITool(tool) {
    const configs = {
      'classify':  { title: 'Analyzing photo…',      sub: 'Computing quality metrics' },
      'remove-bg': { title: 'Removing background…',  sub: 'Running segmentation model' },
      'blur-bg':   { title: 'Blurring background…',  sub: 'Running MiDaS depth estimation' },
      'upscale':   { title: 'Upscaling image…',       sub: 'Running Real-ESRGAN' },
    };
    const cfg = configs[tool];

    showAIOverlay('Connecting to server…', `Checking ${API_BASE}`);
    const online = await checkServer();
    if (!online) {
      hideAIOverlay();
      showServerOfflineModal();
      return;
    }

    showAIOverlay(cfg.title, cfg.sub);

    // Save before state for before/after slider
    const beforeData = baseCtx.getImageData(0, 0, imageW, imageH);

    try {
      const blob = await getCanvasBlob();
      const form = new FormData();
      form.append('file', blob, 'image.png');

      if (tool === 'upscale') {
        form.append('scale', document.getElementById('upscale-scale').value);
      }
      if (tool === 'blur-bg') {
        form.append('blur_radius', document.getElementById('ai-blur-radius').value);
      }

      const endpoints = {
        'classify':  '/api/classify/',
        'remove-bg': '/api/remove-bg/',
        'blur-bg':   '/api/blur-bg/',
        'upscale':   '/api/upscale/',
      };

      const res = await fetch(API_BASE + endpoints[tool], { method: 'POST', body: form });
      if (!res.ok) {
        let msg = 'Server returned ' + res.status;
        try {
          const err = await res.json();
          if (err && err.detail) msg += ': ' + err.detail;
        } catch {}
        throw new Error(msg);
      }

      hideAIOverlay();

      if (tool === 'classify') {
        const data = await res.json();
        showClassifyResult(data);
        return;
      }

      // Image result — parse blob and render
      const imgBlob = await res.blob();
      const url = URL.createObjectURL(imgBlob);
      const method = res.headers.get('X-Upscale-Method'); // realesrgan | lanczos | null

      const img = new Image();
      img.onerror = () => {
        URL.revokeObjectURL(url);
        showToast('Failed to load result image', 'error');
      };

      const prevW = imageW;
      const prevH = imageH;

      img.onload = () => {
        // Preserve the current zoom level across every AI tool so the canvas
        // doesn't visually jump. Canvas bitmap always matches the returned
        // image's native size (so upscale keeps its real 4x pixels).
        setupCanvases(img.width, img.height, /* preserveDisplay */ true);
        baseCtx.clearRect(0, 0, img.width, img.height);
        baseCtx.drawImage(img, 0, 0);
        const afterData = baseCtx.getImageData(0, 0, img.width, img.height);
        baselineImage = afterData;
        canvasInfo.textContent = `${img.width} × ${img.height}`;
        URL.revokeObjectURL(url);
        resetAdjustments(/* suppressApply */ true);

        // For the before/after slider, rescale beforeData to the new dimensions
        // if they differ (happens on upscale). This keeps the slider aligned.
        let beforeForSlider = beforeData;
        if (beforeData.width !== img.width || beforeData.height !== img.height) {
          const tmp = document.createElement('canvas');
          tmp.width = beforeData.width; tmp.height = beforeData.height;
          tmp.getContext('2d').putImageData(beforeData, 0, 0);
          const scaled = document.createElement('canvas');
          scaled.width = img.width; scaled.height = img.height;
          const sctx = scaled.getContext('2d');
          sctx.imageSmoothingEnabled = true;
          sctx.imageSmoothingQuality = 'high';
          sctx.drawImage(tmp, 0, 0, img.width, img.height);
          beforeForSlider = sctx.getImageData(0, 0, img.width, img.height);
        }

        const label = tool === 'upscale'
          ? (method === 'lanczos' ? 'Upscale (LANCZOS fallback)' : `Upscale ×${Math.round(img.width / prevW)}`)
          : cfg.title.replace('…', '');
        pushHistory(label);
        showBeforeAfter(beforeForSlider, afterData);

        if (tool === 'upscale' && method === 'lanczos') {
          showToast('Upscaled (LANCZOS fallback — ESRGAN not loaded on server)', 'error');
        } else {
          showToast(label + ' ✓', 'success');
        }
      };
      img.src = url;

    } catch (err) {
      hideAIOverlay();
      console.error('Error in runAITool:', err);
      showToast('Error: ' + err.message, 'error');
    }
  }

  /* ====================================================================
   *  MODALS + OVERLAYS
   * ================================================================== */
  function showServerOfflineModal() {
    let modal = document.getElementById('server-offline-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'server-offline-modal';
      modal.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:500;display:flex;align-items:center;justify-content:center;backdrop-filter:blur(4px)';
      modal.innerHTML = `
        <div style="background:#fff;border-radius:14px;padding:36px 40px;text-align:left;width:420px;max-width:92vw;box-shadow:0 20px 60px rgba(0,0,0,.3)">
          <div style="font-size:40px;margin-bottom:16px;text-align:center">🔌</div>
          <div style="font-size:18px;font-weight:700;color:#1a1f2e;margin-bottom:12px;text-align:center">Backend Not Reachable</div>
          <div style="font-size:13px;color:#5a6478;margin-bottom:16px;line-height:1.6">
            Cannot reach <code id="offline-url" style="background:#f0f2f5;padding:2px 6px;border-radius:4px"></code>.
            <br><br>
            <strong>Options:</strong>
            <ol style="margin:10px 0 10px 20px;padding:0;font-size:12px">
              <li>Start a local backend: <code style="background:#1a1f2e;color:#4ade80;padding:2px 6px;border-radius:4px;font-size:11px">uvicorn main:app --port 8000</code></li>
              <li>Connect via SSH tunnel to Viper: <code style="background:#1a1f2e;color:#a78bfa;padding:2px 6px;border-radius:4px;font-size:11px">ssh -L 8000:localhost:8000 rracha@viper.cs.kent.edu</code></li>
              <li>Use a Cloudflare Tunnel or ngrok URL and paste it into the server field in the top bar.</li>
            </ol>
          </div>
          <button id="offline-close-btn" style="background:#7c3aed;color:#fff;border:none;padding:10px 28px;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;display:block;margin:0 auto">Got it</button>
        </div>`;
      document.body.appendChild(modal);
      document.getElementById('offline-close-btn').addEventListener('click', () => {
        modal.style.display = 'none';
      });
    }
    document.getElementById('offline-url').textContent = API_BASE;
    modal.style.display = 'flex';
  }

  function showAIOverlay(title, sub) {
    let overlay = document.getElementById('ai-overlay');
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'ai-overlay';
      overlay.innerHTML = `
        <div id="ai-overlay-box">
          <div id="ai-overlay-spinner"><i class="fa-solid fa-wand-magic-sparkles"></i></div>
          <div id="ai-overlay-title"></div>
          <div id="ai-overlay-sub"></div>
          <div id="ai-overlay-bar"><div id="ai-overlay-fill"></div></div>
        </div>`;
      overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:500;display:flex;align-items:center;justify-content:center;backdrop-filter:blur(4px)';
      const box = overlay.querySelector('#ai-overlay-box');
      box.style.cssText = 'background:#fff;border-radius:14px;padding:40px 48px;text-align:center;width:320px;box-shadow:0 20px 60px rgba(0,0,0,.3)';
      overlay.querySelector('#ai-overlay-spinner').style.cssText = 'font-size:32px;color:#7c3aed;margin-bottom:16px;animation:spin 2s linear infinite';
      overlay.querySelector('#ai-overlay-title').style.cssText = 'font-size:17px;font-weight:600;color:#1a1f2e;margin-bottom:6px';
      overlay.querySelector('#ai-overlay-sub').style.cssText = 'font-size:12px;color:#8b949e;margin-bottom:20px';
      overlay.querySelector('#ai-overlay-bar').style.cssText = 'width:100%;height:4px;background:#eaecf0;border-radius:2px;overflow:hidden';
      overlay.querySelector('#ai-overlay-fill').style.cssText = 'height:100%;width:0%;background:linear-gradient(90deg,#6d28d9,#a78bfa);border-radius:2px;transition:width .1s';
      if (!document.getElementById('spin-style')) {
        const s = document.createElement('style');
        s.id = 'spin-style';
        s.textContent = '@keyframes spin{to{transform:rotate(360deg)}}';
        document.head.appendChild(s);
      }
      document.body.appendChild(overlay);
    }
    overlay.querySelector('#ai-overlay-title').textContent = title;
    overlay.querySelector('#ai-overlay-sub').textContent = sub;
    overlay.style.display = 'flex';
    let pct = 0;
    clearInterval(overlay._interval);
    overlay._interval = setInterval(() => {
      pct = Math.min(88, pct + Math.random() * 6);
      overlay.querySelector('#ai-overlay-fill').style.width = pct + '%';
    }, 400);
  }

  function hideAIOverlay() {
    const overlay = document.getElementById('ai-overlay');
    if (!overlay) return;
    clearInterval(overlay._interval);
    overlay.querySelector('#ai-overlay-fill').style.width = '100%';
    setTimeout(() => { overlay.style.display = 'none'; }, 300);
  }

  function showClassifyResult(data) {
    let modal = document.getElementById('classify-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'classify-modal';
      modal.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:500;display:flex;align-items:center;justify-content:center;backdrop-filter:blur(4px)';
      modal.innerHTML = `
        <div style="background:#fff;border-radius:14px;padding:32px 40px;text-align:center;width:340px;position:relative;box-shadow:0 20px 60px rgba(0,0,0,.3)">
          <button id="cm-close" style="position:absolute;top:12px;right:12px;background:#f0f2f5;border:none;border-radius:50%;width:28px;height:28px;cursor:pointer;font-size:13px">✕</button>
          <div style="font-size:16px;font-weight:600;margin-bottom:16px">📸 Photo Score</div>
          <div id="cm-score" style="font-size:52px;font-weight:700;color:#7c3aed;margin-bottom:8px"></div>
          <div id="cm-label" style="font-size:15px;font-weight:600;color:#00897b;margin-bottom:16px"></div>
          <div id="cm-tags" style="display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin-bottom:12px"></div>
          <div id="cm-metrics" style="font-size:11px;color:#8b949e;text-align:left;background:#f5f7fa;border-radius:8px;padding:10px;line-height:1.6"></div>
        </div>`;
      document.body.appendChild(modal);
      document.getElementById('cm-close').addEventListener('click', () => {
        modal.style.display = 'none';
      });
    }
    modal.querySelector('#cm-score').textContent = data.score + '/10';
    modal.querySelector('#cm-label').textContent = data.label;
    const tagsEl = modal.querySelector('#cm-tags');
    tagsEl.innerHTML = '';
    (data.tags || []).forEach(t => {
      const span = document.createElement('span');
      span.textContent = t.tag;
      span.style.cssText = `padding:3px 10px;border-radius:20px;font-size:11px;${t.type==='good'?'background:#d1fae5;color:#065f46':'background:#fee2e2;color:#991b1b'}`;
      tagsEl.appendChild(span);
    });
    const m = data.metrics || {};
    modal.querySelector('#cm-metrics').innerHTML =
      `Sharpness: <b>${m.sharpness}</b> &nbsp;|&nbsp; Brightness: <b>${m.brightness}</b><br>` +
      `Contrast: <b>${m.contrast}</b> &nbsp;|&nbsp; Noise: <b>${m.noise}</b>` +
      (m.has_face ? `<br>😊 ${m.face_count || 1} face(s) detected` : '');
    modal.style.display = 'flex';
  }

  /* ====================================================================
   *  KEYBOARD + UTILITIES
   * ================================================================== */
  function bindKeyboard() {
    document.addEventListener('keydown', e => {
      if (['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;
      if (!e.ctrlKey && !e.metaKey) {
        if (e.key === 'v') setTool('select');
        if (e.key === 'c') setTool('crop');
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); undo(); }
      if ((e.ctrlKey || e.metaKey) && e.key === 'y') { e.preventDefault(); redo(); }
      if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); saveImage(); }
      if (e.key === 'Escape') {
        cropOverlay.classList.add('hidden');
        setTool('select');
        removeBeforeAfter();
      }
    });
  }

  function bindCollapsibles() {
    document.querySelectorAll('.panel-title.collapsible').forEach(t => {
      t.addEventListener('click', () => {
        const el = document.getElementById(t.dataset.target);
        if (!el) return;
        const open = el.style.display !== 'none';
        el.style.display = open ? 'none' : '';
        const ic = t.querySelector('.fa-chevron-down');
        if (ic) ic.style.transform = open ? 'rotate(-90deg)' : '';
      });
    });
  }

  let toastT;
  function showToast(msg, type='info') {
    clearTimeout(toastT);
    toastEl.textContent = msg;
    toastEl.className = 'show ' + type;
    toastT = setTimeout(() => { toastEl.className = ''; }, 3500);
  }

  init();
})();
