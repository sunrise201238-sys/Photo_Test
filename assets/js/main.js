const MAX_DIMENSION = 2048;

const metricsContainer = document.getElementById('metrics');
const originalCanvas = document.getElementById('original-canvas');
const improvedCanvas = document.getElementById('improved-canvas');
const originalMeta = document.getElementById('original-meta');
const improvedMeta = document.getElementById('improved-meta');
const downloadButton = document.getElementById('download-button');
const toggleGrid = document.getElementById('toggle-grid');
const resetButton = document.getElementById('reset-button');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadButton = document.getElementById('upload-button');
const loadingIndicator = document.getElementById('loading-indicator');
const metricTemplate = document.getElementById('metric-template');
const langToggleButtons = document.querySelectorAll('.lang-toggle button');
const analysisSummary = document.getElementById('analysis-summary');
const engineStatus = document.getElementById('engine-status');
const errorToast = document.getElementById('error-toast');

const fallbackDictionaries = {
  'en-US': {
    "app_title": "Photo Aesthetic Assistant",
    "app_tagline": "Private, instant composition feedback in your browser.",
    "analysis_heading": "Analyse & enhance instantly",
    "analysis_subheading": "Upload a single photo to receive automatic composition guidance and a refined suggestion.",
    "drop_instructions": "Drop an image here or click to choose a file.",
    "drop_hint": "JPG, PNG, HEIC up to 12 MP. Processed privately on-device.",
    "select_button": "Select Photo",
    "reset_button": "Reset",
    "original_title": "Original",
    "improved_title": "Improved Suggestion",
    "download_button": "Download Improved Image",
    "toggle_grid": "Show guides",
    "footer_note": "All processing happens locally in your browser. No uploads. No tracking.",
    "loading": "Processing photo…",
    "engine_loading": "Loading vision engine…",
    "engine_ready": "Vision engine ready",
    "engine_error": "Vision engine unavailable",
    "metric_main_subject": "Main subject position",
    "metric_horizon": "Horizon angle",
    "metric_rule_of_thirds": "Rule-of-thirds alignment",
    "metric_sharpness": "Sharpness variance",
    "metric_exposure": "Exposure",
    "metric_contrast": "Contrast",
    "metric_saturation": "Saturation",
    "metric_color_balance": "Color balance",
    "metric_foreground_background": "Foreground vs. background",
    "metric_subject_size": "Subject size",
    "metric_shadow_clipping": "Shadow clipping",
    "metric_highlight_clipping": "Highlight clipping",
    "metric_midtone_balance": "Midtone balance",
    "metric_color_cast_strength": "Color cast",
    "metric_color_cast_warm": "Warm",
    "metric_color_cast_cool": "Cool",
    "metric_leading_lines": "Leading lines",
    "metric_texture": "Texture energy",
    "metric_download_name": "improved-photo",
    "metric_feedback": "Suggestions",
    "analysis_summary_default": "Upload a photo to see composition notes tailored to your scene.",
    "summary_subject_missing": "<strong>Subject:</strong> No dominant subject detected — choose a clearer focal point.",
    "summary_subject_centered": "<strong>Subject:</strong> Nicely balanced near the thirds intersections.",
    "summary_subject_off_center": "<strong>Subject:</strong> Off-centre — cropping will guide the eye to a stronger point.",
    "summary_horizon_level": "<strong>Horizon:</strong> Already level and steady.",
    "summary_horizon_tilted": "<strong>Horizon:</strong> Tilted by {{angle}} — auto rotation applied.",
    "summary_shadow_clipped": "<strong>Light:</strong> Shadows were crushed — lifted to recover detail.",
    "summary_highlight_clipped": "<strong>Light:</strong> Highlights clipped; toned down to protect detail.",
    "summary_exposure_dark": "<strong>Light:</strong> Slightly underexposed; midtones lifted in the suggestion.",
    "summary_exposure_bright": "<strong>Light:</strong> Highlights are bright; toned down for balance.",
    "summary_exposure_balanced": "<strong>Light:</strong> Balanced exposure with healthy midtones.",
    "summary_balance_foreground": "<strong>Depth:</strong> Foreground dominates; background softened for separation.",
    "summary_balance_background": "<strong>Depth:</strong> Background detail is strong; foreground contrast improved.",
    "summary_balance_even": "<strong>Depth:</strong> Foreground and background feel evenly weighted.",
    "summary_sharpness_soft": "<strong>Texture:</strong> Edges read soft — stabilise capture for extra crispness.",
    "summary_color_warm": "<strong>Colour:</strong> Warm cast detected; neutralised for natural tones.",
    "summary_color_cool": "<strong>Colour:</strong> Cool tint spotted; warmed for lifelike skin and skies.",
    "summary_color_balanced": "<strong>Colour:</strong> Palette remains balanced after subtle toning.",
    "summary_leading_lines": "<strong>Flow:</strong> Leading lines draw the eye along a {{angle}} direction.",
    "tip_subject_title": "Find the subject",
    "tip_subject_text": "The assistant highlights the strongest contour to estimate your main subject position.",
    "tip_horizon_title": "Balance the horizon",
    "tip_horizon_text": "We detect dominant lines to level the scene and keep skies straight.",
    "tip_color_title": "Polish the tones",
    "tip_color_text": "Subtle exposure, contrast, and color tweaks keep the improved version natural.",
    "meta_dimensions": "{{width}}×{{height}} px",
    "error_processing": "Unable to process this file. Please try another image.",
    "error_dictionary": "Using built-in language defaults. Some translations may be missing.",
    "feedback_rotation": "Slightly rotate the horizon to level the scene.",
    "feedback_crop": "Consider cropping so the subject sits on a thirds intersection.",
    "feedback_exposure": "Brighten the midtones for better balance.",
    "feedback_contrast": "Increase contrast to emphasize depth.",
    "feedback_saturation": "Boost saturation slightly for richer color.",
    "feedback_sharpness": "Try increasing focus or reducing camera shake.",
    "feedback_balance": "Balance foreground and background elements for clarity.",
    "feedback_highlights": "Recover highlights to protect bright detail.",
    "feedback_shadows": "Lift shadow detail to avoid crushed blacks.",
    "feedback_local_contrast": "Enhance micro-contrast so textures feel more defined.",
    "feedback_vibrance": "Add vibrance for livelier hues without oversaturating.",
    "feedback_color_warm": "Cool down the warm colour cast for neutral tones.",
    "feedback_color_cool": "Warm up the cool colour shift to keep tones natural.",
    "feedback_leading_lines": "Strengthen leading lines or perspective cues to guide the eye.",
    "feedback_vignette": "Add a gentle vignette to spotlight the subject.",
    "feedback_good": "Great balance! Only minor refinements suggested."
  },
  'zh-TW': {
    "app_title": "影像美感助手",
    "app_tagline": "完全在瀏覽器內即時提供構圖建議，隱私零外洩。",
    "analysis_heading": "立即分析並優化",
    "analysis_subheading": "上傳單張照片，即可獲得自動構圖建議與優化版本。",
    "drop_instructions": "拖曳影像到此或點擊選擇檔案。",
    "drop_hint": "支援 JPG、PNG、HEIC，最多 1200 萬像素。所有處理皆在本機完成。",
    "select_button": "選擇照片",
    "reset_button": "重設",
    "original_title": "原始影像",
    "improved_title": "優化建議",
    "download_button": "下載優化影像",
    "toggle_grid": "顯示輔助線",
    "footer_note": "所有處理都在您的瀏覽器內進行，不需上傳、不會追蹤。",
    "loading": "影像分析中…",
    "engine_loading": "視覺引擎載入中…",
    "engine_ready": "視覺引擎就緒",
    "engine_error": "視覺引擎無法使用",
    "metric_main_subject": "主體位置",
    "metric_horizon": "地平線角度",
    "metric_rule_of_thirds": "三分構圖對齊",
    "metric_sharpness": "銳利度變異",
    "metric_exposure": "曝光",
    "metric_contrast": "對比",
    "metric_saturation": "飽和度",
    "metric_color_balance": "色彩平衡",
    "metric_foreground_background": "前景 / 背景比例",
    "metric_subject_size": "主體比例",
    "metric_shadow_clipping": "陰影裁切",
    "metric_highlight_clipping": "高光裁切",
    "metric_midtone_balance": "中間調平衡",
    "metric_color_cast_strength": "色偏",
    "metric_color_cast_warm": "偏暖",
    "metric_color_cast_cool": "偏冷",
    "metric_leading_lines": "引導線",
    "metric_texture": "紋理能量",
    "metric_download_name": "improved-photo",
    "metric_feedback": "建議",
    "analysis_summary_default": "上傳照片即可看到針對場景量身打造的構圖重點。",
    "summary_subject_missing": "<strong>主體：</strong> 未偵測到明顯主體，請選擇更清楚的焦點。",
    "summary_subject_centered": "<strong>主體：</strong> 已落在三分線附近，構圖平衡。",
    "summary_subject_off_center": "<strong>主體：</strong> 稍微偏離三分線，裁切後可更聚焦。",
    "summary_horizon_level": "<strong>地平線：</strong> 已經水平穩定。",
    "summary_horizon_tilted": "<strong>地平線：</strong> 傾斜 {{angle}}，已自動校正。",
    "summary_shadow_clipped": "<strong>光線：</strong> 陰影過暗，優化版本已拉回細節。",
    "summary_highlight_clipped": "<strong>光線：</strong> 高光爆掉，已壓低保留紋理。",
    "summary_exposure_dark": "<strong>光線：</strong> 稍微偏暗，優化版本提升了中間調。",
    "summary_exposure_bright": "<strong>光線：</strong> 高光較亮，已適度壓低。",
    "summary_exposure_balanced": "<strong>光線：</strong> 曝光均衡，中間調健康。",
    "summary_balance_foreground": "<strong>景深：</strong> 前景佔比高，已讓背景更柔和。",
    "summary_balance_background": "<strong>景深：</strong> 背景細節強烈，已提升前景對比。",
    "summary_balance_even": "<strong>景深：</strong> 前景與背景比例平衡。",
    "summary_sharpness_soft": "<strong>細節：</strong> 邊緣略軟，可嘗試穩定拍攝以提高銳利度。",
    "summary_color_warm": "<strong>色彩：</strong> 偵測到暖色色偏，已稍微冷卻讓色調自然。",
    "summary_color_cool": "<strong>色彩：</strong> 偵測到偏冷色調，已加暖讓膚色與天空更自然。",
    "summary_color_balanced": "<strong>色彩：</strong> 維持中性平衡並微調色調。",
    "summary_leading_lines": "<strong>視線：</strong> 引導線形成 {{angle}} 方向的動勢。",
    "tip_subject_title": "鎖定主體",
    "tip_subject_text": "透過強烈輪廓估計最有力的主體位置。",
    "tip_horizon_title": "維持水平",
    "tip_horizon_text": "偵測主要線條自動校正地平線。",
    "tip_color_title": "調整色調",
    "tip_color_text": "細緻的曝光與色彩調整讓畫面自然不失真。",
    "meta_dimensions": "{{width}}×{{height}} px",
    "error_processing": "此檔案無法處理，請改用其他影像。",
    "error_dictionary": "使用內建語系字串，部分翻譯可能缺少。",
    "feedback_rotation": "稍微旋轉地平線以保持水平。",
    "feedback_crop": "調整裁切讓主體落在三分線上。",
    "feedback_exposure": "提升中間調亮度讓畫面更平衡。",
    "feedback_contrast": "增加對比以強化景深層次。",
    "feedback_saturation": "些微提升飽和度讓色彩更鮮明。",
    "feedback_sharpness": "拍攝時保持穩定以獲得更銳利的影像。",
    "feedback_balance": "調整前景與背景比例讓構圖更清晰。",
    "feedback_highlights": "回收高光避免亮部失真。",
    "feedback_shadows": "提亮暗部保留陰影細節。",
    "feedback_local_contrast": "加強微對比讓紋理更立體。",
    "feedback_vibrance": "增加活力飽和讓色彩更生動。",
    "feedback_color_warm": "稍微降低暖色色偏保持自然色調。",
    "feedback_color_cool": "加入暖色減少偏冷色調。",
    "feedback_leading_lines": "強化引導線或透視感來帶動視線。",
    "feedback_vignette": "加上柔和暗角以聚焦主體。",
    "feedback_good": "整體表現優異，只需細部微調。"
  }
};

let dictionaries = cloneFallback();
let currentLang = 'en-US';
let currentMetrics = null;
let currentDownloadUrl = null;
let lastOriginalCanvas = null;
let lastImprovedCanvas = null;
let engineStatusState = 'loading';
let errorTimeoutId = null;
let dictionaryWarningShown = false;

const formatters = {
  percent: value => `${Math.round(value * 100)}%`,
  degrees: value => `${value.toFixed(1)}°`,
  score: value => value.toFixed(2),
  numeric: value => value.toFixed(1)
};

function clamp(value, min = 0, max = 255) {
  return Math.min(max, Math.max(min, value));
}

function cloneFallback() {
  return JSON.parse(JSON.stringify(fallbackDictionaries));
}

function updateStatusBadge() {
  if (!engineStatus) return;
  const dict = dictionaries[currentLang] || {};
  engineStatus.classList.remove('ready', 'error');
  let key = 'engine_loading';
  if (engineStatusState === 'ready') {
    key = 'engine_ready';
    engineStatus.classList.add('ready');
  } else if (engineStatusState === 'error') {
    key = 'engine_error';
    engineStatus.classList.add('error');
  }
  engineStatus.textContent = dict[key] || engineStatus.textContent;
}

function setLoading(isLoading) {
  loadingIndicator.hidden = !isLoading;
  dropZone.setAttribute('aria-busy', String(isLoading));
  dropZone.classList.toggle('processing', isLoading);
}

function setCanvasMeta(element, source) {
  if (!element) return;
  if (!source) {
    element.textContent = '';
    return;
  }
  const dict = dictionaries[currentLang] || {};
  const template = dict['meta_dimensions'] || '{{width}}×{{height}} px';
  element.textContent = template
    .replace('{{width}}', source.width)
    .replace('{{height}}', source.height);
}

function drawGuides(canvas, metrics, options = {}) {
  const { showGuides = true, includeAnnotations = true } = options;
  if (!showGuides) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.save();
  const scale = Math.min(w, h);
  const guideWidth = Math.max(2, scale * 0.005);
  ctx.lineWidth = guideWidth;
  ctx.strokeStyle = 'rgba(91, 192, 255, 0.85)';
  ctx.shadowColor = 'rgba(15, 23, 42, 0.85)';
  ctx.shadowBlur = guideWidth * 1.6;
  ctx.setLineDash([guideWidth * 3, guideWidth * 1.5]);
  for (let i = 1; i <= 2; i++) {
    const x = (w * i) / 3;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    const y = (h * i) / 3;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
    const markerRadius = Math.max(4, scale * 0.03);
    ctx.beginPath();
    ctx.fillStyle = 'rgba(14, 165, 233, 0.18)';
    ctx.strokeStyle = 'rgba(191, 219, 254, 0.9)';
    ctx.lineWidth = guideWidth * 0.9;
    ctx.arc(x, y, markerRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = 'rgba(91, 192, 255, 0.85)';
    ctx.lineWidth = guideWidth;
  }
  ctx.setLineDash([]);
  ctx.shadowBlur = guideWidth;
  if (includeAnnotations && metrics && metrics.subjectRect) {
    const scaleX = w / metrics.imageSize.width;
    const scaleY = h / metrics.imageSize.height;
    ctx.strokeStyle = 'rgba(244, 114, 182, 0.75)';
    ctx.lineWidth = Math.max(1.5, guideWidth * 0.9);
    ctx.shadowBlur = guideWidth * 1.2;
    ctx.strokeRect(
      metrics.subjectRect.x * scaleX,
      metrics.subjectRect.y * scaleY,
      metrics.subjectRect.width * scaleX,
      metrics.subjectRect.height * scaleY
    );
  }
  if (includeAnnotations && metrics && metrics.horizonLine) {
    const scaleX = w / metrics.imageSize.width;
    const scaleY = h / metrics.imageSize.height;
    ctx.strokeStyle = 'rgba(45, 212, 191, 0.9)';
    ctx.lineWidth = Math.max(2, guideWidth * 1.1);
    ctx.shadowBlur = guideWidth * 1.8;
    ctx.beginPath();
    ctx.moveTo(metrics.horizonLine[0].x * scaleX, metrics.horizonLine[0].y * scaleY);
    ctx.lineTo(metrics.horizonLine[1].x * scaleX, metrics.horizonLine[1].y * scaleY);
    ctx.stroke();
  }
  ctx.restore();
}

function showError(messageKey, fallback) {
  if (!errorToast) return;
  const dict = dictionaries[currentLang] || {};
  errorToast.textContent = dict[messageKey] || fallback || 'Something went wrong.';
  errorToast.hidden = false;
  clearTimeout(errorTimeoutId);
  errorTimeoutId = setTimeout(() => {
    errorToast.hidden = true;
  }, 4000);
}

function cleanupCanvases() {
  lastOriginalCanvas = null;
  lastImprovedCanvas = null;
}

function resetInterface() {
  cleanupCanvases();
  currentMetrics = null;
  metricsContainer.innerHTML = '';
  const dict = dictionaries[currentLang] || {};
  analysisSummary.innerHTML = dict['analysis_summary_default'] || '';
  const originalCtx = originalCanvas.getContext('2d');
  originalCtx.clearRect(0, 0, originalCanvas.width, originalCanvas.height);
  const improvedCtx = improvedCanvas.getContext('2d');
  improvedCtx.clearRect(0, 0, improvedCanvas.width, improvedCanvas.height);
  setCanvasMeta(originalMeta, null);
  setCanvasMeta(improvedMeta, null);
  downloadButton.disabled = true;
  revokeDownloadUrl();
}

function revokeDownloadUrl() {
  if (currentDownloadUrl) {
    URL.revokeObjectURL(currentDownloadUrl);
    currentDownloadUrl = null;
  }
}

async function prepareDownload(sourceCanvas) {
  revokeDownloadUrl();
  await new Promise(resolve => {
    sourceCanvas.toBlob(blob => {
      const dict = dictionaries[currentLang] || {};
      const name = dict['metric_download_name'] || 'improved-photo';
      if (blob) {
        currentDownloadUrl = URL.createObjectURL(blob);
        downloadButton.disabled = false;
        downloadButton.dataset.filename = `${name}.png`;
      } else {
        downloadButton.disabled = true;
      }
      resolve();
    }, 'image/png');
  });
}

function renderToCanvas(targetCanvas, sourceCanvas, options = {}) {
  targetCanvas.width = sourceCanvas.width;
  targetCanvas.height = sourceCanvas.height;
  const ctx = targetCanvas.getContext('2d');
  ctx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  ctx.drawImage(sourceCanvas, 0, 0);
  drawGuides(targetCanvas, options.metrics, {
    showGuides: options.showGuides,
    includeAnnotations: options.includeAnnotations
  });
}

function refreshCanvases() {
  if (!currentMetrics) return;
  if (lastOriginalCanvas) {
    renderToCanvas(originalCanvas, lastOriginalCanvas, {
      showGuides: toggleGrid.checked,
      metrics: currentMetrics,
      includeAnnotations: true
    });
    setCanvasMeta(originalMeta, lastOriginalCanvas);
  }
  if (lastImprovedCanvas) {
    renderToCanvas(improvedCanvas, lastImprovedCanvas, {
      showGuides: toggleGrid.checked,
      metrics: currentMetrics,
      includeAnnotations: false
    });
    setCanvasMeta(improvedMeta, lastImprovedCanvas);
  }
}

function translatePage() {
  const dict = dictionaries[currentLang] || {};
  document.documentElement.lang = currentLang;
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (dict[key]) {
      el.textContent = dict[key];
    }
  });
  langToggleButtons.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.lang === currentLang);
  });
  if (!currentMetrics) {
    analysisSummary.innerHTML = dict['analysis_summary_default'] || '';
  } else {
    renderMetrics(currentMetrics);
    updateAnalysisSummary(currentMetrics);
    setCanvasMeta(originalMeta, lastOriginalCanvas);
    setCanvasMeta(improvedMeta, lastImprovedCanvas);
  }
  updateStatusBadge();
}

function renderMetrics(metrics) {
  const dict = dictionaries[currentLang] || {};
  metricsContainer.innerHTML = '';
  const castDirection = metrics.colorCast.bias >= 0 ? (dict['metric_color_cast_warm'] || 'Warm') : (dict['metric_color_cast_cool'] || 'Cool');
  const castValue = `${castDirection} ${formatters.percent(Math.min(1, Math.abs(metrics.colorCast.strength)))}`;
  const entries = [
    ['metric_main_subject', `${formatters.percent(Math.max(0, 0.5 - Math.abs(metrics.subjectOffset.x)))} / ${formatters.percent(Math.max(0, 0.5 - Math.abs(metrics.subjectOffset.y)))}`],
    ['metric_horizon', formatters.degrees(metrics.horizonAngle)],
    ['metric_rule_of_thirds', formatters.score(metrics.ruleOfThirdsScore)],
    ['metric_sharpness', formatters.numeric(metrics.sharpnessVariance)],
    ['metric_exposure', formatters.numeric(metrics.exposure)],
    ['metric_contrast', formatters.numeric(metrics.contrast)],
    ['metric_saturation', formatters.numeric(metrics.saturation)],
    ['metric_color_balance', `${metrics.colorBalance.r.toFixed(0)} / ${metrics.colorBalance.g.toFixed(0)} / ${metrics.colorBalance.b.toFixed(0)}`],
    ['metric_foreground_background', formatters.score(metrics.foregroundBackground)],
    ['metric_subject_size', formatters.percent(metrics.subjectSize)],
    ['metric_shadow_clipping', formatters.percent(metrics.shadowClipping)],
    ['metric_highlight_clipping', formatters.percent(metrics.highlightClipping)],
    ['metric_midtone_balance', formatters.percent(metrics.midtoneBalance)],
    ['metric_color_cast_strength', castValue],
    ['metric_leading_lines', `${formatters.degrees(metrics.leadingLines.angle)} / ${formatters.percent(metrics.leadingLines.strength)}`],
    ['metric_texture', formatters.percent(metrics.textureStrength)]
  ];

  for (const [labelKey, value] of entries) {
    const fragment = metricTemplate.content.cloneNode(true);
    fragment.querySelector('.metric-label').textContent = dict[labelKey] || labelKey;
    fragment.querySelector('.metric-value').textContent = value;
    metricsContainer.appendChild(fragment);
  }

  const feedbackWrap = document.createElement('div');
  feedbackWrap.className = 'metric';
  const label = document.createElement('span');
  label.className = 'metric-label';
  label.textContent = dict['metric_feedback'] || 'Suggestions';
  const valueSpan = document.createElement('span');
  valueSpan.className = 'metric-value';
  valueSpan.innerHTML = metrics.feedback.map(key => dict[key] || key).join('<br>');
  feedbackWrap.append(label, valueSpan);
  metricsContainer.appendChild(feedbackWrap);
}

function updateAnalysisSummary(metrics) {
  const dict = dictionaries[currentLang] || {};
  if (!analysisSummary) return;
  analysisSummary.innerHTML = '';
  if (!metrics) {
    analysisSummary.textContent = dict['analysis_summary_default'] || '';
    return;
  }

  const segments = [];
  if (!metrics.subjectRect) {
    segments.push(dict['summary_subject_missing']);
  } else {
    const offset = Math.max(Math.abs(metrics.subjectOffset.x), Math.abs(metrics.subjectOffset.y));
    if (offset < 0.12) {
      segments.push(dict['summary_subject_centered']);
    } else {
      segments.push(dict['summary_subject_off_center']);
    }
  }

  if (Math.abs(metrics.horizonAngle) <= 1) {
    segments.push(dict['summary_horizon_level']);
  } else {
    const template = dict['summary_horizon_tilted'] || '';
    segments.push(template.replace('{{angle}}', formatters.degrees(Math.abs(metrics.horizonAngle))));
  }

  if (metrics.shadowClipping > 0.04) {
    segments.push(dict['summary_shadow_clipped']);
  } else if (metrics.highlightClipping > 0.04) {
    segments.push(dict['summary_highlight_clipped']);
  } else if (metrics.exposure < 110) {
    segments.push(dict['summary_exposure_dark']);
  } else if (metrics.exposure > 150) {
    segments.push(dict['summary_exposure_bright']);
  } else {
    segments.push(dict['summary_exposure_balanced']);
  }

  if (metrics.foregroundBackground > 1.2) {
    segments.push(dict['summary_balance_foreground']);
  } else if (metrics.foregroundBackground < 0.8) {
    segments.push(dict['summary_balance_background']);
  } else {
    segments.push(dict['summary_balance_even']);
  }

  if (metrics.sharpnessVariance < 120) {
    segments.push(dict['summary_sharpness_soft']);
  }

  if (metrics.colorCast.strength > 0.08) {
    const colorKey = metrics.colorCast.bias >= 0 ? 'summary_color_warm' : 'summary_color_cool';
    segments.push(dict[colorKey]);
  } else {
    segments.push(dict['summary_color_balanced']);
  }

  if (metrics.leadingLines.strength > 0.35) {
    const template = dict['summary_leading_lines'] || '';
    segments.push(template.replace('{{angle}}', formatters.degrees(Math.abs(metrics.leadingLines.angle))));
  }

  const items = segments.filter(Boolean);
  if (!items.length) {
    analysisSummary.textContent = dict['analysis_summary_default'] || '';
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const segment of items) {
    const wrapper = document.createElement('div');
    wrapper.className = 'summary-item';

    const temp = document.createElement('div');
    temp.innerHTML = segment;
    const strong = temp.querySelector('strong');

    if (strong) {
      const label = document.createElement('span');
      label.className = 'summary-label';
      label.textContent = strong.textContent.trim();
      const text = document.createElement('span');
      text.className = 'summary-text';
      strong.remove();
      const detail = temp.textContent.trim();
      text.textContent = detail;
      wrapper.append(label, text);
    } else {
      wrapper.classList.add('summary-item--single');
      const text = document.createElement('span');
      text.className = 'summary-text';
      const detail = temp.textContent.trim();
      text.textContent = detail;
      wrapper.append(text);
    }

    fragment.appendChild(wrapper);
  }

  analysisSummary.appendChild(fragment);
}

function scaleDimensions(width, height, maxSize) {
  if (Math.max(width, height) <= maxSize) {
    return { width, height };
  }
  const ratio = width / height;
  if (ratio > 1) {
    return { width: maxSize, height: Math.round(maxSize / ratio) };
  }
  return { width: Math.round(maxSize * ratio), height: maxSize };
}

function readFileAsArrayBuffer(file, length = 128 * 1024) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file.slice(0, length));
  });
}

async function getExifOrientation(file) {
  try {
    const buffer = await readFileAsArrayBuffer(file);
    const view = new DataView(buffer);
    if (view.getUint16(0, false) !== 0xffd8) {
      return 1;
    }
    let offset = 2;
    const length = view.byteLength;
    while (offset < length) {
      const marker = view.getUint16(offset, false);
      offset += 2;
      if (marker === 0xffe1) {
        const blockLength = view.getUint16(offset, false);
        offset += 2;
        if (view.getUint32(offset, false) !== 0x45786966) {
          break;
        }
        offset += 6;
        const little = view.getUint16(offset, false) === 0x4949;
        offset += view.getUint32(offset + 4, little);
        const tags = view.getUint16(offset, little);
        offset += 2;
        for (let i = 0; i < tags; i++) {
          const tagOffset = offset + i * 12;
          if (view.getUint16(tagOffset, little) === 0x0112) {
            return view.getUint16(tagOffset + 8, little);
          }
        }
      } else if ((marker & 0xff00) !== 0xff00) {
        break;
      } else {
        offset += view.getUint16(offset, false);
      }
    }
  } catch (error) {
    console.warn('Failed to read EXIF orientation', error);
  }
  return 1;
}

function orientImageSource(source, orientation) {
  const width = source.width || source.naturalWidth;
  const height = source.height || source.naturalHeight;
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  switch (orientation) {
    case 2:
      canvas.width = width;
      canvas.height = height;
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
      break;
    case 3:
      canvas.width = width;
      canvas.height = height;
      ctx.translate(width, height);
      ctx.rotate(Math.PI);
      break;
    case 4:
      canvas.width = width;
      canvas.height = height;
      ctx.translate(0, height);
      ctx.scale(1, -1);
      break;
    case 5:
      canvas.width = height;
      canvas.height = width;
      ctx.rotate(0.5 * Math.PI);
      ctx.scale(1, -1);
      break;
    case 6:
      canvas.width = height;
      canvas.height = width;
      ctx.rotate(0.5 * Math.PI);
      ctx.translate(0, -height);
      break;
    case 7:
      canvas.width = height;
      canvas.height = width;
      ctx.rotate(0.5 * Math.PI);
      ctx.translate(width, -height);
      ctx.scale(-1, 1);
      break;
    case 8:
      canvas.width = height;
      canvas.height = width;
      ctx.rotate(-0.5 * Math.PI);
      ctx.translate(-width, 0);
      break;
    default:
      canvas.width = width;
      canvas.height = height;
      break;
  }

  ctx.drawImage(source, 0, 0);
  return canvas;
}

function drawSourceToCanvas(source) {
  const width = source.width || source.naturalWidth;
  const height = source.height || source.naturalHeight;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(source, 0, 0);
  if (typeof source.close === 'function') {
    source.close();
  }
  return canvas;
}

function loadImageElement(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = reader.result;
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function readImageFile(file) {
  const orientation = await getExifOrientation(file);
  let source = null;
  let orientationHandled = false;

  if ('createImageBitmap' in window) {
    try {
      source = await createImageBitmap(file, { imageOrientation: 'from-image' });
      orientationHandled = true;
    } catch (error) {
      console.warn('createImageBitmap failed, falling back to <img>', error);
    }
  }

  if (!source) {
    source = await loadImageElement(file);
  }

  const baseCanvas = orientationHandled ? drawSourceToCanvas(source) : orientImageSource(source, orientation);
  const targetSize = scaleDimensions(baseCanvas.width, baseCanvas.height, MAX_DIMENSION);
  const resizedCanvas = document.createElement('canvas');
  resizedCanvas.width = targetSize.width;
  resizedCanvas.height = targetSize.height;
  const ctx = resizedCanvas.getContext('2d');
  ctx.drawImage(baseCanvas, 0, 0, targetSize.width, targetSize.height);
  const imageData = ctx.getImageData(0, 0, targetSize.width, targetSize.height);
  return { canvas: resizedCanvas, imageData };
}

function computeStatistics(values) {
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
  }
  const mean = sum / values.length;
  let varianceSum = 0;
  for (let i = 0; i < values.length; i++) {
    const diff = values[i] - mean;
    varianceSum += diff * diff;
  }
  return { mean, std: Math.sqrt(varianceSum / values.length) };
}

function samplePercentile(values, percentile) {
  const step = Math.max(1, Math.floor(values.length / 10000));
  const sample = [];
  for (let i = 0; i < values.length; i += step) {
    sample.push(values[i]);
  }
  sample.sort((a, b) => a - b);
  const index = Math.min(sample.length - 1, Math.max(0, Math.floor(percentile * (sample.length - 1))));
  return sample[index] || 0;
}

function computeMetrics(imageData) {
  const { width, height, data } = imageData;
  const pixelCount = width * height;
  const grayscale = new Float32Array(pixelCount);
  const gradX = new Float32Array(pixelCount);
  const gradY = new Float32Array(pixelCount);
  const gradient = new Float32Array(pixelCount);

  const colorBalance = { r: 0, g: 0, b: 0 };
  let shadowClipped = 0;
  let highlightClipped = 0;
  let midtoneSum = 0;
  let midtoneCount = 0;

  for (let i = 0; i < pixelCount; i++) {
    const idx = i * 4;
    const r = data[idx];
    const g = data[idx + 1];
    const b = data[idx + 2];
    colorBalance.r += r;
    colorBalance.g += g;
    colorBalance.b += b;
    const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    grayscale[i] = luminance;
    if (luminance < 12) shadowClipped++;
    if (luminance > 243) highlightClipped++;
    if (luminance > 64 && luminance < 200) {
      midtoneSum += luminance;
      midtoneCount++;
    }
  }

  colorBalance.r /= pixelCount;
  colorBalance.g /= pixelCount;
  colorBalance.b /= pixelCount;

  let gradientSum = 0;
  let orientationSumX = 0;
  let orientationSumY = 0;
  let orientationCount = 0;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const gx =
        -grayscale[idx - width - 1] - 2 * grayscale[idx - 1] - grayscale[idx + width - 1] +
        grayscale[idx - width + 1] + 2 * grayscale[idx + 1] + grayscale[idx + width + 1];
      const gy =
        -grayscale[idx - width - 1] - 2 * grayscale[idx - width] - grayscale[idx - width + 1] +
        grayscale[idx + width - 1] + 2 * grayscale[idx + width] + grayscale[idx + width + 1];
      gradX[idx] = gx;
      gradY[idx] = gy;
      gradient[idx] = Math.hypot(gx, gy);
      gradientSum += gradient[idx];
    }
  }

  const stats = computeStatistics(grayscale);
  const gradientThreshold = samplePercentile(gradient, 0.9);
  const horizonThreshold = samplePercentile(gradient, 0.85);

  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;
  let sumX = 0;
  let sumY = 0;
  let strongCount = 0;

  let horizonAngle = 0;
  let horizonWeight = 0;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const magnitude = gradient[idx];
      if (magnitude > gradientThreshold) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        sumX += x;
        sumY += y;
        strongCount++;
      }
      if (magnitude > horizonThreshold && y > height * 0.25 && y < height * 0.75) {
        const gx = gradX[idx];
        const gy = gradY[idx];
        const angle = ((Math.atan2(gy, gx) * 180) / Math.PI) + 90;
        const normalized = ((angle + 180) % 180) - 90;
        horizonAngle += normalized * magnitude;
        horizonWeight += magnitude;
        const orientation = Math.atan2(gy, gx);
        orientationSumX += Math.cos(2 * orientation) * magnitude;
        orientationSumY += Math.sin(2 * orientation) * magnitude;
        orientationCount += magnitude;
      }
    }
  }

  const metrics = {
    imageSize: { width, height },
    subjectRect: null,
    subjectCenter: { x: width / 2, y: height / 2 },
    subjectOffset: { x: 0, y: 0 },
    subjectSize: 0,
    horizonAngle: horizonWeight ? horizonAngle / horizonWeight : 0,
    horizonLine: null,
    ruleOfThirdsScore: 0,
    sharpnessVariance: 0,
    exposure: stats.mean,
    contrast: stats.std,
    saturation: 0,
    colorBalance,
    foregroundBackground: 0,
    shadowClipping: shadowClipped / pixelCount,
    highlightClipping: highlightClipped / pixelCount,
    midtoneBalance: midtoneCount ? midtoneSum / (midtoneCount * 255) : stats.mean / 255,
    colorCast: { bias: 0, warmBias: 0, coolBias: 0, strength: 0 },
    leadingLines: { angle: 0, strength: 0 },
    textureStrength: gradientSum / Math.max(1, pixelCount * 255),
    feedback: []
  };

  const avgColor = (colorBalance.r + colorBalance.g + colorBalance.b) / 3;
  const warmBias = colorBalance.r - avgColor;
  const coolBias = colorBalance.b - avgColor;
  metrics.colorCast = {
    bias: warmBias - coolBias,
    warmBias,
    coolBias,
    strength: Math.max(Math.abs(warmBias), Math.abs(coolBias)) / 255
  };

  if (orientationCount > 0) {
    const strength = Math.hypot(orientationSumX, orientationSumY) / orientationCount;
    const angle = (Math.atan2(orientationSumY, orientationSumX) / 2) * (180 / Math.PI);
    metrics.leadingLines = { angle, strength };
  }

  if (strongCount > 50) {
    const widthRect = maxX - minX;
    const heightRect = maxY - minY;
    metrics.subjectRect = {
      x: Math.max(0, minX - 4),
      y: Math.max(0, minY - 4),
      width: Math.min(width, widthRect + 8),
      height: Math.min(height, heightRect + 8)
    };
    metrics.subjectCenter = {
      x: sumX / strongCount,
      y: sumY / strongCount
    };
    metrics.subjectOffset = {
      x: metrics.subjectCenter.x / width - 0.5,
      y: metrics.subjectCenter.y / height - 0.5
    };
    metrics.subjectSize = (widthRect * heightRect) / (width * height);
  }

  const thirdsX = [width / 3, (2 * width) / 3];
  const thirdsY = [height / 3, (2 * height) / 3];
  const nearestX = Math.min(...thirdsX.map(x => Math.abs(metrics.subjectCenter.x - x)));
  const nearestY = Math.min(...thirdsY.map(y => Math.abs(metrics.subjectCenter.y - y)));
  metrics.ruleOfThirdsScore = 1 - (nearestX / width + nearestY / height);

  let sharpnessAccumulator = 0;
  for (let i = 0; i < gradient.length; i++) {
    sharpnessAccumulator += gradient[i] * gradient[i];
  }
  metrics.sharpnessVariance = sharpnessAccumulator / gradient.length;

  let saturationSum = 0;
  for (let i = 0; i < pixelCount; i++) {
    const idx = i * 4;
    const r = data[idx] / 255;
    const g = data[idx + 1] / 255;
    const b = data[idx + 2] / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    saturationSum += max - min;
  }
  metrics.saturation = (saturationSum / pixelCount) * 255;

  const half = Math.floor(height / 2);
  let topSum = 0;
  let bottomSum = 0;
  for (let y = 0; y < half; y++) {
    for (let x = 0; x < width; x++) {
      topSum += grayscale[y * width + x];
    }
  }
  for (let y = half; y < height; y++) {
    for (let x = 0; x < width; x++) {
      bottomSum += grayscale[y * width + x];
    }
  }
  const topMean = topSum / (half * width);
  const bottomMean = bottomSum / (Math.max(1, height - half) * width);
  metrics.foregroundBackground = bottomMean / Math.max(1, topMean);

  const center = { x: width / 2, y: height / 2 };
  const angleRad = (metrics.horizonAngle * Math.PI) / 180;
  const lengthLine = Math.max(width, height);
  const dx = Math.cos(angleRad) * lengthLine;
  const dy = Math.sin(angleRad) * lengthLine;
  metrics.horizonLine = [
    { x: center.x - dx / 2, y: center.y - dy / 2 },
    { x: center.x + dx / 2, y: center.y + dy / 2 }
  ];

  const feedback = new Set();
  if (Math.abs(metrics.horizonAngle) > 1.5) {
    feedback.add('feedback_rotation');
  }
  if (metrics.subjectRect && metrics.ruleOfThirdsScore < 0.6) {
    feedback.add('feedback_crop');
  }
  if (metrics.exposure < 110 && metrics.shadowClipping < 0.03) {
    feedback.add('feedback_exposure');
  }
  if (metrics.highlightClipping > 0.035 || metrics.exposure > 165) {
    feedback.add('feedback_highlights');
  }
  if (metrics.shadowClipping > 0.035) {
    feedback.add('feedback_shadows');
  }
  if (metrics.contrast < 45 || metrics.textureStrength < 0.08) {
    feedback.add('feedback_contrast');
    feedback.add('feedback_local_contrast');
  }
  if (metrics.saturation < 50) {
    feedback.add('feedback_saturation');
    feedback.add('feedback_vibrance');
  }
  if (metrics.sharpnessVariance < 120) {
    feedback.add('feedback_sharpness');
  }
  if (metrics.foregroundBackground < 0.8 || metrics.foregroundBackground > 1.2) {
    feedback.add('feedback_balance');
  }
  if (metrics.colorCast.strength > 0.08) {
    if (metrics.colorCast.bias >= 0) {
      feedback.add('feedback_color_warm');
    } else {
      feedback.add('feedback_color_cool');
    }
  }
  if (metrics.leadingLines.strength < 0.18 && metrics.subjectRect) {
    feedback.add('feedback_leading_lines');
  }
  if (metrics.subjectSize < 0.14) {
    feedback.add('feedback_vignette');
  }
  if (feedback.size === 0) {
    feedback.add('feedback_good');
  }
  metrics.feedback = Array.from(feedback);

  return metrics;
}

function improveImage(baseCanvas, metrics) {
  const width = baseCanvas.width;
  const height = baseCanvas.height;
  const crop = computeCropBox(width, height, metrics);

  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = crop.width;
  cropCanvas.height = crop.height;
  const cropCtx = cropCanvas.getContext('2d');
  cropCtx.drawImage(baseCanvas, crop.x, crop.y, crop.width, crop.height, 0, 0, crop.width, crop.height);

  const leveledAngle = clamp(metrics.horizonAngle, -18, 18);
  const rotation = (-leveledAngle * Math.PI) / 180;
  const rotatedCanvas = rotateCanvas(cropCanvas, rotation);

  const cropCenter = { x: crop.width / 2, y: crop.height / 2 };
  const focusRelative = {
    x: crop.focus.x - cropCenter.x,
    y: crop.focus.y - cropCenter.y
  };
  const rotatedFocus = {
    x: rotatedCanvas.width / 2 + focusRelative.x * Math.cos(rotation) - focusRelative.y * Math.sin(rotation),
    y: rotatedCanvas.height / 2 + focusRelative.x * Math.sin(rotation) + focusRelative.y * Math.cos(rotation)
  };

  const perspective = applySubtlePerspective(rotatedCanvas, metrics);
  const perspectiveFocus = {
    x: perspective.margin + rotatedFocus.x + perspective.skewX * rotatedFocus.y,
    y: perspective.margin + rotatedFocus.y + perspective.skewY * rotatedFocus.x
  };

  const finalCanvas = document.createElement('canvas');
  finalCanvas.width = crop.width;
  finalCanvas.height = crop.height;
  const finalCtx = finalCanvas.getContext('2d');

  const distortion = Math.max(Math.abs(perspective.skewX), Math.abs(perspective.skewY));
  const rotationInfluence = Math.abs(rotation);
  const zoomPadding = rotationInfluence * 0.9 + distortion * 1.35 + (metrics.subjectSize < 0.18 ? 0.18 : 0.1);
  const extraScale = 1 + Math.min(0.42, Math.max(0.12, zoomPadding + 0.04));
  const availableWidth = perspective.canvas.width;
  const availableHeight = perspective.canvas.height;
  const targetAspect = crop.width / crop.height;

  const contentBounds = computeContentBounds(perspective.canvas, 4);
  let sampleX = 0;
  let sampleY = 0;
  let sampleWidth = crop.width * extraScale;
  let sampleHeight = crop.height * extraScale;

  if (contentBounds) {
    const insetBaseX = Math.max(12, contentBounds.width * (0.08 + rotationInfluence * 0.16 + distortion * 0.22));
    const insetBaseY = Math.max(12, contentBounds.height * (0.08 + rotationInfluence * 0.16 + distortion * 0.22));

    let safeX = clamp(contentBounds.x + insetBaseX, 0, availableWidth - 1);
    let safeY = clamp(contentBounds.y + insetBaseY, 0, availableHeight - 1);
    let safeWidth = Math.max(1, contentBounds.width - insetBaseX * 2);
    let safeHeight = Math.max(1, contentBounds.height - insetBaseY * 2);

    if (safeX + safeWidth > availableWidth) {
      safeWidth = availableWidth - safeX;
    }
    if (safeY + safeHeight > availableHeight) {
      safeHeight = availableHeight - safeY;
    }

    const usableWidth = Math.max(1, safeWidth * 0.94);
    const usableHeight = Math.max(1, safeHeight * 0.94);

    let candidateWidth = Math.min(usableWidth, crop.width * extraScale * 0.94);
    let candidateHeight = candidateWidth / targetAspect;

    if (candidateHeight > usableHeight) {
      candidateHeight = Math.min(usableHeight, crop.height * extraScale * 0.94);
      candidateWidth = candidateHeight * targetAspect;
    }

    sampleWidth = Math.max(1, candidateWidth);
    sampleHeight = Math.max(1, candidateHeight);

    const halfWidth = sampleWidth / 2;
    const halfHeight = sampleHeight / 2;
    const minFocusX = safeX + halfWidth;
    const maxFocusX = safeX + usableWidth - halfWidth;
    const minFocusY = safeY + halfHeight;
    const maxFocusY = safeY + usableHeight - halfHeight;

    const focusX = usableWidth <= sampleWidth
      ? safeX + (usableWidth - sampleWidth) / 2 + halfWidth
      : clamp(perspectiveFocus.x, minFocusX, maxFocusX);
    const focusY = usableHeight <= sampleHeight
      ? safeY + (usableHeight - sampleHeight) / 2 + halfHeight
      : clamp(perspectiveFocus.y, minFocusY, maxFocusY);

    sampleX = clamp(focusX - halfWidth, safeX, safeX + usableWidth - sampleWidth);
    sampleY = clamp(focusY - halfHeight, safeY, safeY + usableHeight - sampleHeight);
  } else {
    const widthMargin = Math.max(0, (availableWidth - crop.width) / 2);
    const heightMargin = Math.max(0, (availableHeight - crop.height) / 2);
    const guardX = Math.min(
      availableWidth / 2 - 2,
      Math.max(16, widthMargin * 0.7 + rotationInfluence * availableWidth * 0.24 + distortion * availableWidth * 0.34)
    );
    const guardY = Math.min(
      availableHeight / 2 - 2,
      Math.max(16, heightMargin * 0.7 + rotationInfluence * availableHeight * 0.24 + distortion * availableHeight * 0.34)
    );
    const maxWidth = Math.max(crop.width, availableWidth - guardX * 2);
    const maxHeight = Math.max(crop.height, availableHeight - guardY * 2);
    sampleWidth = Math.min(maxWidth, crop.width * extraScale * 0.94);
    sampleHeight = Math.min(maxHeight, crop.height * extraScale * 0.94);
    sampleX = (availableWidth - sampleWidth) / 2;
    sampleY = (availableHeight - sampleHeight) / 2;
  }

  finalCtx.drawImage(
    perspective.canvas,
    sampleX,
    sampleY,
    sampleWidth,
    sampleHeight,
    0,
    0,
    crop.width,
    crop.height
  );

  const focusPoint = {
    x: clamp(((perspectiveFocus.x - sampleX) / sampleWidth) * crop.width, 0, crop.width),
    y: clamp(((perspectiveFocus.y - sampleY) / sampleHeight) * crop.height, 0, crop.height)
  };

  applyToneAndColorAdjustments(finalCanvas, metrics, focusPoint);
  const clarityAmount = metrics.textureStrength < 0.08 ? 0.038 : metrics.textureStrength < 0.12 ? 0.032 : 0.026;
  applyLocalContrast(finalCanvas, clarityAmount);
  const vignetteStrength = metrics.subjectSize < 0.12 ? 0.12 : metrics.subjectSize < 0.25 ? 0.08 : 0.05;
  applyVignette(finalCanvas, vignetteStrength, focusPoint);

  return fillCanvasGutters(finalCanvas);
}

function computeCropBox(width, height, metrics) {
  const aspectPreference = width >= height ? 3 / 2 : 4 / 5;
  const subjectMargin = metrics.subjectRect ? Math.max(0.1, 0.28 - metrics.subjectSize * 1.2) : 0.16;
  let cropWidth = Math.round(width * (1 - subjectMargin));
  let cropHeight = Math.round(height * (1 - subjectMargin));

  if (cropWidth / cropHeight > aspectPreference) {
    cropWidth = Math.round(cropHeight * aspectPreference);
  } else {
    cropHeight = Math.round(cropWidth / aspectPreference);
  }

  cropWidth = Math.min(cropWidth, width);
  cropHeight = Math.min(cropHeight, height);

  const baseX = metrics.subjectRect ? metrics.subjectCenter.x : width / 2;
  const baseY = metrics.subjectRect ? metrics.subjectCenter.y : height / 2;
  const horizontalBias = metrics.subjectRect
    ? metrics.subjectCenter.x < width / 2
      ? 0.32
      : 0.68
    : 0.5;
  const verticalBias = metrics.subjectRect
    ? metrics.subjectCenter.y < height / 2
      ? 0.36
      : 0.64
    : 0.5;
  const offsetInfluenceX = metrics.subjectOffset.x * width * 0.12;
  const offsetInfluenceY = metrics.subjectOffset.y * height * 0.12;
  const targetCenterX = baseX - cropWidth * (horizontalBias - 0.5) + offsetInfluenceX;
  const targetCenterY = baseY - cropHeight * (verticalBias - 0.5) + offsetInfluenceY;
  const alignmentStrengthX = metrics.subjectRect ? 0.7 : 0.45;
  const alignmentStrengthY = metrics.subjectRect ? 0.55 : 0.4;
  const blendedCenterX = baseX * (1 - alignmentStrengthX) + targetCenterX * alignmentStrengthX;
  const blendedCenterY = baseY * (1 - alignmentStrengthY) + targetCenterY * alignmentStrengthY;

  const centerX = clamp(blendedCenterX, cropWidth / 2, width - cropWidth / 2);
  const centerY = clamp(blendedCenterY, cropHeight / 2, height - cropHeight / 2);

  const x = Math.round(centerX - cropWidth / 2);
  const y = Math.round(centerY - cropHeight / 2);

  return {
    x,
    y,
    width: cropWidth,
    height: cropHeight,
    focus: {
      x: metrics.subjectRect ? metrics.subjectCenter.x - x : cropWidth / 2,
      y: metrics.subjectRect ? metrics.subjectCenter.y - y : cropHeight / 2
    }
  };
}

function rotateCanvas(sourceCanvas, rotation) {
  if (Math.abs(rotation) < 0.002) {
    const clone = document.createElement('canvas');
    clone.width = sourceCanvas.width;
    clone.height = sourceCanvas.height;
    clone.getContext('2d').drawImage(sourceCanvas, 0, 0);
    return clone;
  }
  const width = sourceCanvas.width;
  const height = sourceCanvas.height;
  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);
  const rotatedWidth = Math.round(Math.abs(cos) * width + Math.abs(sin) * height);
  const rotatedHeight = Math.round(Math.abs(sin) * width + Math.abs(cos) * height);
  const rotatedCanvas = document.createElement('canvas');
  rotatedCanvas.width = rotatedWidth;
  rotatedCanvas.height = rotatedHeight;
  const ctx = rotatedCanvas.getContext('2d');
  ctx.translate(rotatedWidth / 2, rotatedHeight / 2);
  ctx.rotate(rotation);
  ctx.drawImage(sourceCanvas, -width / 2, -height / 2);
  return rotatedCanvas;
}

function applySubtlePerspective(canvas, metrics) {
  const width = canvas.width;
  const height = canvas.height;
  const maxDimension = Math.max(width, height);
  const attitude = Math.max(Math.abs(metrics.subjectOffset.x), Math.abs(metrics.subjectOffset.y));
  const angleInfluence = Math.min(0.18, Math.abs(metrics.horizonAngle) * 0.01);
  const offsetInfluence = Math.min(0.12, attitude * 0.18);
  const marginRatio = clamp(0.08 + angleInfluence * 0.6 + offsetInfluence * 0.8, 0.08, 0.18);
  const margin = Math.round(maxDimension * marginRatio);
  const skewX = clamp(metrics.subjectOffset.x * 0.05 + metrics.horizonAngle * 0.0012, -0.12, 0.12);
  const skewY = clamp(metrics.subjectOffset.y * 0.045, -0.12, 0.12);
  const warpedCanvas = document.createElement('canvas');
  warpedCanvas.width = width + margin * 2;
  warpedCanvas.height = height + margin * 2;
  const ctx = warpedCanvas.getContext('2d');
  ctx.translate(margin, margin);
  ctx.transform(1, skewY, skewX, 1, 0, 0);
  ctx.drawImage(canvas, 0, 0);
  return { canvas: warpedCanvas, skewX, skewY, margin };
}

function toneCurve(value, options = {}) {
  const {
    shadowBoost = 0,
    highlightPull = 0,
    midtoneBias = 0,
    blackLift = 0,
    brightnessLift = 0
  } = options;
  let v = value;
  if (shadowBoost > 0 && v < 0.6) {
    const influence = (0.6 - v) / 0.6;
    v += influence * shadowBoost * 0.35;
  }
  if (blackLift > 0 && v < 0.4) {
    const influence = (0.4 - v) / 0.4;
    v += influence * blackLift * 0.55;
  }
  if (highlightPull > 0 && v > 0.6) {
    const influence = (v - 0.6) / 0.4;
    v -= influence * highlightPull * 0.5;
  }
  v += midtoneBias;
  v += brightnessLift;
  return Math.min(1, Math.max(0, v));
}

function computeContentBounds(canvas, step = 4) {
  const { width, height } = canvas;
  if (!width || !height) {
    return null;
  }
  const ctx = canvas.getContext('2d');
  const { data } = ctx.getImageData(0, 0, width, height);
  let minX = width;
  let maxX = -1;
  let minY = height;
  let maxY = -1;
  const alphaThreshold = 8;

  for (let y = 0; y < height; y += step) {
    const rowOffset = y * width * 4;
    for (let x = 0; x < width; x += step) {
      if (data[rowOffset + x * 4 + 3] > alphaThreshold) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < minX || maxY < minY) {
    return null;
  }

  const refine = bounds => {
    const startX = Math.max(0, bounds.x - step);
    const endX = Math.min(width - 1, bounds.x + bounds.width + step);
    const startY = Math.max(0, bounds.y - step);
    const endY = Math.min(height - 1, bounds.y + bounds.height + step);

    let top = startY;
    while (top < endY) {
      let hasPixel = false;
      for (let x = startX; x <= endX; x++) {
        if (data[(top * width + x) * 4 + 3] > alphaThreshold) {
          hasPixel = true;
          break;
        }
      }
      if (hasPixel) break;
      top++;
    }

    let bottom = endY;
    while (bottom > top) {
      let hasPixel = false;
      for (let x = startX; x <= endX; x++) {
        if (data[(bottom * width + x) * 4 + 3] > alphaThreshold) {
          hasPixel = true;
          break;
        }
      }
      if (hasPixel) break;
      bottom--;
    }

    let left = startX;
    while (left < endX) {
      let hasPixel = false;
      for (let y = top; y <= bottom; y++) {
        if (data[(y * width + left) * 4 + 3] > alphaThreshold) {
          hasPixel = true;
          break;
        }
      }
      if (hasPixel) break;
      left++;
    }

    let right = endX;
    while (right > left) {
      let hasPixel = false;
      for (let y = top; y <= bottom; y++) {
        if (data[(y * width + right) * 4 + 3] > alphaThreshold) {
          hasPixel = true;
          break;
        }
      }
      if (hasPixel) break;
      right--;
    }

    return {
      x: left,
      y: top,
      width: Math.max(1, right - left + 1),
      height: Math.max(1, bottom - top + 1)
    };
  };

  return refine({
    x: minX,
    y: minY,
    width: Math.max(1, maxX - minX + 1),
    height: Math.max(1, maxY - minY + 1)
  });
}

function fillCanvasGutters(canvas) {
  const bounds = computeContentBounds(canvas, 1);
  if (!bounds) {
    return canvas;
  }

  const tolerance = 1;
  if (
    bounds.x <= tolerance &&
    bounds.y <= tolerance &&
    bounds.width >= canvas.width - tolerance * 2 &&
    bounds.height >= canvas.height - tolerance * 2
  ) {
    return canvas;
  }

  const filled = document.createElement('canvas');
  filled.width = canvas.width;
  filled.height = canvas.height;
  const ctx = filled.getContext('2d');
  ctx.drawImage(
    canvas,
    bounds.x,
    bounds.y,
    bounds.width,
    bounds.height,
    0,
    0,
    filled.width,
    filled.height
  );
  return filled;
}

function applyToneAndColorAdjustments(canvas, metrics, focusPoint) {
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  const imageData = ctx.getImageData(0, 0, width, height);
  const { data } = imageData;
  const focus = focusPoint || { x: width / 2, y: height / 2 };
  const focusRadius = Math.min(
    Math.max(width, height),
    Math.min(width, height) * (metrics.subjectSize > 0 ? Math.min(0.58, Math.max(0.32, Math.sqrt(metrics.subjectSize) * 1.1)) : 0.42)
  );
  const focusLift = metrics.subjectSize < 0.16 ? 0.038 : 0.028;
  const brightnessLift = metrics.exposure < 120 ? 0.009 : metrics.exposure < 150 ? 0.007 : metrics.exposure < 180 ? 0.004 : 0.0025;
  const shadowBoost = 0.028 + (metrics.shadowClipping > 0.035 ? 0.022 : metrics.shadowClipping > 0.02 ? 0.014 : 0.008);
  const blackLift = 0.02 + (metrics.shadowClipping > 0.035 ? 0.018 : metrics.shadowClipping > 0.02 ? 0.012 : 0.006);
  const highlightPull = Math.min(0.026, metrics.highlightClipping > 0.035 ? 0.026 : metrics.highlightClipping > 0.02 ? 0.018 : 0.01);
  const midtoneBias = metrics.midtoneBalance < 0.48 ? 0.008 : metrics.midtoneBalance > 0.6 ? -0.006 : 0.004;
  const colorBias = metrics.colorCast.bias;
  const castStrength = Math.min(0.045, Math.abs(colorBias) / 900);
  let warmShift = colorBias >= 0 ? castStrength : 0;
  let coolShift = colorBias < 0 ? castStrength : 0;
  const naturalWarmth = colorBias <= 16 ? 0.0025 : 0.0015;
  const saturationTarget = metrics.subjectRect ? 118 : 110;
  const saturationDelta = metrics.saturation - saturationTarget;
  let vibrance = 0;
  if (saturationDelta < -18) {
    vibrance = Math.min(0.012, Math.max(0.004, (-saturationDelta) / 540));
  } else if (saturationDelta > 26) {
    vibrance = -Math.min(0.012, Math.max(0.004, saturationDelta / 620));
  } else if (Math.abs(saturationDelta) <= 12 && metrics.textureStrength < 0.1) {
    vibrance = 0.006;
  }
  const gamma = 1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      let r = data[idx];
      let g = data[idx + 1];
      let b = data[idx + 2];

      const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      const tone = luminance / 255;
      const mapped = toneCurve(tone, {
        shadowBoost,
        highlightPull,
        midtoneBias,
        blackLift,
        brightnessLift
      });
      const toneScale = tone > 0 ? mapped / tone : mapped;
      if (toneScale > 0) {
        r = clamp(r * toneScale);
        g = clamp(g * toneScale);
        b = clamp(b * toneScale);
      }

      if (brightnessLift > 0) {
        const lift = brightnessLift * 0.45;
        r = clamp(r + (255 - r) * lift);
        g = clamp(g + (255 - g) * lift);
        b = clamp(b + (255 - b) * lift);
      }

      if (blackLift > 0 && tone < 0.45) {
        const blackFactor = (0.45 - tone) / 0.45;
        const blackGain = blackLift * blackFactor * 14;
        r = clamp(r + blackGain);
        g = clamp(g + blackGain);
        b = clamp(b + blackGain);
      }

      const dx = x - focus.x;
      const dy = y - focus.y;
      const distance = Math.hypot(dx, dy);
      const focusInfluence = focusRadius ? Math.max(0, 1 - distance / focusRadius) : 0;
      if (focusInfluence > 0) {
        const lift = 1 + focusInfluence * focusLift;
        r = clamp(r * lift);
        g = clamp(g * lift);
        b = clamp(b * lift);
      }

      if (gamma !== 1) {
        r = clamp(Math.pow(r / 255, gamma) * 255);
        g = clamp(Math.pow(g / 255, gamma) * 255);
        b = clamp(Math.pow(b / 255, gamma) * 255);
      }

      if (warmShift > 0) {
        r = clamp(r - warmShift * 9);
        b = clamp(b + warmShift * 6);
      }
      if (coolShift > 0) {
        r = clamp(r + coolShift * 6);
        b = clamp(b - coolShift * 9);
      }

      const warmthInfluence = Math.max(0, naturalWarmth - warmShift * 0.3);
      if (warmthInfluence > 0) {
        r = clamp(r + warmthInfluence * 7);
        g = clamp(g + warmthInfluence * 3);
        b = clamp(b - warmthInfluence * 8);
      }

      if (vibrance !== 0) {
        const avg = (r + g + b) / 3;
        const saturationWeight = Math.min(
          1,
          (Math.abs(r - avg) + Math.abs(g - avg) + Math.abs(b - avg)) / 255
        );
        const primaryBoost = 1 + vibrance * saturationWeight;
        const secondaryBoost = 1 + vibrance * saturationWeight * 0.6;
        r = clamp(avg + (r - avg) * primaryBoost);
        g = clamp(avg + (g - avg) * secondaryBoost);
        b = clamp(avg + (b - avg) * primaryBoost);
      }

      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

function applyLocalContrast(canvas, amount = 0.03) {
  if (!amount || amount <= 0) return;
  const { width, height } = canvas;
  if (!width || !height) return;
  const ctx = canvas.getContext('2d');
  const original = ctx.getImageData(0, 0, width, height);
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = width;
  tempCanvas.height = height;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.filter = 'blur(2px)';
  tempCtx.drawImage(canvas, 0, 0, width, height);
  const blurred = tempCtx.getImageData(0, 0, width, height);
  const data = original.data;
  const blurData = blurred.data;
  for (let i = 0; i < data.length; i += 4) {
    data[i] = clamp(data[i] + (data[i] - blurData[i]) * amount);
    data[i + 1] = clamp(data[i + 1] + (data[i + 1] - blurData[i + 1]) * amount);
    data[i + 2] = clamp(data[i + 2] + (data[i + 2] - blurData[i + 2]) * amount);
  }
  ctx.putImageData(original, 0, 0);
}

function applyVignette(canvas, strength = 0.1, focusPoint) {
  if (!strength || strength <= 0) return;
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  const imageData = ctx.getImageData(0, 0, width, height);
  const { data } = imageData;
  const center = focusPoint || { x: width / 2, y: height / 2 };
  const maxDistance = Math.hypot(Math.max(center.x, width - center.x), Math.max(center.y, height - center.y));
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const dist = Math.hypot(x - center.x, y - center.y);
      const influence = Math.min(1, dist / maxDistance);
      const factor = 1 - strength * Math.pow(influence, 1.4);
      data[idx] = clamp(data[idx] * factor);
      data[idx + 1] = clamp(data[idx + 1] * factor);
      data[idx + 2] = clamp(data[idx + 2] * factor);
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

async function processFile(file) {
  setLoading(true);
  downloadButton.disabled = true;
  try {
    const { canvas, imageData } = await readImageFile(file);
    lastOriginalCanvas = canvas;
    currentMetrics = computeMetrics(imageData);
    renderMetrics(currentMetrics);
    updateAnalysisSummary(currentMetrics);

    renderToCanvas(originalCanvas, lastOriginalCanvas, {
      showGuides: toggleGrid.checked,
      metrics: currentMetrics,
      includeAnnotations: true
    });
    setCanvasMeta(originalMeta, lastOriginalCanvas);

    lastImprovedCanvas = improveImage(lastOriginalCanvas, currentMetrics);
    renderToCanvas(improvedCanvas, lastImprovedCanvas, {
      showGuides: toggleGrid.checked,
      metrics: currentMetrics,
      includeAnnotations: false
    });
    setCanvasMeta(improvedMeta, lastImprovedCanvas);

    await prepareDownload(lastImprovedCanvas);
  } catch (error) {
    console.error(error);
    showError('error_processing', 'Unable to process this file. Please try another image.');
  } finally {
    setLoading(false);
  }
}

function handleFileInput(event) {
  const input = event.target;
  if (!input || !input.files || input.files.length === 0) {
    return;
  }
  const file = input.files[0];
  if (file) {
    processFile(file);
  }
}

function handleDrop(event) {
  event.preventDefault();
  dropZone.classList.remove('dragover');
  const transfer = event.dataTransfer;
  if (!transfer || !transfer.files || transfer.files.length === 0) {
    return;
  }
  const file = transfer.files[0];
  if (file) {
    processFile(file);
  }
}

function handleDrag(event) {
  event.preventDefault();
  if (event.type === 'dragover') {
    dropZone.classList.add('dragover');
  } else {
    dropZone.classList.remove('dragover');
  }
}

function initEventListeners() {
  uploadButton.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', handleFileInput);
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      fileInput.click();
    }
  });
  dropZone.addEventListener('dragover', handleDrag);
  dropZone.addEventListener('dragleave', handleDrag);
  dropZone.addEventListener('drop', handleDrop);
  toggleGrid.addEventListener('change', refreshCanvases);
  resetButton.addEventListener('click', resetInterface);
  downloadButton.addEventListener('click', () => {
    if (!currentDownloadUrl) return;
    const a = document.createElement('a');
    a.href = currentDownloadUrl;
    a.download = downloadButton.dataset.filename || 'improved-photo.png';
    a.click();
  });
  langToggleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      if (btn.dataset.lang !== currentLang) {
        currentLang = btn.dataset.lang;
        translatePage();
        refreshCanvases();
      }
    });
  });
}

async function loadDictionaries() {
  const langs = Object.keys(fallbackDictionaries);
  await Promise.all(
    langs.map(async lang => {
      try {
        const response = await fetch(`translations/${lang}.json`, { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`Failed to load dictionary for ${lang}`);
        }
        const data = await response.json();
        dictionaries[lang] = { ...dictionaries[lang], ...data };
      } catch (error) {
        console.warn('Dictionary load failed', error);
        if (!dictionaryWarningShown) {
          showError('error_dictionary', 'Using built-in language defaults. Some translations may be missing.');
          dictionaryWarningShown = true;
        }
      }
    })
  );
  translatePage();
}

async function init() {
  dictionaries = cloneFallback();
  downloadButton.disabled = true;
  translatePage();
  initEventListeners();
  try {
    await loadDictionaries();
  } catch (error) {
    console.warn('Unable to refresh dictionaries', error);
  }
  engineStatusState = 'ready';
  updateStatusBadge();
}

init();

window.addEventListener('beforeunload', () => {
  cleanupCanvases();
  revokeDownloadUrl();
});
