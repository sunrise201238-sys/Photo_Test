import { AIInferenceController } from './ai_inference_controller.js';
import { CompositionEngine, clamp } from './composition_engine.js';
import { estimateSaliency } from './saliency_fallback.js';
import { estimateHorizon } from './horizon_detector.js';

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
const aiPanel = document.getElementById('ai-panel');
const aiStatusText = document.getElementById('ai-status-text');
const aiModeLabel = document.getElementById('ai-mode');
const aiCompositionScore = document.getElementById('ai-composition-score');
const aiAestheticScore = document.getElementById('ai-aesthetic-score');
const aiLatency = document.getElementById('ai-latency');
const aiSuggestions = document.getElementById('ai-suggestions');
const aiProcessingMode = document.getElementById('ai-processing-mode');

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
    "ai_panel_title": "AI composition insights",
    "ai_model_status_loading": "Model loading…",
    "ai_model_status_ready": "Model ready",
    "ai_model_status_rules": "Rules-only mode",
    "ai_model_status_error": "Model unavailable",
    "ai_model_status_cloud": "Cloud scoring active",
    "ai_status_offline": "Offline — using rule-based analysis",
    "ai_status_ready": "Local AI scoring active",
    "ai_status_rules": "Rule-based scoring active",
    "ai_status_cloud": "Cloud AI scoring active",
    "ai_mode_label": "Processing mode",
    "ai_mode_local": "Local",
    "ai_mode_cloud": "Cloud",
    "ai_mode_rules": "Rules only",
    "ai_composition_score": "Composition",
    "ai_aesthetic_score": "Aesthetic",
    "ai_suggestion_crop": "Crop {{width}}×{{height}} px",
    "ai_suggestion_rotation": "Rotate {{angle}}",
    "ai_suggestion_thirds": "Thirds alignment {{score}}",
    "ai_suggestion_saliency": "Subject confidence {{confidence}}",
    "ai_latency_label": "Latency",
    "ai_processing_mode": "Scoring",
    "ai_processing_local": "Local",
    "ai_processing_cloud": "Cloud",
    "ai_processing_rules": "Rules",
    "ai_unavailable": "AI model unavailable – falling back to rules",
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
    "ai_panel_title": "AI 構圖洞察",
    "ai_model_status_loading": "模型載入中…",
    "ai_model_status_ready": "模型就緒",
    "ai_model_status_rules": "僅使用規則模式",
    "ai_model_status_error": "模型無法使用",
    "ai_model_status_cloud": "雲端評分啟用",
    "ai_status_offline": "離線模式 — 使用規則分析",
    "ai_status_ready": "本地 AI 評分啟用",
    "ai_status_rules": "規則評分啟用",
    "ai_status_cloud": "雲端 AI 評分啟用",
    "ai_mode_label": "處理模式",
    "ai_mode_local": "本地",
    "ai_mode_cloud": "雲端",
    "ai_mode_rules": "規則",
    "ai_composition_score": "構圖",
    "ai_aesthetic_score": "美感",
    "ai_suggestion_crop": "裁切 {{width}}×{{height}} px",
    "ai_suggestion_rotation": "旋轉 {{angle}}",
    "ai_suggestion_thirds": "三分線對齊 {{score}}",
    "ai_suggestion_saliency": "主體信心 {{confidence}}",
    "ai_latency_label": "延遲",
    "ai_processing_mode": "評分",
    "ai_processing_local": "本地",
    "ai_processing_cloud": "雲端",
    "ai_processing_rules": "規則",
    "ai_unavailable": "AI 模型不可用，改用規則評估",
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
const compositionEngine = new CompositionEngine({ maxCandidates: 6 });
const aiController = new AIInferenceController({ modelVersion: 'v1.0.0', modelUrl: 'models/emo_aen_v1_quant.onnx' });
let bestCandidate = null;
let aiInferenceResults = null;
let aiStatusState = 'loading';

const formatters = {
  percent: value => `${Math.round(value * 100)}%`,
  degrees: value => `${value.toFixed(1)}°`,
  score: value => value.toFixed(2),
  numeric: value => value.toFixed(1)
};

function cloneFallback() {
  return JSON.parse(JSON.stringify(fallbackDictionaries));
}

function updateStatusBadge() {
  if (!engineStatus) return;
  const dict = dictionaries[currentLang] || {};
  engineStatus.classList.remove('ready', 'error');
  const keyMap = {
    loading: 'ai_model_status_loading',
    ready: 'ai_model_status_ready',
    local: 'ai_model_status_ready',
    cloud: 'ai_model_status_cloud',
    rules: 'ai_model_status_rules',
    error: 'ai_model_status_error'
  };
  let key = keyMap[engineStatusState] || 'ai_model_status_loading';
  if (engineStatusState === 'ready' || engineStatusState === 'local' || engineStatusState === 'cloud') {
    engineStatus.classList.add('ready');
  } else if (engineStatusState === 'error') {
    engineStatus.classList.add('error');
  }
  engineStatus.textContent = dict[key] || engineStatus.textContent;
}

function updateAiPanel(candidate, metrics) {
  if (!aiPanel) return;
  const dict = dictionaries[currentLang] || {};
  const status = aiController.getStatus ? aiController.getStatus() : { mode: 'rules', status: 'loading', latency: 0 };
  const statusKeyMap = {
    loading: 'ai_model_status_loading',
    ready: 'ai_status_ready',
    local: 'ai_status_ready',
    cloud: 'ai_status_cloud',
    rules: 'ai_status_rules',
    error: 'ai_unavailable'
  };
  const modeKeyMap = {
    cloud: 'ai_mode_cloud',
    local: 'ai_mode_local',
    ready: 'ai_mode_local',
    rules: 'ai_mode_rules',
    loading: 'ai_mode_rules',
    error: 'ai_mode_rules'
  };
  const processingKeyMap = {
    cloud: 'ai_processing_cloud',
    local: 'ai_processing_local',
    ready: 'ai_processing_local',
    rules: 'ai_processing_rules',
    loading: 'ai_processing_rules',
    error: 'ai_processing_rules'
  };
  if (aiStatusText) {
    const key = statusKeyMap[aiStatusState] || 'ai_status_offline';
    aiStatusText.textContent = dict[key] || dict['ai_status_offline'] || '';
  }
  if (aiModeLabel) {
    const modeKey = modeKeyMap[status.mode] || 'ai_mode_rules';
    aiModeLabel.textContent = dict[modeKey] || dict['ai_mode_rules'] || '';
  }
  if (aiProcessingMode) {
    const processingKey = processingKeyMap[status.mode] || 'ai_processing_rules';
    aiProcessingMode.textContent = dict[processingKey] || dict['ai_processing_rules'] || '';
  }
  if (aiCompositionScore) {
    aiCompositionScore.textContent = candidate ? formatters.score(candidate.compositionScore) : '—';
  }
  if (aiAestheticScore) {
    aiAestheticScore.textContent = candidate ? formatters.score(candidate.aestheticScore) : '—';
  }
  if (aiLatency) {
    const latency = status.latency ? Math.round(status.latency) : null;
    aiLatency.textContent = latency ? `${latency} ms` : '—';
  }
  if (aiSuggestions) {
    aiSuggestions.innerHTML = '';
    if (candidate && metrics) {
      const crop = candidate.crop || metrics.bestCrop;
      const rotationDegrees = typeof metrics.bestRotation === 'number'
        ? metrics.bestRotation
        : candidate.rotation ?? metrics.horizonAngle ?? 0;
      const thirdsScore = candidate.features?.ruleOfThirdsScore ?? metrics.ruleOfThirdsScore ?? 0;
      const saliencyConfidence = Math.min(
        1,
        Math.max(0, metrics.detectors?.saliency?.confidence ?? metrics.saliencyConfidence ?? 0)
      );
      const suggestions = [
        dict['ai_suggestion_crop']
          ? dict['ai_suggestion_crop']
              .replace('{{width}}', Math.round(crop.width))
              .replace('{{height}}', Math.round(crop.height))
          : `Crop ${Math.round(crop.width)}×${Math.round(crop.height)} px`,
        dict['ai_suggestion_rotation']
          ? dict['ai_suggestion_rotation'].replace(
              '{{angle}}',
              formatters.degrees(Math.abs(rotationDegrees))
            )
          : `Rotate ${formatters.degrees(Math.abs(rotationDegrees))}`,
        dict['ai_suggestion_thirds']
          ? dict['ai_suggestion_thirds'].replace(
              '{{score}}',
              formatters.score(thirdsScore)
            )
          : `Thirds ${formatters.score(thirdsScore)}`,
        dict['ai_suggestion_saliency']
          ? dict['ai_suggestion_saliency'].replace(
              '{{confidence}}',
              formatters.percent(saliencyConfidence)
            )
          : `Saliency ${formatters.percent(saliencyConfidence)}`
      ];
      for (const text of suggestions) {
        const li = document.createElement('li');
        li.textContent = text;
        aiSuggestions.appendChild(li);
      }
    }
  }
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
  const { showGuides = true, includeAnnotations = true, highlightCrop = false } = options;
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
  if (includeAnnotations && highlightCrop && metrics && metrics.bestCrop) {
    const scaleX = w / metrics.imageSize.width;
    const scaleY = h / metrics.imageSize.height;
    const crop = metrics.bestCrop;
    ctx.save();
    ctx.setLineDash([guideWidth * 4, guideWidth * 2]);
    ctx.lineWidth = Math.max(2, guideWidth * 1.4);
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.9)';
    ctx.strokeRect(crop.x * scaleX, crop.y * scaleY, crop.width * scaleX, crop.height * scaleY);
    ctx.restore();
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
  bestCandidate = null;
  aiInferenceResults = null;
  aiStatusState = 'loading';
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
  updateAiPanel(null, null);
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
    includeAnnotations: options.includeAnnotations,
    highlightCrop: options.highlightCrop
  });
}

function refreshCanvases() {
  if (!currentMetrics) return;
  if (lastOriginalCanvas) {
    renderToCanvas(originalCanvas, lastOriginalCanvas, {
      showGuides: toggleGrid.checked,
      metrics: currentMetrics,
      includeAnnotations: true,
      highlightCrop: true
    });
    setCanvasMeta(originalMeta, lastOriginalCanvas);
  }
  if (lastImprovedCanvas) {
    renderToCanvas(improvedCanvas, lastImprovedCanvas, {
      showGuides: toggleGrid.checked,
      metrics: currentMetrics,
      includeAnnotations: false,
      highlightCrop: false
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
  updateAiPanel(bestCandidate, currentMetrics);
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
  if (metrics.ai) {
    entries.unshift(['ai_aesthetic_score', formatters.score(metrics.ai.aesthetic || metrics.ai.composition)]);
    entries.unshift(['ai_composition_score', formatters.score(metrics.ai.composition)]);
  }

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

async function processFile(file) {
  setLoading(true);
  downloadButton.disabled = true;
  bestCandidate = null;
  aiInferenceResults = null;
  aiStatusState = 'loading';
  updateAiPanel(null, null);
  try {
    const { canvas, imageData } = await readImageFile(file);
    lastOriginalCanvas = canvas;
    const saliency = estimateSaliency(imageData);
    const horizon = estimateHorizon(imageData);
    const detectors = { saliency, horizon };
    const metrics = compositionEngine.analyse(imageData, detectors);
    currentMetrics = metrics;

    const candidates = compositionEngine.generateCandidates(canvas, metrics, detectors);
    await aiController.initialize();
    const scores = await aiController.scoreCandidates(candidates, {
      imageSize: metrics.imageSize,
      detectors
    });
    aiInferenceResults = scores;
    const evaluated = compositionEngine.evaluateCandidates(candidates, scores);
    bestCandidate = compositionEngine.selectBestCandidate(evaluated) || evaluated[0] || null;

    if (bestCandidate) {
      metrics.bestCrop = bestCandidate.crop;
      metrics.bestRotation = bestCandidate.rotation;
      metrics.ai = {
        composition: bestCandidate.compositionScore,
        aesthetic: bestCandidate.aestheticScore,
        mode: bestCandidate.mode
      };
    } else if (candidates.length) {
      metrics.bestCrop = candidates[0].crop;
      metrics.bestRotation = candidates[0].rotation;
    }

    const status = aiController.getStatus();
    aiStatusState = status.mode || status.status || 'rules';
    engineStatusState = status.mode || status.status || 'rules';
    updateStatusBadge();

    renderMetrics(metrics);
    updateAnalysisSummary(metrics);

    renderToCanvas(originalCanvas, lastOriginalCanvas, {
      showGuides: toggleGrid.checked,
      metrics,
      includeAnnotations: true,
      highlightCrop: true
    });
    setCanvasMeta(originalMeta, lastOriginalCanvas);

    if (bestCandidate) {
      const improved = compositionEngine.renderCandidate(lastOriginalCanvas, metrics, bestCandidate);
      lastImprovedCanvas = improved.canvas;
      metrics.bestCrop = improved.crop;
      metrics.bestRotation = (bestCandidate.rotation ?? metrics.horizonAngle) || 0;
    } else {
      lastImprovedCanvas = canvas;
    }

    renderToCanvas(improvedCanvas, lastImprovedCanvas, {
      showGuides: toggleGrid.checked,
      metrics,
      includeAnnotations: false,
      highlightCrop: false
    });
    setCanvasMeta(improvedMeta, lastImprovedCanvas);

    await prepareDownload(lastImprovedCanvas);
    updateAiPanel(bestCandidate, metrics);
  } catch (error) {
    console.error(error);
    aiStatusState = 'error';
    engineStatusState = 'error';
    updateStatusBadge();
    updateAiPanel(null, null);
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
  try {
    const status = await aiController.initialize();
    engineStatusState = status.mode || status.status || 'ready';
    aiStatusState = engineStatusState;
  } catch (error) {
    console.warn('AI controller failed to initialize', error);
    engineStatusState = 'rules';
    aiStatusState = 'rules';
  }
  updateStatusBadge();
  updateAiPanel(null, null);
  registerServiceWorker();
}

init();

window.addEventListener('beforeunload', () => {
  cleanupCanvases();
  revokeDownloadUrl();
});

function registerServiceWorker() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker
      .register('./service-worker.js')
      .catch(error => console.warn('Service worker registration failed', error));
  }
}
