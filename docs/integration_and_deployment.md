# Integration & Deployment Guide

## Overview
The front-end now embeds the Emo-AEN lightweight assessment model. The UI exposes a dedicated AI panel, crop overlays, and a rule-based fallback mode. Inference is performed in a Web Worker with optional cloud delegation. This document summarises how to integrate the system and deploy it in production.

## Directory Layout
- `assets/js/ai_inference_controller.js` – orchestrates model loading, inference, fallbacks, and worker lifecycle.
- `assets/js/composition_engine.js` – generates crop candidates, prepares inference features, and renders the final crop.
- `assets/js/worker_infer.js` – worker script that loads the ONNX/TF.js model (when available) and evaluates feature batches.
- `assets/js/horizon_detector.js` / `assets/js/saliency_fallback.js` – lightweight horizon and saliency estimators used prior to inference.
- `assets/css/ai_panel.css` – styles for the AI summary card and overlays.
- `models/emo_aen_v2_int8.onnx` – placeholder for the quantised model artefact (replace with the trained export).
- `service-worker.js` – caches core assets and the model for offline operation.

## Browser Integration Steps
1. **Model asset** – place the quantised ONNX (or TF.js graph) file under `models/`. Update the `AIInferenceController` constructor options in `main.js` if the filename changes.
2. **Translations** – ensure new i18n keys are filled in `translations/en-US.json` and `translations/zh-TW.json`. The fallback dictionary in `main.js` must contain the same keys.
3. **Service worker** – register the worker (already handled in `main.js`). On deployment, serve the site over HTTPS so the worker and model cache function correctly.
4. **CORS** – when hosting the model on a CDN or separate domain, configure CORS headers (`Access-Control-Allow-Origin`) to permit fetches from the web app. Update `PRECACHE_URLS` in `service-worker.js` if paths change.
5. **Optional backend** – set `backendEndpoint` when instantiating `AIInferenceController` (e.g. via an environment-driven global) to enable remote inference.

## Deployment Notes
- **Caching** – bump `CACHE_NAME` in `service-worker.js` whenever you change asset filenames or model versions to avoid stale caches.
- **Compression** – serve the model with gzip or brotli compression; browsers support streaming decompression for ONNX/TFJS weights.
- **Content Security Policy** – ensure the CSP allows `worker-src 'self'` (or relevant CDN) and `connect-src` entries for any backend endpoint used.
- **Version metadata** – embed version strings in the model file (custom metadata) and expose the version via `modelVersion` option in the controller.
- **Offline mode** – the UI automatically falls back to rule-based heuristics when the model cannot load. Ensure translations describe the fallback state for clarity.

## Updating the Model
1. Re-export the trained network to ONNX/TF.js (see training guide).
2. Quantise to FP16 or INT8 (ONNX Runtime quantisation toolkit or TF.js post-training quantisation).
3. Replace `models/emo_aen_v2_int8.onnx` with the new artefact and update `modelVersion` inside `main.js` (and optionally `CACHE_NAME`).
4. Run `npm run build` / bundler steps if applicable, then redeploy.

## Monitoring & Telemetry (Optional)
- Collect anonymous metrics about fallback frequency and latency via the `AIInferenceController.getStatus()` output before/after `scoreCandidates` (add your own analytics layer—none is included by default).
- Expose a UI indicator when the backend endpoint is used so users understand where inference runs.

## Browser Compatibility
- Requires ES modules, Web Workers, and Fetch API support. Modern evergreen browsers (Chromium 94+, Firefox 93+, Safari 15+) meet these requirements.
- For older browsers, polyfill or provide a static fallback experience without AI by disabling the controller instantiation.

## Deployment Checklist
- [ ] Quantised model present and paths correct in service worker & controller.
- [ ] HTTPS hosting configured and CSP updated.
- [ ] Translations reviewed for new AI labels.
- [ ] Service worker cache name bumped on release.
- [ ] Optional backend endpoint secured (authentication/ratelimiting) if enabled.
