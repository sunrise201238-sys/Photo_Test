# Testing Checklist

## Functional Tests
- [ ] Upload sample images and verify the AI panel updates scores, mode, and suggestions.
- [ ] Confirm best crop rectangle renders on the original preview when guides are enabled.
- [ ] Toggle guides and ensure overlays hide/show correctly.
- [ ] Download the improved crop and inspect that dimensions match the AI recommendation.

## Latency & Performance
- [ ] Measure local inference latency via browser devtools; ensure average latency <300 ms on desktop hardware.
- [ ] Force rule-based fallback by removing the model file or blocking the worker; confirm UI shows "Rules-only" mode.
- [ ] (Optional) Configure backend endpoint and simulate slow responses to confirm the app retries and falls back gracefully.

## Offline Support
- [ ] Load the app once, then go offline and refresh. Verify service worker serves cached assets and model.
- [ ] Confirm rule-based analysis still operates offline if the model is unavailable.

## Translations
- [ ] Switch to English and Traditional Chinese; validate all new AI panel strings translate correctly.
- [ ] Ensure RTL or extended characters do not overflow the AI panel layout.

## Visual QA
- [ ] Check AI panel responsiveness on narrow viewports (<768px) and wide desktops.
- [ ] Verify third-line overlays and horizon highlights remain legible on bright/dark images.

## Regression Tests
- [ ] Confirm legacy non-AI features (drop zone, manual reset, tips) still function.
- [ ] Run automated linters/tests if available (e.g. `npm run lint` or custom scripts).
