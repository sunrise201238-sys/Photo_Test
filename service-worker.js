const CACHE_NAME = 'photo-assistant-v1';
const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/assets/css/styles.css',
  '/assets/css/ai_panel.css',
  '/assets/js/main.js',
  '/assets/js/ai_inference_controller.js',
  '/assets/js/composition_engine.js',
  '/assets/js/horizon_detector.js',
  '/assets/js/saliency_fallback.js',
  '/assets/js/worker_infer.js',
  '/translations/en-US.json',
  '/translations/zh-TW.json',
  '/models/emo_aen_v1_quant.onnx'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(PRECACHE_URLS)).catch(error => console.warn('Precache failed', error))
  );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') {
    return;
  }
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) {
        return cached;
      }
      return fetch(event.request)
        .then(response => {
          const copy = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
          return response;
        })
        .catch(error => {
          if (event.request.mode === 'navigate') {
            return caches.match('/index.html');
          }
          throw error;
        });
    })
  );
});
