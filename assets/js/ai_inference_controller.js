const DEFAULT_MODEL_URL = 'models/emo_aen_v2_int8.onnx';
const WORKER_PATH = new URL('./worker_infer.js', import.meta.url);

function now() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function heuristicScore(feature) {
  const thirds = feature.ruleOfThirdsScore ?? 0.5;
  const saliency = feature.saliencyConfidence ?? 0.5;
  const horizon = Math.max(0, 1 - Math.abs(feature.horizonAngle ?? 0) / 45);
  const texture = Math.min(1, (feature.textureStrength ?? 0.18) * 3);
  const crop = Math.min(1, feature.cropArea ?? 1);
  const balance = 1 - Math.min(1, Math.abs((feature.balanceRatio ?? 1) - 1) * 0.5);
  const composition = Math.min(0.99, 0.24 + thirds * 0.4 + saliency * 0.2 + horizon * 0.1 + texture * 0.08 + balance * 0.05);
  const aesthetic = Math.min(0.98, composition * 0.65 + texture * 0.15 + (feature.colorHarmony ?? 0.5) * 0.2);
  return { composition, aesthetic };
}

export class AIInferenceController {
  constructor(options = {}) {
    this.modelUrl = options.modelUrl || DEFAULT_MODEL_URL;
    this.modelFormat = options.modelFormat || 'onnx';
    this.modelVersion = options.modelVersion || 'v2.0.0';
    this.backendEndpoint = options.backendEndpoint || null;
    this.worker = null;
    this.status = 'idle';
    this.mode = 'rules';
    this.lastError = null;
    this.initialized = false;
    this.latency = 0;
    this.lastInference = null;
    this.useWorker = options.useWorker !== false;
  }

  async initialize() {
    if (this.initialized) {
      return { status: this.status, mode: this.mode };
    }
    this.status = 'loading';
    try {
      if (this.useWorker && typeof Worker !== 'undefined') {
        this.worker = new Worker(WORKER_PATH, { type: 'module' });
        const success = await this._loadModelWithWorker();
        if (success) {
          this.status = 'ready';
          this.mode = success.backendMode || 'local';
          this.initialized = true;
          return { status: this.status, mode: this.mode };
        }
      }
      this.status = 'ready';
      this.mode = 'rules';
      this.initialized = true;
      return { status: this.status, mode: this.mode };
    } catch (error) {
      console.error('[AIInferenceController] initialize failed', error);
      this.status = 'error';
      this.mode = 'rules';
      this.lastError = error;
      this.initialized = true;
      return { status: this.status, mode: this.mode };
    }
  }

  terminate() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.initialized = false;
    this.status = 'idle';
    this.mode = 'rules';
  }

  async scoreCandidates(candidates, context = {}) {
    const features = candidates.map(candidate => candidate.features);
    if (!features.length) {
      return [];
    }
    if (!this.initialized) {
      await this.initialize();
    }

    const start = now();

    if (this.mode === 'cloud' && this.backendEndpoint) {
      try {
        const response = await fetch(this.backendEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            version: this.modelVersion,
            features,
            context
          })
        });
        if (response.ok) {
          const payload = await response.json();
          const results = (payload.results || []).map((result, idx) => ({
            composition: result.composition ?? heuristicScore(features[idx]).composition,
            aesthetic: result.aesthetic ?? result.composition ?? heuristicScore(features[idx]).composition,
            mode: 'cloud'
          }));
          this.latency = now() - start;
          this.lastInference = { timestamp: Date.now(), results };
          return results;
        }
        throw new Error(`Cloud inference failed (${response.status})`);
      } catch (error) {
        console.warn('[AIInferenceController] Cloud inference error, reverting to rules', error);
        this.mode = 'rules';
      }
    }

    if (this.worker && this.mode !== 'rules') {
      try {
        const payload = features.map(feature => ({
          ...feature,
          vector: feature.vector || this._featureVector(feature)
        }));
        const workerId = typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
        const results = await this._postToWorker({
          type: 'infer',
          features: payload,
          id: workerId
        });
        this.mode = results.backendMode || this.mode;
        this.latency = now() - start;
        const scored = results.results.map((result, idx) => ({
          composition: result.composition,
          aesthetic: result.aesthetic,
          mode: results.backendMode || 'local'
        }));
        this.lastInference = { timestamp: Date.now(), results: scored };
        return scored;
      } catch (error) {
        console.warn('[AIInferenceController] Worker inference failed, fallback to heuristics', error);
        this.mode = 'rules';
      }
    }

    const fallback = features.map(feature => ({
      ...heuristicScore(feature),
      mode: 'rules'
    }));
    this.latency = now() - start;
    this.lastInference = { timestamp: Date.now(), results: fallback };
    return fallback;
  }

  getStatus() {
    return {
      status: this.status,
      mode: this.mode,
      error: this.lastError,
      latency: this.latency,
      initialized: this.initialized,
      lastInference: this.lastInference
    };
  }

  async _loadModelWithWorker() {
    if (!this.worker) return false;
    const message = {
      type: 'loadModel',
      url: this.modelUrl,
      format: this.modelFormat,
      version: this.modelVersion,
      options: { executionMode: 'parallel' }
    };
    try {
      const response = await this._postToWorker(message, true);
      if (response.success) {
        this.mode = response.backendMode || 'local';
      } else {
        this.mode = 'rules';
      }
      return response;
    } catch (error) {
      console.warn('[AIInferenceController] worker model load failed', error);
      this.mode = 'rules';
      return false;
    }
  }

  _postToWorker(message, expectFull = false) {
    return new Promise((resolve, reject) => {
      if (!this.worker) {
        return reject(new Error('Worker not available'));
      }
      const handleMessage = event => {
        const { data } = event;
        if (!data) return;
        if (expectFull && data.type === 'loadModelResult') {
          this.worker.removeEventListener('message', handleMessage);
          this.worker.removeEventListener('error', handleError);
          resolve(data);
        } else if (!expectFull && data.type === 'inferenceResult' && (!message.id || data.id === message.id)) {
          this.worker.removeEventListener('message', handleMessage);
          this.worker.removeEventListener('error', handleError);
          resolve(data);
        }
      };
      const handleError = error => {
        this.worker.removeEventListener('message', handleMessage);
        this.worker.removeEventListener('error', handleError);
        reject(error);
      };
      this.worker.addEventListener('message', handleMessage);
      this.worker.addEventListener('error', handleError);
      this.worker.postMessage(message);
    });
  }

  _featureVector(feature) {
    return [
      feature.ruleOfThirdsScore ?? 0.5,
      feature.saliencyConfidence ?? 0.5,
      feature.horizonAngle ?? 0,
      feature.textureStrength ?? 0.1,
      feature.balanceRatio ?? 1,
      feature.cropArea ?? 1,
      feature.colorHarmony ?? 0.5,
      feature.horizonConfidence ?? 0.5,
      feature.subjectSize ?? 0.2,
      feature.leadingLineStrength ?? 0.2
    ];
  }
}
