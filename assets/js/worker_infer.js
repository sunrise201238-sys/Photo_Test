let modelSession = null;
let backendMode = 'rules';
let modelVersion = null;

async function tryLoadOnnx(buffer, options = {}) {
  try {
    const ort = await import('onnxruntime-web');
    const session = await ort.InferenceSession.create(buffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      ...options
    });
    backendMode = 'local';
    return session;
  } catch (error) {
    console.warn('[worker] ONNX load failed, falling back to heuristics', error);
    return null;
  }
}

async function tryLoadTf(modelUrl) {
  try {
    const tf = await import('@tensorflow/tfjs');
    await import('@tensorflow/tfjs-backend-wasm');
    await tf.setBackend('wasm');
    const model = await tf.loadGraphModel(modelUrl);
    backendMode = 'local';
    return model;
  } catch (error) {
    console.warn('[worker] TF.js load failed, falling back to heuristics', error);
    return null;
  }
}

function heuristicScore(feature) {
  const thirds = feature.ruleOfThirdsScore ?? 0.5;
  const saliency = feature.saliencyConfidence ?? 0.5;
  const horizonPenalty = Math.max(0, 1 - Math.abs(feature.horizonAngle ?? 0) / 45);
  const texture = Math.min(1, (feature.textureStrength ?? 0.2) * 3);
  const balance = 1 - Math.min(1, Math.abs(feature.balanceRatio ?? 1 - 1) * 0.5);
  const cropArea = Math.min(1, feature.cropArea ?? 1);
  const composition = Math.min(
    0.99,
    0.22 + thirds * 0.35 + saliency * 0.18 + horizonPenalty * 0.12 + texture * 0.08 + balance * 0.05
  );
  const aesthetic = Math.min(
    0.98,
    composition * 0.6 + texture * 0.18 + (feature.colorHarmony ?? 0.5) * 0.22
  );
  return { composition, aesthetic };
}

async function runModel(featureBatch) {
  if (!modelSession) {
    return featureBatch.map(heuristicScore);
  }
  if (backendMode === 'local' && modelSession.run) {
    try {
      const ort = await import('onnxruntime-web');
      const inputs = new ort.Tensor('float32', Float32Array.from(featureBatch.flatMap(f => f.vector)), [
        featureBatch.length,
        featureBatch[0].vector.length
      ]);
      const result = await modelSession.run({ input: inputs });
      const comp = result.composition_score.data;
      const aesth = result.aesthetic_score?.data || comp;
      return featureBatch.map((feature, idx) => ({
        composition: comp[idx],
        aesthetic: aesth[idx] ?? comp[idx]
      }));
    } catch (error) {
      console.warn('[worker] ONNX inference failed, reverting to heuristics', error);
      backendMode = 'rules';
      modelSession = null;
      return featureBatch.map(heuristicScore);
    }
  }
  if (backendMode === 'local' && modelSession.executeAsync) {
    try {
      const tf = await import('@tensorflow/tfjs');
      const tensor = tf.tensor2d(featureBatch.map(feature => feature.vector));
      const result = await modelSession.executeAsync({ input: tensor });
      const compositionTensor = result[0];
      const aestheticTensor = result[1] || compositionTensor;
      const compositionData = await compositionTensor.data();
      const aestheticData = await aestheticTensor.data();
      tf.dispose([tensor, ...result]);
      return featureBatch.map((feature, idx) => ({
        composition: compositionData[idx],
        aesthetic: aestheticData[idx] ?? compositionData[idx]
      }));
    } catch (error) {
      console.warn('[worker] TF.js inference failed, reverting to heuristics', error);
      backendMode = 'rules';
      modelSession = null;
      return featureBatch.map(heuristicScore);
    }
  }
  return featureBatch.map(heuristicScore);
}

self.addEventListener('message', async event => {
  const { data } = event;
  if (!data) return;
  if (data.type === 'loadModel') {
    backendMode = 'rules';
    modelSession = null;
    modelVersion = data.version || null;
    try {
      if (data.format === 'onnx') {
        const response = await fetch(data.url);
        if (!response.ok) throw new Error(`Failed to fetch model ${data.url}`);
        const buffer = await response.arrayBuffer();
        modelSession = await tryLoadOnnx(buffer, data.options);
      } else if (data.format === 'tfjs') {
        modelSession = await tryLoadTf(data.url);
      }
      if (!modelSession) {
        backendMode = 'rules';
      }
      self.postMessage({
        type: 'loadModelResult',
        success: Boolean(modelSession),
        backendMode,
        version: modelVersion
      });
    } catch (error) {
      console.error('[worker] Model load error', error);
      backendMode = 'rules';
      modelSession = null;
      self.postMessage({ type: 'loadModelResult', success: false, backendMode, version: modelVersion, error: error.message });
    }
  } else if (data.type === 'infer') {
    const features = Array.isArray(data.features) ? data.features : [];
    const results = await runModel(features);
    self.postMessage({ type: 'inferenceResult', id: data.id, results, backendMode, version: modelVersion });
  } else if (data.type === 'terminate') {
    close();
  }
});
