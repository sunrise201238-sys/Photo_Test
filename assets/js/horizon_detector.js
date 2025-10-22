function toGrayscale(imageData) {
  const { data, width, height } = imageData;
  const grayscale = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      grayscale[y * width + x] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
  }
  return grayscale;
}

function computeGradient(grayscale, width, height) {
  const gradX = new Float32Array(width * height);
  const gradY = new Float32Array(width * height);
  const magnitude = new Float32Array(width * height);
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
      magnitude[idx] = Math.hypot(gx, gy);
    }
  }
  return { gradX, gradY, magnitude };
}

function weightedAverage(values, weights) {
  let sum = 0;
  let weightSum = 0;
  for (let i = 0; i < values.length; i++) {
    const w = weights[i];
    sum += values[i] * w;
    weightSum += w;
  }
  return weightSum ? sum / weightSum : 0;
}

export function estimateHorizon(imageData) {
  const { width, height } = imageData;
  if (!width || !height) {
    return {
      angle: 0,
      confidence: 0,
      line: [
        { x: 0, y: height / 2 },
        { x: width, y: height / 2 }
      ]
    };
  }

  const grayscale = toGrayscale(imageData);
  const { gradX, gradY, magnitude } = computeGradient(grayscale, width, height);
  const angles = [];
  const weights = [];
  const maskTop = Math.floor(height * 0.2);
  const maskBottom = Math.floor(height * 0.8);

  for (let y = maskTop; y < maskBottom; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const mag = magnitude[idx];
      if (mag < 48) continue;
      const gx = gradX[idx];
      const gy = gradY[idx];
      const angle = Math.atan2(gy, gx);
      const normalised = ((angle * 180) / Math.PI + 180) % 180;
      const horizonAngle = normalised > 90 ? normalised - 180 : normalised;
      const alignment = 1 - Math.abs(horizonAngle) / 90;
      const weight = mag * alignment;
      if (!weight) continue;
      angles.push(horizonAngle);
      weights.push(weight);
    }
  }

  const angle = weightedAverage(angles, weights);
  const normalizedConfidence = Math.min(1, weights.reduce((a, b) => a + b, 0) / (width * height * 22));

  const center = { x: width / 2, y: height / 2 };
  const length = Math.max(width, height);
  const radians = (angle * Math.PI) / 180;
  const dx = Math.cos(radians) * length;
  const dy = Math.sin(radians) * length;
  const line = [
    { x: center.x - dx / 2, y: center.y - dy / 2 },
    { x: center.x + dx / 2, y: center.y + dy / 2 }
  ];

  return { angle, confidence: normalizedConfidence, line };
}
