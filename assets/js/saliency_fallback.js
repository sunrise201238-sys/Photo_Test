const SOBEL_X = [
  -1, 0, 1,
  -2, 0, 2,
  -1, 0, 1
];

const SOBEL_Y = [
  -1, -2, -1,
   0,  0,  0,
   1,  2,  1
];

function clamp(value, min = 0, max = 255) {
  return Math.min(max, Math.max(min, value));
}

function toGrayscale(imageData) {
  const { data, width, height } = imageData;
  const grayscale = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      grayscale[y * width + x] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
  }
  return grayscale;
}

function convolveSobel(grayscale, width, height) {
  const gradX = new Float32Array(width * height);
  const gradY = new Float32Array(width * height);
  const magnitude = new Float32Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0;
      let gy = 0;
      let k = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const sample = grayscale[(y + ky) * width + (x + kx)];
          gx += sample * SOBEL_X[k];
          gy += sample * SOBEL_Y[k];
          k++;
        }
      }
      const idx = y * width + x;
      gradX[idx] = gx;
      gradY[idx] = gy;
      magnitude[idx] = Math.hypot(gx, gy);
    }
  }

  return { gradX, gradY, magnitude };
}

function normalize(values) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  const normalised = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    normalised[i] = (values[i] - min) / range;
  }
  return normalised;
}

function computeIntegralMap(magnitude, width, height) {
  const integral = new Float32Array((width + 1) * (height + 1));
  for (let y = 1; y <= height; y++) {
    let rowSum = 0;
    for (let x = 1; x <= width; x++) {
      const value = magnitude[(y - 1) * width + (x - 1)];
      rowSum += value;
      const idx = y * (width + 1) + x;
      integral[idx] = integral[idx - (width + 1)] + rowSum;
    }
  }
  return integral;
}

function sumArea(integral, width, x, y, w, h) {
  const stride = width + 1;
  const x2 = x + w;
  const y2 = y + h;
  return (
    integral[y2 * stride + x2] -
    integral[y2 * stride + x] -
    integral[y * stride + x2] +
    integral[y * stride + x]
  );
}

export function estimateSaliency(imageData) {
  const { width, height } = imageData;
  if (!width || !height) {
    return {
      center: { x: width / 2 || 0, y: height / 2 || 0 },
      rect: { x: 0, y: 0, width, height },
      confidence: 0,
      map: null
    };
  }

  const grayscale = toGrayscale(imageData);
  const { gradX, gradY, magnitude } = convolveSobel(grayscale, width, height);
  const normMag = normalize(magnitude);
  const integral = computeIntegralMap(normMag, width, height);

  const step = Math.max(12, Math.round(Math.min(width, height) * 0.08));
  let bestRect = { x: 0, y: 0, width, height };
  let bestScore = -Infinity;

  for (let hSize = step; hSize <= height; hSize += step) {
    for (let wSize = step; wSize <= width; wSize += step) {
      for (let y = 0; y + hSize < height; y += Math.max(8, Math.round(step / 2))) {
        for (let x = 0; x + wSize < width; x += Math.max(8, Math.round(step / 2))) {
          const total = sumArea(integral, width, x, y, wSize, hSize);
          const area = wSize * hSize;
          const density = total / Math.max(1, area);
          const edgePerimeter =
            (sumArea(integral, width, Math.max(0, x - 6), Math.max(0, y - 6), wSize + 12, 6) +
              sumArea(integral, width, Math.max(0, x - 6), y + hSize, wSize + 12, 6) +
              sumArea(integral, width, Math.max(0, x - 6), y, 6, hSize) +
              sumArea(integral, width, x + wSize, y, 6, hSize)) /
            Math.max(1, wSize * hSize * 0.15);
          const score = density - edgePerimeter * 0.45;
          if (score > bestScore) {
            bestScore = score;
            bestRect = { x, y, width: wSize, height: hSize };
          }
        }
      }
    }
  }

  const center = {
    x: bestRect.x + bestRect.width / 2,
    y: bestRect.y + bestRect.height / 2
  };

  let energy = 0;
  let energyTotal = 0;
  for (let y = bestRect.y; y < bestRect.y + bestRect.height; y++) {
    for (let x = bestRect.x; x < bestRect.x + bestRect.width; x++) {
      const idx = y * width + x;
      const value = normMag[idx];
      energy += value;
      energyTotal += Math.max(0, Math.abs(gradX[idx]) + Math.abs(gradY[idx]));
    }
  }
  const confidence = Math.min(1, energy / Math.max(1, energyTotal * 0.6));

  return {
    center,
    rect: bestRect,
    confidence: isFinite(confidence) ? confidence : 0,
    map: normMag,
    gradX,
    gradY
  };
}

export function drawSaliencyHeatmap(canvas, saliency, color = 'rgba(244, 114, 182, 0.4)') {
  if (!canvas || !saliency || !saliency.map) return;
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  const heat = saliency.map;
  const image = ctx.getImageData(0, 0, width, height);
  const { data } = image;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const weight = clamp(heat[idx] * 255, 0, 255);
      const offset = idx * 4;
      data[offset] = Math.max(data[offset], weight);
      data[offset + 1] = Math.max(data[offset + 1], weight * 0.6);
      data[offset + 2] = Math.max(data[offset + 2], weight * 0.9);
      data[offset + 3] = clamp(weight * 0.35);
    }
  }
  ctx.putImageData(image, 0, 0);
}
