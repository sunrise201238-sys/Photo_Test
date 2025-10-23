import { estimateSaliency } from './saliency_fallback.js';
import { estimateHorizon } from './horizon_detector.js';

export function clamp(value, min = 0, max = 255) {
  return Math.min(max, Math.max(min, value));
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
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.round(percentile * (sorted.length - 1)));
  return sorted[index];
}

function computeHistogram(grayscale, bins = 32) {
  const histogram = new Array(bins).fill(0);
  for (let i = 0; i < grayscale.length; i++) {
    const bin = Math.min(bins - 1, Math.floor((grayscale[i] / 255) * bins));
    histogram[bin]++;
  }
  return histogram;
}

function analyseColorBalance(imageData) {
  const { data } = imageData;
  let r = 0;
  let g = 0;
  let b = 0;
  const pixelCount = data.length / 4;
  for (let i = 0; i < data.length; i += 4) {
    r += data[i];
    g += data[i + 1];
    b += data[i + 2];
  }
  return { r: r / pixelCount, g: g / pixelCount, b: b / pixelCount };
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

function sobel(grayscale, width, height) {
  const gradX = new Float32Array(width * height);
  const gradY = new Float32Array(width * height);
  const gradient = new Float32Array(width * height);
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
    }
  }
  return { gradX, gradY, gradient };
}

export function computeMetrics(imageData, detectors = {}) {
  const { width, height, data } = imageData;
  const pixelCount = width * height;
  const grayscale = toGrayscale(imageData);
  const histogram = computeHistogram(grayscale);
  const stats = computeStatistics(grayscale);
  const percentile95 = samplePercentile(grayscale, 0.95);
  const percentile05 = samplePercentile(grayscale, 0.05);
  const contrast = percentile95 - percentile05;
  const colorBalance = analyseColorBalance(imageData);
  const { gradX, gradY, gradient } = sobel(grayscale, width, height);

  const gradientThreshold = samplePercentile(gradient, 0.82);
  const horizonThreshold = samplePercentile(gradient, 0.75);

  let orientationSumX = 0;
  let orientationSumY = 0;
  let orientationCount = 0;
  let gradientSum = 0;
  let shadowClipped = 0;
  let highlightClipped = 0;
  let midtoneSum = 0;
  let midtoneCount = 0;
  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;
  let sumX = 0;
  let sumY = 0;
  let strongCount = 0;
  let horizonAngle = 0;
  let horizonWeight = 0;

  for (let i = 0; i < pixelCount; i++) {
    const g = gradient[i];
    gradientSum += g;
    if (grayscale[i] < 15) shadowClipped++;
    if (grayscale[i] > 240) highlightClipped++;
    if (grayscale[i] > 96 && grayscale[i] < 192) {
      midtoneSum += grayscale[i];
      midtoneCount++;
    }
  }

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
    saliencyConfidence: 0,
    horizonAngle: horizonWeight ? horizonAngle / horizonWeight : 0,
    horizonConfidence: horizonWeight ? clamp(horizonWeight / (pixelCount * 6), 0, 1) : 0,
    horizonLine: null,
    ruleOfThirdsScore: 0,
    sharpnessVariance: 0,
    exposure: stats.mean,
    contrast: stats.std,
    saturation: 0,
    colorBalance,
    colorHarmony: 0,
    foregroundBackground: 0,
    shadowClipping: shadowClipped / pixelCount,
    highlightClipping: highlightClipped / pixelCount,
    midtoneBalance: midtoneCount ? midtoneSum / (midtoneCount * 255) : stats.mean / 255,
    colorCast: { bias: 0, warmBias: 0, coolBias: 0, strength: 0 },
    leadingLines: { angle: 0, strength: 0 },
    textureStrength: gradientSum / Math.max(1, pixelCount * 255),
    feedback: [],
    histogram,
    detectors
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
  metrics.colorHarmony = 1 - Math.min(1, metrics.colorCast.strength * 1.4);

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

  metrics.feedback = buildFeedback(metrics);

  if (!metrics.subjectRect || (detectors.saliency && detectors.saliency.confidence > metrics.saliencyConfidence)) {
    if (detectors.saliency) {
      metrics.saliencyConfidence = detectors.saliency.confidence;
      metrics.subjectRect = detectors.saliency.rect;
      metrics.subjectCenter = detectors.saliency.center;
      metrics.subjectOffset = {
        x: detectors.saliency.center.x / width - 0.5,
        y: detectors.saliency.center.y / height - 0.5
      };
      metrics.subjectSize = (detectors.saliency.rect.width * detectors.saliency.rect.height) / (width * height);
    }
  }

  if (detectors.horizon) {
    metrics.horizonAngle = detectors.horizon.angle;
    metrics.horizonConfidence = detectors.horizon.confidence;
    metrics.horizonLine = detectors.horizon.line;
  }

  return metrics;
}

function buildFeedback(metrics) {
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
  return Array.from(feedback);
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
  const offsetMagnitude = Math.hypot(metrics.subjectOffset.x, metrics.subjectOffset.y);
  const angleInfluence = Math.min(0.14, Math.abs(metrics.horizonAngle) * 0.0075);
  const offsetInfluence = Math.min(0.1, offsetMagnitude * 0.16);
  const marginRatio = clamp(0.1 + angleInfluence * 0.55 + offsetInfluence * 0.7, 0.1, 0.22);
  const margin = Math.round(maxDimension * marginRatio);
  const skewLimit = metrics.subjectRect ? 0.07 : 0.055;
  const skewX = clamp(metrics.subjectOffset.x * 0.032 + metrics.horizonAngle * 0.00065, -skewLimit, skewLimit);
  const skewY = clamp(metrics.subjectOffset.y * 0.03, -skewLimit, skewLimit);
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
  const bounds = computeContentBounds(canvas, 2);
  if (!bounds) {
    return canvas;
  }
  const paddedCanvas = document.createElement('canvas');
  paddedCanvas.width = bounds.width;
  paddedCanvas.height = bounds.height;
  const ctx = paddedCanvas.getContext('2d');
  ctx.drawImage(canvas, bounds.x, bounds.y, bounds.width, bounds.height, 0, 0, bounds.width, bounds.height);
  return paddedCanvas;
}

function applyToneAndColorAdjustments(canvas, metrics, focusPoint) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const { data } = imageData;
  const focus = focusPoint || { x: canvas.width / 2, y: canvas.height / 2 };
  const focusRadius = Math.min(canvas.width, canvas.height) * 0.35;
  const focusLift = metrics.subjectSize < 0.18 ? 0.35 : 0.18;
  const vibrance = metrics.saturation < 55 ? 0.18 : 0.08;
  const gamma = metrics.exposure < 110 ? 0.95 : metrics.exposure > 170 ? 1.05 : 1;
  const shadowBoost = metrics.shadowClipping > 0.035 ? 0.32 : 0.18;
  const highlightPull = metrics.highlightClipping > 0.035 ? 0.28 : 0.12;
  const midtoneBias = metrics.exposure < 115 ? 0.04 : metrics.exposure > 170 ? -0.03 : 0;
  const blackLift = metrics.shadowClipping > 0.045 ? 0.14 : 0.06;
  const brightnessLift = metrics.exposure < 110 ? 0.06 : 0.02;
  const warmShift = metrics.colorCast.bias > 0 ? Math.min(0.16, metrics.colorCast.bias / 255) : 0;
  const coolShift = metrics.colorCast.bias < 0 ? Math.min(0.14, Math.abs(metrics.colorCast.bias) / 255) : 0;
  const naturalWarmth = metrics.colorCast.bias < 0 ? Math.abs(metrics.colorCast.bias) * 0.0007 : 0;

  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const idx = (y * canvas.width + x) * 4;
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

function computeCropBox(width, height, metrics, options = {}) {
  const aspectPreference = width >= height ? 3 / 2 : 4 / 5;
  const subjectMarginBase = metrics.subjectRect ? Math.max(0.1, 0.28 - metrics.subjectSize * 1.2) : 0.16;
  const subjectMargin = clamp(subjectMarginBase + (options.marginOffset || 0), 0.08, 0.35);
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
  const horizontalBias = options.horizontalBias ?? (metrics.subjectRect
    ? metrics.subjectCenter.x < width / 2
      ? 0.32
      : 0.68
    : 0.5);
  const verticalBias = options.verticalBias ?? (metrics.subjectRect
    ? metrics.subjectCenter.y < height / 2
      ? 0.36
      : 0.64
    : 0.5);
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

function improveImage(baseCanvas, metrics, candidateOptions = {}) {
  const width = baseCanvas.width;
  const height = baseCanvas.height;
  const crop = candidateOptions.crop || computeCropBox(width, height, metrics, candidateOptions.variation);

  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = crop.width;
  cropCanvas.height = crop.height;
  const cropCtx = cropCanvas.getContext('2d');
  cropCtx.drawImage(baseCanvas, crop.x, crop.y, crop.width, crop.height, 0, 0, crop.width, crop.height);

  const desiredAngle = typeof candidateOptions.rotation === 'number' ? candidateOptions.rotation : metrics.horizonAngle;
  const leveledAngle = clamp(desiredAngle, -18, 18);
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
  const rotationDegrees = Math.abs(leveledAngle);
  const zoomPadding = rotationDegrees * 0.017 + distortion * 1.65 + (metrics.subjectSize < 0.18 ? 0.22 : 0.12);
  const extraScale = 1 + Math.min(0.6, Math.max(0.2, zoomPadding + 0.08));
  const availableWidth = perspective.canvas.width;
  const availableHeight = perspective.canvas.height;
  const targetAspect = crop.width / crop.height;

  const contentBounds = computeContentBounds(perspective.canvas, 4);
  let sampleX = contentBounds ? contentBounds.x : 0;
  let sampleY = contentBounds ? contentBounds.y : 0;
  let sampleWidth = contentBounds ? contentBounds.width : availableWidth;
  let sampleHeight = contentBounds ? contentBounds.height : availableHeight;

  const desiredWidth = Math.min(sampleWidth, Math.round(crop.width * extraScale));
  const desiredHeight = Math.min(sampleHeight, Math.round(crop.height * extraScale));

  let adjustedWidth = desiredWidth;
  let adjustedHeight = desiredHeight;

  if (adjustedWidth / adjustedHeight > targetAspect) {
    adjustedWidth = Math.round(adjustedHeight * targetAspect);
  } else {
    adjustedHeight = Math.round(adjustedWidth / targetAspect);
  }

  adjustedWidth = Math.max(crop.width, Math.min(adjustedWidth, sampleWidth));
  adjustedHeight = Math.max(crop.height, Math.min(adjustedHeight, sampleHeight));

  const widthExcess = sampleWidth - adjustedWidth;
  const heightExcess = sampleHeight - adjustedHeight;

  if (widthExcess > 0) {
    sampleX += Math.floor(widthExcess / 2);
    sampleWidth = adjustedWidth;
  }
  if (heightExcess > 0) {
    sampleY += Math.floor(heightExcess / 2);
    sampleHeight = adjustedHeight;
  }

  const safetyRatio = Math.min(0.24, rotationDegrees * 0.018 + distortion * 0.65);
  const marginXLimit = Math.max(0, Math.floor((sampleWidth - crop.width) / 2));
  const marginYLimit = Math.max(0, Math.floor((sampleHeight - crop.height) / 2));
  let marginX = Math.round(sampleWidth * (0.03 + safetyRatio * 0.5));
  let marginY = Math.round(sampleHeight * (0.025 + safetyRatio * 0.45));
  marginX = Math.min(marginX, marginXLimit);
  marginY = Math.min(marginY, marginYLimit);

  if (marginX > 0) {
    sampleX += marginX;
    sampleWidth -= marginX * 2;
  }
  if (marginY > 0) {
    sampleY += marginY;
    sampleHeight -= marginY * 2;
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

  return {
    canvas: fillCanvasGutters(finalCanvas),
    crop,
    rotation,
    focusPoint
  };
}

function candidateFeatures(metrics, candidate, detectors = {}) {
  const { imageSize } = metrics;
  const thirdsX = [imageSize.width / 3, (2 * imageSize.width) / 3];
  const thirdsY = [imageSize.height / 3, (2 * imageSize.height) / 3];
  const subject = metrics.subjectRect || {
    x: candidate.crop.x + candidate.crop.width / 2,
    y: candidate.crop.y + candidate.crop.height / 2,
    width: candidate.crop.width * 0.2,
    height: candidate.crop.height * 0.2
  };
  const subjectCenter = {
    x: subject.x + subject.width / 2,
    y: subject.y + subject.height / 2
  };
  const nearestX = Math.min(...thirdsX.map(x => Math.abs(subjectCenter.x - x)));
  const nearestY = Math.min(...thirdsY.map(y => Math.abs(subjectCenter.y - y)));
  const ruleScore = 1 - (nearestX / imageSize.width + nearestY / imageSize.height);

  const cropArea = (candidate.crop.width * candidate.crop.height) / (imageSize.width * imageSize.height);
  const subjectRatio = (subject.width * subject.height) / (candidate.crop.width * candidate.crop.height);

  return {
    ruleOfThirdsScore: clamp(ruleScore, 0, 1),
    saliencyConfidence: detectors.saliency?.confidence ?? metrics.saliencyConfidence,
    horizonAngle: metrics.horizonAngle,
    horizonConfidence: detectors.horizon?.confidence ?? metrics.horizonConfidence,
    textureStrength: metrics.textureStrength,
    balanceRatio: metrics.foregroundBackground,
    cropArea,
    colorHarmony: metrics.colorHarmony,
    subjectSize: subjectRatio,
    leadingLineStrength: metrics.leadingLines.strength,
    vector: [
      clamp(ruleScore, 0, 1),
      detectors.saliency?.confidence ?? metrics.saliencyConfidence,
      metrics.horizonAngle / 90,
      metrics.textureStrength,
      metrics.foregroundBackground,
      cropArea,
      metrics.colorHarmony,
      detectors.horizon?.confidence ?? metrics.horizonConfidence,
      subjectRatio,
      metrics.leadingLines.strength
    ]
  };
}

function variationList() {
  return [
    { id: 'base', marginOffset: 0 },
    { id: 'tight', marginOffset: -0.06 },
    { id: 'wide', marginOffset: 0.08 },
    { id: 'leftThird', horizontalBias: 0.32 },
    { id: 'rightThird', horizontalBias: 0.68 },
    { id: 'topThird', verticalBias: 0.38 },
    { id: 'bottomThird', verticalBias: 0.62 }
  ];
}

export class CompositionEngine {
  constructor(options = {}) {
    this.maxCandidates = options.maxCandidates || 6;
  }

  analyse(imageData, detectors = {}) {
    const enrichedDetectors = {
      saliency: detectors.saliency || estimateSaliency(imageData),
      horizon: detectors.horizon || estimateHorizon(imageData)
    };
    const metrics = computeMetrics(imageData, enrichedDetectors);
    metrics.detectors = enrichedDetectors;
    return metrics;
  }

  generateCandidates(baseCanvas, metrics, detectors = {}) {
    const variations = variationList().slice(0, this.maxCandidates);
    const candidates = variations.map(variation => {
      const crop = computeCropBox(baseCanvas.width, baseCanvas.height, metrics, variation);
      const rotation = metrics.horizonAngle;
      const features = candidateFeatures(metrics, { crop }, detectors);
      return {
        id: variation.id,
        crop,
        rotation,
        features,
        variation
      };
    });
    return candidates;
  }

  evaluateCandidates(candidates, scores) {
    return candidates.map((candidate, idx) => ({
      ...candidate,
      compositionScore: scores[idx]?.composition ?? 0.5,
      aestheticScore: scores[idx]?.aesthetic ?? 0.5,
      mode: scores[idx]?.mode || 'rules'
    }));
  }

  selectBestCandidate(scoredCandidates) {
    if (!scoredCandidates.length) return null;
    return scoredCandidates.reduce((best, candidate) => {
      if (!best) return candidate;
      if (candidate.compositionScore > best.compositionScore) {
        return candidate;
      }
      if (
        candidate.compositionScore === best.compositionScore &&
        candidate.aestheticScore > best.aestheticScore
      ) {
        return candidate;
      }
      return best;
    }, null);
  }

  renderCandidate(baseCanvas, metrics, candidate) {
    return improveImage(baseCanvas, metrics, candidate);
  }
}
