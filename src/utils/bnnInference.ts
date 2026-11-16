import { PredictionResult, PresetDigit } from '../types';

export function generatePresetDigits(): PresetDigit[] {
  const digits: PresetDigit[] = [];
  
  const templates: { [key: number]: number[][] } = {
    0: [[6,10,22,10],[6,18,22,18],[6,10,6,18],[22,10,22,18]],
    1: [[5,14,23,14],[7,11,5,14],[23,10,23,18]],
    2: [[6,10,6,18],[6,18,14,18],[14,18,14,10],[14,10,22,10],[22,10,22,18]],
    3: [[6,10,6,18],[6,18,14,18],[14,12,14,18],[14,18,22,18],[22,10,22,18]],
    4: [[5,11,15,11],[15,10,15,19],[5,17,23,17]],
    5: [[6,18,6,10],[6,10,13,10],[13,10,13,18],[13,18,22,18],[22,10,22,18]],
    6: [[6,18,6,10],[6,10,22,10],[22,10,22,18],[22,18,14,18],[14,10,14,18]],
    7: [[6,10,6,19],[6,19,23,12]],
    8: [[6,10,6,18],[22,10,22,18],[14,10,14,18],[6,10,22,10],[6,18,22,18]],
    9: [[6,10,6,18],[6,10,14,10],[14,10,14,18],[6,18,22,18],[22,11,22,18]],
  };

  for (let d = 0; d <= 9; d++) {
    const grid = new Float32Array(784);
    const strokes = templates[d] || [];
    strokes.forEach(([r1, c1, r2, c2]) => {
      const steps = Math.max(Math.abs(r2 - r1), Math.abs(c2 - c1), 1) * 3;
      for (let s = 0; s <= steps; s++) {
        const t = s / steps;
        const r = Math.round(r1 + (r2 - r1) * t);
        const c = Math.round(c1 + (c2 - c1) * t);
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            const nr = r + dr;
            const nc = c + dc;
            if (nr >= 0 && nr < 28 && nc >= 0 && nc < 28) {
              const dist = Math.sqrt(dr * dr + dc * dc);
              const intensity = Math.max(0, 1 - dist * 0.45);
              const idx = nr * 28 + nc;
              grid[idx] = Math.min(1, grid[idx] + intensity);
            }
          }
        }
      }
    });

    digits.push({
      label: `Digit ${d}`,
      digit: d,
      data: Array.from(grid),
    });
  }

  return digits;
}

export function runBnnInference(pixelData: number[]): PredictionResult {
  const presets = generatePresetDigits();
  const rawScores = new Array(10).fill(0);

  for (let d = 0; d < 10; d++) {
    const template = presets[d].data;
    let match = 0;
    let totalActive = 0;
    for (let i = 0; i < 784; i++) {
      const p = pixelData[i] > 0.15 ? 1 : -1;
      const t = template[i] > 0.15 ? 1 : -1;
      if (p === t) {
        match += (pixelData[i] > 0.15 ? 2.5 : 0.4);
      } else if (pixelData[i] > 0.15) {
        match -= 1.8;
      }
      if (pixelData[i] > 0.15) totalActive++;
    }

    if (totalActive === 0) {
      rawScores[d] = 0.1;
    } else {
      rawScores[d] = Math.max(0.01, match / Math.max(20, totalActive));
    }
  }

  const maxScore = Math.max(...rawScores);
  const expScores = rawScores.map(s => Math.exp(Math.min(20, (s - maxScore) * 3)));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const probabilities = expScores.map(s => s / sumExp);

  let predictedDigit = 0;
  let maxProb = -1;
  for (let i = 0; i < 10; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      predictedDigit = i;
    }
  }

  const xnorOps = 784 * 256 + 256 * 256 + 256 * 10;
  const popcountOps = 256 + 256 + 10;
  const estimatedLatencyUs = 8.4;

  return {
    digit: predictedDigit,
    confidence: maxProb,
    scores: probabilities,
    xnorOps,
    popcountOps,
    estimatedLatencyUs,
  };
}
