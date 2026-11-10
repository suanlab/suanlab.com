/** Known-scale, one-dimensional laboratories. No statistical guarantee is inferred from a finite sweep. */
const clip = (value: number, radius: number) => Math.max(-radius, Math.min(radius, value));
const huberLoss = (residual: number, threshold: number) => Math.abs(residual) <= threshold
  ? residual * residual / 2
  : threshold * (Math.abs(residual) - threshold / 2);
function fraction(numerator: number, denominator: number) {
  let a = Math.abs(numerator), b = denominator;
  while (b) [a, b] = [b, a % b];
  return {numerator, denominator, value: numerator / denominator,
    label: denominator / a === 1 ? String(numerator / a) : `${numerator / a}/${denominator / a}`};
}
function quarter(value: number, min: number, max: number) {
  if (!Number.isFinite(value) || value < min || value > max || !Number.isInteger(value * 4)) {
    throw new Error('Use the stated quarter-step robust-statistics controls.');
  }
}
/** Endpoints of the full minimizer interval, including even-sample plateaus.
 * Supported numerical domain: |sample| <= 1e6, 1e-6 <= threshold <= 1e6. */
export function huberInterval(sample: readonly number[], threshold: number): [number, number] {
  if (!sample.length || sample.some(x => !Number.isFinite(x) || Math.abs(x) > 1e6)
    || !Number.isFinite(threshold) || threshold < 1e-6 || threshold > 1e6) {
    throw new Error('Use a nonempty finite bounded sample and a supported positive Huber threshold.');
  }
  const endpoint = (right: boolean) => {
    let low = sample.reduce((a, b) => Math.min(a, b)) - threshold;
    let high = sample.reduce((a, b) => Math.max(a, b)) + threshold;
    for (let i = 0; i < 90; i++) {
      const mid = low + (high - low) / 2;
      if (mid === low || mid === high) break;
      const score = sample.reduce((sum, x) => sum + clip(x - mid, threshold), 0);
      if (score > 0 || (right && score === 0)) low = mid;
      else high = mid;
    }
    return low + (high - low) / 2;
  };
  return [endpoint(false), endpoint(true)];
}
/** Replace the highest m original lattice observations with z; n=21, clean target=0.
 * Replacement identity is retained even if z equals the original value. */
export function robustLocation(m: number, z: number, k: number, threshold: number) {
  if (!Number.isInteger(m) || m < 0 || m > 21 || !Number.isInteger(k) || k < 0 || k > 9) {
    throw new Error('Replacement count must be0..21 and trimming per tail0..9, both integers.');
  }
  quarter(z, -100, 100); quarter(threshold, .25, 4);
  // All permitted observations lie on a 1/20 lattice. Sum integer masses before dividing.
  const cleanUnits = Array.from({length: 21}, (_, i) => 4 * (i - 10));
  const sampleUnits = cleanUnits.map((x, i) => i >= 21 - m ? z * 20 : x);
  const ordered = sampleUnits.slice().sort((a, b) => a - b);
  const meanExact = fraction(sampleUnits.reduce((a, b) => a + b, 0), 420);
  const medianExact = fraction(ordered[10], 20);
  const trimmedExact = fraction(ordered.slice(k, 21 - k).reduce((a, b) => a + b, 0), 20 * (21 - 2 * k));
  const clean = cleanUnits.map(x => x / 20), sample = sampleUnits.map(x => x / 20);
  // Solve in integer lattice coordinates as well, preserving exact symmetric zero scores.
  // With21 observations a zero score cannot persist on a flat interval.
  // Recognize the exact lattice root rather than rounding arbitrary small estimates.
  const exactZero = sampleUnits.reduce((sum, x) => sum + clip(x, threshold * 20), 0) === 0;
  const intervalUnits = exactZero ? [0, 0] : huberInterval(sampleUnits, threshold * 20);
  const interval: [number, number] = [intervalUnits[0] / 20, intervalUnits[1] / 20];
  const huber = (interval[0] + interval[1]) / 2;
  const scoreResidual = sampleUnits.reduce((sum, x) => sum + clip(x - huber * 20, threshold * 20), 0) / 20;
  return {clean, sample, replaced: sample.map((_, i) => i >= 21 - m), meanExact, medianExact, trimmedExact,
    mean: meanExact.value, median: medianExact.value, trimmed: trimmedExact.value, interval, huber,
    lossAtHuber: sample.reduce((sum, x) => sum + huberLoss(x - huber, threshold), 0), scoreResidual,
    meanFormula: (m * z * 20 - 2 * m * (21 - m)) / 420,
    cleanTarget: 0, empiricalFraction: m / 21,
    replacementBreakdown: {mean: 1 / 21, median: 11 / 21, trimmed: (k + 1) / 21, huberKnownScale: 11 / 21}};
}
/** Population mixture (1-epsilon)Uniform[-1,1] + epsilon delta_z; known scale1.
 * Bias relative to the CLEAN target0, not sampling error or a confidence interval. */
export function contaminatedUniform(epsilon: number, z: number, threshold: number) {
  if (!Number.isFinite(epsilon) || epsilon < 0 || epsilon > .49) {
    throw new Error('Use an explicitly submajority mixture fraction0..0.49.');
  }
  quarter(z, -100, 100); quarter(threshold, .25, 4);
  const cleanScore = (theta: number) => (huberLoss(1 - theta, threshold) - huberLoss(1 + theta, threshold)) / 2;
  const score = (theta: number) => (1 - epsilon) * cleanScore(theta) + epsilon * clip(z - theta, threshold);
  let low = -1 - threshold, high = 1 + threshold;
  for (let i = 0; i < 90; i++) {
    const mid = low + (high - low) / 2;
    if (mid === low || mid === high) break;
    if (score(mid) > 0) low = mid; else high = mid;
  }
  const huber = epsilon === 0 || z === 0 ? 0 : low + (high - low) / 2;
  const medianBound = epsilon / (1 - epsilon), denominator = Math.min(threshold, 1);
  const influence = {mean: z, median: Math.sign(z), huber: clip(z, threshold) / denominator};
  return {epsilon, z, threshold, mean: epsilon * z, median: clip(z, medianBound), huber, influence,
    firstOrder: {mean: epsilon * influence.mean, median: epsilon * influence.median, huber: epsilon * influence.huber},
    cleanAsymptoticVariance: {mean: 1 / 3, median: 1, huber: threshold < 1 ? 1 - 2 * threshold / 3 : 1 / 3},
    scoreResidual: score(huber), cleanTarget: 0, uniquePopulationHuberRoot: true};
}
