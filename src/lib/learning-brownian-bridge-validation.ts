/** Independent reconstruction from temporal incidence and Gaussian elimination.
 * Response strings never supply arithmetic operands; only bounded owned input does.
 * No producer or producer arithmetic is imported at runtime.
 */
import { Rational as Q, Z, O, sum } from './learning-graph-gauge-validation-math';
import { readBrownianBridgeInput, type BrownianBridgeInput } from './learning-brownian-bridge-input';
import { snapshotQubitWire } from './learning-qubit-wire';
import type { BrownianBridgeResult } from './learning-brownian-bridge';
type Matrix = Q[][];
const integer = (n: number) => new Q(BigInt(n));
const dot = (a: Q[], b: Q[]) => sum(a.map((x, i) => x.mul(b[i])));
const vector = (matrix: Matrix, values: Q[]) => matrix.map(row => dot(row, values));
const wire = (matrix: Matrix) => matrix.map(row => row.map(x => x.text()));
function inverse(matrix: Matrix) {
  const size = matrix.length, rows = matrix.map((row, i) => [...row, ...Array.from({ length: size }, (_, j) => i === j ? O : Z)]), pivots: Q[] = [];
  for (let k = 0; k < size; k++) {
    const pivot = rows[k][k]; if (pivot.n <= BigInt(0)) throw Error('Invalid bridge precision');
    pivots.push(pivot); rows[k] = rows[k].map(value => value.div(pivot));
    for (let i = 0; i < size; i++) if (i !== k) { const coefficient = rows[i][k]; rows[i] = rows[i].map((value, j) => value.sub(coefficient.mul(rows[k][j]))); }
  }
  return { covariance: rows.map(row => row.slice(size)), pivots };
}
function reconstruct(input: Readonly<BrownianBridgeInput>) {
  const n = input.intervals, size = n - 1, time = Q.read(input.time), diffusion = Q.read(input.diffusion), start = Q.read(input.start), end = Q.read(input.end), source = Q.read(input.source), probe = input.probe, step = time.div(integer(n));
  const scale = integer(2).mul(diffusion).mul(step);
  const incidence = Array.from({ length: n }, (_, edge) => Array.from({ length: size }, (__, j) => integer(Number(edge + 1 === j + 1) - Number(edge === j + 1))));
  const precision = Array.from({ length: size }, (_, i) => Array.from({ length: size }, (__, j) => sum(incidence.map(row => row[i].mul(row[j]))).div(scale)));
  const { covariance: innerCovariance, pivots } = inverse(precision);
  const covariance = Array.from({ length: n + 1 }, (_, i) => Array.from({ length: n + 1 }, (__, j) => i > 0 && i < n && j > 0 && j < n ? innerCovariance[i - 1][j - 1] : Z));
  const forcing = Array.from({ length: size }, () => Z);
  forcing[0] = forcing[0].add(start.div(scale)); forcing[size - 1] = forcing[size - 1].add(end.div(scale));
  const tiltedForcing = forcing.slice(); if (probe > 0 && probe < n) tiltedForcing[probe - 1] = tiltedForcing[probe - 1].add(source);
  const innerMean = vector(innerCovariance, forcing), tiltedInnerMean = vector(innerCovariance, tiltedForcing);
  const baselineMean = [start, ...innerMean, end], mean = [start, ...tiltedInnerMean, end];
  let logGenerating = dot(tiltedForcing, tiltedInnerMean).sub(dot(forcing, innerMean)).div(integer(2));
  if (probe === 0 || probe === n) logGenerating = logGenerating.add(source.mul(baselineMean[probe]));
  const baselineAction = sum(baselineMean.slice(1).map((m, i) => m.sub(baselineMean[i]).mul(m.sub(baselineMean[i])))).div(integer(4).mul(diffusion).mul(step));
  const determinant = pivots.reduce((a, b) => a.mul(b), O);
  // Square of the N increment prefactors times the (N-1)-dimensional Gaussian integral, with pi^-1 removed.
  let prefactorSquareTimesPi = O.div(determinant);
  for (let i = 0; i < n; i++) prefactorSquareTimesPi = prefactorSquareTimesPi.div(integer(4).mul(diffusion).mul(step));
  for (let i = 0; i < size; i++) prefactorSquareTimesPi = prefactorSquareTimesPi.mul(integer(2));
  const incrementMean = mean.slice(1).map((m, i) => m.sub(mean[i]));
  const incrementCovariance = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (__, j) => covariance[i + 1][j + 1].sub(covariance[i + 1][j]).sub(covariance[i][j + 1]).add(covariance[i][j])));
  const areaWeights = Array.from({ length: n + 1 }, (_, i) => i === 0 || i === n ? step.div(integer(2)) : step);
  const areaMean = dot(areaWeights, mean), areaVariance = dot(areaWeights, vector(covariance, areaWeights));
  const limitingAreaVariance = diffusion.mul(time).mul(time).mul(time).div(integer(6));
  const variation = sum(incrementMean.map((x, i) => x.mul(x).add(incrementCovariance[i][i])));
  const refinements = [1, 2, 4, 8].map(factor => {
    const count = n * factor, delta = time.div(integer(count)), q = probe * factor;
    const beta = Array.from({ length: count }, (_, i) => delta.mul(integer(count - i).sub(O.div(integer(2)))));
    const sourceIncrement = Array.from({ length: count }, (_, i) => integer(2).mul(diffusion).mul(delta).mul(integer(Number(i < q)).sub(integer(q).div(integer(count)))));
    const meanIncrement = sourceIncrement.map(c => end.sub(start).div(integer(count)).add(source.mul(c)));
    const incrementVariance = integer(2).mul(diffusion).mul(delta).mul(O.sub(O.div(integer(count))));
    const expectedVariation = sum(meanIncrement.map(x => x.mul(x).add(incrementVariance)));
    const betaSum = sum(beta);
    const areaVar = integer(2).mul(diffusion).mul(delta).mul(dot(beta, beta).sub(betaSum.mul(betaSum).div(integer(count))));
    return {
      intervals: count, integrationDimension: count - 1, matchedProbeIndex: q, timeStep: delta.text(), probeTime: integer(q).mul(delta).text(),
      meanAtProbe: start.add(sum(meanIncrement.slice(0, q))).text(),
      varianceAtProbe: integer(2).mul(diffusion).mul(delta).mul(integer(q).sub(integer(q * q).div(integer(count)))).text(),
      areaMean: time.mul(start).add(dot(beta, meanIncrement)).text(), areaVariance: areaVar.text(), areaVarianceGap: limitingAreaVariance.sub(areaVar).text(),
      expectedQuadraticVariation: expectedVariation.text(), expectedAction: expectedVariation.div(integer(4).mul(diffusion).mul(delta)).text(), fullMatrixCalculated: factor === 1,
    };
  });
  return {
    input, integrationDimension: size, timeStep: step.text(), times: Array.from({ length: n + 1 }, (_, i) => integer(i).mul(step).text()), baselineMean: baselineMean.map(x => x.text()), mean: mean.map(x => x.text()),
    precision: wire(precision), covariance: wire(covariance),
    factorization: { diagonal: pivots.map(x => x.text()), subdiagonal: Array.from({ length: size - 1 }, (_, i) => precision[i + 1][i].div(pivots[i]).text()), determinant: determinant.text() },
    kernel: { prefactorSquaredTimesPi: prefactorSquareTimesPi.text(), baselineNegativeLogExponential: baselineAction.text(), tiltedLogExponential: logGenerating.sub(baselineAction).text(), formula: 'K_tau(b,a)=exp[-(b-a)^2/(4 D tau)]/sqrt(4 pi D tau)' },
    source: { convention: 'dimensionless-exponent-point-source', probeTime: integer(probe).mul(step).text(), baselineMeanAtProbe: baselineMean[probe].text(), varianceAtProbe: covariance[probe][probe].text(), logGenerating: logGenerating.text(), meanResponse: covariance.map(row => row[probe].text()) },
    area: { baselineMean: dot(areaWeights, baselineMean).text(), mean: areaMean.text(), variance: areaVariance.text(), analyticLimitingVariance: limitingAreaVariance.text(), varianceGap: limitingAreaVariance.sub(areaVariance).text() },
    increments: { mean: incrementMean.map(x => x.text()), covariance: wire(incrementCovariance), deterministicQuadraticVariation: dot(incrementMean, incrementMean).text(), expectedQuadraticVariation: variation.text(), expectedAction: variation.div(integer(4).mul(diffusion).mul(step)).text(), baselineExpectedAction: sum(baselineMean.slice(1).map((x, i) => x.sub(baselineMean[i]).mul(x.sub(baselineMean[i])).add(incrementCovariance[i][i]))).div(integer(4).mul(diffusion).mul(step)).text(), fluctuationAction: sum(incrementCovariance.map((row, i) => row[i])).div(integer(4).mul(diffusion).mul(step)).text() },
    refinements,
    certificates: { meanResidual: precision.map((row, i) => dot(row, tiltedInnerMean).sub(tiltedForcing[i]).text()), inverseResidual: precision.map((row, i) => row.map((_, j) => sum(row.map((x, k) => x.mul(innerCovariance[k][j]))).sub(i === j ? O : Z).text())) },
    scope: { fullMatrixIntervals: n, largestAnalyticRefinementIntervals: n * 8, continuousMeasureExistenceProved: false, randomPathsGenerated: false, transcendentalDecimalEnclosuresComputed: false },
  };
}
function same(actual: unknown, expected: unknown): boolean {
  if (expected === null || typeof expected !== 'object') return actual === expected;
  if (!actual || typeof actual !== 'object' || Array.isArray(actual) !== Array.isArray(expected)) return false;
  if (Array.isArray(expected)) return Array.isArray(actual) && actual.length === expected.length && expected.every((value, i) => same(actual[i], value));
  const left = actual as Record<string, unknown>, right = expected as Record<string, unknown>, keys = Object.keys(right);
  return Object.keys(left).length === keys.length && keys.every(key => Object.hasOwn(left, key) && same(left[key], right[key]));
}
const cache = new Map<string, unknown>();
export function validateBrownianBridgeResult(source: unknown, value: unknown): BrownianBridgeResult | null {
  try {
    const input = readBrownianBridgeInput(source); if (!input) return null;
    const owned = snapshotQubitWire(value), key = JSON.stringify(input);
    let expected = cache.get(key);
    if (!expected) { expected = snapshotQubitWire(reconstruct(input)); if (cache.size === 4) cache.delete(cache.keys().next().value!); cache.set(key, expected); }
    return same(owned, expected) ? owned as BrownianBridgeResult : null;
  } catch { return null; }
}
