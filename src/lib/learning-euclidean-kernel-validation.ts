/** Independent incidence/LDL solves and Gaussian kernel convolution.
 * Only trusted owned input supplies operands. Candidate fractions are never parsed.
 */
import { Rational as Q, Z, O, T, sum } from './learning-graph-gauge-validation-math';
import { readEuclideanKernelInput, type EuclideanKernelInput } from './learning-euclidean-kernel-input';
import { snapshotQubitWire } from './learning-qubit-wire';
import type { EuclideanKernelResult } from './learning-euclidean-kernel';
const integer = (n: number) => new Q(BigInt(n));
const square = (x: Q) => x.mul(x);
const dot = (a: Q[], b: Q[]) => sum(a.map((x, i) => x.mul(b[i])));
const wire = (matrix: Q[][]) => matrix.map(row => row.map(x => x.text()));
type Kernel = { p: Q; left: Q; right: Q; cross: Q };
function compose(after: Kernel, before: Kernel): Kernel {
  const denominator = after.right.add(before.left), fourDenominator = integer(4).mul(denominator);
  return { p: after.p.mul(before.p).div(denominator), left: after.left.sub(square(after.cross).div(fourDenominator)), right: before.right.sub(square(before.cross).div(fourDenominator)), cross: after.cross.mul(before.cross).div(T.mul(denominator)) };
}
function kernels(step: Q, mass: Q, hbar: Q, omega: Q) {
  const scale = mass.div(hbar.mul(step)), diagonal = scale.div(T).add(mass.mul(omega).mul(step).div(integer(4).mul(hbar)));
  const cache = new Map<number, Kernel>([[1, { p: scale.div(T), left: diagonal, right: diagonal, cross: scale }]]);
  const get = (count: number): Kernel => {
    const existing = cache.get(count); if (existing) return existing;
    const half = get(Math.floor(count / 2)), doubled = compose(half, half), result = count % 2 ? compose(cache.get(1)!, doubled) : doubled;
    if (result.left.text() !== result.right.text()) throw Error('Asymmetric harmonic convolution');
    cache.set(count, result); return result;
  };
  return get;
}
function solveFactorization(matrix: Q[][]) {
  const size = matrix.length, diagonal: Q[] = [], lower: Q[] = [];
  for (let i = 0; i < size; i++) {
    if (i > 0) lower.push(matrix[i][i - 1].div(diagonal[i - 1]));
    diagonal.push(i ? matrix[i][i].sub(square(lower[i - 1]).mul(diagonal[i - 1])) : matrix[i][i]);
    if (diagonal[i].n <= BigInt(0)) throw Error('Nonpositive pinned precision');
  }
  const solve = (rhs: Q[]) => {
    const forward: Q[] = [];
    for (let i = 0; i < size; i++) forward.push(i ? rhs[i].sub(lower[i - 1].mul(forward[i - 1])) : rhs[i]);
    const answer = forward.map((x, i) => x.div(diagonal[i]));
    for (let i = size - 2; i >= 0; i--) answer[i] = answer[i].sub(lower[i].mul(answer[i + 1]));
    return answer;
  };
  const columns = Array.from({ length: size }, (_, j) => solve(Array.from({ length: size }, (__, i) => i === j ? O : Z)));
  return { diagonal, lower, solve, covariance: Array.from({ length: size }, (_, i) => columns.map(column => column[i])) };
}
function indexed(matrix: Q[][]) {
  const values: string[] = [], ids = new Map<string, number>();
  const indices = matrix.map(row => row.map(value => {
    const key = value.text(); if (!ids.has(key)) { ids.set(key, values.length); values.push(key); } return ids.get(key)!;
  }));
  return { values: Array.from({ length: Math.ceil(values.length / 128) }, (_, i) => values.slice(i * 128, (i + 1) * 128)), indices };
}
function reconstruct(input: Readonly<EuclideanKernelInput>) {
  const n = input.intervals, size = n - 1, time = Q.read(input.time), mass = Q.read(input.mass), hbar = Q.read(input.hbar), omega = Q.read(input.omegaSquared), start = Q.read(input.start), end = Q.read(input.end), step = time.div(integer(n));
  const scale = mass.div(hbar.mul(step)), potentialPrecision = mass.mul(omega).mul(step).div(hbar);
  const incidence = Array.from({ length: n }, (_, edge) => Array.from({ length: size }, (__, i) => integer(Number(edge + 1 === i + 1) - Number(edge === i + 1))));
  const fixed = Array.from({ length: n }, (_, i) => (i === 0 ? start.neg() : Z).add(i === n - 1 ? end : Z));
  const precision = Array.from({ length: size }, (_, i) => Array.from({ length: size }, (__, j) => scale.mul(sum(incidence.map(row => row[i].mul(row[j])))).add(i === j ? potentialPrecision : Z)));
  const forcing = Array.from({ length: size }, (_, i) => scale.mul(sum(incidence.map((row, edge) => row[i].mul(fixed[edge])))).neg());
  const factor = solveFactorization(precision), innerMean = factor.solve(forcing), mean = [start, ...innerMean, end];
  const covariance = Array.from({ length: n + 1 }, (_, i) => Array.from({ length: n + 1 }, (__, j) => i > 0 && j > 0 && i < n && j < n ? factor.covariance[i - 1][j - 1] : Z));
  const determinant = factor.diagonal.reduce((a, b) => a.mul(b), O);
  let prefactor = O.div(determinant);
  for (let i = 0; i < n; i++) prefactor = prefactor.mul(scale.div(T));
  for (let i = 0; i < size; i++) prefactor = prefactor.mul(T);
  const g = scale.div(T).add(potentialPrecision.div(integer(4))).sub(square(scale).mul(factor.covariance[0][0]).div(T));
  const h = square(scale).mul(factor.covariance[0][size - 1]);
  const constant = scale.mul(dot(fixed, fixed)).div(T).add(potentialPrecision.mul(square(start).add(square(end))).div(integer(4)));
  const minimum = constant.sub(dot(forcing, innerMean).div(T));
  const rightShift = step.mul(mass).mul(omega).mul(square(start).sub(square(end))).div(integer(4).mul(hbar));
  const kinetic = scale.mul(sum(mean.slice(1).map((x, i) => square(x.sub(mean[i])).add(covariance[i + 1][i + 1]).add(covariance[i][i]).sub(T.mul(covariance[i][i + 1]))))).div(T);
  const potential = potentialPrecision.mul(square(start).add(square(end)).div(T).add(sum(innerMean.map((x, i) => square(x).add(factor.covariance[i][i]))))).div(T);
  const isFree = omega.n === BigInt(0), curvature = T.mul(g).sub(h);
  let principal = O;
  const recurrence = [O, ...factor.diagonal.map(pivot => { principal = principal.mul(pivot.div(scale)); return principal; })];
  const refinements = [1, 2, 4, 8].map(multiplier => {
    const count = n * multiplier, probe = input.probe * multiplier, delta = time.div(integer(count)), get = kernels(delta, mass, hbar, omega), all = get(count);
    let mu = probe === 0 ? start : end, variance = Z;
    if (probe > 0 && probe < count) {
      const left = get(probe), right = get(count - probe), denominator = left.left.add(right.right);
      mu = left.cross.mul(start).add(right.cross.mul(end)).div(T.mul(denominator)); variance = O.div(T.mul(denominator));
    }
    return { intervals: count, integrationDimension: count - 1, matchedProbeIndex: probe, timeStep: delta.text(), probeTime: integer(probe).mul(delta).text(), meanAtProbe: mu.text(), varianceAtProbe: variance.text(), prefactorSquaredTimesPi: all.p.text(), diagonalExponentCoefficient: all.left.text(), crossExponentCoefficient: all.cross.text(), minimumAction: all.left.mul(square(end)).add(all.right.mul(square(start))).sub(all.cross.mul(start).mul(end)).text(), rightLogCorrection: rightShift.div(integer(multiplier)).text(), traceSquared: isFree ? null : all.p.div(all.left.add(all.right).sub(all.cross)).text(), fullMatrixCalculated: multiplier === 1 };
  });
  return {
    input, integrationDimension: size, timeStep: step.text(), inverseTemperature: time.div(hbar).text(), times: Array.from({ length: n + 1 }, (_, i) => step.mul(integer(i)).text()), mean: mean.map(x => x.text()), precision: wire(precision), covariance: indexed(covariance), endpointForcing: forcing.map(x => x.text()),
    factorization: { diagonal: factor.diagonal.map(x => x.text()), subdiagonal: factor.lower.map(x => x.text()), determinant: determinant.text(), recurrence: recurrence.map(x => x.text()) },
    kernel: { prefactorSquaredTimesPi: prefactor.text(), diagonalExponentCoefficient: g.text(), crossExponentCoefficient: h.text(), actionConstant: constant.text(), minimumAction: minimum.text(), rightLogCorrection: rightShift.text(), rightLogExponential: rightShift.sub(minimum).text(), symmetricUnderEndpointReversal: true },
    trace: { status: isFree ? 'divergent-unbounded-line' : 'finite', diagonalCurvature: curvature.text(), squared: isFree ? null : prefactor.div(curvature).text() },
    endpointIntegral: { prefactorSquared: prefactor.div(g).text(), negativeLogQuadraticCoefficient: g.sub(square(h).div(integer(4).mul(g))).text(), conservesMass: isFree },
    expectedAction: { kinetic: kinetic.text(), potential: potential.text(), total: kinetic.add(potential).text(), minimum: minimum.text(), fluctuation: integer(size).div(T).text() },
    refinements, certificates: { meanResidual: precision.map((row, i) => dot(row, innerMean).sub(forcing[i]).text()), inverseResidual: precision.map((row, i) => row.map((_, j) => sum([i - 1, i, i + 1].filter(k => k >= 0 && k < size).map(k => row[k].mul(factor.covariance[k][j]))).sub(i === j ? O : Z).text())) },
    scope: { fullMatrixIntervals: n, largestAnalyticRefinementIntervals: n * 8, covarianceRepresentation: 'indexed-exact-fractions', exactFiniteRule: 'trapezoidal-potential', continuousKernelEvaluated: false, transcendentalDecimalEnclosuresComputed: false, generalOperatorConvergenceProved: false, randomPathsGenerated: false },
  };
}
function same(actual: unknown, expected: unknown): boolean {
  if (expected === null || typeof expected !== 'object') return actual === expected;
  if (!actual || typeof actual !== 'object' || Array.isArray(actual) !== Array.isArray(expected)) return false;
  if (Array.isArray(expected)) return Array.isArray(actual) && actual.length === expected.length && expected.every((value, i) => same(actual[i], value));
  const a = actual as Record<string, unknown>, b = expected as Record<string, unknown>, keys = Object.keys(b);
  return Object.keys(a).length === keys.length && keys.every(key => Object.hasOwn(a, key) && same(a[key], b[key]));
}
const cache = new Map<string, unknown>();
export function validateEuclideanKernelResult(source: unknown, value: unknown): EuclideanKernelResult | null {
  try {
    const input = readEuclideanKernelInput(source); if (!input) return null;
    const owned = snapshotQubitWire(value), key = JSON.stringify(input);
    let expected = cache.get(key);
    if (!expected) { expected = snapshotQubitWire(reconstruct(input)); if (cache.size === 4) cache.delete(cache.keys().next().value!); cache.set(key, expected); }
    return same(owned, expected) ? owned as EuclideanKernelResult : null;
  } catch { return null; }
}
