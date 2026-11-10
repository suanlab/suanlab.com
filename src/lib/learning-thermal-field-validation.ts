/** Independent complete reconstruction. No producer engine, arithmetic or display imports.
 * Geometry: incidence rows; algebra: Gauss-Jordan; thermodynamics: log-det derivatives.
 */
import { CheckerRational as Q, checkerZero as Z, checkerOne as O, checkerInteger as int, checkerCanonicalReader } from './learning-real-time-validation-rational';
import { createRealTimeCheckerInterval, type CheckerInterval } from './learning-real-time-checker-interval';
import { readThermalFieldInput, thermalFieldData, type ThermalFieldInput, type ThermalJetInput } from './learning-thermal-field-input';
import { snapshotQubitWire } from './learning-qubit-wire';
import type { ThermalFieldResult } from './learning-thermal-field-protocol';

type Matrix = Q[][];
const read = Q.input, sum = (xs: Q[]) => xs.reduce((a, b) => a.add(b), Z);
const zeros = (n: number, m = n): Matrix => Array.from({ length: n }, () => Array.from({ length: m }, () => Z));
const dot = (a: Q[], b: Q[]) => sum(a.map((x, i) => x.mul(b[i])));
const transpose = (a: Matrix) => a[0].map((_, j) => a.map(row => row[j]));
const product = (a: Matrix, b: Matrix): Matrix => { const cols = transpose(b); return a.map(row => cols.map(col => dot(row, col))); };
const matvec = (a: Matrix, v: Q[]) => a.map(row => dot(row, v));
const trace = (a: Matrix) => sum(a.map((row, i) => row[i]));
const wire = (a: Matrix) => a.map(row => row.map(x => x.text()));
function gaussJordan(source: Matrix) {
  const n = source.length, a = source.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => i === j ? O : Z)]);
  let determinant = O;
  for (let k = 0; k < n; k++) {
    const pivot = a[k][k];
    if (pivot.n <= BigInt(0)) return { kind: 'not-positive' as const, index: k, pivot };
    determinant = determinant.mul(pivot);
    a[k] = a[k].map(x => x.div(pivot));
    for (let i = 0; i < n; i++) if (i !== k) {
      const coefficient = a[i][k];
      a[i] = a[i].map((x, j) => x.sub(coefficient.mul(a[k][j])));
    }
  }
  return { kind: 'positive' as const, inverse: a.map(row => row.slice(n)), determinant };
}
function reconstruct(input: ThermalFieldInput) {
  const shape = input.shape, n = shape.reduce((a, b) => a * b, 1), incidence: Matrix = [];
  // Decode each coordinate independently; no producer stride loop is reused.
  const coordinates = Array.from({ length: n }, (_, index) => {
    let remaining = index;
    return shape.map(length => { const at = remaining % length; remaining = Math.floor(remaining / length); return at; });
  });
  const address = new Map(coordinates.map((coordinate, i) => [coordinate.join(','), i]));
  for (const [i, coordinate] of coordinates.entries()) for (const [axis, length] of shape.entries()) {
    if (coordinate[axis] < length - 1 || input.boundary === 'periodic') {
      const neighbor = coordinate.slice(); neighbor[axis] = (neighbor[axis] + 1) % length;
      const row = Array.from({ length: n }, () => Z); row[i] = O.neg(); row[address.get(neighbor.join(','))!] = O; incidence.push(row);
    }
    if (input.boundary === 'dirichlet') for (const face of [0, length - 1]) if (coordinate[axis] === face) {
      const row = Array.from({ length: n }, () => Z); row[i] = O; incidence.push(row);
    }
  }
  const laplacian = product(transpose(incidence), incidence), spacing = read(input.spacing), T = read(input.temperature);
  const cell = shape.reduce(value => value.mul(spacing), O), volume = cell.mul(int(n));
  const precision = (key: keyof ThermalJetInput): Matrix => laplacian.map((row, i) => row.map((x, j) => cell.mul(
    read(input.stiffness[key]).mul(x).div(spacing.mul(spacing)).add(i === j ? read(input.mass[key]) : Z))));
  const K = precision('value'), K1 = precision('first'), K2 = precision('second');
  const source = (key: keyof ThermalJetInput) => input.sourceDensity.map(x => cell.mul(read(x[key])));
  const J = source('value'), J1 = source('first'), J2 = source('second');
  const E0 = volume.mul(read(input.backgroundDensity.value)), E1 = volume.mul(read(input.backgroundDensity.first)), E2 = volume.mul(read(input.backgroundDensity.second));
  const constrained = input.constraint === 'mean-zero', q = n - Number(constrained), B = zeros(n, q);
  for (let j = 0; j < q; j++) { B[j][j] = O; if (constrained) B[n - 1][j] = O.neg(); }
  const A = product(product(transpose(B), K), B), inverse = gaussJordan(A);
  if (inverse.kind !== 'positive') return { kind: 'not-normalizable' as const, dimension: q, failedPivotIndex: inverse.index,
    failedPivot: inverse.pivot.text(), precision: wire(K), laplacian: wire(laplacian),
    explanation: 'Not positive definite on the declared space; a zero pivot alone does not classify singularity.' };
  const G = product(product(B, inverse.inverse), transpose(B)), mean = matvec(G, J);
  const covariance = G.map(row => row.map(x => x.mul(T))), gram = constrained ? int(n) : O, determinant = inverse.determinant.div(gram);
  const minimum = E0.sub(dot(J, mean).div(int(2)));
  const K1mean = matvec(K1, mean), responseSource = J1.map((x, i) => x.sub(K1mean[i]));
  const minimumFirst = E1.sub(dot(J1, mean)).add(dot(mean, K1mean).div(int(2)));
  const minimumSecond = E2.sub(dot(J2, mean)).sub(dot(responseSource, matvec(G, responseSource))).add(dot(mean, matvec(K2, mean)).div(int(2)));
  const GK1 = product(G, K1), detLogFirst = trace(GK1), detLogSecond = trace(product(G, K2)).sub(trace(product(GK1, GK1)));
  // Differentiate q/2 log(T) - 1/2 log(det K_H) - minimum(T)/T.
  const U = minimum.sub(T.mul(minimumFirst)).add(int(q).mul(T).div(int(2))).sub(T.mul(T).mul(detLogFirst).div(int(2)));
  const Cv = int(q).div(int(2)).sub(T.mul(detLogFirst)).sub(T.mul(T).mul(detLogSecond).div(int(2))).sub(T.mul(minimumSecond));
  const expectedSecond = E2.sub(dot(J2, mean)).add(trace(product(K2, covariance)).add(dot(mean, matvec(K2, mean))).div(int(2)));
  return { kind: 'normalizable' as const, dimension: q, sites: n, cellVolume: cell.text(), volume: volume.text(),
    laplacian: wire(laplacian), precision: wire(K), precisionFirst: wire(K1), precisionSecond: wire(K2), source: J.map(x => x.text()),
    mean: mean.map(x => x.text()), covariance: wire(covariance), inverseOnSpace: wire(G), restrictedDeterminant: determinant.text(),
    coordinateDeterminant: inverse.determinant.text(), gramDeterminant: gram.text(), minimumEnergy: minimum.text(),
    expectedEnergy: minimum.add(int(q).mul(T).div(int(2))).text(), thermalEnergy: U.text(),
    varianceAdjustedEnergy: Cv.add(T.mul(expectedSecond)).mul(T).mul(T).text(), expectedSecondThermalDerivative: expectedSecond.text(), heatCapacity: Cv.text(),
    logPartition: { dimension: q, temperature: T.text(), restrictedDeterminant: determinant.text(), referenceScale: input.referenceScale, constant: minimum.neg().div(T).text() },
    entropyRationalAddend: U.div(T).text(), freeEnergyMultiplier: T.neg().text() };
}
type Expected = ReturnType<typeof reconstruct>;
const context = createRealTimeCheckerInterval(2048);
function observationTruth(exact: Extract<Expected, { kind: 'normalizable' }>) {
  const T = read(exact.logPartition.temperature), logRatio = (q: Q) => context.sub(
    context.log(context.point(q.n * context.S)), context.log(context.point(q.d * context.S)));
  // Logging integers separately avoids rounding a tiny positive determinant to zero.
  const form = exact.logPartition, log2pi = context.log(context.scale(context.pi(), 2));
  let logZ = context.divideInteger(context.scale(context.add(log2pi, logRatio(T)), form.dimension), 2);
  logZ = context.sub(logZ, context.divideInteger(logRatio(read(form.restrictedDeterminant)), 2));
  logZ = context.add(context.sub(logZ, context.scale(logRatio(read(form.referenceScale)), form.dimension)), context.from(read(form.constant)));
  return { logPartition: logZ, freeEnergy: context.multiply(context.from(T.neg()), logZ),
    thermodynamicEntropy: context.add(logZ, context.from(read(exact.entropyRationalAddend))),
    gibbsEntropy: context.add(logZ, context.from(read(exact.expectedEnergy).div(T))) };
}
function same(a: unknown, b: unknown): boolean {
  if (!b || typeof b !== 'object') return a === b;
  if (!a || typeof a !== 'object' || Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(b)) return Array.isArray(a) && a.length === b.length && b.every((x, i) => same(a[i], x));
  const left = a as Record<string, unknown>, right = b as Record<string, unknown>, keys = Object.keys(right);
  return Object.keys(left).length === keys.length && keys.every(k => Object.hasOwn(left, k) && same(left[k], right[k]));
}
const canonical = checkerCanonicalReader(), quantumDenominator = BigInt(1) << BigInt(512);
function encloses(value: unknown, truth: CheckerInterval) {
  const d = thermalFieldData(value, ['lower', 'upper']);
  if (!d) return false;
  const lower = canonical(d.lower), upper = canonical(d.upper);
  if (quantumDenominator % lower.d || quantumDenominator % upper.d) return false;
  const width = upper.sub(lower);
  return width.n >= BigInt(0) && width.n * quantumDenominator <= BigInt(6) * width.d &&
    lower.n * context.S <= truth.lower * lower.d && upper.n * context.S >= truth.upper * upper.d;
}
const cache = new Map<string, { exact: Expected; truth: ReturnType<typeof observationTruth> | null }>();
export function validateThermalFieldResult(source: ThermalFieldInput, value: unknown): ThermalFieldResult | null {
  try {
    const input = readThermalFieldInput(source);
    if (!input) return null;
    const owned = snapshotQubitWire(value), d = thermalFieldData(owned, ['exact', 'observations']);
    if (!d) return null;
    const key = JSON.stringify(input);
    let expected = cache.get(key);
    if (!expected) {
      const exact = reconstruct(input), truth = exact.kind === 'normalizable' ? observationTruth(exact) : null;
      expected = { exact, truth };
      if (cache.size === 4) cache.delete(cache.keys().next().value!);
      cache.set(key, expected);
    }
    if (!same(d.exact, expected.exact)) return null;
    if (!expected.truth) return d.observations === null ? owned as ThermalFieldResult : null;
    const observations = thermalFieldData(d.observations, ['precisionBits', 'interpretation', 'logPartition', 'freeEnergy', 'thermodynamicEntropy', 'gibbsEntropy']);
    if (!observations || observations.precisionBits !== 512 || observations.interpretation !== 'finite-gaussian-fixed-reference-measure') return null;
    for (const key of ['logPartition', 'freeEnergy', 'thermodynamicEntropy', 'gibbsEntropy'] as const) if (!encloses(observations[key], expected.truth[key])) return null;
    return owned as ThermalFieldResult;
  } catch { return null; }
}
