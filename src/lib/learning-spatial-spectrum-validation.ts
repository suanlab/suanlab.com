/** Independent entire-response checking. Exact finite values come from an
 * incidence Gram matrix,bordered Gauss-Jordan inverse and Bareiss minors.
 * Fourier sums use combined phases,not the producer's two-pass transform.
 * All arithmetic operands are owned input or locally derived expectations.
 */
import { readSpatialSpectrumInput, type SpatialSpectrumInput } from './learning-spatial-spectrum-input';
import type { SpatialSpectrumResult } from './learning-spatial-spectrum';
import { snapshotQubitWire } from './learning-qubit-wire';
import { f, int, zero, one, add, sub, mul, div, neg, eq, le, point, plus, minus, times, quotient, squared, root, scale, rationalTotal, text, parse, inputFraction, pi, trigContext, independentSpatialLog, independentSpatialLogPoint, type F, type Range } from './learning-spatial-spectrum-validation-math';

type Matrix = F[][];
const zeros = (n: number, m = n): Matrix => Array.from({ length: n }, () => Array.from({ length: m }, () => zero));
const transpose = (a: Matrix) => a[0].map((_, j) => a.map(row => row[j]));
const product = (a: Matrix, b: Matrix): Matrix => a.map(row => b[0].map((_, j) => rationalTotal(row.map((v, k) => mul(v, b[k][j])))));
const wire = (a: Matrix) => a.map(row => row.map(text));
function determinant(source: Matrix): F {
  if (!source.length) return one;
  const a = source.map(row => row.slice()), n = a.length;
  let previous = one, sign = one;
  for (let k = 0; k < n - 1; k++) {
    let row = k; while (row < n && eq(a[row][k], zero)) row++;
    if (row === n) return zero;
    if (row !== k) { [a[row], a[k]] = [a[k], a[row]]; sign = neg(sign); }
    const pivot = a[k][k];
    for (let i = k + 1; i < n; i++) for (let j = k + 1; j < n; j++) a[i][j] = div(sub(mul(pivot, a[i][j]), mul(a[i][k], a[k][j])), previous);
    for (let i = k + 1; i < n; i++) a[i][k] = zero;
    previous = pivot;
  }
  return mul(sign, a[n - 1][n - 1]);
}
function inverse(source: Matrix): Matrix {
  const n = source.length, a = source.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => i === j ? one : zero)]);
  for (let k = 0; k < n; k++) {
    let row = k; while (row < n && eq(a[row][k], zero)) row++;
    if (row === n) throw Error('Singular declared ensemble');
    [a[k], a[row]] = [a[row], a[k]];
    const pivot = a[k][k]; a[k] = a[k].map(v => div(v, pivot));
    for (let i = 0; i < n; i++) if (i !== k) { const factor = a[i][k]; a[i] = a[i].map((v, j) => sub(v, mul(factor, a[k][j]))); }
  }
  return a.map(row => row.slice(n));
}

class Enclosure {
  constructor(readonly value: Range, readonly exactZero = false, readonly nonnegativeMaximum = false) {}
}
const enclosure = (value: Range) => new Enclosure(value);
const exactZero = () => new Enclosure(point(zero), true);
const abs = (x: F) => x.n < BigInt(0) ? neg(x) : x;
function matches(actual: unknown, expected: unknown): boolean {
  if (expected instanceof Enclosure) {
    if (!actual || typeof actual !== 'object' || Array.isArray(actual)) return false;
    const row = actual as Record<string, unknown>;
    if (Object.keys(row).length !== 2 || typeof row.lower !== 'string' || typeof row.upper !== 'string') return false;
    const lower = parse(row.lower), upper = parse(row.upper), grid = BigInt(1) << BigInt(512);
    if (grid % lower.d !== BigInt(0) || grid % upper.d !== BigInt(0) || !le(lower, upper)) return false;
    if (expected.exactZero) return eq(lower, zero) && eq(upper, zero);
    if (expected.nonnegativeMaximum && !eq(lower, zero)) return false;
    const tolerance = mul(f(BigInt(1), BigInt(10) ** BigInt(90)), add(one, add(abs(expected.value.lower), abs(expected.value.upper))));
    return le(lower, expected.value.lower) && le(expected.value.upper, upper) && le(sub(upper, lower), tolerance);
  }
  if (expected === null || typeof expected !== 'object') return actual === expected;
  if (!actual || typeof actual !== 'object' || Array.isArray(actual) !== Array.isArray(expected)) return false;
  if (Array.isArray(expected)) return Array.isArray(actual) && actual.length === expected.length && expected.every((v, i) => matches(actual[i], v));
  const a = actual as Record<string, unknown>, e = expected as Record<string, unknown>, keys = Object.keys(e);
  return Object.keys(a).length === keys.length && keys.every(key => Object.hasOwn(a, key) && matches(a[key], e[key]));
}

function reconstruct(input: Readonly<SpatialSpectrumInput>) {
  const n = input.sites, a = inputFraction(input.spacing), mass = inputFraction(input.massSquared), c2 = inputFraction(input.speedSquared), T = inputFraction(input.temperature);
  const M = mul(a, mass), kappa = div(c2, a), periodic = input.boundary === 'periodic', pinned = input.boundary === 'dirichlet', constrained = input.constraint === 'mean-zero', dimension = constrained ? n - 1 : n;
  const incidence: Matrix = [];
  for (let j = 0; j < n - 1; j++) { const edge = Array.from({ length: n }, () => zero); edge[j] = neg(one); edge[j + 1] = one; incidence.push(edge); }
  if (periodic) { const edge = Array.from({ length: n }, () => zero); edge[n - 1] = neg(one); edge[0] = one; incidence.push(edge); }
  if (pinned) for (const j of [0, n - 1]) { const pin = Array.from({ length: n }, () => zero); pin[j] = one; incidence.push(pin); }
  const laplacian = product(transpose(incidence), incidence), K = laplacian.map((row, i) => row.map((v, j) => add(mul(kappa, v), i === j ? M : zero)));
  const expanded = constrained ? [...K.map(row => [...row, one]), [...Array.from({ length: n }, () => one), zero]] : K;
  const G = inverse(expanded).slice(0, n).map(row => row.slice(0, n)), C = G.map(row => row.map(v => mul(T, v)));
  const B = zeros(n, dimension), P = zeros(n);
  for (let j = 0; j < dimension; j++) { B[j][j] = one; if (constrained) B[n - 1][j] = neg(one); }
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) P[i][j] = sub(i === j ? one : zero, constrained ? div(one, int(n)) : zero);
  const reduced = product(product(transpose(B), K), B), diagonal: string[] = [];
  let previous = one;
  for (let j = 1; j <= dimension; j++) { const det = determinant(reduced.slice(0, j).map(row => row.slice(0, j))), pivot = div(det, previous); if (pivot.n <= BigInt(0)) throw Error('Nonpositive reduced precision'); diagonal.push(text(pivot)); previous = det; }
  const green = product(product(P, K), C).map((row, i) => row.map((v, j) => sub(v, mul(T, P[i][j]))));
  const projection = product(P, C).map((row, i) => row.map((v, j) => sub(v, C[i][j])));
  if ([...green, ...projection].some(row => row.some(v => !eq(v, zero)))) throw Error('Independent Green account failed');

  const trig = trigContext(), PI = pi(), cz = () => ({ real: exactZero(), imaginary: exactZero() });
  // Combine both phase factors before summation. This differs from the
  // producer's sequence of right and left matrix products.
  function transform(matrix: Matrix, covariance: boolean) {
    const coefficients = matrix.map(row => row.map(v => point(div(v, int(n)))));
    const phases = Array.from({ length: n }, (_, j) => trig(f(BigInt(2 * j), BigInt(n))));
    return Array.from({ length: n }, (_, l) => Array.from({ length: n }, (_, m) => {
      if ((periodic && l !== m) || (covariance && constrained && (l === 0 || m === 0))) return cz();
      // Accumulate coefficients sharing the same combined phase before
      // multiplication. This preserves directed rounding and the full matrix
      // while avoiding repeated large-rational products at maximum input.
      const groups = Array.from({ length: n }, () => point(zero));
      for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
        const index = ((m * k - l * j) % n + n) % n;
        groups[index] = plus(groups[index], coefficients[j][k]);
      }
      let real = point(zero), imaginary = point(zero);
      for (let j = 0; j < n; j++) {
        real = plus(real, times(phases[j].cos, groups[j]));
        if (l !== m) imaginary = plus(imaginary, times(phases[j].sin, groups[j]));
      }
      return { real: enclosure(real), imaginary: l === m ? exactZero() : enclosure(imaginary) };
    }));
  }
  const spectral = transform(C, true), operator = transform(K, false);
  function exactSinSquared(sites: number, wave: number): F | null {
    const q = f(BigInt((wave % sites + sites) % sites), BigInt(sites));
    if (eq(q, zero)) return zero;
    if (eq(q, f(BigInt(1), BigInt(2)))) return one;
    if ([1, 5].some(k => eq(q, f(BigInt(k), BigInt(6))))) return f(BigInt(1), BigInt(4));
    if ([1, 3].some(k => eq(q, f(BigInt(k), BigInt(4))))) return f(BigInt(1), BigInt(2));
    if ([1, 2].some(k => eq(q, f(BigInt(k), BigInt(3))))) return f(BigInt(3), BigInt(4));
    return null;
  }
  function symbol(sites: number, spacing: F, wave: number) {
    const ss = exactSinSquared(sites, wave);
    const exact = ss === null ? null : add(mul(spacing, mass), mul(div(mul(int(4), c2), spacing), ss));
    // Use1-cos(2θ),rather than the producer's squared half-angle sine.
    const lambda = exact === null ? plus(point(mul(spacing, mass)), scale(minus(point(one), trig(f(BigInt(2 * wave), BigInt(sites))).cos), div(mul(int(2), c2), spacing))) : point(exact);
    return { lambda, exact };
  }
  const waveNumber = (wave: number) => scale(PI, div(int(2 * wave), mul(a, int(n))));
  const modes = Array.from({ length: n }, (_, index) => {
    const signedWave = 2 * index <= n ? index : index - n, s = periodic ? symbol(n, a, index) : null, power = spectral[index][index].real;
    return { index, signedWave, aliasPartner: (n - index) % n, wavenumber: enclosure(waveNumber(signedWave)), power, continuumNormalizedPower: power.exactZero ? exactZero() : enclosure(scale(power.value, a)), excludedConstant: constrained && index === 0, periodicEnergyEigenvalue: s ? enclosure(s.lambda) : null, exactPeriodicEnergyEigenvalue: s?.exact ? text(s.exact) : null };
  });
  const curve = Array.from({ length: n }, (_, d) => { const site = periodic ? (input.origin + d) % n : d, steps = periodic ? d : site - input.origin; return { site, steps, distance: text(mul(a, int(steps))), covariance: text(C[input.origin][site]) }; });
  const localRatios = curve.slice(0, -1).map((p, d) => {
    const q = curve[d + 1], x = C[input.origin][p.site], y = C[input.origin][q.site];
    const status = !periodic ? 'boundary-nonuniform' : x.n <= BigInt(0) || y.n <= BigInt(0) ? 'nonpositive' : eq(x, y) ? 'equal' : le(x, y) ? 'increasing' : 'positive-local-ratio-only';
    return { from: p.site, to: q.site, status, length: status === 'positive-local-ratio-only' ? enclosure(quotient(point(a), independentSpatialLogPoint(div(x, y)))) : null };
  });

  const infinite = (() => {
    if (eq(mass, zero)) return { status: 'massless-no-stationary-field', decay: null, amplitude: null, correlationLength: null, continuumLength: null, periodicImageApplicable: false, curve: null, periodicImage: null };
    if (eq(c2, zero)) {
      const amp = div(T, M);
      return { status: 'independent-infinite-reference', decay: exactZero(), amplitude: enclosure(point(amp)), correlationLength: exactZero(), continuumLength: exactZero(), periodicImageApplicable: false, curve: Array.from({ length: n }, (_, j) => j === (periodic ? 0 : input.origin) ? enclosure(point(amp)) : exactZero()), periodicImage: null };
    }
    const u = div(M, kappa), discriminant = root(point(mul(u, add(u, int(4)))));
    const r = scale(minus(point(add(int(2), u)), discriminant), f(BigInt(1), BigInt(2)));
    const amp = quotient(point(T), root(point(mul(M, add(M, mul(int(4), kappa))))));
    const coshArgument = add(one, div(u, int(2))), alpha = independentSpatialLog(plus(point(coshArgument), root(point(sub(mul(coshArgument, coshArgument), one)))));
    const powers = [point(one)]; for (let j = 1; j <= n; j++) powers.push(times(powers[j - 1], r));
    const image = periodic ? Array.from({ length: n }, (_, d) => {
      const unconditioned = quotient(times(amp, plus(powers[d], powers[n - d])), minus(point(one), powers[n]));
      return enclosure(constrained ? minus(unconditioned, point(div(T, mul(int(n), M)))) : unconditioned);
    }) : null;
    return { status: 'massive-correlated', decay: enclosure(r), amplitude: enclosure(amp), correlationLength: enclosure(quotient(point(a), alpha)), continuumLength: enclosure(root(point(div(c2, mass)))), periodicImageApplicable: periodic, curve: Array.from({ length: n }, (_, j) => enclosure(times(amp, powers[periodic ? j : Math.abs(j - input.origin)]))), periodicImage: image };
  })();
  const signed = 2 * input.selectedMode <= n ? input.selectedMode : input.selectedMode - n, k = waveNumber(signed), continuum = plus(point(mass), scale(squared(k), c2));
  const refinements = periodic ? Array.from({ length: 4 }, (_, level) => {
    const sites = n * 2 ** level, spacing = div(a, int(2 ** level)), s = symbol(sites, spacing, signed), excluded = constrained && signed === 0;
    const power = excluded ? point(zero) : quotient(point(T), s.lambda);
    return { level, sites, spacing: text(spacing), fixedBoxLength: text(mul(a, int(n))), signedWave: signed, arrayIndex: (signed % sites + sites) % sites, wavenumber: enclosure(k), operatorEigenvalue: enclosure(scale(s.lambda, div(one, spacing))), continuumOperatorEigenvalue: enclosure(continuum), operatorGap: enclosure(minus(continuum, scale(s.lambda, div(one, spacing)))), unitaryPower: excluded ? exactZero() : enclosure(power), continuumNormalizedPower: excluded ? exactZero() : enclosure(scale(power, spacing)) };
  }) : null;
  const massless = periodic && eq(mass, zero) && constrained ? Array.from({ length: n }, (_, d) => text(mul(div(T, kappa), sub(f(BigInt(n * n - 1), BigInt(12 * n)), f(BigInt(d * (n - d)), BigInt(2 * n)))))) : null;
  const trace = rationalTotal(C.map((row, i) => row[i]));
  return {
    input, integrationDimension: dimension, precisionBits: 512, workBits: 768, squareRootBits: 512, spacing: text(a), boxLength: text(mul(a, int(pinned ? n + 1 : n))), positions: Array.from({ length: n }, (_, j) => text(mul(a, pinned ? int(j + 1) : periodic ? int(j) : f(BigInt(2 * j + 1), BigInt(2))))), onsiteCoefficient: text(M), edgeCoefficient: text(kappa), precision: wire(K), laplacian: wire(laplacian), basis: wire(B), projector: wire(P), reducedPrecision: wire(reduced), factorization: { diagonal, reducedDeterminant: text(previous) }, covariance: wire(C), fourierCovariance: spectral, fourierPrecision: operator, modes,
    selectedReconstruction: C[input.origin].map(v => ({ real: enclosure(point(v)), imaginary: enclosure(point(zero)) })), masslessPeriodicGreen: massless, curve, localRatios, infinite, refinements,
    certificates: { projectedGreenResidual: wire(green), projectionResidual: wire(projection), trace: text(trace), traceResidual: enclosure(point(zero)), hermitianResidualMaximum: new Enclosure(point(zero), false, true), covarianceRowSums: C.map(row => text(rationalTotal(row))), periodicTranslationInvariant: periodic && C.every((row, i) => row.every((v, j) => eq(v, C[0][(j - i + n) % n]))) },
    contract: 'Physical Gaussian energy K=a*mSquared*I+(speedSquared/a)*L on3..16 sites, zero source. Counting coordinates with Lebesgue measure; mean-zero is a distinct ensemble. Full unitary DFT includes boundary mixing; a*power is the continuum-comparable normalization. Infinite-chain length is not a local fit. Refinement computes periodic symbols only, fixes box and signed physical mode, and does not construct larger covariance matrices. Rational finite data, directed768-bit work with512-bit root bounds and outward512-bit endpoint quantum; this is not a uniform relative-accuracy guarantee.'
  };
}

const cache = new Map<string, ReturnType<typeof reconstruct>>();
export function validateSpatialSpectrumResult(source: SpatialSpectrumInput, value: unknown): SpatialSpectrumResult | null {
  try {
    const input = readSpatialSpectrumInput(source); if (!input) return null;
    const owned = snapshotQubitWire(value), key = JSON.stringify(input);
    let expected = cache.get(key);
    if (!expected) { expected = reconstruct(input); if (cache.size === 8) cache.delete(cache.keys().next().value!); cache.set(key, expected); }
    return matches(owned, expected) ? owned as SpatialSpectrumResult : null;
  } catch { return null; }
}
