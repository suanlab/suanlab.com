/** Internal directed arithmetic for stationary jump-path entropy.
 * Mantissa normalization accounts for its log factor; it never renormalizes a
 * path law or makes a tilted operator stochastic. No React or browser imports. */
import { markovFraction as fraction, exactMarkovObservation, type MarkovFraction as F } from './learning-finite-markov-exact';
import { createTranscendentalContext, type FixedInterval as I } from './learning-interval-transcendentals';
import { partitionObservationFromRange, type PartitionObservation as Observation } from './learning-partition-exact-core';
export type { F, I, Observation };
export { fraction };
const Z = BigInt(0), O = BigInt(1), TWO = BigInt(2);
export const entropyPrecisionBits = 256, entropyScale = O << BigInt(entropyPrecisionBits);
const S = entropyScale, ctx = createTranscendentalContext(entropyPrecisionBits);
export const zero = fraction(Z), one = fraction(O);
export const add = (a: F, b: F) => fraction(a.n * b.d + b.n * a.d, a.d * b.d);
export const neg = (a: F) => fraction(-a.n, a.d);
export const sub = (a: F, b: F) => add(a, neg(b));
export const mul = (a: F, b: F) => fraction(a.n * b.n, a.d * b.d);
export const div = (a: F, b: F) => fraction(a.n * b.d, a.d * b.n);
export const sum = (xs: readonly F[]) => xs.reduce(add, zero);
export const equal = (a: F, b: F) => a.n * b.d === b.n * a.d;
export const floor = (n: bigint, d: bigint) => n >= Z ? n / d : -((-n + d - O) / d);
export const ceil = (n: bigint, d: bigint) => -floor(-n, d);
export const min = (a: bigint, b: bigint) => a < b ? a : b;
export const max = (a: bigint, b: bigint) => a > b ? a : b;
const abs = (a: bigint) => a < Z ? -a : a;
const binary = (coefficient: bigint) => ({ coefficient, exponent: 0 });
export const exact = (a: F) => exactMarkovObservation(a.n, a.d);
export const point = (a: F): I => ({ lower: floor(a.n * S, a.d), upper: ceil(a.n * S, a.d) });
export const plus = (a: I, b: I): I => ({ lower: a.lower + b.lower, upper: a.upper + b.upper });
export const negative = (a: I): I => ({ lower: -a.upper, upper: -a.lower });
export const minus = (a: I, b: I) => plus(a, negative(b));
export const scale = (a: I, b: F): I => b.n >= Z
  ? { lower: floor(a.lower * b.n, b.d), upper: ceil(a.upper * b.n, b.d) }
  : { lower: floor(a.upper * b.n, b.d), upper: ceil(a.lower * b.n, b.d) };
export const positiveProduct = (a: I, b: I): I => ({ lower: floor(a.lower * b.lower, S), upper: ceil(a.upper * b.upper, S) });
export function square(a: I): I {
  const lower = a.lower >= Z ? a.lower * a.lower : a.upper <= Z ? a.upper * a.upper : Z;
  const upper = max(a.lower * a.lower, a.upper * a.upper);
  return { lower: floor(lower, S), upper: ceil(upper, S) };
}
export function observeRange(lower: F, upper: F): Observation {
  if (lower.n * upper.d > upper.n * lower.d) throw new RangeError('Inverted entropy enclosure');
  if (equal(lower, upper)) return exact(lower);
  return partitionObservationFromRange({ lower: { n: binary(lower.n), d: binary(lower.d) }, upper: { n: binary(upper.n), d: binary(upper.d) } });
}
export const observe = (a: I) => observeRange(fraction(a.lower, S), fraction(a.upper, S));
export function logarithm(a: F): I {
  if (a.n <= Z || a.d <= Z) throw new RangeError('Logarithm needs a positive exact ratio');
  if (a.n === a.d) return { lower: Z, upper: Z };
  return ctx.logRatio(binary(a.n), binary(a.d));
}
function expPoint(x: bigint, bits: number) {
  const context = bits === entropyPrecisionBits ? ctx : createTranscendentalContext(bits), scale = context.scale;
  if (x === Z) return { lower: fraction(O), upper: fraction(O) };
  const value = context.expNegativeRatio(binary(abs(x)), binary(S));
  return x < Z ? { lower: fraction(value.lower, scale), upper: fraction(value.upper, scale) }
    : { lower: fraction(scale, value.upper), upper: value.lower === Z ? null : fraction(scale, value.lower) };
}
/** Off-diagonal powers use the256bit model context. Their input range is
 * bounded by the public rational rate/tilt bounds, not by floating exponentials. */
export function exponentialFixed(a: I): I | null {
  const lower = expPoint(a.lower, entropyPrecisionBits), upper = expPoint(a.upper, entropyPrecisionBits);
  if (!upper.upper) return null;
  return { lower: floor(lower.lower.n * S, lower.lower.d), upper: ceil(upper.upper.n * S, upper.upper.d) };
}
export function momentFromLog(a: I) {
  const magnitude = max(abs(a.lower), abs(a.upper));
  const bits = [256, 512, 1024, 2048].find(b => TWO * magnitude <= BigInt(b - 80) * S) ?? 2048;
  const lo = expPoint(a.lower, bits), hi = expPoint(a.upper, bits);
  return { observation: hi.upper ? observeRange(lo.lower, hi.upper) : null, precisionBits: bits,
    status: hi.upper ? 'directed-positive-exponential-enclosure' as const : 'unresolved-positive-exponential-upper-bound' as const };
}
type Matrix = I[][];
const identity = (): Matrix => Array.from({ length: 3 }, (_, i) => Array.from({ length: 3 }, (_, j) => ({ lower: i === j ? S : Z, upper: i === j ? S : Z })));
const rowUpper = (matrix: Matrix) => matrix.reduce((total, row) => max(total, row.reduce((s, x) => s + x.upper, Z)), Z);
const matrixProduct = (a: Matrix, b: Matrix): Matrix => a.map(row => b[0].map((_, j) => row.reduce((total, x, k) => plus(total, positiveProduct(x, b[k][j])), { lower: Z, upper: Z })));
function normalize(matrix: Matrix) {
  const row = rowUpper(matrix);
  if (row <= Z || row > BigInt(3) * S) throw new RangeError('Positive-series normalization bound failed');
  // The16-quantum cushion exceeds the three per-row upward-rounding errors.
  // All true mantissa entries are divided by this same exact positive factor.
  const denominator = row + BigInt(16), factor = fraction(denominator, S);
  const mantissa = matrix.map(line => line.map(x => ({ lower: floor(x.lower * S, denominator), upper: ceil(x.upper * S, denominator) })));
  if (rowUpper(mantissa) > S) throw new RangeError('Mantissa norm proof failed');
  return { mantissa, logFactor: logarithm(factor) };
}
const cutoff = 64;
let factorial = O;
for (let k = 2; k <= cutoff + 1; k++) factorial *= BigInt(k);
const tail = fraction(BigInt(2 * (cutoff + 2)), (O << BigInt(cutoff + 1)) * factorial * BigInt(2 * (cutoff + 2) - 1));
const tailUpper = ceil(tail.n * S, tail.d);
export interface EntropyTilt {
  tiltedGenerator: Observation[][];
  shifted: Matrix;
  clock: F;
  tilt: F;
  equilibrium: boolean;
}
/** rates and pi are the validated exact positive generator and stationary law
 * of the public three-state model. Integer tilt powers are kept exact. */
export function buildEntropyTilt(rates: readonly F[], pi: readonly F[], tilt: F): EntropyTilt {
  const [a, b, c, d, e, f] = rates, outgoing = [[a, f], [d, b], [c, e]], exits = outgoing.map(sum);
  const Q = [[neg(exits[0]), a, f], [d, neg(exits[1]), b], [c, e, neg(exits[2])]];
  const clock = exits.reduce((largest, x) => x.n * largest.d > largest.n * x.d ? x : largest, zero);
  const equilibrium = equal(mul(mul(a, b), c), mul(mul(d, e), f));
  const exactPower = tilt.n % tilt.d === Z ? tilt.n / tilt.d : null;
  const tiltedGenerator: Observation[][] = [], shifted: Matrix = [];
  for (let i = 0; i < 3; i++) {
    const line: Observation[] = [], positive: I[] = [];
    for (let j = 0; j < 3; j++) {
      if (i === j) { line.push(exact(Q[i][j])); positive.push(point(add(clock, Q[i][j]))); continue; }
      const reverseRatio = div(mul(pi[j], Q[j][i]), mul(pi[i], Q[i][j]));
      if (reverseRatio.n === reverseRatio.d) {
        line.push(exact(Q[i][j])); positive.push(point(Q[i][j]));
      } else if (exactPower !== null) {
        let power = one;
        for (let k = Z; k < abs(exactPower); k++) power = mul(power, reverseRatio);
        if (exactPower < Z) power = div(one, power);
        const value = mul(Q[i][j], power); line.push(exact(value)); positive.push(point(value));
      } else {
        const factor = exponentialFixed(scale(logarithm(reverseRatio), tilt));
        if (!factor) throw new RangeError('Tilted positive rate upper bound unavailable within the declared input domain');
        const value = scale(factor, Q[i][j]); line.push(observe(value)); positive.push(value);
      }
    }
    tiltedGenerator.push(line); shifted.push(positive);
  }
  return { tiltedGenerator, shifted, clock, tilt, equilibrium };
}
export function entropyMoment(tilted: EntropyTilt, pi: readonly F[], exitRates: readonly F[], time: F) {
  const { tilt } = tilted;
  const identityReason = time.n === Z ? 'zero-time' : tilted.equilibrium ? 'detailed-balance-path-identity' : tilt.n === Z ? 'zero-tilt-normalization' : tilt.n === tilt.d ? 'adjoint-normalization' : null;
  if (identityReason) return { moment: exact(one), logMoment: exact(zero), logRange: { lower: Z, upper: Z }, linearStatus: 'analytic-exact-one' as const,
    proof: { method: 'normalization-identity' as const, identityReason, squarings: 0, retainedTerms: 0, baseTailUpperBound: exact(zero), logScale: exact(zero), linearPrecisionBits: entropyPrecisionBits } };
  const largest = max(rowUpper(tilted.shifted), point(tilted.clock).upper);
  let denominator = S * time.d, squarings = 0;
  while (BigInt(4) * largest * time.n > denominator) { denominator *= TWO; squarings++; }
  const delta = fraction(time.n, time.d * (O << BigInt(squarings)));
  const C = tilted.shifted.map(row => row.map(x => scale(x, delta)));
  if (rowUpper(C) > S / TWO) throw new RangeError('Positive Taylor base norm is not bounded by1/2');
  let term = identity(), total = identity();
  for (let k = 1; k <= cutoff; k++) {
    term = matrixProduct(term, C).map(row => row.map(x => scale(x, fraction(O, BigInt(k)))));
    total = total.map((row, i) => row.map((x, j) => plus(x, term[i][j])));
  }
  total = total.map(row => row.map(x => ({ lower: x.lower, upper: x.upper + tailUpper })));
  let normalized = normalize(total), mantissa = normalized.mantissa;
  let logScale = minus(normalized.logFactor, point(mul(tilted.clock, delta)));
  for (let k = 0; k < squarings; k++) {
    normalized = normalize(matrixProduct(mantissa, mantissa)); mantissa = normalized.mantissa;
    logScale = plus({ lower: TWO * logScale.lower, upper: TWO * logScale.upper }, normalized.logFactor);
  }
  const scalar = mantissa.reduce((total, row, i) => plus(total, scale(row.reduce(plus, { lower: Z, upper: Z }), pi[i])), { lower: Z, upper: Z });
  if (scalar.upper <= Z) throw new RangeError('Positive moment upper enclosure lost');
  const noJumpLower = pi.reduce((lower, p, i) => max(lower, minus(logarithm(p), point(mul(exitRates[i], time))).lower), minus(logarithm(pi[0]), point(mul(exitRates[0], time))).lower);
  const lower = scalar.lower > Z ? max(noJumpLower, logScale.lower + logarithm(fraction(scalar.lower, S)).lower) : noJumpLower;
  const upper = logScale.upper + logarithm(fraction(scalar.upper, S)).upper;
  // Convexity and M(0)=M(1)=1 bound log M above within[0,1], below outside.
  const interior = tilt.n > Z && tilt.n < tilt.d;
  const logRange = { lower: interior ? lower : max(Z, lower), upper: interior ? min(Z, upper) : upper };
  if (logRange.lower > logRange.upper) throw new RangeError('Fluctuation convexity enclosure is inconsistent');
  const projection = momentFromLog(logRange);
  return { moment: projection.observation, logMoment: observe(logRange), logRange, linearStatus: projection.status,
    proof: { method: 'shifted-positive-series-log-normalization' as const, identityReason: null, squarings, retainedTerms: cutoff + 1,
      baseTailUpperBound: exact(tail), logScale: observe(logScale), linearPrecisionBits: projection.precisionBits } };
}
