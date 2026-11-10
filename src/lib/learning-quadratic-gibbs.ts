/** Quadratic Gibbs density on dimensionless R² relative to Lebesgue dx.
 * Exact binary rational definiteness, means and covariances; directed pi,
 * logarithm and relative square-root bounds. Positive temperature only.
 */
import { binaryProduct, type BinaryValue } from './learning-binary-products';
import { createTranscendentalContext } from './learning-interval-transcendentals';
import { directedPi } from './learning-directed-pi';
import { partitionObservationFromRange, type PartitionObservation } from './learning-partition-exact-core';
import { canonicalBoltzmannConstant as kB } from './learning-canonical-spectrum';
export interface QuadraticGibbsParameters {
  firstStiffness: number;
  crossStiffness: number;
  secondStiffness: number;
  firstSourceEnergy: number;
  secondSourceEnergy: number;
  energyOffset: number;
  temperature: number;
  coordinateScale?: readonly [number, number];
  coordinateShift?: readonly [number, number];
}
export interface QuadraticGibbsOptions { minimumPrecision?: number; maximumPrecision?: number }
interface Fraction { n: bigint; d: bigint }
interface Interval { lower: Fraction; upper: Fraction }
const Z = BigInt(0), O = BigInt(1), TWO = BigInt(2);
const abs = (v: bigint) => v < Z ? -v : v;
function gcd(a: bigint, b: bigint) { a = abs(a); while (b) { const r = a % b; a = b; b = r; } return a; }
function q(n: bigint, d = O): Fraction {
  if (d <= Z) throw new RangeError('Expected positive rational denominator');
  const g = gcd(n, d); return { n: n / g, d: d / g };
}
function binary(v: BinaryValue): Fraction { return v.exponent >= 0 ? q(v.coefficient << BigInt(v.exponent)) : q(v.coefficient, O << BigInt(-v.exponent)); }
const scalar = (v: number) => binary(binaryProduct(v));
const add = (a: Fraction, b: Fraction) => q(a.n * b.d + b.n * a.d, a.d * b.d);
const neg = (a: Fraction): Fraction => ({ n: -a.n, d: a.d });
const sub = (a: Fraction, b: Fraction) => add(a, neg(b));
const mul = (a: Fraction, b: Fraction) => q(a.n * b.n, a.d * b.d);
function div(a: Fraction, b: Fraction): Fraction {
  if (b.n === Z) throw new RangeError('Zero rational denominator');
  return q(a.n * b.d * (b.n < Z ? -O : O), a.d * abs(b.n));
}
const cmp = (a: Fraction, b: Fraction) => a.n * b.d - b.n * a.d;
const point = (v: Fraction): Interval => ({ lower: v, upper: v });
const sum = (a: Interval, b: Interval): Interval => ({ lower: add(a.lower, b.lower), upper: add(a.upper, b.upper) });
const negative = (a: Interval): Interval => ({ lower: neg(a.upper), upper: neg(a.lower) });
const difference = (a: Interval, b: Interval) => sum(a, negative(b));
function multiply(a: Interval, b: Interval): Interval {
  const terms = [mul(a.lower, b.lower), mul(a.lower, b.upper), mul(a.upper, b.lower), mul(a.upper, b.upper)];
  return { lower: terms.reduce((a, b) => cmp(a, b) < Z ? a : b), upper: terms.reduce((a, b) => cmp(a, b) > Z ? a : b) };
}
function dividePositive(a: Interval, b: Interval): Interval {
  if (b.lower.n <= Z) throw new RangeError('Expected strictly positive interval denominator');
  return multiply(a, { lower: div(q(O), b.upper), upper: div(q(O), b.lower) });
}
const dyadic = (coefficient: bigint, exponent = 0): BinaryValue => ({ coefficient, exponent });
const observe = (v: Interval): PartitionObservation => partitionObservationFromRange({
  lower: { n: dyadic(v.lower.n), d: dyadic(v.lower.d) }, upper: { n: dyadic(v.upper.n), d: dyadic(v.upper.d) },
});
function integerSqrt(n: bigint) {
  if (n < TWO) return n;
  let x = O << BigInt(Math.ceil(n.toString(2).length / 2));
  for (;;) { const y = (x + n / x) / TWO; if (y >= x) return x; x = y; }
}
/** Relative mantissa precision; tiny square roots are not rounded to zero
 * merely because their absolute value lies below 2^-precisionBits. */
function sqrtRange(v: Fraction, precisionBits: number): Interval {
  const delta = v.n.toString(2).length - v.d.toString(2).length;
  const below = delta >= 0 ? v.n < (v.d << BigInt(delta)) : (v.n << BigInt(-delta)) < v.d;
  const highest = delta - (below ? 1 : 0), quantum = Math.floor(highest / 2) - precisionBits;
  const shift = -2 * quantum;
  const n = shift >= 0 ? v.n << BigInt(shift) : v.n, d = shift < 0 ? v.d << BigInt(-shift) : v.d;
  const root = integerSqrt(n / d), exact = root * root * d === n;
  return { lower: binary(dyadic(root, quantum)), upper: binary(dyadic(exact ? root : root + O, quantum)) };
}
const infinity = (): PartitionObservation => ({ value: null, sign: 1, logMagnitude: null, logResolution: 'not-applicable', resolution: 'mathematical-infinity', bounds: null });
const precisions = [128, 256, 512, 1024, 2048, 4096, 8192];
export function quadraticGibbsModel(input: QuadraticGibbsParameters, options: QuadraticGibbsOptions = {}) {
  if (!input || ![input.firstStiffness, input.crossStiffness, input.secondStiffness, input.firstSourceEnergy, input.secondSourceEnergy, input.energyOffset, input.temperature].every(Number.isFinite)
    || input.temperature <= 0) throw new RangeError('Expected finite quadratic energies and strictly positive finite temperature; a zero-temperature delta limit is not a Lebesgue density');
  const scales = input.coordinateScale === undefined ? [1, 1] : input.coordinateScale, shifts = input.coordinateShift === undefined ? [0, 0] : input.coordinateShift;
  if (!Array.isArray(scales) || scales.length !== 2 || !Array.from(scales).every(v => Number.isFinite(v) && v !== 0)
    || !Array.isArray(shifts) || shifts.length !== 2 || !Array.from(shifts).every(Number.isFinite)) throw new RangeError('Expected two nonzero finite coordinate scales and two finite shifts');
  const minimum = options.minimumPrecision ?? 128, maximum = options.maximumPrecision ?? 8192;
  if (!precisions.includes(minimum) || !precisions.includes(maximum) || minimum > maximum) throw new RangeError('Invalid precision range');
  const a = scalar(input.firstStiffness), b = scalar(input.crossStiffness), d = scalar(input.secondStiffness);
  const determinant = sub(mul(a, d), mul(b, b)), determinantSign = determinant.n < Z ? -1 : determinant.n > Z ? 1 : 0;
  const parameters = { firstStiffness: input.firstStiffness, crossStiffness: input.crossStiffness, secondStiffness: input.secondStiffness,
    firstSourceEnergy: input.firstSourceEnergy, secondSourceEnergy: input.secondSourceEnergy, energyOffset: input.energyOffset, temperature: input.temperature,
    coordinateScale: [...scales] as [number, number], coordinateShift: [...shifts] as [number, number] };
  const common = { scope: 'continuous-quadratic-gibbs-lebesgue-r2' as const, referenceMeasure: 'Lebesgue dx on dimensionless R²' as const,
    boltzmannConstant: kB, parameters, determinant: observe(point(determinant)), exactDeterminantSign: determinantSign,
    definitenessContract: 'Strict positive definiteness uses exact binary A11>0 and det(A)>0, without eigenvalue clipping or a pseudoinverse.' };
  if (a.n <= Z || determinant.n <= Z) return { ...common, normalizable: false as const, classification: 'not-positive-definite' as const,
    reason: determinant.n === Z && a.n >= Z && d.n >= Z ? 'flat-or-linear-unconfined-direction' as const : 'negative-curvature-direction' as const,
    logPartition: infinity(), precisionBits: 0, attempts: [] as number[], fullyResolved: true,
    nonNormalizationContract: 'The full R² integral diverges at positive T. A finite plotting box, lower-dimensional singular Gaussian or pseudoinverse does not normalize this Lebesgue density.' };
  const h = [scalar(input.firstSourceEnergy), scalar(input.secondSourceEnergy)], offset = scalar(input.energyOffset);
  const thermal = binary(binaryProduct(kB, input.temperature));
  const inv = [[div(d, determinant), div(neg(b), determinant)], [div(neg(b), determinant), div(a, determinant)]];
  const mean = inv.map(row => add(mul(row[0], h[0]), mul(row[1], h[1])));
  const covariance = inv.map(row => row.map(v => mul(thermal, v)));
  const precisionMatrix = [[a, b], [b, d]].map(row => row.map(v => div(v, thermal)));
  const minimumEnergy = sub(offset, div(add(mul(h[0], mean[0]), mul(h[1], mean[1])), scalar(2)));
  const meanEnergy = add(minimumEnergy, thermal);
  const coordinateScale = scales.map(scalar), coordinateShift = shifts.map(scalar);
  const signedJacobian = mul(coordinateScale[0], coordinateScale[1]), jacobian = q(abs(signedJacobian.n), signedJacobian.d);
  const transformedMean = mean.map((v, i) => div(sub(v, coordinateShift[i]), coordinateScale[i]));
  const transformedCovariance = covariance.map((row, i) => row.map((v, j) => div(v, mul(coordinateScale[i], coordinateScale[j]))));
  const exactOutputs = { thermalEnergy: observe(point(thermal)), mean: mean.map(v => observe(point(v))), covariance: covariance.map(row => row.map(v => observe(point(v)))),
    precisionMatrix: precisionMatrix.map(row => row.map(v => observe(point(v)))), minimumEnergy: observe(point(minimumEnergy)), meanEnergy: observe(point(meanEnergy)), heatCapacityInKB: observe(point(q(O))) };
  function calculate(P: number) {
    const ctx = createTranscendentalContext(P), S = O << BigInt(P);
    const fixed = (v: { lower: bigint; upper: bigint }): Interval => ({ lower: q(v.lower, S), upper: q(v.upper, S) });
    const log = (v: Fraction) => fixed(ctx.logRatio(dyadic(v.n), dyadic(v.d)));
    const logRange = (v: Interval): Interval => ({ lower: log(v.lower).lower, upper: log(v.upper).upper });
    const pi = fixed(directedPi(P)), sqrtDet = sqrtRange(determinant, P);
    const denominator = multiply(multiply(point(scalar(2)), pi), point(thermal));
    const shape = difference(logRange(denominator), multiply(point(scalar(.5)), log(determinant)));
    const logZ = difference(shape, point(div(minimumEnergy, thermal)));
    const freeEnergy = difference(point(minimumEnergy), multiply(point(thermal), shape));
    const entropy = sum(point(q(O)), shape), peak = dividePositive(sqrtDet, denominator), logJacobian = log(jacobian);
    const transformed = { mean: transformedMean.map(v => observe(point(v))), covariance: transformedCovariance.map(row => row.map(v => observe(point(v)))),
      logAbsoluteJacobian: observe(logJacobian), logPartition: observe(difference(logZ, logJacobian)),
      differentialEntropyNats: observe(difference(entropy, logJacobian)), densityAtMean: observe(multiply(peak, point(jacobian))) };
    const correlation = b.n === Z ? point(q(Z)) : dividePositive(point(neg(b)), sqrtRange(mul(a, d), P));
    const outputs = { logPartition: observe(logZ), freeEnergy: observe(freeEnergy), differentialEntropyNats: observe(entropy), densityAtMean: observe(peak),
      correlation: observe(correlation), transformedCoordinates: transformed };
    const all = [outputs.logPartition, outputs.freeEnergy, outputs.differentialEntropyNats, outputs.densityAtMean, outputs.correlation,
      transformed.logAbsoluteJacobian, transformed.logPartition, transformed.differentialEntropyNats, transformed.densityAtMean];
    return { ...outputs, precisionBits: P, fullyResolved: all.every(q => q.resolution !== 'unresolved-at-precision-ceiling') };
  }
  const attempts: number[] = []; let latest: ReturnType<typeof calculate> | undefined;
  for (const P of precisions.filter(p => p >= minimum && p <= maximum)) { attempts.push(P); latest = calculate(P); if (latest.fullyResolved) break; }
  return { ...common, normalizable: true as const, classification: 'positive-definite' as const, ...exactOutputs, ...latest!, attempts,
    coordinateConvention: 'x_i=coordinateScale_i*z_i+coordinateShift_i; transformed density is relative to dz, with absolute Jacobian.',
    approximationContract: 'Exact supplied binary64 rational Hamiltonian, means, covariance, precision and coordinate transforms; stored binary64 kB. Directed Machin pi, relative square-root and logarithm bounds. Resolved nonzero intervals certify sign and relative width <=2^-48. Nonzero outside display range is null with sign; unresolved is not zero. Fixed A,h,c temperature derivatives give C/kB=1 for positive T; no zero-temperature Lebesgue density is claimed.' };
}
