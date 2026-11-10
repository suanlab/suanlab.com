/** Uniform preparation on the ground manifold: D=log1p(tail/g).
 * Work in log D so an arbitrarily cold positive tail is not a physical zero.
 */
import { createTranscendentalContext, type FixedInterval } from './learning-interval-transcendentals';
import { binaryProduct as product, binaryMultiply as multiply, binarySum as sum,
  binaryNegate as negate, binaryObservation as observe,
  type BinaryValue } from './learning-binary-products';
const Z = BigInt(0), O = BigInt(1);
const floor = (n: bigint, d: bigint) => n >= Z ? n / d : -((-n + d - O) / d);
const ceil = (n: bigint, d: bigint) => -floor(-n, d);
const subtract = (a: BinaryValue, b: BinaryValue) => sum([a, negate(b)]);
function scaled(n: BinaryValue, d: BinaryValue, P: number): FixedInterval {
  let a = n.coefficient, b = d.coefficient;
  const shift = n.exponent - d.exponent + P;
  if (shift >= 0) a <<= BigInt(shift); else b <<= BigInt(-shift);
  return { lower: floor(a, b), upper: ceil(a, b) };
}
/** Internal: caller validates nonnegative gaps, positive thermal energy,
 * normalized-weight eligibility and an integer nonnegative copy count.
 */
export function canonicalGroundTail(gaps: readonly BinaryValue[], thermal: BinaryValue,
  trialWeights: readonly number[], copies: number) {
  const grounds = gaps.map((gap, i) => ({ gap, weight: trialWeights[i] })).filter(x => x.gap.coefficient === Z);
  const excited = gaps.filter(g => g.coefficient !== Z);
  if (!excited.length || !grounds.length || grounds[0].weight <= 0
    || !grounds.every(g => g.weight === grounds[0].weight)
    || gaps.some((gap, i) => gap.coefficient !== Z && trialWeights[i] !== 0)) return null;
  const minimum = excited.reduce((a, b) => subtract(a, b).coefficient <= Z ? a : b);
  // An exact domain condition for the logarithmic expansion, never equilibrium.
  if (subtract(minimum, multiply(thermal, product(100))).coefficient < Z) return null;
  const P = 256, c = createTranscendentalContext(P), g = grounds.length;
  let B: FixedInterval = { lower: Z, upper: Z };
  for (const gap of excited) {
    const term = c.expNegativeRatio(subtract(gap, minimum), thermal);
    B = { lower: B.lower + term.lower, upper: B.upper + term.upper };
  }
  const denominator = product(g);
  const lowerB = c.logRatio({ coefficient: B.lower, exponent: -P }, denominator);
  const upperB = c.logRatio({ coefficient: B.upper, exponent: -P }, denominator);
  const amin = scaled(minimum, thermal, P);
  const expMinimum = c.expNegativeRatio(minimum, thermal);
  const upperX = ceil(expMinimum.upper * BigInt(excited.length), BigInt(g));
  // x=tail/g <= upperX*2^-P <1. For 0<x<=1:
  // -x <= log(log1p(x)/x) <=0. Thus log D is enclosed below.
  if (upperX >= c.scale) throw new RangeError('Ground-tail expansion domain failure');
  const logD = { lower: -amin.upper + lowerB.lower - upperX, upper: -amin.lower + upperB.upper };
  // exp(width)-1 <= 2*width for width<=1. Retain the same 2^-40
  // relative enclosure criterion as the general gap path.
  if (((logD.upper - logD.lower) << BigInt(41)) > c.scale) return null;
  const logMidpoint = { coefficient: logD.lower + logD.upper, exponent: -P - 1 };
  const logObservation = observe(logMidpoint);
  const exactLogInterval = { lowerCoefficient: logD.lower.toString(), upperCoefficient: logD.upper.toString(), exponent: -P };
  const tailMetadata = { groundCount: g, excitedCount: excited.length,
    expansion: 'log(log1p(x))=log(x)+correction; -x<=correction<=0' as const,
    upperX: observe({ coefficient: upperX, exponent: -P }) };
  if (logD.upper < -BigInt(10000) * c.scale) {
    // Even MAX_VALUE*kB*MAX_VALUE replicas cannot rescue this linear value.
    const logThermal = observe(thermal).logMagnitude!;
    const positive = (count: number) => count === 0 ? observe(product(0)) : {
      value: null, sign: 1, logMagnitude: logObservation.value === null ? null : logObservation.value + logThermal + Math.log(count) };
    return { resolution: 'positive-log-tail-resolved' as const, precisionBits: P,
      divergence: { value: null, sign: 1, logMagnitude: logObservation.value },
      perCopyGap: positive(1), totalGap: positive(copies),
      bounds: { lower: { value: null, sign: 1, logMagnitude: observe({ coefficient: logD.lower, exponent: -P }).value },
        upper: { value: null, sign: 1, logMagnitude: observe({ coefficient: logD.upper, exponent: -P }).value } },
      exactInterval: null, exactLogInterval, logDivergence: logObservation, tailMetadata, attempts: [P] };
  }
  // Obtain a scaled D interval without exponentiating a rounded large log.
  // D=2^k exp(-(k ln2-logD)); exact checking establishes a nonnegative residual.
  const ln2 = c.logRatio(product(2), product(1));
  let k = Math.ceil(logObservation.value! / Math.LN2) + 1;
  let residual: FixedInterval;
  for (;;) {
    const factor = BigInt(k);
    const low = factor >= Z ? factor * ln2.lower : factor * ln2.upper;
    const high = factor >= Z ? factor * ln2.upper : factor * ln2.lower;
    residual = { lower: low - logD.upper, upper: high - logD.lower };
    if (residual.lower >= Z) break;
    k++;
  }
  const lower = c.expNegativeRatio({ coefficient: residual.upper, exponent: -P }, product(1)).lower;
  const upper = c.expNegativeRatio({ coefficient: residual.lower, exponent: -P }, product(1)).upper;
  if (lower <= Z || ((upper - lower) << BigInt(40)) > lower) return null;
  const midpoint = { coefficient: lower + upper, exponent: k - P - 1 };
  return { resolution: 'positive-log-tail-resolved' as const, precisionBits: P,
    divergence: observe(midpoint), perCopyGap: observe(multiply(midpoint, thermal)),
    totalGap: observe(multiply(multiply(midpoint, thermal), product(copies))),
    bounds: { lower: observe({ coefficient: lower, exponent: k - P }), upper: observe({ coefficient: upper, exponent: k - P }) },
    exactInterval: { lowerCoefficient: lower.toString(), upperCoefficient: upper.toString(), exponent: k - P },
    exactLogInterval, logDivergence: logObservation, tailMetadata, attempts: [P] };
}
