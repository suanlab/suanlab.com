/** Directed fixed-point enclosures for positive rational logarithms and
 * negative exponentials. Internal arithmetic for the canonical Helmholtz gap;
 * BigInt endpoints are not reader observations and must not reach JSON props.
 */
import type { BinaryValue } from './learning-binary-products';
export interface FixedInterval { lower: bigint; upper: bigint }
const ZERO = BigInt(0), ONE = BigInt(1), TWO = BigInt(2), THREE = BigInt(3);
const floor = (n: bigint, d: bigint) => n >= ZERO ? n / d : -((-n + d - ONE) / d);
const ceil = (n: bigint, d: bigint) => -floor(-n, d);
const bits = (n: bigint) => n.toString(2).length;
function validate(v: BinaryValue, allowZero = false) {
  if (typeof v?.coefficient !== 'bigint' || v.coefficient < ZERO || (!allowZero && v.coefficient === ZERO)
    || !Number.isSafeInteger(v.exponent) || Math.abs(v.exponent) > 32768 || bits(v.coefficient) > 32768)
    throw new RangeError('Expected a bounded nonnegative binary rational');
}
/** precisionBits controls the absolute endpoint quantum 2^-precisionBits.
 * The internal 32 guard bits reduce accumulated interval width, but correctness
 * comes from directed operations and the explicit analytic tail bounds.
 */
export function createTranscendentalContext(precisionBits = 256) {
  if (!Number.isInteger(precisionBits) || precisionBits < 64 || precisionBits > 8192)
    throw new RangeError('Expected 64..8192 precision bits');
  const guard = 32, workBits = precisionBits + guard;
  const S = ONE << BigInt(workBits), scale = ONE << BigInt(precisionBits);
  const point = (x: bigint): FixedInterval => ({ lower: x, upper: x });
  const add = (a: FixedInterval, b: FixedInterval): FixedInterval => ({ lower: a.lower + b.lower, upper: a.upper + b.upper });
  const positiveMultiply = (a: FixedInterval, b: FixedInterval): FixedInterval => ({
    lower: floor(a.lower * b.lower, S), upper: ceil(a.upper * b.upper, S) });
  const divideInteger = (a: FixedInterval, d: bigint): FixedInterval => ({ lower: floor(a.lower, d), upper: ceil(a.upper, d) });
  function rational(n: bigint, d: bigint, exponent = 0): FixedInterval {
    if (exponent >= 0) n <<= BigInt(exponent);
    else d <<= BigInt(-exponent);
    return { lower: floor(n * S, d), upper: ceil(n * S, d) };
  }
  const shrink = (v: FixedInterval): FixedInterval => ({ lower: floor(v.lower, ONE << BigInt(guard)), upper: ceil(v.upper, ONE << BigInt(guard)) });
  /** ln((1+z)/(1-z)) = 2 sum z^(2j+1)/(2j+1), 0<=z<=1/3.
   * Tail <= 3*(1/3)^(2N+1). The latter bound is rounded UP exactly.
   */
  function logSeries(n: bigint, d: bigint): FixedInterval {
    if (n === ZERO) return point(ZERO);
    const z = rational(n, d), z2 = positiveMultiply(z, z);
    const terms = Math.ceil((workBits + 4) / 3);
    let power = z, result = point(ZERO);
    for (let j = 0; j < terms; j++) {
      result = add(result, divideInteger(power, BigInt(2 * j + 1)));
      power = positiveMultiply(power, z2);
    }
    const tail = ceil(THREE * S, THREE ** BigInt(2 * terms + 1));
    return { lower: TWO * result.lower, upper: TWO * result.upper + tail };
  }
  let logTwo: FixedInterval | undefined;
  function logarithm(n: BinaryValue, d: BinaryValue): FixedInterval {
    validate(n); validate(d);
    const difference = bits(n.coefficient) - bits(d.coefficient);
    const below = difference >= 0 ? n.coefficient < (d.coefficient << BigInt(difference))
      : (n.coefficient << BigInt(-difference)) < d.coefficient;
    const exponent = n.exponent - d.exponent;
    const k = difference + exponent - (below ? 1 : 0);
    let numerator = n.coefficient, denominator = d.coefficient;
    if (exponent - k >= 0) numerator <<= BigInt(exponent - k);
    else denominator <<= BigInt(k - exponent);
    // 1<=numerator/denominator<2, so 0<=z<1/3.
    const reduced = logSeries(numerator - denominator, numerator + denominator);
    if (k === 0) return reduced;
    logTwo ??= logSeries(ONE, THREE);
    const factor = BigInt(k);
    const multiple = k >= 0 ? { lower: factor * logTwo.lower, upper: factor * logTwo.upper }
      : { lower: factor * logTwo.upper, upper: factor * logTwo.lower };
    return add(reduced, multiple);
  }
  function expNegativeRatio(n: BinaryValue, d: BinaryValue): FixedInterval {
    validate(n, true); validate(d);
    if (n.coefficient === ZERO) return { lower: scale, upper: scale };
    // n/d < 2^upperPower. Range reduction gives 0<x<=1/8.
    const upperPower = bits(n.coefficient) - bits(d.coefficient) + n.exponent - d.exponent + 1;
    const squarings = Math.max(0, upperPower + 3);
    const x = rational(n.coefficient, d.coefficient, n.exponent - d.exponent - squarings);
    const terms = Math.ceil((workBits + 4) / 3);
    let term = point(S), result = point(S);
    for (let j = 1; j <= terms; j++) {
      term = divideInteger(positiveMultiply(term, x), BigInt(j));
      result = add(result, term);
    }
    // exp(x) tail <= 2*(1/8)^(N+1); inversion gives exp(-x).
    result.upper += ceil(TWO * S, ONE << BigInt(3 * (terms + 1)));
    let negative = { lower: floor(S * S, result.upper), upper: ceil(S * S, result.lower) };
    // exp(-a)<=1 is an analytic bound, including when x is below one quantum.
    if (negative.upper > S) negative.upper = S;
    for (let j = 0; j < squarings; j++) negative = positiveMultiply(negative, negative);
    return shrink(negative);
  }
  return { precisionBits, scale,
    logRatio: (n: BinaryValue, d: BinaryValue): FixedInterval => shrink(logarithm(n, d)),
    expNegativeRatio,
  };
}
