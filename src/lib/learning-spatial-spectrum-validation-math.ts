/** Independent positive logarithms on the existing3072-bit verification grid.
 * Uses a fixed atanh series and explicit positive geometric remainder after
 * exact binary reduction; no producer logarithm, trig or evolution core.
 */
import { f, int, zero, one, add, sub, mul, div, eq, le, point, plus, times, scale, inputFraction, type F, type Range } from './learning-field-modes-validation-math';
export { f, int, zero, one, add, sub, mul, div, neg, eq, le, point, plus, minus, times, quotient, squared, root, scale, total, rationalTotal, text, parse, inputFraction, pi, trigContext } from './learning-field-modes-validation-math';
export type { F, Range } from './learning-field-modes-validation-math';

const terms = 640;
function mantissaLog(x: F): Range {
  if (!le(one, x) || !le(x, int(2))) throw Error('Log mantissa outside[1,2]');
  if (eq(x, one)) return point(zero);
  const t = div(sub(x, one), add(x, one)), t2 = point(mul(t, t));
  let power = point(t), sum = point(zero);
  for (let j = 0; j < terms; j++) {
    sum = plus(sum, scale(power, f(BigInt(2), BigInt(2 * j + 1))));
    power = times(power, t2);
  }
  // The first omitted power is t^(2N+1),0<=t<=1/3.
  // 2 sum_{j>=N} t^(2j+1)/(2j+1) <= 2 t^(2N+1)/((2N+1)(1-t²)).
  const tail = scale(power, div(int(2), mul(int(2 * terms + 1), sub(one, mul(t, t)))));
  return plus(sum, point(zero, tail.upper));
}
let logarithmTwo: Range | null = null;
export function independentSpatialLogPoint(value: F): Range {
  if (value.n <= BigInt(0)) throw Error('Positive logarithm required');
  let exponent = value.n.toString(2).length - value.d.toString(2).length;
  let mantissa = exponent >= 0 ? f(value.n, value.d << BigInt(exponent)) : f(value.n << BigInt(-exponent), value.d);
  if (!le(one, mantissa)) { exponent--; mantissa = mul(mantissa, int(2)); }
  if (!le(mantissa, int(2))) throw Error('Binary log normalization failed');
  logarithmTwo ??= mantissaLog(int(2));
  return plus(mantissaLog(mantissa), scale(logarithmTwo, int(exponent)));
}
export function independentSpatialLog(value: Range): Range {
  return { lower: independentSpatialLogPoint(value.lower).lower, upper: independentSpatialLogPoint(value.upper).upper };
}

// A small direct account is useful when checking this arithmetic separately.
export const spatialLogFraction = (value: string) => independentSpatialLogPoint(inputFraction(value));
