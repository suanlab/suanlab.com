/** Certified plotting intervals. Worker/CLI only: never convert the two
 * quadratic coefficients separately to floating point before adding them. */
import { f, quadratic, qSign, qSub, type Q, type C } from './learning-qubit-channel-algebra';

export type QubitDisplayReal = {
  sign: -1 | 0 | 1;
  status: 'enclosed' | 'outside';
  /** Dyadic endpoints are exactly representable as JS numbers. */
  lower: number | null;
  upper: number | null;
  coordinate: number | null;
  exact: boolean;
};
export type QubitDisplayComplex = { real: QubitDisplayReal; imaginary: QubitDisplayReal };
export const qubitDisplayBits = 40;
const scale = BigInt(2) ** BigInt(qubitDisplayBits);

export function requireQubitDisplayLimit(limit: number) {
  if (!Number.isInteger(limit) || limit < 1 || limit > 1024 || (limit & (limit - 1)) !== 0)
    throw new RangeError('Display limit must be a power of two from 1 to 1024');
}
function requireQuadratic(value: Q) {
  if (!Array.isArray(value) || value.length !== 2 || value.some(v => !v
    || typeof v.n !== 'bigint' || typeof v.d !== 'bigint' || v.d <= BigInt(0)
    || v.n.toString().length > 65536 || v.d.toString().length > 65536))
    throw new RangeError('Bounded exact quadratic scalar required');
}
export function observeQubitReal(value: Q, limit = 1): QubitDisplayReal {
  requireQubitDisplayLimit(limit);
  requireQuadratic(value);
  const compare = (numerator: bigint) => qSign(qSub(value, quadratic(f(numerator, scale))));
  const sign = qSign(value) as -1 | 0 | 1;
  if (sign === 0)
    return { sign, status: 'enclosed', lower: 0, upper: 0, coordinate: 0, exact: true };
  let low = -BigInt(limit) * scale, high = BigInt(limit) * scale;
  if (compare(low) < 0 || compare(high) > 0)
    return { sign, status: 'outside', lower: null, upper: null, coordinate: null, exact: false };
  // Find floor(value * 2^40) using exact comparisons, including cancellation.
  if (value[1].n === BigInt(0)) {
    const n = value[0].n * scale, d = value[0].d;
    low = n >= BigInt(0) ? n / d : -((-n + d - BigInt(1)) / d);
  } else while (low < high) {
      const middle = low + (high - low + BigInt(1)) / BigInt(2);
      if (compare(middle) >= 0) low = middle; else high = middle - BigInt(1);
    }
  const exact = compare(low) === 0;
  const lower = Number(low) / Number(scale);
  const upper = Number(exact ? low : low + BigInt(1)) / Number(scale);
  return { sign, status: 'enclosed', lower, upper, coordinate: (lower + upper) / 2, exact };
}
export function observeQubitComplex(value: C, limit = 1): QubitDisplayComplex {
  return { real: observeQubitReal(value.re, limit), imaginary: observeQubitReal(value.im, limit) };
}
