/** Independent certificate checks. Worker/CLI only. Does not call the producer
 * or import its arithmetic implementation; checks integer inequalities. */
import type { Q } from './learning-qubit-channel-algebra';
import type { QubitDisplayReal } from './learning-qubit-display';

const scale = BigInt(2) ** BigInt(40);
function compare(value: Q, dyadic: bigint): -1 | 0 | 1 {
  // value - dyadic/scale = (a + b sqrt(2))/positive denominator.
  const [a, b] = value;
  const rational = (a.n * scale - dyadic * a.d) * b.d;
  const radical = b.n * a.d * scale;
  const sign = (n: bigint): -1 | 0 | 1 => n > BigInt(0) ? 1 : n < BigInt(0) ? -1 : 0;
  const sa = sign(rational), sb = sign(radical);
  if (!sb) return sa;
  if (!sa || sa === sb) return sb;
  const difference = rational * rational - BigInt(2) * radical * radical;
  return difference === BigInt(0) ? 0 : difference > BigInt(0) ? sa : sb;
}
export function validateQubitDisplayReal(value: Q, observed: QubitDisplayReal, limit = 1): boolean {
  try {
    if (!Number.isInteger(limit) || limit < 1 || limit > 1024 || (limit & (limit - 1)) !== 0) return false;
    if (!Array.isArray(value) || value.length !== 2 || value.some(v => !v
      || typeof v.n !== 'bigint' || typeof v.d !== 'bigint' || v.d <= BigInt(0)
      || v.n.toString().length > 65536 || v.d.toString().length > 65536)) return false;
    if (!observed || observed.sign !== compare(value, BigInt(0))) return false;
    const outside = compare(value, -BigInt(limit) * scale) < 0 || compare(value, BigInt(limit) * scale) > 0;
    if (outside) return observed.status === 'outside' && observed.lower === null && observed.upper === null
      && observed.coordinate === null && observed.exact === false;
    if (observed.status !== 'enclosed') return false;
    const { lower, upper, coordinate } = observed;
    if (typeof lower !== 'number' || typeof upper !== 'number' || typeof coordinate !== 'number'
      || !Number.isFinite(lower) || !Number.isFinite(upper) || !Number.isFinite(coordinate)
      || Object.is(lower, -0) || Object.is(upper, -0) || Object.is(coordinate, -0)
      || lower < -limit || upper > limit || lower > upper) return false;
    const lo = lower * Number(scale), hi = upper * Number(scale);
    if (!Number.isSafeInteger(lo) || !Number.isSafeInteger(hi) || hi - lo > 1) return false;
    const lowComparison = compare(value, BigInt(lo)), highComparison = compare(value, BigInt(hi));
    if (lowComparison < 0 || highComparison > 0 || coordinate !== (lower + upper) / 2) return false;
    // Exact endpoints must use the unique singleton, not a needlessly wide cell.
    return observed.exact === (lo === hi) && (lo === hi ? lowComparison === 0 : lowComparison > 0 && highComparison < 0);
  } catch { return false; }
}
