/** Directed dyadic bounds for pi from Machin's identity.
 * Each alternating arctangent term is rounded outwards; the first omitted
 * term bounds the alternating remainder. No Math.PI or floating summation.
 */
import type { FixedInterval } from './learning-interval-transcendentals';
const Z = BigInt(0), O = BigInt(1), TWO = BigInt(2);
const floor = (n: bigint, d: bigint) => n >= Z ? n / d : -((-n + d - O) / d);
const ceil = (n: bigint, d: bigint) => -floor(-n, d);
const cache = new Map<number, FixedInterval>();
export function directedPi(precisionBits: number): FixedInterval {
  if (!Number.isInteger(precisionBits) || precisionBits < 64 || precisionBits > 8192) throw new RangeError('Expected 64..8192 pi precision bits');
  const saved = cache.get(precisionBits); if (saved) return { ...saved };
  const guard = 32, S = O << BigInt(precisionBits + guard);
  function atanInverse(k: bigint): FixedInterval {
    let power = k, j = Z, lower = Z, upper = Z;
    for (;;) {
      const denominator = (TWO * j + O) * power;
      if (denominator > (S << BigInt(8))) {
        const tail = ceil(S, denominator);
        if (j % TWO === Z) upper += tail; else lower -= tail;
        return { lower, upper };
      }
      const lo = floor(S, denominator), hi = ceil(S, denominator);
      if (j % TWO === Z) { lower += lo; upper += hi; } else { lower -= hi; upper -= lo; }
      power *= k * k; j++;
    }
  }
  const a = atanInverse(BigInt(5)), b = atanInverse(BigInt(239)), shrink = O << BigInt(guard);
  const result = { lower: floor(BigInt(16) * a.lower - BigInt(4) * b.upper, shrink), upper: ceil(BigInt(16) * a.upper - BigInt(4) * b.lower, shrink) };
  cache.set(precisionBits, result); return { ...result };
}
