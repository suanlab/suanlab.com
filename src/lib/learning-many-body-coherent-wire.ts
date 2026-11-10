/** Independent exp(-x) certificates: direct alternating Taylor sums, no range
 * reduction, inversion, squaring or production transcendental engine.
 */
import { f, int, zero, one, two, sub, mul, div, le, eq, lo, hi, type F } from './learning-tensor-wire-math';
import { field, fieldSign, fieldScale, fieldSub, fieldNeg, type WireField } from './learning-many-body-field-wire';
import { expectedValidated, expectedRational, type ManyBodyExpected, type ManyBodyWireChecks } from './learning-many-body-wire-match';
import type { PartitionObservation as Q } from './learning-partition-exact-core';
export type CoherentRange = Readonly<{ lower: F; upper: F }>;
const scale = BigInt(1) << BigInt(2048), terms = 512;
const floor = (n: bigint, d: bigint) => n >= BigInt(0) ? n / d : -((-n + d - BigInt(1)) / d);
const ceil = (n: bigint, d: bigint) => -floor(-n, d);
const cache = new Map<string, CoherentRange>();
export function coherentExpCertificate(x: F): CoherentRange {
  if (!le(zero, x) || !le(x, int(32))) throw new RangeError('Exponent in [0,32] required');
  const key = `${x.n}/${x.d}`, prior = cache.get(key); if (prior) return prior;
  let lowerTerm = scale, upperTerm = scale, lowerSum = scale, upperSum = scale;
  for (let n = 1; n <= terms; n++) {
    lowerTerm = floor(lowerTerm * x.n, x.d * BigInt(n)); upperTerm = ceil(upperTerm * x.n, x.d * BigInt(n));
    if (n % 2) { lowerSum -= upperTerm; upperSum -= lowerTerm; } else { lowerSum += lowerTerm; upperSum += upperTerm; }
  }
  // After term512 the alternating tail decreases since x<=32. The even
  // partial sum is an upper bound; subtracting the first omitted positive
  // term gives a lower bound, including all directed recurrence error.
  const omittedUpper = ceil(upperTerm * x.n, x.d * BigInt(terms + 1));
  const lower = f(lowerSum - omittedUpper, scale), upper = f(upperSum, scale);
  const result = Object.freeze({ lower: Object.freeze(le(zero, lower) ? lower : zero), upper: Object.freeze(le(upper, one) ? upper : one) });
  if (!le(result.lower, result.upper)) throw new RangeError('Inconsistent exponential certificate');
  if (cache.size >= 64) cache.delete(cache.keys().next().value!); cache.set(key, result); return result;
}
export const coherentScaleRange = (r: CoherentRange, a: F): CoherentRange => a.n >= BigInt(0) ? { lower: mul(r.lower, a), upper: mul(r.upper, a) } : { lower: mul(r.upper, a), upper: mul(r.lower, a) };
export const coherentComplementRange = (r: CoherentRange): CoherentRange => ({ lower: sub(one, r.upper), upper: sub(one, r.lower) });
export function coherentRetainedRange(mu: F, series: F): CoherentRange {
  const r = coherentScaleRange(coherentExpCertificate(mu), series);
  // Exact coherent-series probabilities are in [0,1]. Intersect the proved
  // enclosure with that mathematical range; never alter a supplied state.
  return { lower: le(zero, r.lower) ? r.lower : zero, upper: le(r.upper, one) ? r.upper : one };
}
const widthFactor = f(BigInt(10) ** BigInt(125));
function boundedWidth(q: Q, scaleValue = one) { return le(mul(sub(hi(q), lo(q)), widthFactor), scaleValue); }
export const expectedCoherentRange = (r: CoherentRange): ManyBodyExpected => eq(r.lower, r.upper) ? expectedRational(r.lower) : expectedValidated((actual, checks) => {
  if (!checks.q(actual)) return false; const q = actual as Q;
  return le(zero, lo(q)) && le(hi(q), one) && le(lo(q), r.lower) && le(r.upper, hi(q)) && boundedWidth(q);
});
export const expectedCoherentZeroResidual = (): ManyBodyExpected => expectedValidated((actual, checks) => {
  if (!checks.q(actual)) return false; const q = actual as Q;
  return le(lo(q), zero) && le(zero, hi(q)) && boundedWidth(q);
});
/** A field coefficient times exp(-mu/2), including complex-component signs. */
export const expectedCoherentAmplitude = (target: WireField, mu: F): ManyBodyExpected => expectedValidated((actual, checks) => {
  if (!checks.q(actual)) return false; const q = actual as Q, sign = fieldSign(target);
  if (!sign) return checks.exact(q) && eq(lo(q), zero);
  if (mu.n === BigInt(0) && target.slice(1).every(v => v.n === BigInt(0))) return checks.exact(q) && eq(lo(q), target[0]);
  const r = coherentExpCertificate(div(mu, two));
  const lower = fieldScale(target, sign > 0 ? r.lower : r.upper), upper = fieldScale(target, sign > 0 ? r.upper : r.lower);
  if (fieldSign(fieldSub(lower, field(lo(q)))) < 0 || fieldSign(fieldSub(field(hi(q)), upper)) < 0) return false;
  const absolute = sign > 0 ? target : fieldNeg(target), width = field(mul(sub(hi(q), lo(q)), widthFactor));
  return fieldSign(fieldSub(width, fieldSign(fieldSub(absolute, field(one))) > 0 ? absolute : field(one))) <= 0;
});
function validRootObservation(actual: unknown, checks: ManyBodyWireChecks): actual is Q { return checks.q(actual) && boundedWidth(actual as Q); }
export const expectedCoherentOverlap = (retained: CoherentRange): ManyBodyExpected => eq(retained.lower, one) && eq(retained.upper, one) ? expectedRational(one) : expectedValidated((actual, checks) => {
  if (!validRootObservation(actual, checks)) return false;
  const a = lo(actual), b = hi(actual);
  return le(zero, a) && le(b, one) && le(mul(a, a), retained.lower) && le(retained.upper, mul(b, b));
});
export const expectedCoherentDistance = (retained: CoherentRange): ManyBodyExpected => eq(retained.lower, one) && eq(retained.upper, one) ? expectedRational(zero) : expectedValidated((actual, checks) => {
  if (!validRootObservation(actual, checks)) return false;
  const lowerComplement = sub(one, div(lo(actual), two)), upperComplement = sub(one, div(hi(actual), two));
  return le(zero, lo(actual)) && le(hi(actual), two) && le(zero, lowerComplement) && le(retained.upper, mul(lowerComplement, lowerComplement))
    && (le(upperComplement, zero) || le(mul(upperComplement, upperComplement), retained.lower));
});
