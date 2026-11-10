/** Reader-side biquadratic accounts via Q(sqrt2)[sqrt3], without root engines.
 * Observation bounds are compared to the exact algebraic target by signs;
 * neither a rounded Number nor a supplied field sign is used as an oracle.
 */
import { f, int, zero, one, add, sub, mul, div, neg, eq, le, lo, hi, matches, type F } from './learning-tensor-wire-math';
import { expectedValidated, type ManyBodyExpected, type ManyBodyWireChecks } from './learning-many-body-wire-match';
import type { PartitionObservation as Q } from './learning-partition-exact-core';

export type WireField = readonly [F, F, F, F];
type Quadratic = readonly [F, F];
const qadd = (a: Quadratic, b: Quadratic): Quadratic => [add(a[0], b[0]), add(a[1], b[1])];
const qscale = (a: Quadratic, b: F): Quadratic => [mul(a[0], b), mul(a[1], b)];
const qmul = (a: Quadratic, b: Quadratic): Quadratic => [add(mul(a[0], b[0]), mul(int(2), mul(a[1], b[1]))), add(mul(a[0], b[1]), mul(a[1], b[0]))];
const signQ = (a: F) => a.n === BigInt(0) ? 0 : a.n > BigInt(0) ? 1 : -1;
function quadraticSign(a: Quadratic): number {
  const sa = signQ(a[0]), sb = signQ(a[1]);
  if (!sa) return sb; if (!sb || sa === sb) return sa;
  return sa * signQ(sub(mul(a[0], a[0]), mul(int(2), mul(a[1], a[1]))));
}
export const field = (a: F, b = zero, c = zero, d = zero): WireField => [a, b, c, d];
export const fieldZero = field(zero), fieldOne = field(one), fieldInt = (n: number) => field(int(n));
export const fieldAdd = (a: WireField, b: WireField): WireField => [add(a[0], b[0]), add(a[1], b[1]), add(a[2], b[2]), add(a[3], b[3])];
export const fieldNeg = (a: WireField): WireField => [neg(a[0]), neg(a[1]), neg(a[2]), neg(a[3])];
export const fieldSub = (a: WireField, b: WireField) => fieldAdd(a, fieldNeg(b));
export const fieldScale = (a: WireField, b: F): WireField => [mul(a[0], b), mul(a[1], b), mul(a[2], b), mul(a[3], b)];
export function fieldMul(a: WireField, b: WireField): WireField {
  if (b.slice(1).every(v => v.n === BigInt(0))) return fieldScale(a, b[0]);
  if (a.slice(1).every(v => v.n === BigInt(0))) return fieldScale(b, a[0]);
  const u: Quadratic = [a[0], a[1]], v: Quadratic = [a[2], a[3]], s: Quadratic = [b[0], b[1]], t: Quadratic = [b[2], b[3]];
  const left = qadd(qmul(u, s), qscale(qmul(v, t), int(3))), right = qadd(qmul(u, t), qmul(v, s));
  return [left[0], left[1], right[0], right[1]];
}
export function fieldSign(a: WireField): number {
  const u: Quadratic = [a[0], a[1]], v: Quadratic = [a[2], a[3]], su = quadraticSign(u), sv = quadraticSign(v);
  if (!su) return sv; if (!sv || su === sv) return su;
  const difference = qadd(qmul(u, u), qscale(qmul(v, v), int(-3)));
  return su * quadraticSign(difference);
}
export const fieldEqual = (a: WireField, b: WireField) => a.every((v, i) => eq(v, b[i]));
export const fieldRational = (a: WireField): F => { if (a.slice(1).some(v => v.n !== BigInt(0))) throw new RangeError('Rational field value required'); return a[0]; };
export function fieldDivide(a: WireField, b: WireField): WireField {
  if (!fieldSign(b)) throw new RangeError('Nonzero field denominator required');
  const u: Quadratic = [b[0], b[1]], v: Quadratic = [b[2], b[3]], w = qadd(qmul(u, u), qscale(qmul(v, v), int(-3)));
  const denominator = sub(mul(w[0], w[0]), mul(int(2), mul(w[1], w[1])));
  const inverseW: Quadratic = [div(w[0], denominator), div(neg(w[1]), denominator)];
  const first = qmul(u, inverseW), second = qscale(qmul(v, inverseW), int(-1));
  return fieldMul(a, field(first[0], first[1], second[0], second[1]));
}
export const fieldSum = (values: WireField[]) => values.reduce(fieldAdd, fieldZero);

/** sign(bound - target/sqrt(norm)), by exact signed squaring. */
function compareNormalized(bound: F, target: WireField, norm: F): number {
  const st = fieldSign(target), sb = signQ(bound);
  if (!st) return sb; if (!sb) return -st; if (st !== sb) return sb;
  return st * fieldSign(fieldSub(field(mul(mul(bound, bound), norm)), fieldMul(target, target)));
}
const widthFactor = f(BigInt(10) ** BigInt(125));
function normalizedObservation(actual: unknown, target: WireField, norm: F, checks: ManyBodyWireChecks): boolean {
  if (norm.n <= BigInt(0) || !checks.q(actual)) return false;
  const q = actual as Q;
  if (!fieldSign(target)) return checks.exact(q) && matches(q, zero);
  const lower = lo(q), upper = hi(q);
  if (compareNormalized(lower, target, norm) > 0 || compareNormalized(upper, target, norm) < 0) return false;
  const absolute = fieldSign(target) < 0 ? fieldNeg(target) : target, width = mul(sub(upper, lower), widthFactor);
  return compareNormalized(one, absolute, norm) < 0 ? compareNormalized(width, absolute, norm) <= 0 : le(width, one);
}
export const expectedField = (target: WireField): ManyBodyExpected => expectedValidated((actual, checks) => {
  if (!actual || typeof actual !== 'object' || Array.isArray(actual)) return false;
  const x = actual as Record<string, unknown>;
  if (Object.keys(x).length !== 3 || !['coefficients', 'exactSign', 'observation'].every(k => Object.hasOwn(x, k)) || !Array.isArray(x.coefficients) || x.coefficients.length !== 4) return false;
  if (!x.coefficients.every((v, i) => checks.exact(v) && matches(v as Q, target[i])) || x.exactSign !== fieldSign(target)) return false;
  if (target.slice(1).every(v => v.n === BigInt(0))) return checks.exact(x.observation) && matches(x.observation as Q, target[0]);
  return normalizedObservation(x.observation, target, one, checks);
});
export const expectedNormalizedField = (target: WireField, norm: F): ManyBodyExpected => expectedValidated((actual, checks) => normalizedObservation(actual, target, norm, checks));
