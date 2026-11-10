/** Exact semantic trees for bounded response schemas and input attribution. */
import { createQuantumVariationalWireChecks } from './learning-quantum-variational-wire-schema';
import { zero, one, sub, mul, le, lo, hi, matches, range, point, rmul, contains, type F, type C } from './learning-tensor-wire-math';
import type { PartitionObservation as Q } from './learning-partition-exact-core';

const observation = Symbol('expected-observation');
const validated = Symbol('expected-validated-value');
export type ManyBodyWireChecks = ReturnType<typeof createQuantumVariationalWireChecks>;
type ExpectedValidated = { [validated]: true; check: (actual: unknown, checks: ManyBodyWireChecks) => boolean };
type ExpectedObservation = { [observation]: true; coefficient: F; squaredNorm: F | null };
export type ManyBodyExpected = null | boolean | number | string | ExpectedObservation | ExpectedValidated | ManyBodyExpected[] | { [key: string]: ManyBodyExpected };
/** Trusted mathematical predicate in an expected tree, never executable wire data. */
export const expectedValidated = (check: ExpectedValidated['check']): ExpectedValidated => ({ [validated]: true, check });
export const expectedRational = (coefficient: F): ExpectedObservation => ({ [observation]: true, coefficient, squaredNorm: null });
export const expectedComplex = (z: C): ManyBodyExpected => ({ real: expectedRational(z.re), imaginary: expectedRational(z.im) });
export const expectedNormalized = (z: C, squaredNorm: F): ManyBodyExpected => ({
  real: { [observation]: true, coefficient: z.re, squaredNorm },
  imaginary: { [observation]: true, coefficient: z.im, squaredNorm },
});

/** Call only with a descriptor-only snapshot, never the caller's object. */
export function matchManyBodyExpected(actual: unknown, expected: ManyBodyExpected): boolean {
  const base = createQuantumVariationalWireChecks();
  const checks = { ...base, q: (value: unknown) => {
    if (!base.q(value)) return false;
    const q = value as Q;
    if (q.resolution !== 'interval-resolved' && q.resolution !== 'unresolved-at-precision-ceiling') return true;
    const a = lo(q), b = hi(q), near = a.n > BigInt(0) ? a : b.n < BigInt(0) ? { n: -b.n, d: b.d } : zero;
    const resolved = near.n > BigInt(0) && le(mul(sub(b, a), { n: BigInt(1) << BigInt(48), d: BigInt(1) }), near);
    // A tightly resolved interval must not be relabelled as unresolved to
    // suppress its display value/log checks. Mirrors the public observation
    // contract using exact rational comparisons, without an interval engine.
    return (q.resolution === 'interval-resolved') === resolved;
  } };
  function match(actual: unknown, expected: ManyBodyExpected): boolean {
    if (expected === null || typeof expected !== 'object') return actual === expected;
    if (validated in expected) return (expected as ExpectedValidated).check(actual, checks);
    if (observation in expected) {
      const target = expected as ExpectedObservation, a = target.coefficient;
      if (!checks.q(actual)) return false;
      const q = actual as Q;
      if (target.squaredNorm === null || a.n === BigInt(0)) return checks.exact(q) && matches(q, a);
      const N = target.squaredNorm;
      if (q.sign !== (a.n > BigInt(0) ? 1 : -1) || !contains(rmul(rmul(range(q), range(q)), point(N)), mul(a, a))) return false;
      // Relative width uses |a|/sqrt(N), not the raw coordinate |a|.
      // Squaring positive quantities handles arbitrary rational N exactly.
      const width = mul(sub(hi(q), lo(q)), { n: BigInt(10) ** BigInt(125), d: BigInt(1) });
      return le(zero, width) && (le(mul(a, a), N) ? le(width, one) : le(mul(mul(width, width), N), mul(a, a)));
    }
    if (Array.isArray(expected)) return Array.isArray(actual) && actual.length === expected.length && expected.every((v, i) => match(actual[i], v));
    if (!actual || typeof actual !== 'object' || Array.isArray(actual)) return false;
    const keys = Object.keys(expected), data = actual as Record<string, unknown>;
    return Object.keys(data).length === keys.length && keys.every(key => Object.hasOwn(data, key) && match(data[key], expected[key]));
  }
  return match(actual, expected);
}
