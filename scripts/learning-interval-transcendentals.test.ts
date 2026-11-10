import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/interval-transcendentals-independent.json';
import { createTranscendentalContext as context, type FixedInterval } from '../src/lib/learning-interval-transcendentals';
import { binaryProduct as product, binarySum as sum } from '../src/lib/learning-binary-products';
function contains(interval: FixedInterval, reference: { lower: string; upper: string }, bits: number) {
  assert.ok(interval.lower <= interval.upper);
  assert.ok((interval.lower << BigInt(bits)) <= BigInt(reference.lower), 'lower endpoint excludes reference envelope');
  assert.ok((interval.upper << BigInt(bits)) >= BigInt(reference.upper), 'upper endpoint excludes reference envelope');
}
for (const [i, row] of refs.cases.entries()) test(`independent rational transcendental enclosure ${i + 1} at ${row.precisionBits} bits`, () => {
  const c = context(row.precisionBits), n = product(...row.numeratorFactors), d = product(...row.denominatorFactors);
  if (row.logarithm !== null) {
    const log = c.logRatio(n, d);
    contains(log, row.logarithm, row.fineScaleBits);
    assert.ok(log.upper - log.lower <= BigInt(16));
  }
  const exp = c.expNegativeRatio(n, d);
  contains(exp, row.negativeExponential, row.fineScaleBits);
  assert.ok(exp.lower >= BigInt(0)); assert.ok(exp.upper <= c.scale);
  assert.ok(exp.upper - exp.lower <= BigInt(16));
  assert.ok(exp.upper > BigInt(0), 'a finite exponential cannot have an exact zero interval');
});
test('exact identities remain points rather than numerical zero tolerances', () => {
  const c = context(256);
  assert.deepEqual(c.logRatio(product(1), product(1)), { lower: BigInt(0), upper: BigInt(0) });
  assert.deepEqual(c.expNegativeRatio(product(0), product(1)), { lower: c.scale, upper: c.scale });
});
test('adjacent probabilities remain distinguished at 64 bits', () => {
  const c = context(64), above = c.logRatio(product(1 + Number.EPSILON), product(1));
  const below = c.logRatio(product(1), product(1 + Number.EPSILON));
  assert.ok(above.lower > BigInt(0)); assert.ok(below.upper < BigInt(0));
  assert.equal(above.lower, -below.upper); assert.equal(above.upper, -below.lower);
});
test('subnormal energy coordinates can require higher precision without implying equality', () => {
  const n = product(Number.MIN_VALUE), d = product(1.380649e-23, Number.MAX_VALUE);
  const low = context(256), high = context(4096);
  const a = low.expNegativeRatio(n, d), b = high.expNegativeRatio(n, d);
  assert.equal(a.upper, low.scale); assert.ok(a.lower < low.scale);
  assert.ok(b.upper < high.scale); assert.ok(b.lower > BigInt(0));
  assert.ok((a.lower << BigInt(4096 - 256)) <= b.lower);
  assert.ok((a.upper << BigInt(4096 - 256)) >= b.upper);
});
test('large energy ratios retain a nonzero upper bound without evaluating an infinite argument', () => {
  const c = context(256);
  const result = c.expNegativeRatio(product(Number.MAX_VALUE), product(1.380649e-23, Number.MIN_VALUE));
  assert.equal(result.lower, BigInt(0)); assert.equal(result.upper, BigInt(1));
});
test('logarithm intervals narrow consistently as precision increases', () => {
  const low = context(64).logRatio(product(7), product(3));
  const high = context(1024).logRatio(product(7), product(3));
  assert.ok((low.lower << BigInt(960)) <= high.lower);
  assert.ok((low.upper << BigInt(960)) >= high.upper);
});
test('8192-bit range is available for a later adaptive residual calculation', () => {
  const c = context(8192), l = c.logRatio(product(2), product(1));
  const e = c.expNegativeRatio(product(1), product(8));
  assert.ok(l.lower > BigInt(0)); assert.ok(l.upper < c.scale);
  assert.ok(e.lower > BigInt(0)); assert.ok(e.upper < c.scale);
  assert.ok(l.upper - l.lower <= BigInt(16)); assert.ok(e.upper - e.lower <= BigInt(16));
});
test('exact pre-projection sums can be logarithm inputs after ordinary normalizers overflow', () => {
  const c = context(256), W = sum([product(Number.MAX_VALUE), product(Number.MAX_VALUE)]);
  const result = c.logRatio(W, product(Number.MAX_VALUE));
  assert.deepEqual(result, c.logRatio(product(2), product(1)));
});
test('internal bounds reject unsupported precision and invalid rational inputs', () => {
  for (const P of [0, 63, 8193, 256.5, NaN, Infinity]) assert.throws(() => context(P), RangeError);
  const c = context();
  for (const n of [product(0), product(-1), { coefficient: BigInt(1), exponent: Infinity }])
    assert.throws(() => c.logRatio(n, product(1)), RangeError);
  assert.throws(() => c.expNegativeRatio(product(-1), product(1)), RangeError);
  assert.throws(() => c.expNegativeRatio(product(1), product(0)), RangeError);
});
