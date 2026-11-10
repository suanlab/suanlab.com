import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/canonical-gap-independent.json';
import { finiteCanonicalHelmholtzGap as evaluate } from '../src/lib/learning-canonical-gap';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
import type { BinaryObservation } from '../src/lib/learning-binary-products';
function observation(actual: BinaryObservation | null, expected: BinaryObservation) {
  assert.notEqual(actual, null);
  assert.equal(actual!.sign, expected.sign);
  if (expected.value === null) assert.equal(actual!.value, null);
  else {
    assert.notEqual(actual!.value, null);
    if (expected.sign !== 0) assert.notEqual(actual!.value, 0);
    assert.ok(Math.abs(actual!.value! - expected.value) <= Math.max(2 * Number.MIN_VALUE, 6e-12 * Math.abs(expected.value)), `${actual!.value} vs ${expected.value}`);
  }
  if (expected.logMagnitude === null) assert.equal(actual!.logMagnitude, null);
  else { assert.notEqual(actual!.logMagnitude, null); assert.ok(Math.abs(actual!.logMagnitude! - expected.logMagnitude) < 2e-9); }
}
for (const [family, rows] of Object.entries({ ordinary: refs.ordinary, limits: refs.limits })) {
  for (const [i, row] of rows.entries()) test(`${family} independent fixed-Hamiltonian microstate KL ${i + 1}`, () => {
    const r = evaluate(row.energies, row.temperature, row.trialWeights, row.copies);
    assert.notEqual(r.resolution, 'unresolved-at-precision-ceiling');
    assert.ok(r.exactInterval !== null);
    const shift = row.referenceBits + r.exactInterval.exponent;
    assert.ok(shift >= 0);
    assert.ok((BigInt(r.exactInterval.lowerCoefficient) << BigInt(shift)) <= BigInt(row.divergenceEnvelope.lower));
    assert.ok((BigInt(r.exactInterval.upperCoefficient) << BigInt(shift)) >= BigInt(row.divergenceEnvelope.upper));
    for (const key of ['divergence', 'perCopyGap', 'totalGap'] as const) observation(r[key], row[key]);
    if (row.divergence.sign !== 0) assert.equal(r.bounds.lower.sign, 1);
  });
}
test('the rounded canonical weights are not silently substituted as the target', () => {
  const E = [0, kB * 300 * Math.log(3)];
  const r = evaluate(E, 300, [3, 1]);
  assert.equal(r.trial.canonical.states[0].probability.value, .75);
  assert.equal(r.resolution, 'positive-interval-resolved');
  assert.equal(r.divergence!.sign, 1); assert.ok(r.divergence!.value! > 0);
  assert.ok(r.precisionBits >= 256);
});
test('an almost degenerate Hamiltonian triggers precision escalation and retains its tiny gap', () => {
  const r = evaluate([0, Number.MIN_VALUE], Number.MAX_VALUE, [1, 1], Number.MAX_VALUE);
  assert.equal(r.resolution, 'positive-interval-resolved');
  assert.ok(r.precisionBits >= 4096);
  assert.equal(r.divergence!.value, null); assert.equal(r.divergence!.sign, 1);
  assert.equal(r.perCopyGap!.sign, 1); assert.ok(r.divergence!.logMagnitude! < -2000);
});
test('tiny absolute temperature does not lose a representable energy gap through KL overflow', () => {
  const r = evaluate([0, 1], Number.MIN_VALUE, [1, 1]);
  assert.equal(r.divergence!.value, null); assert.equal(r.divergence!.sign, 1);
  assert.ok(Math.abs(r.perCopyGap!.value! - .5) < 1e-12);
});
test('cold canonical tails resolve logarithmically instead of becoming physical zero', () => {
  const r = evaluate([0, 1], Number.MIN_VALUE, [1, 0]);
  assert.equal(r.resolution, 'positive-log-tail-resolved');
  assert.equal(r.precisionBits, 256); assert.equal(r.divergence!.value, null); assert.equal(r.perCopyGap!.value, null);
  assert.equal(r.divergence!.sign, 1); assert.equal(r.perCopyGap!.sign, 1);
  assert.equal(r.bounds.lower.sign, 1); assert.equal(r.bounds.upper.sign, 1);
  const none = evaluate([0, 1], Number.MIN_VALUE, [1, 0], 0);
  assert.equal(none.totalGap!.value, 0);
});
test('structural equality uses exact input conditions, not a numerical residual tolerance', () => {
  const equal = evaluate([7, 7, 7], 300, [Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE]);
  assert.equal(equal.resolution, 'structural-equilibrium'); assert.equal(equal.divergence!.value, 0);
  assert.deepEqual(equal.attempts, []);
  const distinct = evaluate([7, 7, 7], 300, [1, 1, 1 + Number.EPSILON]);
  assert.equal(distinct.resolution, 'positive-interval-resolved'); assert.equal(distinct.divergence!.sign, 1);
});
test('gauge shift, permutation, copy scaling and JSON contracts', () => {
  const a = evaluate([-4, 0, 4], 1e23, [1, 2, 1], 8), b = evaluate([4, 8, 12], 1e23, [1, 2, 1], 8);
  assert.deepEqual(a.divergence, b.divergence); assert.deepEqual(a.perCopyGap, b.perCopyGap);
  assert.ok(Math.abs(a.totalGap!.value! - 8 * a.perCopyGap!.value!) < 1e-12);
  const permuted = evaluate([4, -4, 0], 1e23, [1, 1, 2], 8);
  observation(permuted.divergence, a.divergence!);
  assert.doesNotThrow(() => JSON.stringify(a));
  assert.throws(() => evaluate([0, 1], 0, [1, 1]), RangeError);
  assert.throws(() => evaluate([0, 1], 300, [0, 0]), RangeError);
});
