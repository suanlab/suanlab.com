import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/relative-entropy-independent.json';
import { relativeEntropyFromWeights as evaluate } from '../src/lib/learning-relative-entropy';
import type { BinaryObservation } from '../src/lib/learning-binary-products';
interface Reference { sign: number; projected: number | null; logMagnitudeProjected: number | null }
const keys = ['divergence', 'trialEntropy', 'targetEntropy', 'crossEntropy'] as const;
function near(actual: number, expected: number) {
  assert.ok(Number.isFinite(actual));
  if (expected !== 0) assert.notEqual(actual, 0);
  assert.ok(Math.abs(actual - expected) <= Math.max(2 * Number.MIN_VALUE, 4e-12 * Math.abs(expected)), `${actual} vs ${expected}`);
}
function observation(actual: BinaryObservation, expected: Reference, label: string) {
  try {
    assert.equal(actual.sign, expected.sign);
    if (expected.projected === null) assert.equal(actual.value, null);
    else { assert.notEqual(actual.value, null); near(actual.value!, expected.projected); }
    if (expected.logMagnitudeProjected === null) assert.equal(actual.logMagnitude, null);
    else {
      assert.notEqual(actual.logMagnitude, null);
      if (Math.abs(expected.logMagnitudeProjected) < 1e-5) near(actual.logMagnitude!, expected.logMagnitudeProjected);
      else assert.ok(Math.abs(actual.logMagnitude! - expected.logMagnitudeProjected) < 3e-10);
    }
  } catch (error) { throw new Error(label, { cause: error }); }
}
for (const [family, rows] of Object.entries({ ordinary: refs.ordinary, limits: refs.limits })) {
  for (const [i, row] of rows.entries()) test(`${family} independent curvature quadrature ${i + 1}`, () => {
    const result = evaluate(row.trialWeights, row.targetWeights, row.scales);
    assert.equal(result.sameDistribution, row.sameDistribution);
    for (const key of keys) observation(result[key], row.measures[key] as unknown as Reference, key);
    for (const [j, expected] of row.coordinates.entries()) {
      const actual = result.coordinates[j];
      assert.equal(actual.index, expected.index); assert.equal(actual.regime, expected.regime);
      for (const [key, value] of Object.entries(expected)) {
        if (key === 'index' || key === 'regime') continue;
        const measured = (actual as unknown as Record<string, BinaryObservation | null>)[key];
        if (value === null) assert.equal(measured, null);
        else { assert.notEqual(measured, null); observation(measured!, value as Reference, `coordinate ${j} ${key}`); }
      }
    }
    for (const [name, expected] of Object.entries(row.scaledMeasures))
      for (const key of keys) observation(result.scaledMeasures[name][key], expected[key] as unknown as Reference, `${name}.${key}`);
  });
}
test('oracle covers 44 cases, 32 states, extreme scales and four isothermal identities', () => {
  assert.equal(refs.ordinary.length, 24); assert.equal(refs.limits.length, 20);
  assert.equal(refs.ordinary.filter(x => x.thermodynamicIdentityCheck).length, 4);
  assert.equal(Math.max(...refs.ordinary.map(x => x.trialWeights.length)), 32);
});
test('exact normalization recognizes proportional weights even when normalizers overflow', () => {
  for (const [w, v] of [[[1, 3, 7], [2, 6, 14]], [[Number.MAX_VALUE, Number.MAX_VALUE], [1, 1]]]) {
    const r = evaluate(w, v); assert.equal(r.sameDistribution, true);
    assert.deepEqual(r.divergence, { value: 0, sign: 0, logMagnitude: null });
  }
});
test('adjacent weights retain positive KL when subtracting displayed entropies loses positivity', () => {
  const r = evaluate([1, 1], [1, 1 + Number.EPSILON]);
  assert.ok(r.crossEntropy.value! - r.trialEntropy.value! <= 0);
  assert.equal(r.divergence.sign, 1); assert.ok(r.divergence.value! > 0);
});
test('underflowed probabilities retain entropy and logarithmic evidence', () => {
  const r = evaluate([Number.MAX_VALUE, Number.MIN_VALUE], [Number.MAX_VALUE, 2 * Number.MIN_VALUE], { rescue: [1e308, 1e308] });
  assert.equal(r.coordinates[1].trialProbability.value, null);
  assert.equal(r.coordinates[1].trialProbability.sign, 1);
  assert.equal(r.coordinates[1].trialEntropyContribution.sign, 1);
  assert.equal(r.divergence.value, null); assert.equal(r.divergence.sign, 1);
  assert.ok(r.scaledMeasures.rescue.divergence.value! > 0);
  assert.equal(r.coordinates[0].logTrialProbability!.value, null);
  assert.equal(r.coordinates[0].logTrialProbability!.sign, -1);
});
test('permutation and splitting a state preserve aggregate information', () => {
  const a = evaluate([1, 3], [3, 1]), b = evaluate([3, 1], [1, 3]);
  const split = evaluate([.5, .5, 3], [1.5, 1.5, 1]);
  near(a.divergence.value!, b.divergence.value!); near(a.divergence.value!, split.divergence.value!);
  near(split.trialEntropy.value! - a.trialEntropy.value!, .25 * Math.log(2));
});
test('KL is directional, admits zero trial probabilities, and one state has zero entropy', () => {
  assert.notEqual(evaluate([1, 4], [1, 1]).divergence.value, evaluate([1, 1], [1, 4]).divergence.value);
  const r = evaluate([1, 0], [1, 1]); near(r.divergence.value!, Math.log(2));
  assert.equal(r.coordinates[1].logProbabilityRatio, null);
  for (const key of keys) assert.equal(evaluate([Number.MIN_VALUE], [Number.MAX_VALUE])[key].value, 0);
});
test('input validation rejects sparse, nonfinite, negative, empty and unsupported vectors', () => {
  const invalid = [[], [0, 0], [-1, 1], [NaN, 1], [Infinity, 1], new Array(2), Array(33).fill(1)];
  for (const w of invalid) assert.throws(() => evaluate(w, [1, 1]), RangeError);
  for (const v of [[0, 1], [-1, 1], [Infinity, 1], new Array(2)]) assert.throws(() => evaluate([1, 1], v), RangeError);
  assert.throws(() => evaluate([1], [1, 1]), RangeError);
  for (const factors of [[-1], [NaN], [Infinity], new Array(2), Array(9).fill(1)])
    assert.throws(() => evaluate([1], [1], { bad: factors }), RangeError);
  assert.throws(() => evaluate([1], [1], null as never), RangeError);
});
test('returned data is serializable, does not mutate inputs and declares finite-weight scope', () => {
  const w = Object.freeze([1, 2]), v = Object.freeze([2, 1]);
  const r = evaluate(w, v, Object.fromEntries([['__proto__', [2]]]));
  assert.equal(r.scope, 'exactly-normalized-finite-weight-distributions');
  assert.ok(Object.prototype.hasOwnProperty.call(r.scaledMeasures, '__proto__'));
  assert.doesNotThrow(() => JSON.stringify(r));
  assert.notEqual(r.trialWeights, w); assert.deepEqual(w, [1, 2]);
  assert.equal('hamiltonian' in r, false); assert.equal('temperature' in r, false);
});
