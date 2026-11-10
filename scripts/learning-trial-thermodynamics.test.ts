import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/trial-thermodynamics-independent.json';
import { finiteTrialThermodynamics as evaluate } from '../src/lib/learning-trial-thermodynamics';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
import { binaryProduct as product, binarySum as sum, binaryNegate as negate, binaryObservation as observe, type BinaryObservation } from '../src/lib/learning-binary-products';
function near(a: number, b: number, scale = Math.abs(b)) {
  assert.ok(Number.isFinite(a));
  assert.ok(Math.abs(a - b) <= Math.max(2 * Number.MIN_VALUE, 6e-12 * scale), `${a} vs ${b}`);
}
function observation(a: BinaryObservation, b: BinaryObservation, label: string) {
  try {
    assert.equal(a.sign, b.sign);
    if (b.value === null) assert.equal(a.value, null);
    else { assert.notEqual(a.value, null); if (b.value !== 0) assert.notEqual(a.value, 0); near(a.value!, b.value); }
    if (b.logMagnitude === null) assert.equal(a.logMagnitude, null);
    else { assert.notEqual(a.logMagnitude, null); assert.ok(Math.abs(a.logMagnitude! - b.logMagnitude) < 5e-10); }
  } catch (cause) { throw new Error(label, { cause }); }
}
for (const [family, rows] of Object.entries({ ordinary: refs.ordinary, limits: refs.limits })) {
  for (const [i, row] of rows.entries()) test(`${family} independent trial energy/entropy ${i + 1}`, () => {
    const r = evaluate(row.energies, row.temperature, row.trialWeights, row.copies);
    for (const group of ['perCopy', 'total'] as const) {
      for (const key of ['meanEnergy', 'informationEntropy', 'entropy', 'temperatureTimesEntropy'] as const)
        observation(r[group][key], row[group][key], `${group}.${key}`);
      const actual = r[group].helmholtz, expected = row[group].helmholtz;
      const scale = Math.max(Math.abs(row[group].meanEnergy.value ?? 0), Math.abs(row[group].temperatureTimesEntropy.value ?? 0));
      if (expected.value === null) assert.equal(actual.value, null);
      else {
        assert.notEqual(actual.value, null); near(actual.value!, expected.value, scale);
        if (Math.abs(expected.value) > Math.max(4 * Number.MIN_VALUE, scale * 3e-11)) assert.equal(actual.sign, expected.sign);
      }
      if (actual.sign === 0 && expected.sign !== 0) assert.equal(actual.zeroStatus, 'unresolved-transcendental-cancellation');
    }
    for (const [j, expected] of row.states.entries()) {
      for (const key of ['probability', 'meanEnergyContribution', 'informationEntropyContribution'] as const)
        observation(r.states[j][key], expected[key], `${j}.${key}`);
    }
  });
}
test('reference-temperature uniform trial reproduces U, H, S and F with physical units', () => {
  const r = evaluate([0, kB * 300 * Math.log(3)], 300, [1, 1]);
  near(r.perCopy.meanEnergy.value! / (kB * 300), Math.log(3) / 2);
  near(r.perCopy.informationEntropy.value!, Math.log(2));
  near(r.perCopy.entropy.value!, kB * Math.log(2));
  near(r.perCopy.helmholtz.value! / (kB * 300), Math.log(3) / 2 - Math.log(2));
  assert.equal(r.units.helmholtz, 'J'); assert.equal(r.units.entropy, 'J/K');
});
test('mean energy uses exact normalization even when weighted products overflow', () => {
  const r = evaluate([Number.MAX_VALUE, -Number.MAX_VALUE], 300, [Number.MAX_VALUE, Number.MAX_VALUE]);
  assert.equal(r.perCopy.meanEnergy.value, 0);
  near(r.perCopy.helmholtz.value!, -kB * 300 * Math.log(2));
});
test('nearly concentrated distributions retain entropy and total scaling rescues underflow', () => {
  const r = evaluate([0, 0], Number.MIN_VALUE, [1, 1], Number.MAX_VALUE);
  assert.equal(r.perCopy.temperatureTimesEntropy.value, null); assert.equal(r.perCopy.temperatureTimesEntropy.sign, 1);
  assert.ok(r.total.temperatureTimesEntropy.value! > 0);
  const tiny = evaluate([0, 0], Number.MAX_VALUE, [Number.MAX_VALUE, Number.MIN_VALUE], Number.MAX_VALUE);
  assert.equal(tiny.perCopy.informationEntropy.value, null);
  assert.equal(tiny.perCopy.informationEntropy.sign, 1);
  assert.ok(tiny.total.informationEntropy.value! > 0);
  assert.equal(tiny.total.entropy.value, null); assert.equal(tiny.total.entropy.sign, 1);
  assert.ok(tiny.total.temperatureTimesEntropy.value! > 0);
});
test('constant shifts change mean energy and Helmholtz while preserving entropy and target', () => {
  const a = evaluate([-4, 0, 4], 1e23, [1, 2, 1], 8), b = evaluate([4, 8, 12], 1e23, [1, 2, 1], 8);
  near(b.perCopy.meanEnergy.value! - a.perCopy.meanEnergy.value!, 8);
  near(b.perCopy.helmholtz.value! - a.perCopy.helmholtz.value!, 8);
  near(b.total.helmholtz.value! - a.total.helmholtz.value!, 64);
  assert.deepEqual(a.perCopy.entropy, b.perCopy.entropy);
  for (let i = 0; i < 3; i++) assert.deepEqual(a.canonical.states[i].probability, b.canonical.states[i].probability);
});
test('rounded transcendental cancellation is distinct from a structural zero', () => {
  const exactApproximation = product(2, kB, 256, Math.LN2);
  const high = observe(exactApproximation).value!;
  const low = observe(sum([exactApproximation, negate(product(high))])).value!;
  const cancelled = evaluate([high, low], 256, [1, 1]);
  assert.equal(cancelled.perCopy.helmholtz.value, 0);
  assert.equal(cancelled.perCopy.helmholtz.zeroStatus, 'unresolved-transcendental-cancellation');
  const structural = evaluate([0, 5], 300, [1, 0]);
  assert.equal(structural.perCopy.helmholtz.zeroStatus, 'structural-zero');
  const noCopies = evaluate([1, 2], 300, [1, 1], 0);
  assert.equal(noCopies.total.helmholtz.zeroStatus, 'structural-zero');
});
test('permutation and proportional input weights preserve trial functionals', () => {
  const a = evaluate([1, 2, 4], 300, [1, 3, 2]), b = evaluate([4, 1, 2], 300, [4, 2, 6]);
  for (const key of ['meanEnergy', 'informationEntropy', 'entropy', 'temperatureTimesEntropy', 'helmholtz'] as const)
    near(a.perCopy[key].value!, b.perCopy[key].value!);
});
test('boundary, singleton and validation contracts remain explicit and serializable', () => {
  const one = evaluate([5], Number.MIN_VALUE, [Number.MIN_VALUE]);
  assert.equal(one.perCopy.entropy.value, 0); assert.equal(one.perCopy.helmholtz.value, 5);
  for (const w of [[0, 0], [-1, 1], [NaN, 1], [Infinity, 1], new Array(2), [1]])
    assert.throws(() => evaluate([0, 1], 300, w), RangeError);
  for (const copies of [-1, .5, Infinity, NaN]) assert.throws(() => evaluate([0], 300, [1], copies), RangeError);
  const weights = Object.freeze([1, 2]);
  assert.doesNotThrow(() => JSON.stringify(evaluate(Object.freeze([0, 1]), 300, weights)));
  assert.deepEqual(weights, [1, 2]);
});
