import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/canonical-spectrum-independent.json';
import { canonicalFiniteSpectrum as evaluate, canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
import type { BinaryObservation } from '../src/lib/learning-binary-products';
function near(a: number, b: number) {
  if (b !== 0) assert.notEqual(a, 0);
  assert.ok(Math.abs(a - b) <= Math.max(2 * Number.MIN_VALUE, 6e-12 * Math.abs(b)), `${a} vs ${b}`);
}
function observation(a: BinaryObservation, b: BinaryObservation, label: string) {
  try {
    assert.equal(a.sign, b.sign);
    if (b.value === null) assert.equal(a.value, null);
    else { assert.notEqual(a.value, null); near(a.value!, b.value); }
    if (b.logMagnitude === null) assert.equal(a.logMagnitude, null);
    else {
      assert.notEqual(a.logMagnitude, null);
      assert.ok(Math.abs(a.logMagnitude! - b.logMagnitude) < 5e-10 || Math.abs((a.logMagnitude! - b.logMagnitude) / b.logMagnitude) < 6e-12,
        `${a.logMagnitude} vs ${b.logMagnitude}`);
    }
  } catch (cause) { throw new Error(label, { cause }); }
}
for (const [family, rows] of Object.entries({ ordinary: refs.ordinary, limits: refs.limits })) {
  for (const [i, row] of rows.entries()) test(`${family} independent canonical sum ${i + 1}`, () => {
    const r = evaluate(row.energies, row.temperature);
    assert.equal(r.groundCount, row.groundCount);
    observation(r.thermalEnergy, row.thermalEnergy, 'thermalEnergy');
    observation(r.centeredLogPartition, row.centeredLogPartition, 'centeredLogPartition');
    for (const [j, expected] of row.states.entries()) {
      const actual = r.states[j]; assert.equal(actual.groundState, expected.groundState);
      for (const key of ['energyGap', 'scaledEnergyGap', 'logProbability', 'probability'] as const)
        observation(actual[key], expected[key], `${j}.${key}`);
      const raw = actual.exactScaledEnergyGap;
      const exponent = raw.numerator.exponent - raw.denominator.exponent;
      let numerator = BigInt(raw.numerator.coefficient), denominator = BigInt(raw.denominator.coefficient);
      if (exponent >= 0) numerator <<= BigInt(exponent);
      else denominator <<= BigInt(-exponent);
      assert.equal(numerator * BigInt(expected.exactScaledEnergyGap.denominator),
        denominator * BigInt(expected.exactScaledEnergyGap.numerator));
    }
  });
}
test('fixed Hamiltonian temperature changes move the canonical probabilities', () => {
  const energies = Object.freeze([0, kB * 300 * Math.log(3)]);
  const a = evaluate(energies, 300), b = evaluate(energies, 600);
  near(a.states[0].probability.value!, .75);
  near(b.states[0].probability.value!, Math.sqrt(3) / (1 + Math.sqrt(3)));
  assert.deepEqual(a.energies, b.energies);
});
test('exact common energy shifts and state permutations preserve probabilities', () => {
  const a = evaluate([-4, 0, 4], 1e23), b = evaluate([4, 8, 12], 1e23), c = evaluate([4, -4, 0], 1e23);
  for (let i = 0; i < 3; i++) {
    assert.deepEqual(a.states[i].probability, b.states[i].probability);
    assert.deepEqual(a.states[i].probability, c.states[[1, 2, 0][i]].probability);
  }
});
test('overflowed energy differences and thermal products are retained as rational coordinates', () => {
  const r = evaluate([-Number.MAX_VALUE, Number.MAX_VALUE], Number.MIN_VALUE);
  assert.equal(r.thermalEnergy.value, null); assert.equal(r.thermalEnergy.sign, 1);
  const s = r.states[1]; assert.equal(s.energyGap.value, null); assert.equal(s.scaledEnergyGap.value, null);
  assert.equal(s.scaledEnergyGap.sign, 1); assert.equal(s.logProbability.sign, -1);
  assert.equal(s.probability.value, null); assert.equal(s.probability.sign, 1);
  assert.equal(r.states[0].probability.value, 1);
  assert.equal(r.states[0].logProbability.value, null); assert.equal(r.states[0].logProbability.sign, -1);
  assert.notEqual(BigInt(s.exactScaledEnergyGap.numerator.coefficient), BigInt(0));
  assert.notEqual(BigInt(s.exactScaledEnergyGap.denominator.coefficient), BigInt(0));
});
test('finite temperature never erases an excited state or creates an exact ground probability one', () => {
  const r = evaluate([0, kB * 300 * 1000], 300);
  assert.equal(r.states[1].probability.value, null); assert.equal(r.states[1].probability.sign, 1);
  assert.ok(r.states[1].probability.logMagnitude! < -999);
  assert.equal(r.states[0].probability.value, 1);
  assert.equal(r.states[0].logProbability.sign, -1);
  assert.equal(r.centeredLogPartition.sign, 1); assert.equal(r.centeredLogPartition.value, null);
});
test('degenerate microstates count once and a singleton has structural zero log probability', () => {
  const r = evaluate([0, 0, kB * 300 * Math.log(2)], 300);
  for (const [i, p] of [.4, .4, .2].entries()) near(r.states[i].probability.value!, p);
  const single = evaluate([42], 300);
  assert.deepEqual(single.states[0].logProbability, { value: 0, sign: 0, logMagnitude: null });
  assert.equal(single.states[0].probability.value, 1);
});
test('validation rejects unsupported states, invalid temperatures and sparse arrays', () => {
  for (const e of [[], new Array(2), [NaN], [Infinity], Array(33).fill(0)]) assert.throws(() => evaluate(e, 300), RangeError);
  for (const T of [0, -1, NaN, Infinity]) assert.throws(() => evaluate([0, 1], T), RangeError);
  assert.doesNotThrow(() => JSON.stringify(evaluate([-Number.MAX_VALUE, Number.MAX_VALUE], Number.MIN_VALUE)));
});
