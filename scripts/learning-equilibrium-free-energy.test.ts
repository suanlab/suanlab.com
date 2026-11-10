import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/equilibrium-free-energy-independent.json';
import { finiteEquilibriumFreeEnergy as evaluate } from '../src/lib/learning-equilibrium-free-energy';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
import type { BinaryObservation } from '../src/lib/learning-binary-products';
function near(a: number, b: number, scale = Math.abs(b)) {
  assert.ok(Number.isFinite(a));
  assert.ok(Math.abs(a - b) <= Math.max(2 * Number.MIN_VALUE, 8e-12 * scale), `${a} vs ${b}`);
}
function correction(a: BinaryObservation, b: BinaryObservation) {
  assert.equal(a.sign, b.sign);
  if (b.value === null) assert.equal(a.value, null);
  else { assert.notEqual(a.value, null); if (b.sign !== 0) assert.notEqual(a.value, 0); near(a.value!, b.value); }
  if (b.logMagnitude === null) assert.equal(a.logMagnitude, null);
  else { assert.notEqual(a.logMagnitude, null); near(a.logMagnitude!, b.logMagnitude, Math.max(1, Math.abs(b.logMagnitude))); }
}
for (const [family, rows] of Object.entries({ ordinary: refs.ordinary, limits: refs.limits })) {
  for (const [i, row] of rows.entries()) test(`${family} independent equilibrium free energy ${i + 1}`, () => {
    const r = evaluate(row.energies, row.temperature, row.copies);
    for (const name of ['perCopy', 'total'] as const) {
      correction(r[name].centeredPartitionEnergyCorrection, row[name].centeredPartitionEnergyCorrection);
      const actual = r[name].helmholtz, expected = row[name].helmholtz;
      if (expected.value === null) { assert.equal(actual.value, null); assert.equal(actual.sign, expected.sign); }
      else {
        assert.notEqual(actual.value, null);
        const scale = Math.max(Math.abs(Math.min(...row.energies) * (name === 'total' ? row.copies : 1)), Math.abs(row[name].centeredPartitionEnergyCorrection.value ?? 0));
        near(actual.value!, expected.value, scale);
        if (Math.abs(expected.value) > Math.max(4 * Number.MIN_VALUE, scale * 4e-11)) assert.equal(actual.sign, expected.sign);
        if (r.correctionRegime === 'logarithmic-infinitesimal') assert.equal(actual.value, expected.value);
      }
      if (actual.sign === 0 && expected.sign !== 0) assert.equal(actual.zeroStatus, 'unresolved-transcendental-cancellation');
      if (expected.logMagnitude !== null && actual.logMagnitude !== null)
        near(actual.logMagnitude, expected.logMagnitude, Math.max(1, Math.abs(expected.logMagnitude)));
    }
  });
}
test('reference-temperature spectrum reproduces minus kBT log Z', () => {
  const r = evaluate([0, kB * 300 * Math.log(3)], 300, 16);
  near(r.perCopy.helmholtz.value!, -kB * 300 * Math.log(4 / 3));
  near(r.total.helmholtz.value!, 16 * r.perCopy.helmholtz.value!);
});
test('unrepresentable centered partition logarithm can yield a finite energy correction', () => {
  const r = evaluate([0, kB * Number.MAX_VALUE * 1000], Number.MAX_VALUE);
  assert.equal(r.canonical.centeredLogPartition.value, null);
  assert.equal(r.correctionRegime, 'reconstructed-before-scaling');
  assert.ok(r.perCopy.centeredPartitionEnergyCorrection.value! > 0);
  assert.ok(r.perCopy.helmholtz.value! < 0);
  const replicas = evaluate([0, kB * 300 * 1000], 300, Number.MAX_VALUE);
  assert.equal(replicas.perCopy.centeredPartitionEnergyCorrection.value, null);
  assert.ok(replicas.total.centeredPartitionEnergyCorrection.value! > 0);
});
test('subnormal centered logarithm is reconstructed before scaling', () => {
  const r = evaluate([0, kB * Number.MAX_VALUE * 744], Number.MAX_VALUE);
  assert.ok(r.canonical.centeredLogPartition.value! < 2 ** -1022);
  assert.ok(r.perCopy.centeredPartitionEnergyCorrection.value! > 0);
  assert.equal(r.correctionRegime, 'reconstructed-before-scaling');
});
test('arbitrarily small positive corrections do not become physical zero', () => {
  const r = evaluate([0, Number.MAX_VALUE], Number.MIN_VALUE, Number.MAX_VALUE);
  assert.equal(r.correctionRegime, 'logarithmic-infinitesimal');
  assert.equal(r.total.centeredPartitionEnergyCorrection.value, null);
  assert.equal(r.total.centeredPartitionEnergyCorrection.sign, 1);
  assert.equal(r.total.helmholtz.sign, -1); assert.equal(r.total.helmholtz.value, null);
  assert.equal(r.total.helmholtz.zeroStatus, 'not-zero-approximation');
});
test('infinitesimal negative corrections decide exact anchor rounding midpoints', () => {
  const anchor = 1 + Number.EPSILON;
  const r = evaluate([anchor, 2], Number.MIN_VALUE, 3);
  assert.equal(anchor * 3, 3.000000000000001);
  assert.equal(r.total.helmholtz.value, 3.0000000000000004);
  assert.ok(r.total.helmholtz.value! < anchor * 3);
  const negative = evaluate([-(1 + 3 * Number.EPSILON), 0], Number.MIN_VALUE, 3);
  assert.equal(negative.total.helmholtz.value, -3.000000000000002);
  const overflowAnchor = 6004799503160661 * 2 ** 970;
  assert.equal(overflowAnchor * 3, Infinity);
  assert.equal(evaluate([overflowAnchor, Number.MAX_VALUE], Number.MIN_VALUE, 3).total.helmholtz.value, Number.MAX_VALUE);
  assert.equal(evaluate([-overflowAnchor, 0], Number.MIN_VALUE, 3).total.helmholtz.value, null);
});
test('energy reference changes Feq and total Feq while preserving centered correction', () => {
  const a = evaluate([-4, 0, 4], 1e23, 8), b = evaluate([4, 8, 12], 1e23, 8);
  near(b.perCopy.helmholtz.value! - a.perCopy.helmholtz.value!, 8);
  near(b.total.helmholtz.value! - a.total.helmholtz.value!, 64);
  assert.deepEqual(a.perCopy.centeredPartitionEnergyCorrection, b.perCopy.centeredPartitionEnergyCorrection);
});
test('degenerate, singleton, zero-copy and serialization contracts', () => {
  near(evaluate([0, 0, 0], 300).perCopy.helmholtz.value!, -kB * 300 * Math.log(3));
  const singleton = evaluate([0], Number.MIN_VALUE);
  assert.equal(singleton.perCopy.helmholtz.zeroStatus, 'structural-zero');
  assert.equal(singleton.perCopy.centeredPartitionEnergyCorrection.sign, 0);
  const zero = evaluate([1, 2], 300, 0);
  assert.equal(zero.total.helmholtz.zeroStatus, 'structural-zero');
  assert.equal(zero.total.centeredPartitionEnergyCorrection.sign, 0);
  assert.doesNotThrow(() => JSON.stringify(evaluate([-Number.MAX_VALUE, Number.MAX_VALUE], Number.MIN_VALUE, Number.MAX_VALUE)));
  for (const copies of [-1, .5, NaN, Infinity]) assert.throws(() => evaluate([0], 300, copies), RangeError);
});
