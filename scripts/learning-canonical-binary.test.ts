import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/canonical-binary-independent.json';
import { canonicalBinaryModel as evaluate, type CanonicalBinaryParameters } from '../src/lib/learning-canonical-binary';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
import type { SpectrumQuantity } from '../src/lib/learning-gapped-spectrum';
import { binomialLogit, binomialRawQuantity } from '../src/lib/learning-binomial-logit';
import { binaryProduct } from '../src/lib/learning-binary-products';
interface Reference { value: string | null; sign: number; logMagnitude: string | null;
  logNegativeLogMagnitude: string | null; logMagnitudeStatus: string; logNegativeLogMagnitudeStatus: string }
function near(a: number, b: number, log = false) {
  assert.ok(Number.isFinite(a));
  assert.ok(Math.abs(a - b) <= Math.max(log ? 4e-10 : 2 * Number.MIN_VALUE, 8e-12 * Math.abs(b)), `${a} vs ${b}`);
}
function observation(a: SpectrumQuantity | null, b: Reference | null, probability = false) {
  if (b === null) { assert.equal(a, null); return; }
  assert.notEqual(a, null); assert.equal(a!.sign, b.sign);
  if (b.value === null) assert.equal(a!.value, null);
  else { assert.notEqual(a!.value, null); near(a!.value!, Number(b.value)); }
  if (b.sign !== 0) assert.notEqual(a!.value, 0);
  // Nested logs amplify rounding near a unit-valued moment. They are used as
  // tail diagnostics, not a certificate of departure from one for moments.
  const stableLog = probability || b.logMagnitude === null || Math.abs(Number(b.logMagnitude)) > 1e-9 || b.value === null;
  if (stableLog) {
    assert.equal(a!.logMagnitudeStatus, b.logMagnitudeStatus);
    if (b.logMagnitude === null) assert.equal(a!.logMagnitude, null);
    else { assert.notEqual(a!.logMagnitude, null); near(a!.logMagnitude!, Number(b.logMagnitude), true); }
    assert.equal(a!.logNegativeLogMagnitudeStatus, b.logNegativeLogMagnitudeStatus);
    if (b.logNegativeLogMagnitude === null) assert.equal(a!.logNegativeLogMagnitude, null);
    else { assert.notEqual(a!.logNegativeLogMagnitude, null); near(a!.logNegativeLogMagnitude!, Number(b.logNegativeLogMagnitude), true); }
  } else if (b.logMagnitude !== null) near(a!.logMagnitude ?? 0, Number(b.logMagnitude), true);
}
for (const [i, reference] of refs.cases.entries()) test(`independent canonical state sums/log bounds ${i + 1}`, () => {
  const r = evaluate(reference.parameters);
  for (const [key, value] of Object.entries(reference.expected))
    observation((r as unknown as Record<string, SpectrumQuantity | null>)[key], value, key === 'oneProbability' || key === 'zeroProbability');
  assert.equal(r.distribution.length, reference.distribution.length);
  r.distribution.forEach((row, k) => {
    assert.equal(row.multiplicity, reference.distribution[k].multiplicity);
    observation(row.probability, reference.distribution[k].probability, true);
  });
});
test('full distributions reproduce moments and multiplicities at ordinary scales', () => {
  for (const n of [1, 4, 32, 128, 4096]) {
    const r = evaluate({ sites: n, energyGap: kB, temperature: 1 });
    const total = r.distribution.reduce((s, row) => s + (row.probability.value ?? 0), 0);
    const first = r.distribution.reduce((s, row) => s + row.excitations * (row.probability.value ?? 0), 0);
    const variance = r.distribution.reduce((s, row) => s + (row.excitations - r.meanExcitations.value!) ** 2 * (row.probability.value ?? 0), 0);
    near(total, 1); near(first, r.meanExcitations.value!); near(variance, r.excitationVariance.value!);
    assert.equal(r.distribution.reduce((s, row) => s + BigInt(row.multiplicity), BigInt(0)).toString(), r.totalMicrostates);
  }
});
test('temperature derivative and fluctuation response agree without introducing dynamics', () => {
  for (const T of [.5, 1, 2, 10, -.5, -1, -2, -10]) {
    const p = { sites: 8, energyGap: kB * 2, temperature: T }, r = evaluate(p), h = T * 1e-4;
    const derivative = (evaluate({ ...p, temperature: T + h }).meanEnergy.value! - evaluate({ ...p, temperature: T - h }).meanEnergy.value!) / (2 * h);
    assert.ok(Math.abs(derivative - r.heatCapacity.value!) <= 1e-7 * r.heatCapacity.value!);
    near(r.energyVariance.value!, kB * T * T * r.heatCapacity.value!);
  }
});
test('positive tails can disappear from a probability display and remain in physical moments', () => {
  const r = evaluate({ sites: 2, energyGap: 1e280, temperature: 1e280 / (kB * 800) });
  assert.equal(r.oneProbability.value, null); assert.equal(r.oneProbability.sign, 1);
  assert.ok(r.meanEnergy.value! > 0); assert.ok(r.energyVariance.value! > 0);
  assert.equal(r.distribution[0].probability.value, 1);
  assert.notEqual(r.distribution[0].probability.logNegativeLogMagnitude, null);
});
test('underflowed ratios retain positive capacity, unlike a true zero gap', () => {
  const tiny = evaluate({ sites: 2, energyGap: Number.MIN_VALUE, temperature: Number.MAX_VALUE });
  const degenerate = evaluate({ sites: 2, energyGap: 0, temperature: Number.MAX_VALUE });
  assert.equal(tiny.scaledEnergyGap!.value, null); assert.equal(tiny.scaledEnergyGap!.sign, 1);
  assert.equal(tiny.heatCapacity.value, null); assert.equal(tiny.heatCapacity.sign, 1);
  assert.equal(degenerate.heatCapacity.value, 0); assert.equal(degenerate.heatCapacity.sign, 0);
});
test('finite-temperature logarithmic tails differ from the explicit zero-temperature limit', () => {
  const p = { sites: 4, energyGap: 1, temperature: Number.MIN_VALUE }, r = evaluate(p);
  assert.equal(r.oneProbability.value, null); assert.equal(r.oneProbability.sign, 1);
  assert.equal(r.oneProbability.logMagnitude, null);
  assert.ok(r.oneProbability.logNegativeLogMagnitude! > 790);
  const limit = evaluate({ ...p, temperature: 0 });
  assert.equal(limit.oneProbability.sign, 0); assert.equal(limit.entropy.sign, 0);
  const zeroGap = evaluate({ ...p, energyGap: 0, temperature: 0 });
  near(zeroGap.entropyInKB.value!, 4 * Math.LN2);
  assert.equal(zeroGap.relativeEnergyFluctuation, null);
});
test('all quantities serialize without nonfinite numbers at full physical input bounds', () => {
  const walk = (value: unknown): void => {
    if (typeof value === 'number') assert.ok(Number.isFinite(value));
    else if (value && typeof value === 'object') Object.values(value).forEach(walk);
  };
  for (const gap of [0, Number.MIN_VALUE, 1, Number.MAX_VALUE]) for (const T of [0, Number.MIN_VALUE, 1, Number.MAX_VALUE, -Number.MIN_VALUE, -1, -Number.MAX_VALUE]) {
    const r = evaluate({ sites: 4, energyGap: gap, temperature: T }); walk(r); assert.doesNotThrow(() => JSON.stringify(r));
  }
  for (const T of [Number.MIN_VALUE, -Number.MIN_VALUE]) walk(evaluate({ sites: 4096, energyGap: Number.MAX_VALUE, temperature: T }));
});
test('invalid counts and physical parameters are rejected', () => {
  const base: CanonicalBinaryParameters = { sites: 4, energyGap: kB, temperature: 1 };
  for (const key of ['sites', 'energyGap', 'temperature'] as const) for (const value of [NaN, Infinity, -Infinity])
    assert.throws(() => evaluate({ ...base, [key]: value }), RangeError);
  assert.throws(() => evaluate({ ...base, energyGap: -1 }), RangeError);
  for (const value of [-1, 1.5, 4097]) assert.throws(() => evaluate({ ...base, sites: value }), RangeError);
});
test('signed logit kernel reflects the distribution without losing positive tails', () => {
  for (const logit of [0, .25, 1, 40, 800, 4097, Number.MAX_VALUE]) {
    const a = binomialLogit(8, binaryProduct(-logit), binaryProduct(1));
    const b = binomialLogit(8, binaryProduct(logit), binaryProduct(1));
    assert.deepEqual(a.oneProbability, b.zeroProbability);
    assert.deepEqual(a.distribution.map(r => r.probability), b.distribution.map(r => r.probability).reverse());
    assert.equal(a.oneProbability.sign, 1); assert.equal(b.zeroProbability.sign, 1);
  }
  const exactUnit = binomialRawQuantity(binaryProduct(1));
  assert.equal(exactUnit.logMagnitude, 0); assert.equal(exactUnit.logNegativeLogMagnitudeStatus, 'not-applicable');
  assert.throws(() => binomialLogit(1, binaryProduct(1), binaryProduct(0)), RangeError);
});
test('bounded negative temperatures invert populations and preserve entropy and capacity', () => {
  const p = { sites: 8, energyGap: kB * 2, temperature: 1 };
  const a = evaluate(p), b = evaluate({ ...p, temperature: -1 });
  assert.equal(b.mode, 'bounded-negative-temperature'); assert.equal(b.scaledEnergyGap!.sign, -1);
  near(a.meanEnergy.value! + b.meanEnergy.value!, p.sites * p.energyGap);
  near(a.entropy.value!, b.entropy.value!); near(a.heatCapacity.value!, b.heatCapacity.value!);
  near(a.energyVariance.value!, b.energyVariance.value!);
  near(a.relativeCountFluctuation!.value! * b.relativeCountFluctuation!.value!, 1 / p.sites);
});
