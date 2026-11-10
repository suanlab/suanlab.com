import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/grand-occupancy-independent.json';
import { grandCanonicalOccupancyModel as evaluate, type GrandOccupancyParameters } from '../src/lib/learning-grand-canonical-occupancy';
import { canonicalBinaryModel } from '../src/lib/learning-canonical-binary';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
import type { SpectrumQuantity } from '../src/lib/learning-gapped-spectrum';
interface Reference { value: string | null; sign: number; logMagnitude: string | null;
  logNegativeLogMagnitude: string | null; logMagnitudeStatus: string; logNegativeLogMagnitudeStatus: string }
function near(a: number, b: number, log = false) {
  assert.ok(Number.isFinite(a));
  assert.ok(Math.abs(a - b) <= Math.max(log ? 5e-10 : 2 * Number.MIN_VALUE, 8e-12 * Math.abs(b)), `${a} vs ${b}`);
}
function observation(a: SpectrumQuantity | null, b: Reference | null, probability = false) {
  if (b === null) { assert.equal(a, null); return; }
  assert.notEqual(a, null); assert.equal(a!.sign, b.sign);
  if (b.value === null) assert.equal(a!.value, null);
  else { assert.notEqual(a!.value, null); near(a!.value!, Number(b.value)); }
  if (b.sign !== 0) assert.notEqual(a!.value, 0);
  const stable = probability || b.logMagnitude === null || Math.abs(Number(b.logMagnitude)) > 1e-9 || b.value === null;
  if (stable) {
    assert.equal(a!.logMagnitudeStatus, b.logMagnitudeStatus);
    if (b.logMagnitude === null) assert.equal(a!.logMagnitude, null);
    else { assert.notEqual(a!.logMagnitude, null); near(a!.logMagnitude!, Number(b.logMagnitude), true); }
    assert.equal(a!.logNegativeLogMagnitudeStatus, b.logNegativeLogMagnitudeStatus);
    if (b.logNegativeLogMagnitude === null) assert.equal(a!.logNegativeLogMagnitude, null);
    else { assert.notEqual(a!.logNegativeLogMagnitude, null); near(a!.logNegativeLogMagnitude!, Number(b.logNegativeLogMagnitude), true); }
  } else if (b.logMagnitude !== null) near(a!.logMagnitude ?? 0, Number(b.logMagnitude), true);
}
for (const [i, reference] of refs.cases.entries()) test(`independent grand occupancy sums/log bounds ${i + 1}`, () => {
  const r = evaluate(reference.parameters);
  for (const [key, value] of Object.entries(reference.expected))
    observation((r as unknown as Record<string, SpectrumQuantity | null>)[key], value, key === 'oneProbability' || key === 'zeroProbability');
  assert.equal(r.logGrandPartitionStatus, reference.logGrandPartitionStatus);
  assert.equal(r.susceptibilityStatus, reference.susceptibilityStatus);
  r.distribution.forEach((row, k) => {
    const expected = reference.distribution[k]; assert.equal(row.multiplicity, expected.multiplicity);
    observation(row.probability, expected.probability, true);
    observation(row.conditionalPatternProbability, expected.conditionalPatternProbability, true);
  });
});
test('balanced occupancy has a binomial particle law and uniform conditionals', () => {
  const r = evaluate({ sites: 2, energyGap: kB, chemicalPotential: kB, temperature: 1 });
  r.distribution.forEach((row, k) => near(row.probability.value!, [1, 2, 1][k] / 4));
  near(r.distribution[1].conditionalPatternProbability!.value!, .5);
  near(r.meanParticles.value!, 1); near(r.numberVariance.value!, .5);
  near(r.entropyInKB.value!, 2 * Math.LN2);
});
test('mu zero recovers the canonical preparation while energy/number covariance retains units', () => {
  for (const T of [1, -1]) {
    const p = { sites: 8, energyGap: kB, temperature: T }, a = canonicalBinaryModel(p), b = evaluate({ ...p, chemicalPotential: 0 });
    near(a.oneProbability.value!, b.oneProbability.value!);
    near(a.meanExcitations.value!, b.meanParticles.value!);
    near(a.energyVariance.value!, b.energyVariance.value!);
    near(a.entropy.value!, b.entropy.value!);
    near(b.energyNumberCovariance.value!, p.energyGap * b.numberVariance.value!);
    near(b.numberSusceptibility!.value!, b.numberVariance.value! / (kB * T));
  }
});
test('chemical response and grand-potential derivative agree with independent finite differences', () => {
  for (const T of [.5, 1, -1]) {
    const p = { sites: 4, energyGap: kB, chemicalPotential: kB * .25, temperature: T }, h = kB * 1e-4;
    const r = evaluate(p), lo = evaluate({ ...p, chemicalPotential: p.chemicalPotential - h }), hi = evaluate({ ...p, chemicalPotential: p.chemicalPotential + h });
    const susceptibility = (hi.meanParticles.value! - lo.meanParticles.value!) / (2 * h);
    assert.ok(Math.abs(susceptibility - r.numberSusceptibility!.value!) < 1e-7 * Math.abs(r.numberSusceptibility!.value!));
    const potentialSlope = (hi.grandPotential.value! - lo.grandPotential.value!) / (2 * h);
    assert.ok(Math.abs(potentialSlope + r.meanParticles.value!) < 1e-7);
    near(r.grandPotential.value!, r.meanEnergy.value! - T * r.entropy.value! - p.chemicalPotential * r.meanParticles.value!);
  }
});
test('common energy/chemical shift preserves weights and exact large cancellation preserves balance', () => {
  const p = { sites: 4, energyGap: 2, chemicalPotential: 1, temperature: 1 / kB }, a = evaluate(p), b = evaluate({ ...p, energyGap: 10, chemicalPotential: 9 });
  near(a.oneProbability.value!, b.oneProbability.value!); near(a.entropy.value!, b.entropy.value!);
  near(a.logGrandPartition!.value!, b.logGrandPartition!.value!);
  near(b.meanEnergy.value!, a.meanEnergy.value! + 8 * a.meanParticles.value!);
  const extreme = evaluate({ sites: 4, energyGap: Number.MAX_VALUE, chemicalPotential: Number.MAX_VALUE, temperature: Number.MIN_VALUE });
  assert.equal(extreme.scaledChemicalBias!.value, 0); assert.equal(extreme.scaledChemicalBias!.sign, 0);
  near(extreme.oneProbability.value!, .5); near(extreme.meanParticles.value!, 2);
  assert.equal(extreme.grandPotential.value, null); assert.equal(extreme.grandPotential.sign, -1);
});
test('zero-temperature empty, full and threshold preparations have different supports and responses', () => {
  const p = { sites: 4, energyGap: 1, chemicalPotential: 1, temperature: 0 };
  const threshold = evaluate(p), full = evaluate({ ...p, chemicalPotential: 2 }), empty = evaluate({ ...p, chemicalPotential: 0 });
  near(threshold.numberVariance.value!, 1); near(threshold.entropyInKB.value!, 4 * Math.LN2);
  assert.equal(threshold.numberSusceptibility, null);
  assert.equal(threshold.susceptibilityStatus, 'undefined-at-zero-temperature-threshold');
  near(full.meanParticles.value!, 4); assert.equal(full.numberVariance.sign, 0);
  assert.equal(full.logGrandPartition, null); assert.equal(full.logGrandPartitionStatus, 'infinite-zero-temperature-limit');
  near(full.grandPotential.value!, -4);
  assert.equal(full.distribution[0].conditionalPatternProbability, null);
  near(full.distribution[4].conditionalPatternProbability!.value!, 1);
  assert.equal(empty.meanParticles.sign, 0); assert.equal(empty.logGrandPartition!.sign, 0);
  assert.equal(empty.relativeNumberFluctuation, null);
});
test('positive susceptibility survives probability underflow; half-MIN covariance stays positive outside linear range', () => {
  const rare = evaluate({ sites: 2, energyGap: 1e-297, chemicalPotential: 0, temperature: 1e-297 / (kB * 800) });
  assert.equal(rare.oneProbability.value, null); assert.equal(rare.oneProbability.sign, 1);
  assert.ok(rare.numberSusceptibility!.value! > 0);
  const tie = evaluate({ sites: 2, energyGap: Number.MIN_VALUE, chemicalPotential: Number.MIN_VALUE, temperature: 1 });
  assert.equal(tie.energyNumberCovariance.value, null); assert.equal(tie.energyNumberCovariance.sign, 1);
});
test('full supported sizes normalize and all physical bounds serialize without nonfinite numbers', () => {
  const r = evaluate({ sites: 4096, energyGap: kB, chemicalPotential: kB, temperature: 1 });
  near(r.distribution.reduce((s, row) => s + (row.probability.value ?? 0), 0), 1);
  near(r.meanParticles.value!, 2048); near(r.numberVariance.value!, 1024);
  assert.equal(r.distribution.reduce((s, row) => s + BigInt(row.multiplicity), BigInt(0)).toString(), r.totalMicrostates);
  const walk = (value: unknown): void => {
    if (typeof value === 'number') assert.ok(Number.isFinite(value));
    else if (value && typeof value === 'object') Object.values(value).forEach(walk);
  };
  for (const epsilon of [0, Number.MIN_VALUE, Number.MAX_VALUE]) for (const mu of [-Number.MAX_VALUE, 0, Number.MAX_VALUE]) for (const T of [0, Number.MIN_VALUE, -Number.MIN_VALUE, Number.MAX_VALUE]) {
    const q = evaluate({ sites: 4, energyGap: epsilon, chemicalPotential: mu, temperature: T }); walk(q); assert.doesNotThrow(() => JSON.stringify(q));
  }
});
test('invalid sizes and nonfinite physical inputs are rejected', () => {
  const p: GrandOccupancyParameters = { sites: 4, energyGap: kB, chemicalPotential: 0, temperature: 1 };
  for (const key of Object.keys(p) as (keyof GrandOccupancyParameters)[]) for (const value of [NaN, Infinity, -Infinity])
    assert.throws(() => evaluate({ ...p, [key]: value }), RangeError);
  for (const value of [-1, 1.5, 4097]) assert.throws(() => evaluate({ ...p, sites: value }), RangeError);
  assert.throws(() => evaluate({ ...p, energyGap: -1 }), RangeError);
});
