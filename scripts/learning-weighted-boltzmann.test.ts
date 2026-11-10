import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { weightedBoltzmannModel as model, type WeightedBoltzmannParameters } from '../src/lib/learning-weighted-boltzmann';
import type { PartitionObservation, PartitionRational } from '../src/lib/learning-partition-exact-core';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
interface Fixture { id: string; parameters: WeightedBoltzmannParameters; confirmationPrecisionDigits: number; reference: Record<string, string | string[]> }
const cases = (JSON.parse(readFileSync('scripts/fixtures/weighted-boltzmann-oracle.json', 'utf8')) as { cases: Fixture[] }).cases;
const Z = BigInt(0), O = BigInt(1);
interface Q { n: bigint; d: bigint }
const add = (a: Q, b: Q): Q => ({ n: a.n * b.d + b.n * a.d, d: a.d * b.d });
const cmp = (a: Q, b: Q) => a.n * b.d - b.n * a.d;
function decimal(s: string): Q {
  const match = /^([+-]?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s); assert.ok(match, s);
  const n = BigInt(`${match[1] === '-' ? '-' : ''}${match[2]}${match[3] ?? ''}`), exponent = Number(match[4] ?? 0) - (match[3]?.length ?? 0);
  return exponent >= 0 ? { n: n * BigInt(10) ** BigInt(exponent), d: O } : { n, d: BigInt(10) ** BigInt(-exponent) };
}
function decode(q: PartitionRational): Q {
  let n = BigInt(q.numerator.coefficient), d = BigInt(q.denominator.coefficient);
  const shift = q.numerator.exponent - q.denominator.exponent;
  if (shift >= 0) n <<= BigInt(shift); else d <<= BigInt(-shift);
  assert.ok(d > Z); return { n, d };
}
function check(q: PartitionObservation, reference: string, precision: number, label: string) {
  if (reference.endsWith('infinity')) {
    assert.equal(q.resolution, 'mathematical-infinity', label); assert.equal(q.sign, reference[0] === '-' ? -1 : 1); return;
  }
  assert.notEqual(q.resolution, 'unresolved-at-precision-ceiling', label);
  const truth = decimal(reference), truthSign = truth.n < Z ? -1 : truth.n > Z ? 1 : 0;
  assert.equal(q.sign, truthSign, label);
  if (!truth.n) { assert.equal(q.value, 0, label); assert.equal(q.resolution, 'structural-zero', label); }
  else {
    const expected = Number(reference);
    if (expected === 0 || !Number.isFinite(expected)) assert.equal(q.value, null, label);
    else {
      assert.notEqual(q.value, null, label);
      assert.ok(Math.abs(q.value! - expected) <= Math.max(2 * Number.MIN_VALUE, 1e-13 * Math.abs(expected)), `${label}: ${q.value} vs ${expected}`);
    }
  }
  assert.ok(q.bounds, label);
  const lower = decode(q.bounds.lower), upper = decode(q.bounds.upper);
  assert.ok(cmp(lower, upper) <= Z, label);
  // The oracle is independently recomputed at 400..1600+80 Decimal digits.
  // Allow 150 relative decimal digits plus a much smaller absolute rounding budget.
  const error = add({ n: truth.n < Z ? -truth.n : truth.n, d: truth.d * BigInt(10) ** BigInt(150) }, { n: O, d: BigInt(10) ** BigInt(precision - 110) });
  assert.ok(cmp(lower, add(truth, error)) <= Z && cmp(upper, add(truth, { n: -error.n, d: error.d })) >= Z, `${label}: independent reference must lie in directed bounds within its stated precision`);
  if (q.resolution === 'interval-resolved') {
    assert.ok(lower.n > Z || upper.n < Z, label);
    const width = upper.n * lower.d - lower.n * upper.d;
    const near = lower.n > Z ? lower.n * upper.d : -upper.n * lower.d;
    assert.ok((width << BigInt(48)) <= near, `${label}: relative width`);
  }
}
for (const c of cases) test(`${c.id}: actual binary64 independent reference, signs and directed bounds`, () => {
  const result = model(c.parameters); assert.equal(result.fullyResolved, true);
  for (const [key, value] of Object.entries(c.reference)) {
    if (key === 'probabilities' || key === 'logProbabilities') {
      for (const [i, reference] of (value as string[]).entries()) check(result.states[i][key === 'probabilities' ? 'probability' : 'logProbability'], reference, c.confirmationPrecisionDigits, `${c.id}.${key}[${i}]`);
    } else check((result as unknown as Record<string, PartitionObservation>)[key], value as string, c.confirmationPrecisionDigits, `${c.id}.${key}`);
  }
  const visit = (v: unknown): void => {
    if (typeof v === 'number') assert.ok(Number.isFinite(v));
    else if (Array.isArray(v)) v.forEach(visit);
    else if (v && typeof v === 'object') Object.values(v).forEach(visit);
  };
  visit(result); assert.doesNotThrow(() => JSON.stringify(result));
});
const normal: WeightedBoltzmannParameters = { energyLevels: [0, kB, 2 * kB], referenceWeights: [1, 2, 1], energyOffset: 0, temperature: 1 };
function close(a: number | null, b: number | null) { assert.notEqual(a, null); assert.notEqual(b, null); assert.ok(Math.abs(a! - b!) <= 1e-12 * Math.max(Math.abs(a!), Math.abs(b!), Number.MIN_VALUE)); }
test('integer multiplicities reproduce explicit microstate marginalization, not label counting', () => {
  const weighted = model(normal), expanded = model({ ...normal, energyLevels: [0, kB, kB, 2 * kB], referenceWeights: [1, 1, 1, 1] });
  close(weighted.states[1].probability.value, 2 * expanded.states[1].probability.value!);
  close(weighted.relativeEntropyInKB.value, expanded.labelEntropyNats.value);
  assert.notEqual(weighted.relativeEntropyInKB.value, weighted.labelEntropyNats.value);
});
test('reference rescaling preserves p and label entropy but shifts relative entropy and free energy', () => {
  const a = model(normal), b = model({ ...normal, referenceWeights: normal.referenceWeights.map(m => m / 4) });
  a.states.forEach((s, i) => close(s.probability.value, b.states[i].probability.value));
  close(a.labelEntropyNats.value, b.labelEntropyNats.value);
  close(a.relativeEntropyInKB.value! - b.relativeEntropyInKB.value!, Math.log(4));
  close(b.freeEnergy.value! - a.freeEnergy.value!, kB * Math.log(4));
  assert.ok(b.relativeEntropyInKB.value! < 0);
});
test('energy offset and common energy/temperature scaling preserve probabilities and fluctuations', () => {
  const a = model(normal), b = model({ ...normal, energyOffset: 2 ** 60 }), c = model({ ...normal, energyLevels: normal.energyLevels.map(e => e * 8), temperature: 8 });
  for (const r of [b, c]) a.states.forEach((s, i) => close(s.probability.value, r.states[i].probability.value));
  close(a.energyVariance.value, b.energyVariance.value); close(a.labelEntropyNats.value, b.labelEntropyNats.value);
  close(a.heatCapacityInKB.value, c.heatCapacityInKB.value); close(c.energyVariance.value, 64 * a.energyVariance.value!);
});
test('zero-temperature tied ground probabilities use reference weights, not a uniform label rule', () => {
  const r = model({ ...normal, energyLevels: [0, 0, kB], referenceWeights: [1, 3, .25], temperature: 0 });
  assert.deepEqual(r.states.map(s => s.probability.value), [.25, .75, 0]);
  close(r.relativeEntropyInKB.value, Math.log(4));
  assert.equal(r.heatCapacityInKB.value, 0); assert.equal(r.energyVariance.value, 0);
});
test('finite-temperature rare nonzero probabilities do not become forbidden zeros', () => {
  const r = model(cases.find(c => c.id === 'rare-mass-physical-energy')!.parameters);
  assert.equal(r.states[1].probability.value, null); assert.equal(r.states[1].probability.sign, 1);
  assert.equal(r.states[1].probability.resolution, 'interval-resolved');
  assert.ok(r.energyVariance.value! > 1e200); assert.ok(r.meanEnergy.value! > 0);
  assert.ok(r.precisionBits >= 2048);
});
test('precision ceiling preserves unresolved probabilities without claiming exact zero', () => {
  const r = model(cases.find(c => c.id === 'rare-mass-physical-energy')!.parameters, { maximumPrecision: 128 });
  assert.equal(r.fullyResolved, false); assert.equal(r.states[1].probability.value, null);
  assert.equal(r.states[1].probability.sign, 1); assert.equal(r.states[1].probability.resolution, 'unresolved-at-precision-ceiling');
});
test('positive-side zero-temperature log partition distinguishes an energy shift from finite-temperature overflow', () => {
  for (const offset of [-1, 0, 1]) {
    const r = model({ energyLevels: [0, 0], referenceWeights: [1, 3], energyOffset: offset, temperature: 0 });
    assert.equal(r.meanEnergy.value, offset); assert.equal(r.freeEnergy.value, offset);
    if (offset) { assert.equal(r.logPartition.resolution, 'mathematical-infinity'); assert.equal(r.logPartition.sign, -Math.sign(offset)); }
    else close(r.logPartition.value, Math.log(4));
  }
});
test('invalid, sparse, mismatched and empty-support inputs are rejected', () => {
  const bad: WeightedBoltzmannParameters[] = [
    { ...normal, energyLevels: [] }, { ...normal, energyLevels: [0] }, { ...normal, referenceWeights: [0, 0, 0] },
    { ...normal, referenceWeights: [-1, 1, 1] }, { ...normal, referenceWeights: [1, NaN, 1] },
    { ...normal, temperature: Infinity }, { ...normal, energyOffset: NaN },
    { ...normal, energyLevels: Array(3) }, { ...normal, referenceWeights: Array(3) },
    { ...normal, energyLevels: Array(33).fill(0), referenceWeights: Array(33).fill(1) },
  ];
  bad.forEach(p => assert.throws(() => model(p), RangeError));
  assert.throws(() => model(normal, { maximumPrecision: 64 }), RangeError);
  assert.throws(() => model(normal, { minimumPrecision: 512, maximumPrecision: 128 }), RangeError);
});
test('log probability survives projection of a dominant probability to one', () => {
  const r = model(cases.find(c => c.id === 'extreme-reference-ratio')!.parameters);
  assert.equal(r.states[0].probability.value, 1); assert.equal(r.states[0].logProbability.sign, -1);
  assert.ok(r.states[0].logProbability.value! < 0);
  assert.equal(r.states[0].probability.logMagnitude, r.states[0].logProbability.value);
});
test('a high-temperature unequal reference approaches its normalized reference instead of a uniform law', () => {
  const r = model({ ...normal, temperature: Number.MAX_VALUE });
  r.states.forEach((s, i) => close(s.probability.value, [.25, .5, .25][i]));
});
test('fixed-reference finite-temperature energy response agrees with the variance heat capacity', () => {
  const base = model(normal);
  for (const h of [1e-3, 1e-4]) {
    const above = model({ ...normal, temperature: 1 + h }), below = model({ ...normal, temperature: 1 - h });
    const derivative = (above.meanEnergy.value! - below.meanEnergy.value!) / (2 * h * kB);
    assert.ok(Math.abs(derivative - base.heatCapacityInKB.value!) < 2e-7);
  }
});
test('an extreme finite cold preparation preserves proved signs at the ceiling and never invents mathematical infinities', () => {
  const r = model({ energyLevels: [0, Number.MAX_VALUE], referenceWeights: [1, 1], energyOffset: 0, temperature: Number.MIN_VALUE }, { maximumPrecision: 128 });
  assert.equal(r.fullyResolved, false);
  for (const q of [r.energyVariance, r.heatCapacityInKB, r.labelEntropyNats, r.states[1].probability]) {
    assert.equal(q.value, null); assert.equal(q.sign, 1); assert.equal(q.resolution, 'unresolved-at-precision-ceiling');
  }
  assert.equal(r.states[0].logProbability.sign, -1);
  assert.equal(r.states[1].logProbability.sign, -1); assert.equal(r.states[1].logProbability.value, null);
  assert.notEqual(r.states[1].logProbability.resolution, 'mathematical-infinity');
});
