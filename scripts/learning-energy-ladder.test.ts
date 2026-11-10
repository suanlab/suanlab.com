import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { energyLadderModel as model, type EnergyLadderParameters } from '../src/lib/learning-energy-ladder';
import type { PartitionObservation, PartitionRational } from '../src/lib/learning-partition-exact-core';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
type Fraction = { n: bigint; d: bigint };
const O = BigInt(1), Z = BigInt(0);
function rational(v: PartitionRational): Fraction {
  let n = BigInt(v.numerator.coefficient), d = BigInt(v.denominator.coefficient);
  const e = v.numerator.exponent - v.denominator.exponent;
  if (e >= 0) n <<= BigInt(e); else d <<= BigInt(-e);
  return { n, d };
}
function decimal(text: string): Fraction {
  const [mantissa, power = '0'] = text.toLowerCase().split('e'), dot = mantissa.indexOf('.');
  const exponent = Number(power) - (dot < 0 ? 0 : mantissa.length - dot - 1), n = BigInt(mantissa.replace('.', ''));
  return exponent >= 0 ? { n: n * BigInt(10) ** BigInt(exponent), d: O } : { n, d: BigInt(10) ** BigInt(-exponent) };
}
const sub = (a: Fraction, c: Fraction): Fraction => ({ n: a.n * c.d - c.n * a.d, d: a.d * c.d });
const compare = (a: Fraction, c: Fraction) => a.n * c.d - c.n * a.d;
function check(q: PartitionObservation, reference: string, digits: number) {
  if (reference.endsWith('infinity')) { assert.equal(q.resolution, 'mathematical-infinity'); assert.equal(q.sign, reference[0] === '+' ? 1 : -1); return; }
  const truth = decimal(reference), value = Number(reference);
  assert.notEqual(q.resolution, 'unresolved-at-precision-ceiling');
  assert.equal(q.sign, truth.n === Z ? 0 : truth.n > Z ? 1 : -1);
  if (q.value === null && Math.abs(value) === Number.MIN_VALUE) {
    // A finite Decimal reference can lie just above an exact ties-to-even
    // midpoint. Confirm the actual enclosure lies in the zero rounding cell.
    assert.ok(q.bounds);
    const endpoint = rational(q.sign === 1 ? q.bounds.upper : q.bounds.lower);
    assert.ok(compare({ n: endpoint.n < Z ? -endpoint.n : endpoint.n, d: endpoint.d }, { n: O, d: O << BigInt(1075) }) <= Z);
  } else if (value === 0 && truth.n !== Z || !Number.isFinite(value)) assert.equal(q.value, null);
  else assert.ok(q.value !== null && Math.abs(q.value - value) <= Math.max(2 * Number.MIN_VALUE, Math.abs(value) * 8e-12), `${q.value} versus ${reference}`);
  // Cold logarithms use explicit asymptotics; physical omission is bounded,
  // but the Math.log/exp projection is deliberately not an interval claim.
  if (q.bounds) {
    const lower = rational(q.bounds.lower), upper = rational(q.bounds.upper);
    const slack = truth.n === Z ? decimal(`1e-${digits - 100}`) : { n: truth.n < Z ? -truth.n : truth.n, d: truth.d * BigInt(10) ** BigInt(digits - 100) };
    assert.ok(compare(sub(truth, lower), { n: -slack.n, d: slack.d }) >= Z, 'Independent reference below lower endpoint');
    assert.ok(compare(sub(upper, truth), { n: -slack.n, d: slack.d }) >= Z, 'Independent reference above upper endpoint');
    assert.ok(compare(lower, upper) <= Z);
    if (q.resolution === 'interval-resolved' && q.sign !== 0) {
      const width = sub(upper, lower), near = q.sign === 1 ? lower : { n: -upper.n, d: upper.d };
      assert.ok(near.n > Z);
      assert.ok(compare({ n: width.n << BigInt(48), d: width.d }, near) <= Z);
    }
  }
}
type Scalars = Record<string, string>;
interface Case { id: string; parameters: EnergyLadderParameters; confirmationPrecisionDigits: number; reference: { finite: Scalars; infinite: Scalars | null; comparison: Scalars | null; samples: { n: number; finiteProbability: string; infiniteProbability: string | null }[] } }
const fixture = JSON.parse(readFileSync('scripts/fixtures/energy-ladder-oracle.json', 'utf8')) as { cases: Case[] };
for (const row of fixture.cases) test(`${row.id}: independent direct finite enumeration and hyperbolic infinite reference`, () => {
  const r = model(row.parameters); assert.equal(r.fullyResolved, true);
  for (const section of ['finite', 'infinite', 'comparison'] as const) {
    const observed = section === 'infinite' ? r.infinite.quantities : r[section], truth = row.reference[section];
    if (truth === null) { assert.equal(observed, null); continue; }
    assert.ok(observed);
    for (const [key, value] of Object.entries(truth)) check((observed as unknown as Record<string, PartitionObservation>)[key], value, row.confirmationPrecisionDigits);
  }
  assert.equal(r.samples.length, row.reference.samples.length);
  row.reference.samples.forEach((s, i) => {
    assert.equal(r.samples[i].n, s.n);
    check(r.samples[i].finiteProbability, s.finiteProbability, row.confirmationPrecisionDigits);
    if (s.infiniteProbability === null) assert.equal(r.samples[i].infiniteProbability, null);
    else check(r.samples[i].infiniteProbability!, s.infiniteProbability, row.confirmationPrecisionDigits);
  });
});
test('ideal log-two example separates normalization from truncation error', () => {
  const r = model({ gapEnergy: kB * Math.log(2), temperature: 1, cutoff: 2 });
  assert.ok(Math.abs(r.infinite.quantities!.meanIndex.value! - 1) < 1e-14);
  assert.ok(Math.abs(r.infinite.quantities!.indexVariance.value! - 2) < 1e-14);
  assert.ok(Math.abs(r.finite.meanIndex.value! - 4 / 7) < 1e-14);
  assert.ok(Math.abs(r.finite.indexVariance.value! - 26 / 49) < 1e-14);
  assert.ok(Math.abs(r.comparison!.missingMass.value! - 1 / 8) < 1e-14);
  assert.ok(Math.abs(r.comparison!.meanIndexBias.value! - 3 / 7) < 1e-14);
  assert.ok(Math.abs(r.comparison!.indexVarianceBias.value! - 72 / 49) < 1e-14);
});
test('q rounding to one preserves a normalizable infinite law and finite energy/heat response', () => {
  const r = model({ gapEnergy: Number.MIN_VALUE, temperature: Number.MAX_VALUE, cutoff: 2 });
  assert.equal(r.q.value, 1); assert.equal(r.infinite.status, 'normalizable');
  assert.equal(r.infinite.quantities!.meanIndex.value, null);
  assert.equal(r.infinite.quantities!.meanIndex.sign, 1);
  assert.ok(Number.isFinite(r.infinite.quantities!.meanEnergy.value));
  assert.equal(r.infinite.quantities!.heatCapacityInKB.value, 1);
  assert.equal(r.comparison!.retainedMass.value, null); assert.equal(r.comparison!.retainedMass.sign, 1);
});
test('zero gap and negative temperature do not create a fictitious infinite distribution', () => {
  for (const parameters of [{ gapEnergy: 0, temperature: 1, cutoff: 3 }, { gapEnergy: kB, temperature: -1, cutoff: 3 }]) {
    const r = model(parameters);
    assert.equal(r.infinite.status, 'not-normalizable'); assert.equal(r.infinite.quantities, null); assert.equal(r.comparison, null);
    assert.ok(r.samples.every(s => s.infiniteProbability === null));
    assert.ok(Math.abs(r.samples.reduce((a, s) => a + s.finiteProbability.value!, 0) - 1) < 1e-14);
  }
});
test('negative temperature uses decreasing deficit weights but physical q exceeds one', () => {
  const p = { gapEnergy: kB * .7, temperature: 1, cutoff: 7 }, positive = model(p), negative = model({ ...p, temperature: -1 });
  assert.ok(negative.q.value! > 1);
  assert.equal(negative.x!.sign, -1);
  assert.ok(Math.abs(negative.x!.value! + .7) < 1e-14);
  assert.ok(negative.anchoredDecayRatio!.value! < 1);
  assert.ok(Math.abs(positive.finite.meanIndex.value! + negative.finite.meanIndex.value! - 7) < 1e-14);
  assert.ok(Math.abs(positive.finite.indexVariance.value! - negative.finite.indexVariance.value!) < 1e-14);
});
test('finite cutoff outside support has true zero while tiny infinite tail remains positive', () => {
  const r = model({ gapEnergy: kB * 800, temperature: 1, cutoff: 0 }), tail = r.samples.find(s => s.n === 1)!;
  assert.equal(tail.finiteProbability.resolution, 'structural-zero');
  assert.equal(tail.infiniteProbability!.value, null); assert.equal(tail.infiniteProbability!.sign, 1);
  assert.notEqual(tail.infiniteProbability!.resolution, 'structural-zero');
  assert.equal(r.comparison!.reverseKL.resolution, 'mathematical-infinity');
});
test('subnormal zero-point energy and positive finite partition differ from mathematical limits', () => {
  const limit = model({ gapEnergy: Number.MIN_VALUE, temperature: 0, cutoff: 2 });
  assert.equal(limit.finite.meanEnergy.value, null); assert.equal(limit.finite.meanEnergy.sign, 1);
  assert.equal(limit.finite.partitionFunction.value, 0);
  assert.equal(limit.comparison!.reverseKL.value, 0);
  const finite = model({ gapEnergy: Number.MAX_VALUE, temperature: Number.MIN_VALUE, cutoff: 2 });
  assert.equal(finite.fullyResolved, true);
  assert.equal(finite.finite.partitionFunction.value, null); assert.equal(finite.finite.partitionFunction.sign, 1);
  assert.notEqual(finite.finite.logPartition.resolution, 'mathematical-infinity');
  assert.equal(finite.infinite.quantities!.heatCapacityInKB.value, null);
  assert.equal(finite.infinite.quantities!.heatCapacityInKB.sign, 1);
});
test('sampled large ladders retain original probabilities without a sample renormalization', () => {
  const r = model({ gapEnergy: kB * .001, temperature: 1, cutoff: 4096 });
  assert.equal(r.finiteStateCount, 4097); assert.equal(r.sampledAllFiniteStates, false);
  assert.ok(r.samples.length <= 64);
  assert.ok(r.samples.some(s => s.n === 0)); assert.ok(r.samples.some(s => s.n === 4096));
  assert.ok(r.samples.filter(s => s.n <= 4096).reduce((a, s) => a + s.finiteProbability.value!, 0) < .1);
});
test('cold entropy includes one plus x, and asymptotic omissions have a rational bound', () => {
  const r = model({ gapEnergy: kB * 5000, temperature: 1, cutoff: 2 });
  const expectedLog = -5000 + Math.log1p(5000);
  assert.ok(Math.abs(r.infinite.quantities!.entropyInKB.logMagnitude! - expectedLog) < 1e-9);
  assert.equal(r.infinite.quantities!.entropyInKB.resolution, 'log-tail-approximation');
  // q<exp(-4096)<2^-4096; 16q bounds the omitted relative
  // moment/entropy corrections for L>=1, hence is below 2^-100.
  assert.ok((BigInt(16) << BigInt(100)) < (O << BigInt(4096)));
});
test('fixed-cutoff high temperature and cutoff removal have different heat-capacity limits', () => {
  const r = model({ gapEnergy: kB * 1e-12, temperature: 1, cutoff: 2 });
  assert.ok(r.finite.heatCapacityInKB.value! < 1e-20);
  assert.ok(Math.abs(r.infinite.quantities!.heatCapacityInKB.value! - 1) < 1e-14);
  assert.ok(Math.abs(r.finite.meanIndex.value! - 1) < 1e-10);
});
test('invalid or unsupported parameter values are rejected explicitly', () => {
  for (const patch of [{ gapEnergy: -1 }, { gapEnergy: Infinity }, { temperature: NaN }, { cutoff: -.1 }, { cutoff: 4097 }, { cutoff: 2.5 }])
    assert.throws(() => model({ gapEnergy: kB, temperature: 1, cutoff: 2, ...patch }), RangeError);
});
