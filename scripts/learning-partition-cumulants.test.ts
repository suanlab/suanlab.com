import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { partitionCumulantModel as model, type PartitionObservation, type PartitionRational } from '../src/lib/learning-partition-cumulants';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
interface Reference { kind: string; value?: string; sign: number; logMagnitude?: string | null }
interface Case { id: string; parameters: { energyLevels: number[]; energyOffset: number; temperature: number }; reference: Record<string, Reference | Reference[]> }
const fixture = JSON.parse(readFileSync('scripts/fixtures/partition-cumulant-oracle.json', 'utf8')) as { cases: Case[] };
const close = (a: number | null, b: number, log = false) => { assert.notEqual(a, null); assert.ok(Number.isFinite(a)); assert.ok(Math.abs(a! - b) <= Math.max(log ? 3e-9 : 2 * Number.MIN_VALUE, 8e-12 * Math.abs(b)), `${a} versus ${b}`); };
function check(q: PartitionObservation, reference: Reference) {
  assert.equal(q.sign, reference.sign);
  if (reference.kind === 'infinite') { assert.equal(q.resolution, 'mathematical-infinity'); assert.equal(q.value, null); return; }
  if (reference.kind === 'zero') { assert.equal(q.value, 0); assert.equal(q.resolution, 'structural-zero'); return; }
  const value = Number(reference.value);
  if (!Number.isFinite(value) || value === 0) assert.equal(q.value, null); else close(q.value, value);
  assert.notEqual(q.resolution, 'unresolved-at-precision-ceiling');
  assert.ok(reference.logMagnitude !== null && reference.logMagnitude !== undefined); close(q.logMagnitude, Number(reference.logMagnitude), true);
}
function fraction(q: PartitionRational) {
  let n = BigInt(q.numerator.coefficient), d = BigInt(q.denominator.coefficient);
  const power = q.numerator.exponent - q.denominator.exponent;
  if (power >= 0) n <<= BigInt(power); else d <<= BigInt(-power);
  assert.ok(d > BigInt(0)); return { n, d };
}
function assertEnclosure(q: PartitionObservation) {
  if (!q.bounds) return;
  const a = fraction(q.bounds.lower), b = fraction(q.bounds.upper), width = b.n * a.d - a.n * b.d;
  assert.ok(width >= BigInt(0));
  if (q.resolution === 'interval-resolved') {
    assert.ok(a.n > BigInt(0) || b.n < BigInt(0), 'A resolved nonzero sign must exclude zero');
    const near = a.n > BigInt(0) ? a.n * b.d : -b.n * a.d;
    assert.ok((width << BigInt(48)) <= near, 'Relative enclosure width must meet the stated 48-bit contract');
  }
}
for (const c of fixture.cases) test(`${c.id}: actual-input independent Decimal reference and directed enclosures`, () => {
  const result = model(c.parameters); assert.equal(result.fullyResolved, true);
  assert.equal(result.states.length, c.parameters.energyLevels.length);
  for (const [key, reference] of Object.entries(c.reference)) {
    if (key === 'probabilities') continue;
    const q = (result as unknown as Record<string, PartitionObservation>)[key]; check(q, reference as Reference); assertEnclosure(q);
  }
  for (const [index, reference] of (c.reference.probabilities as Reference[]).entries()) {
    const state = result.states[index]; assertEnclosure(state.probability); assertEnclosure(state.logProbability);
    if (reference.kind === 'zero') { check(state.probability, reference); assert.equal(state.logProbability.resolution, 'mathematical-infinity'); }
    else {
      const p = Number(reference.value); assert.equal(state.probability.sign, 1);
      if (!Number.isFinite(p) || p === 0) assert.equal(state.probability.value, null); else close(state.probability.value, p);
      const lp = Number(reference.logMagnitude);
      if (lp === 0 && reference.logMagnitude && Number(reference.value) === 1 && /^-/.test(reference.logMagnitude)) {
        assert.equal(state.logProbability.value, null); assert.equal(state.logProbability.sign, -1); assert.equal(state.probability.logMagnitude, null);
      } else { close(state.logProbability.value, lp, true); close(state.probability.logMagnitude, lp, true); }
    }
  }
  const logZ = c.reference.logPartition as Reference;
  if (logZ.kind === 'infinite') assert.equal(result.partitionFunction.resolution, logZ.sign > 0 ? 'mathematical-infinity' : 'structural-zero');
  else {
    const value = Math.exp(Number(logZ.value ?? 0)); assert.equal(result.partitionFunction.sign, 1);
    if (!Number.isFinite(value) || value === 0) assert.equal(result.partitionFunction.value, null); else close(result.partitionFunction.value, value);
  }
  assert.equal(result.logZBetaDerivatives.first.value, result.meanEnergy.value === null ? null : result.meanEnergy.value === 0 ? 0 : -result.meanEnergy.value);
  assert.equal(result.logZBetaDerivatives.second.value, result.energyVariance.value);
  assert.equal(result.logZBetaDerivatives.third.value, result.thirdEnergyCumulant.value === null ? null : result.thirdEnergyCumulant.value === 0 ? 0 : -result.thirdEnergyCumulant.value);
  const finiteNumbers = (v: unknown): void => { if (typeof v === 'number') assert.ok(Number.isFinite(v)); else if (Array.isArray(v)) v.forEach(finiteNumbers); else if (v && typeof v === 'object') Object.values(v).forEach(finiteNumbers); };
  finiteNumbers(result);
  assert.doesNotThrow(() => JSON.parse(JSON.stringify(result)));
  assert.doesNotMatch(JSON.stringify(result), /NaN|Infinity/);
});
test('separate offset retains physical gaps already lost in displayed energies or flat binary64 inputs', () => {
  const a = model({ energyLevels: [0, 1, 2], energyOffset: 2 ** 60, temperature: 1 / kB });
  const b = model({ energyLevels: [2 ** 60, 2 ** 60 + 1, 2 ** 60 + 2], energyOffset: 0, temperature: 1 / kB });
  assert.equal(new Set(a.states.map(s => s.energy.value)).size, 1);
  assert.ok(a.energyVariance.value! > 0); assert.equal(b.energyVariance.value, 0);
  assert.equal(b.energyVariance.resolution, 'structural-zero');
});
test('higher precision preserves a representable third cumulant despite nearly uniform displayed probabilities', () => {
  const a = model({ energyLevels: [-1e70, 0, 1e70], energyOffset: 0, temperature: Number.MAX_VALUE });
  assert.ok(a.precisionBits >= 1024); assert.equal(new Set(a.states.map(s => s.probability.value)).size, 1);
  assert.ok(a.thirdEnergyCumulant.value! > 1e-6); assert.ok(a.meanEnergy.value! < 0);
});
test('structured cancellation can require 8192 bits while remaining nonzero below the linear display range', () => {
  const a = model({ energyLevels: [-Number.MIN_VALUE, Number.MIN_VALUE, 0, 0, 0, 0], energyOffset: 0, temperature: Number.MAX_VALUE });
  assert.equal(a.precisionBits, 8192); assert.equal(a.fullyResolved, true);
  assert.equal(a.thirdEnergyCumulant.value, null); assert.equal(a.thirdEnergyCumulant.sign, 1); assert.equal(a.thirdEnergyCumulant.resolution, 'interval-resolved');
  assertEnclosure(a.thirdEnergyCumulant);
});
test('the precision ceiling preserves a certified sign while withholding an insufficiently precise value', () => {
  const a = model({ energyLevels: [0, Number.MIN_VALUE, Number.MAX_VALUE], energyOffset: 0, temperature: Number.MAX_VALUE });
  assert.equal(a.fullyResolved, false); assert.equal(a.precisionBits, 8192);
  assert.equal(a.thirdEnergyCumulant.resolution, 'unresolved-at-precision-ceiling'); assert.equal(a.thirdEnergyCumulant.value, null); assert.equal(a.thirdEnergyCumulant.sign, 1);
  const bounds = a.thirdEnergyCumulant.bounds!, lo = fraction(bounds.lower), hi = fraction(bounds.upper);
  assert.ok(lo.n > BigInt(0)); assert.ok(((hi.n * lo.d - lo.n * hi.d) << BigInt(48)) > lo.n * hi.d);
  assert.equal(a.energyVariance.sign, 1); assert.ok(a.states.every(s => s.probability.sign === 1));
});
test('common gauge shifts preserve probabilities, entropy, variance, capacity and higher cumulants', () => {
  const p = { energyLevels: [0, kB, 2 * kB, 3 * kB], energyOffset: 0, temperature: 1 };
  const a = model(p), b = model({ ...p, energyOffset: 1 });
  for (const key of ['energyVariance', 'thirdEnergyCumulant', 'heatCapacity', 'heatCapacityInKB', 'entropy', 'entropyInKB'] as const) assert.equal(a[key].value, b[key].value);
  assert.deepEqual(a.states.map(s => s.probability.value), b.states.map(s => s.probability.value));
  close(b.meanEnergy.value! - a.meanEnergy.value!, 1); close(b.freeEnergy.value! - a.freeEnergy.value!, 1);
});
test('negative temperature reverses symmetric odd cumulants and retains positive variance and capacity', () => {
  const p = { energyLevels: [-kB, 0, kB], energyOffset: 0, temperature: 1 };
  const a = model(p), b = model({ ...p, temperature: -1 });
  close(a.meanEnergy.value, -b.meanEnergy.value!); close(a.thirdEnergyCumulant.value, -b.thirdEnergyCumulant.value!);
  close(a.energyVariance.value, b.energyVariance.value!); close(a.heatCapacity.value, b.heatCapacity.value!); close(a.entropy.value, b.entropy.value!);
  assert.equal(b.inverseTemperature.sign, -1); assertEnclosure(b.inverseTemperature);
});
for (const direction of [1, -1]) test(`analytic cold tails at temperature sign ${direction} stay distinct from zero-temperature limits`, () => {
  const a = model({ energyLevels: [0, Number.MAX_VALUE], energyOffset: 0, temperature: direction * Number.MIN_VALUE });
  assert.equal(a.coldTailApproximation, true); assert.equal(a.fullyResolved, true);
  assert.equal(a.energyVariance.value, null); assert.equal(a.energyVariance.sign, 1);
  assert.equal(a.thirdEnergyCumulant.sign, direction); assert.equal(a.thirdEnergyCumulant.resolution, 'log-tail-approximation');
  assert.ok(a.states.every(s => s.probability.sign === 1));
  const b = model({ energyLevels: [0, Number.MAX_VALUE], energyOffset: 0, temperature: 0 });
  assert.equal(b.states[1].probability.value, 0); assert.equal(b.states[1].probability.sign, 0);
  assert.equal(b.energyVariance.value, 0); assert.equal(b.thirdEnergyCumulant.value, 0);
});
test('tiny distinctions between enormous log probabilities survive in the directed bounds', () => {
  const a = model({ energyLevels: [-Number.MAX_VALUE, 0, Number.MIN_VALUE], energyOffset: Number.MAX_VALUE, temperature: Number.MAX_VALUE });
  assert.equal(a.meanEnergy.sign, 1); assert.equal(a.meanEnergy.value, null);
  const p = fraction(a.states[1].logProbability.bounds!.lower), q = fraction(a.states[2].logProbability.bounds!.upper);
  assert.ok(p.n * q.d > q.n * p.d, 'State 1 has a strictly larger log probability even if both displays coincide');
  assert.equal(a.states[1].logProbability.value, a.states[2].logProbability.value);
});
test('all cold ground degeneracy remains entropy while suppressed states remain positive', () => {
  const a = model({ energyLevels: [0, 0, Number.MAX_VALUE], energyOffset: 0, temperature: Number.MIN_VALUE });
  close(a.entropyInKB.value, Math.log(2)); assert.equal(a.freeEnergy.sign, -1); assert.equal(a.freeEnergy.value, null);
  assert.equal(a.states[2].probability.sign, 1); assert.equal(a.states[2].probability.logResolution, 'outside-number-range');
  assert.ok(a.states[2].logProbability.logMagnitude! > 1000);
});
test('caller mutation, frozen inputs and successive results do not share mutable preparations', () => {
  const levels = [0, kB, 2 * kB], input = { energyLevels: levels, energyOffset: 0, temperature: 1 };
  const a = model(input); levels[1] = 3;
  assert.equal(a.parameters.energyLevels[1], kB);
  const b = model(Object.freeze({ ...input, energyLevels: Object.freeze([0, kB, 2 * kB]) }));
  a.parameters.energyLevels[1] = 7; assert.equal(b.parameters.energyLevels[1], kB);
});
test('malformed, sparse, empty, oversized and nonfinite preparations are rejected', () => {
  const base = { energyLevels: [0, 1], energyOffset: 0, temperature: 1 };
  for (const energyLevels of [[], new Array(2), Array(33).fill(0), [0, NaN], [Infinity], ['1'], {}]) assert.throws(() => model({ ...base, energyLevels } as never), RangeError);
  for (const key of ['energyOffset', 'temperature'] as const) for (const value of [NaN, Infinity, -Infinity, null, '1', undefined]) assert.throws(() => model({ ...base, [key]: value } as never), RangeError);
});
const coldFixture = JSON.parse(readFileSync('scripts/fixtures/partition-cumulant-cold-reference.json', 'utf8')) as { cases: { id: string; parameters: Case['parameters']; reference: { logs: Record<string, { sign: number; logMagnitude: string }>; logProbabilities: { sign: number; value: string | null; logMagnitude: string }[] } }[] };
for (const c of coldFixture.cases) test(`${c.id}: independent analytic leading logs and bounded cold omissions`, () => {
  const a = model(c.parameters); assert.equal(a.fullyResolved, true); assert.equal(a.coldTailApproximation, true);
  for (const [key, ref] of Object.entries(c.reference.logs)) {
    const q = (a as unknown as Record<string, PartitionObservation>)[key];
    assert.equal(q.sign, ref.sign); assert.equal(q.resolution, 'log-tail-approximation'); assert.equal(q.value, null);
    const log = Number(ref.logMagnitude);
    if (Number.isFinite(log)) close(q.logMagnitude, log, true); else { assert.equal(q.logMagnitude, null); assert.equal(q.logResolution, 'outside-number-range'); }
  }
  for (const [i, ref] of c.reference.logProbabilities.entries()) {
    const q = a.states[i].logProbability; assert.equal(q.sign, ref.sign);
    if (ref.value === null) { assert.equal(q.value, null); assert.equal(q.resolution, 'log-tail-approximation'); }
    else { const value = Number(ref.value); if (Number.isFinite(value)) close(q.value, value, true); else assert.equal(q.value, null); }
    const log = Number(ref.logMagnitude);
    if (Number.isFinite(log)) close(q.logMagnitude, log, true); else assert.equal(q.logMagnitude, null);
    assert.equal(a.states[i].probability.sign, 1); assert.notEqual(a.states[i].probability.value, 0);
  }
});
test('excited-state degeneracy can make the third cumulant negative at positive temperature', () => {
  const gap = kB * Math.log(2), a = model({ energyLevels: [0, gap, gap, gap], energyOffset: 0, temperature: 1 });
  assert.equal(a.thirdEnergyCumulant.sign, -1); assert.equal(a.thirdEnergyCumulant.resolution, 'interval-resolved');
  // Ideal t=ln2 gives excited mass3/5 and kappa3(K)=-6/125.
  // The supplied binary64 gap differs slightly, so this is an approximate hand check.
  close(a.thirdEnergyCumulant.value, -6 / 125 * gap ** 3); close(a.energyVariance.value, 6 / 25 * gap ** 2);
});
