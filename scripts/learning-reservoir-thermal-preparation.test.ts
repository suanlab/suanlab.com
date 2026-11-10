import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { reservoirTwoStateThermodynamics as run, isReservoirThermalParameters as valid, type ReservoirThermalParameters as Parameters } from '../src/lib/learning-reservoir-two-state-thermodynamics';
import type { PartitionObservation as Q, PartitionRational } from '../src/lib/learning-partition-exact-core';
const fixtures = JSON.parse(readFileSync('scripts/fixtures/reservoir-two-state-thermal-reference.json', 'utf8')) as { cases: { label: string; input: Parameters; expected: unknown; channelDetailedBalance: boolean }[] };
type Fraction = { n: bigint; d: bigint };
const f = (n: bigint, d = BigInt(1)): Fraction => ({ n, d });
const le = (a: Fraction, b: Fraction) => a.n * b.d <= b.n * a.d;
function parse(value: string): Fraction {
  if (value.includes('/')) { const [n, d] = value.split('/'); return f(BigInt(n), BigInt(d)); }
  const m = /^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(value); assert(m, value);
  const n = BigInt(m[1] + m[2] + (m[3] || '')), e = Number(m[4] || 0) - (m[3] || '').length;
  return e >= 0 ? f(n * BigInt(10) ** BigInt(e)) : f(n, BigInt(10) ** BigInt(-e));
}
function endpoint(q: PartitionRational): Fraction {
  const n = BigInt(q.numerator.coefficient), d = BigInt(q.denominator.coefficient), e = q.numerator.exponent - q.denominator.exponent;
  return e >= 0 ? f(n << BigInt(e), d) : f(n, d << BigInt(-e));
}
let certificates = 0;
function compare(actual: unknown, expected: unknown, label: string) {
  if (typeof expected === 'string' && actual && typeof actual === 'object' && 'bounds' in actual) {
    const q = actual as Q; assert(q.bounds, label); const lo = endpoint(q.bounds.lower), hi = endpoint(q.bounds.upper), truth = parse(expected), magnitude = truth.n < BigInt(0) ? -truth.n : truth.n;
    const un = magnitude > truth.d ? magnitude : truth.d, ud = truth.d * BigInt(10) ** BigInt(160);
    // Only uncertainty of the stored180-digit external reference; model bounds are unchanged.
    const below = f(truth.n * ud - un * truth.d, truth.d * ud), above = f(truth.n * ud + un * truth.d, truth.d * ud);
    assert(le(lo, hi) && le(lo, above) && le(below, hi), label + ': ' + expected);
    assert(q.value === null || Number.isFinite(q.value)); if (q.value === 0) assert.equal(q.resolution, 'structural-zero');
    if (q.resolution === 'unresolved-at-precision-ceiling') assert.equal(q.value, null);
    certificates++; return;
  }
  if (Array.isArray(expected)) { assert(Array.isArray(actual), label); assert.equal(actual.length, expected.length, label); expected.forEach((v, i) => compare(actual[i], v, label + '.' + i)); return; }
  if (expected && typeof expected === 'object') { assert(actual && typeof actual === 'object', label); for (const [k, v] of Object.entries(expected)) compare((actual as Record<string, unknown>)[k], v, label + '.' + k); return; }
  assert.deepEqual(actual, expected, label);
}
const find = (label: string) => fixtures.cases.find(c => c.label === label)!.input;
for (const c of fixtures.cases) test(`independent reservoir-resolved reference: ${c.label}`, () => {
  const r = run(c.input); compare(r, c.expected, c.label);
  assert.equal(r.classification.channelDetailedBalance, c.channelDetailedBalance); assert(r.classification.aggregateDetailedBalance);
  assert.equal(r.physicalEntropyRate.sign, c.channelDetailedBalance ? 0 : 1); assert.equal(r.aggregateObservedEntropyRate.resolution, 'structural-zero');
  assert.equal(r.path.conditionalBathWordProbability.sign, 1); assert.notEqual(r.path.conditionalBathWordProbability.resolution, 'structural-zero');
  assert(r.classification.stationaryInitialization); assert.equal(r.classification.sampledPaths, false);
});
test('stationary two-state observations hide a positive bath-resolved heat current', () => {
  const r = run(find('worked/word2/jumps2'));
  assert.equal(r.stationaryLaw[0].value, 33 / 40); assert.equal(r.stationaryLaw[1].value, 7 / 40);
  assert.equal(r.channels[0].current.value, 3 / 40); assert.equal(r.channels[1].current.value, -3 / 40);
  assert(r.channels[0].bathEntropyRate.value! < 0); assert(r.channels[1].bathEntropyRate.value! > 0);
  assert(Math.abs(r.physicalEntropyRate.value! - 3 * Math.log(3) / 40) < 1e-14);
  assert(r.conditionalJeffreysDivergence.value! > 0); assert.deepEqual(r.reversedFilmAggregateGenerator, r.generator);
  assert.notDeepEqual(r.channels.map(c => c.upRate), r.channels.map(c => c.reversedFilmUpRate));
});
test('path reversal changes heat and entropy signs without swapping temperatures', () => {
  const p = find('worked/word2/jumps2'), a = run(p), b = run({ ...p, bathCode: 1 });
  assert(a.path.totalEntropy.value! > 0); assert(b.path.totalEntropy.value! < 0); assert.deepEqual(a.path.reversedBaths, b.path.baths);
  assert.deepEqual(a.reservoirTemperaturesK, b.reservoirTemperaturesK); assert.deepEqual(a.history, b.history);
  assert.equal(a.path.aggregateObservedEntropy.value, 0); assert.equal(b.path.aggregateObservedEntropy.value, 0);
});
test('bath relabeling preserves physics while temperature order is derived from beta', () => {
  const p = find('worked/word2/jumps2'), a = run(p), b = run({ ...p, betaA: p.betaB, betaB: p.betaA, clockANumerator: p.clockBNumerator, clockADenominator: p.clockBDenominator, clockBNumerator: p.clockANumerator, clockBDenominator: p.clockADenominator, bathCode: 1 });
  assert.deepEqual(a.physicalEntropyRate, b.physicalEntropyRate); assert.deepEqual(a.stationaryLaw, b.stationaryLaw); assert.deepEqual(a.path.totalEntropy, b.path.totalEntropy);
  assert.equal(a.classification.hotterChannel, 'A'); assert.equal(b.classification.hotterChannel, 'B');
});
test('exact equilibrium, zero-gap and no-jump controls differ from small production', () => {
  for (const label of ['equal temperatures/word2/jumps2', 'zero gap/word2/jumps2']) { const r = run(find(label)); assert.equal(r.physicalEntropyRate.resolution, 'structural-zero'); assert.equal(r.path.totalEntropy.resolution, 'structural-zero'); assert.equal(r.conditionalJeffreysDivergence.resolution, 'structural-zero'); }
  const noJumps = run(find('no-jump example')); assert.equal(noJumps.path.totalEntropy.value, 0); assert(noJumps.current.meanTotalEntropyProduction.value! > 0);
  const tiny = run(find('tiny physical heat below fixed quantum')); assert(tiny.current.meanHeatIntoSystemAJ.value! > 0 && tiny.current.meanHeatIntoSystemAJ.value! < 2 ** -256); assert.equal(tiny.current.meanHeatIntoSystemAJ.resolution, 'interval-resolved'); assert(tiny.current.meanTotalEntropyProduction.value! > 0 && tiny.current.meanTotalEntropyProduction.value! < 2 ** -256);
});
test('positive conditional bath words below binary64 retain exact bounds and magnitude logs', () => {
  const r = run(find('underflowed conditional bath word')), q = r.path.conditionalBathWordProbability;
  assert.equal(q.value, null); assert.equal(q.sign, 1); assert.equal(q.resolution, 'exact-rational'); assert(q.logMagnitude! < -745); assert(q.bounds && BigInt(q.bounds.lower.numerator.coefficient) > BigInt(0));
  assert.equal(r.path.totalEntropy.value, 0); assert(r.physicalEntropyRate.value! > 0);
});
test('reference temperature changes physical energies together with bath temperatures', () => {
  const a = run(find('worked/word2/jumps2')), b = run(find('double reference temperature'));
  assert.deepEqual(a.physicalEntropyRate, b.physicalEntropyRate); assert.deepEqual(a.path.totalEntropy, b.path.totalEntropy);
  assert(Math.abs(b.current.meanHeatIntoSystemAJ.value! / a.current.meanHeatIntoSystemAJ.value! - 2) < 1e-14);
  assert.equal(b.reservoirTemperaturesK[0].value, 600); assert.equal(b.reservoirTemperaturesK[1].value, 300);
});
test('record density changes neither the endpoint nor physical clocks and input is owned', () => {
  const p = find('worked/word2/jumps2'), saved = JSON.stringify(p), a = run(p), b = run({ ...p, sampleSteps: 64 });
  assert.deepEqual({ ...a.current, step: 64 }, b.current); assert.deepEqual(a.path, b.path); assert.equal(JSON.stringify(p), saved); assert(Object.isFrozen(a.parameters));
});
test('sixteen-field validation rejects unsupported bath words, rates and zero-time jumps', () => {
  const p = find('worked/word2/jumps2'); assert(valid(p));
  for (const v of [null, {}, { ...p, extra: 1 }, { ...p, betaA: 0 }, { ...p, betaB: 9 }, { ...p, betaA: 1.5 }, { ...p, clockANumerator: 0 }, { ...p, initialState: 2 }, { ...p, jumpCount: 17 }, { ...p, bathCode: 4 }, { ...p, timeNumerator: 0 }, { ...p, sampleSteps: 65 }, { ...p, timeDenominator: Infinity }]) { assert.equal(valid(v), false); assert.throws(() => run(v as Parameters), RangeError); }
  assert(valid({ ...p, timeNumerator: 0, jumpCount: 0, bathCode: 0 }));
});
test('every independent stored scalar certificate is compared', () => {
  function count(v: unknown): number { if (typeof v === 'string') return /^-?\d+(?:\.\d*)?(?:e[+-]?\d+)?(?:\/\d+)?$/i.test(v) ? 1 : 0; return v && typeof v === 'object' ? Object.values(v).reduce<number>((n, x) => n + count(x), 0) : 0; }
  assert.equal(certificates, fixtures.cases.reduce((n, c) => n + count(c.expected), 0)); console.log({ independentPreparations: fixtures.cases.length, scalarCertificates: certificates });
});
