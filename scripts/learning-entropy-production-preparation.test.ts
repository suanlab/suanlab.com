import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { threeStateEntropyProductionModel as run, isThreeStateEntropyParameters as valid, type ThreeStateEntropyParameters as Parameters } from '../src/lib/learning-three-state-entropy-production';
import type { PartitionObservation as Observation, PartitionRational } from '../src/lib/learning-partition-exact-core';
interface F { n: bigint; d: bigint }
const f = (n: bigint, d = BigInt(1)): F => ({ n, d });
const add = (a: F, b: F) => f(a.n * b.d + b.n * a.d, a.d * b.d);
const sub = (a: F, b: F) => f(a.n * b.d - b.n * a.d, a.d * b.d);
const le = (a: F, b: F) => a.n * b.d <= b.n * a.d;
function parse(value: string): F {
  if (value.includes('/')) { const [n, d] = value.split('/'); return f(BigInt(n), BigInt(d)); }
  const m = /^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(value); assert(m, value);
  const n = BigInt(m[1] + m[2] + (m[3] || '')), e = Number(m[4] || 0) - (m[3] || '').length;
  return e >= 0 ? f(n * BigInt(10) ** BigInt(e)) : f(n, BigInt(10) ** BigInt(-e));
}
function endpoint(q: PartitionRational): F {
  const n = BigInt(q.numerator.coefficient), d = BigInt(q.denominator.coefficient), e = q.numerator.exponent - q.denominator.exponent;
  return e >= 0 ? f(n << BigInt(e), d) : f(n, d << BigInt(-e));
}
function enclosure(q: Observation, value: string, label: string) {
  assert(q.bounds, label); const lo = endpoint(q.bounds.lower), hi = endpoint(q.bounds.upper), reference = parse(value);
  const abs = f(reference.n < BigInt(0) ? -reference.n : reference.n, reference.d), magnitude = le(abs, f(BigInt(1))) ? f(BigInt(1)) : abs;
  // Uncertainty of stored180-digit external references, not a production tolerance.
  const uncertainty = f(magnitude.n, magnitude.d * BigInt(10) ** BigInt(160));
  assert(le(lo, hi) && le(lo, add(reference, uncertainty)) && le(sub(reference, uncertainty), hi), label);
  if (q.resolution === 'structural-zero') assert(lo.n === BigInt(0) && hi.n === BigInt(0));
  if (q.resolution === 'unresolved-at-precision-ceiling') assert.equal(q.value, null);
}
type Case = { label: string; input: Parameters; stationary: string[]; current: string; affinity: string; entropyRate: string; pathTotal: string; pathSign: number; rows: { time: string; meanEntropy: string; logMoment: string }[] };
const fixtures = JSON.parse(readFileSync('scripts/fixtures/entropy-production-preparation.json', 'utf8')) as { cases: Case[] };
for (const c of fixtures.cases) test(`independent entropy/tilted reference: ${c.label}`, () => {
  const r = run(c.input); assert.equal(r.history.length, c.rows.length); assert(Object.isFrozen(r.parameters));
  for (let i = 0; i < 3; i++) enclosure(r.stationaryLaw[i], c.stationary[i], `pi${i}`);
  enclosure(r.stationaryCurrent, c.current, 'J'); enclosure(r.cycleAffinity, c.affinity, 'affinity'); enclosure(r.entropyRate, c.entropyRate, 'cycle-rate'); enclosure(r.pairedEntropyRate, c.entropyRate, 'paired-rate');
  enclosure(r.path.totalEntropy, c.pathTotal, 'path entropy'); assert.equal(r.path.totalEntropyExactSign, c.pathSign);
  assert.equal(r.path.totalEntropy.resolution === 'structural-zero', c.pathSign === 0);
  assert(r.adjointMassResidual.every(q => q.resolution === 'structural-zero'));
  assert.equal(r.classification.sampledPaths, false); assert.equal(r.classification.thermalBathSpecified, false); assert.equal(r.classification.pathLawOrTiltRenormalized, false);
  for (const [i, row] of r.history.entries()) {
    enclosure(row.time, c.rows[i].time, 'time'); enclosure(row.meanEntropy, c.rows[i].meanEntropy, 'mean'); enclosure(row.logMoment, c.rows[i].logMoment, 'logM'); enclosure(row.logSymmetryDifference, '0', 'symmetry');
    assert(row.symmetryEnclosuresConsistent); assert(row.moment === null || row.moment.resolution !== 'structural-zero');
    if (row.moment) assert(row.moment.value === null || row.moment.value > 0);
    if (row.expectedLogMomentSign !== 0) assert.notEqual(row.logMoment.resolution, 'structural-zero');
  }
});
const base: Parameters = { a: 2, b: 3, c: 4, d: 1, e: 1, f: 1, rateDenominator: 1, timeNumerator: 4, timeDenominator: 1, sampleSteps: 4, initialState: 0, pathCode: 0, jumpCount: 3, tiltNumerator: 1, tiltDenominator: 4 };
test('open nonuniform equilibrium paths cancel their boundary exactly', () => {
  const r = run({ ...base, a: 2, b: 3, c: 1, d: 1, e: 2, f: 3, rateDenominator: 3, jumpCount: 1, tiltNumerator: 17, tiltDenominator: 31 });
  assert(r.classification.detailedBalance); assert.equal(r.entropyRate.resolution, 'structural-zero');
  assert.equal(r.path.totalEntropy.resolution, 'structural-zero'); assert(r.path.rateOnlyEntropy.value! > 0); assert(r.path.systemBoundaryEntropy.value! < 0);
  assert.equal(r.path.totalViaBoundarySum.resolution, 'unresolved-at-precision-ceiling');
  assert.deepEqual(r.tiltedGenerator, r.generator); assert.deepEqual(r.adjointGenerator, r.generator);
  assert(r.history.every(row => row.logMoment.resolution === 'structural-zero' && row.moment?.value === 1));
});
test('negative individual paths coexist with nonnegative production', () => {
  const cw = run(base), ccw = run({ ...base, pathCode: 7 });
  assert(cw.path.totalEntropy.value! > 0); assert(ccw.path.totalEntropy.value! < 0); assert(cw.entropyRate.value! > 0);
  const reference = fixtures.cases.find(c => c.input.a === base.a && c.input.b === base.b && c.input.c === base.c && c.input.d === base.d && c.input.e === base.e && c.input.f === base.f && c.input.rateDenominator === base.rateDenominator)!;
  enclosure(cw.path.totalEntropy, reference.affinity, 'clockwise affinity');
  enclosure(ccw.path.totalEntropy, '-' + reference.affinity, 'opposite affinity');
  assert.deepEqual(cw.path.states, [0, 1, 2, 0]); assert.deepEqual(ccw.path.states, [0, 2, 1, 0]);
});
test('ensemble moments do not depend on the deterministic example word', () => {
  const a = run(base), b = run({ ...base, initialState: 2, jumpCount: 1, pathCode: 1 });
  assert.deepEqual(a.history, b.history); assert.notDeepEqual(a.path.states, b.path.states);
  const dense = run({ ...base, sampleSteps: 64 }); assert.deepEqual(a.current, { ...dense.current, step: a.current.step });
});
test('analytic normalizations differ from computed nontrivial tilted moments', () => {
  for (const tiltNumerator of [0, 1]) {
    const r = run({ ...base, tiltNumerator, tiltDenominator: 1 });
    assert(r.history.every(x => x.momentProof.method === 'normalization-identity' && x.moment?.value === 1 && x.logMoment.resolution === 'structural-zero'));
  }
  const r = run(base); assert.equal(r.current.momentProof.method, 'shifted-positive-series-log-normalization'); assert.equal(r.current.momentProof.retainedTerms, 65); assert(r.current.logMoment.value! < 0);
  // A tilted operator is not forced to have generator row sums.
  assert(r.tiltedGenerator.some(row => Math.abs(row.reduce((s, q) => s + q.value!, 0)) > .01));
});
test('exact cycle products preserve tiny positive production and unresolved tiny log moments', () => {
  const c = fixtures.cases.find(c => c.label === 'floating cycle products coincide')!, p = c.input, r = run(p);
  assert.equal(p.a * p.b * p.c, p.d * p.e * p.f); assert.equal(r.classification.detailedBalance, false); assert.equal(r.classification.entropyRateExactSign, 1); assert(r.entropyRate.value! > 0);
  const tiny = run(fixtures.cases.find(c => c.label === 'Cassini tiny signed log moment')!.input);
  assert.equal(tiny.current.expectedLogMomentSign, -1); assert.notEqual(tiny.current.logMoment.resolution, 'structural-zero');
});
test('large/tiny positive moments retain logs, explicit availability and subnormal projections', () => {
  const find = (label: string) => run(fixtures.cases.find(c => c.label === label)!.input);
  const large = find('huge positive moment'); assert.equal(large.current.moment, null); assert(large.current.logMoment.value! > 1e10);
  const small = find('tiny positive moment'); assert(small.current.logMoment.value! < -1e9); assert(small.current.moment); assert.equal(small.current.moment.value, null); assert.notEqual(small.current.moment.resolution, 'structural-zero');
  const finite = find('large finite linear moment'); assert(finite.current.moment?.value && Number.isFinite(finite.current.moment.value)); assert(finite.current.momentProof.linearPrecisionBits > 256);
  const outside = find('finite moment beyond binary64'); assert(outside.current.moment); assert.equal(outside.current.moment.value, null); assert.equal(outside.current.moment.sign, 1); assert.equal(outside.current.moment.resolution, 'interval-resolved');
  const subnormal = find('positive subnormal projection'); assert(subnormal.current.moment?.value && subnormal.current.moment.value < 2.2250738585072014e-308);
});
test('homogeneous count references require homogeneity and distinguish traffic from entropy', () => {
  const r = run({ ...base, a: 3, b: 3, c: 3, d: 1, e: 1, f: 1 });
  assert(r.classification.homogeneousRates); assert.equal(r.current.homogeneousClockwiseCountMean!.value, 12); assert.equal(r.current.homogeneousReverseCountMean!.value, 4);
  assert(Math.abs(r.current.homogeneousEntropyVariance!.value! - 16 * Math.log(3) ** 2) < 1e-12);
  const nonuniform = run(base); assert.equal(nonuniform.current.homogeneousEntropyVariance, null); assert.equal(nonuniform.current.homogeneousCountReferenceStatus, 'not-applicable-state-dependent-rates');
  const equilibrium = run({ ...base, a: 3, b: 3, c: 3, d: 3, e: 3, f: 3 }); assert.equal(equilibrium.entropyRate.value, 0); assert.equal(equilibrium.current.homogeneousClockwiseCountMean!.value, 12);
});
test('fifteen-field validation, zero duration, unused path bits and caller ownership', () => {
  assert(valid(base));
  for (const value of [null, [], { ...base, extra: 1 }, { ...base, a: 0 }, { ...base, a: 1.5 }, { ...base, a: 1000000001 }, { ...base, rateDenominator: 0 }, { ...base, timeNumerator: 0 }, { ...base, jumpCount: 17 }, { ...base, pathCode: 8 }, { ...base, initialState: 3 }, { ...base, tiltNumerator: 3, tiltDenominator: 1 }, { ...base, tiltNumerator: -2, tiltDenominator: 1 }, { ...base, sampleSteps: 65 }, { ...base, timeNumerator: NaN }]) {
    assert.equal(valid(value), false); assert.throws(() => run(value as Parameters), RangeError);
  }
  const zeroTime = run({ ...base, timeNumerator: 0, jumpCount: 0 }); assert(zeroTime.history.every(x => x.moment?.value === 1 && x.meanEntropy.value === 0));
  const p = { ...base }, saved = JSON.stringify(p), r = run(p); assert.equal(JSON.stringify(p), saved); p.a++; assert.notEqual(p.a, r.parameters.a);
});
