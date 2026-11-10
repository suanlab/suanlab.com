import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { mpsModel as model, createMpsParameters as create, snapshotMpsParameters as snapshot, isMpsParameters as valid, mpsParameterKeys as keys, type MpsParameters as P } from '../src/lib/learning-mps';
import { spinChainModel, createSpinChainParameters, type SpinChainParameters } from '../src/lib/learning-spin-chain';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
ensureTestArtifactDirectory('.runtime/learning/mps-foundation', { recursive: true });


const references = JSON.parse(readFileSync('scripts/fixtures/mps-reference.json', 'utf8')) as { cases: { label: string; input: P; expected: unknown }[] };
type F = { n: bigint; d: bigint };
const frac = (n: bigint, d = BigInt(1)): F => ({ n, d });
const le = (a: F, b: F) => a.n * b.d <= b.n * a.d;
const equal = (a: F, b: F) => a.n * b.d === b.n * a.d;
const add = (a: F, b: F) => frac(a.n * b.d + b.n * a.d, a.d * b.d);
const multiply = (a: F, b: F) => frac(a.n * b.n, a.d * b.d);
function parse(raw: string): F {
  if (raw.includes('/')) { const [n, d] = raw.split('/'); return frac(BigInt(n), BigInt(d)); }
  const m = /^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(raw); assert(m, raw);
  const n = BigInt(m[1] + m[2] + (m[3] ?? '')), exponent = Number(m[4] ?? 0) - (m[3] ?? '').length;
  return exponent >= 0 ? frac(n * BigInt(10) ** BigInt(exponent)) : frac(n, BigInt(10) ** BigInt(-exponent));
}
function endpoint(b: NonNullable<Q['bounds']>['lower']): F {
  const n = BigInt(b.numerator.coefficient), d = BigInt(b.denominator.coefficient), e = b.numerator.exponent - b.denominator.exponent;
  return e >= 0 ? frac(n * BigInt(10) ** BigInt(e), d) : frac(n, d * BigInt(10) ** BigInt(-e));
}
const exact = (q: Q) => { assert(q.bounds); const lo = endpoint(q.bounds.lower), hi = endpoint(q.bounds.upper); assert(equal(lo, hi)); return lo; };
let exactChecks = 0, directedChecks = 0;
function observation(actual: unknown, expected: string, path: string) {
  assert(isMasterObservation(actual), path); const q = actual as Q; assert(q.bounds, path);
  const lo = endpoint(q.bounds.lower), hi = endpoint(q.bounds.upper); assert(le(lo, hi), path);
  const directed = expected.startsWith('decimal:'), value = parse(directed ? expected.slice(8) : expected);
  const absolute = value.n < BigInt(0) ? -value.n : value.n, scale = absolute > value.d ? absolute : value.d;
  if (directed) {
    const eps = BigInt(10) ** BigInt(205);
    assert(le(lo, frac(value.n * eps + scale, value.d * eps)) && le(frac(value.n * eps - scale, value.d * eps), hi), path + ' reference overlap');
    directedChecks++;
  } else { assert(equal(lo, value) && equal(hi, value), path + ' exact equality'); exactChecks++; }
  const width = frac(hi.n * lo.d - lo.n * hi.d, hi.d * lo.d);
  assert(le(width, frac(scale, value.d * BigInt(10) ** BigInt(125))), path + ' directed width');
}
function compare(actual: unknown, expected: unknown, path = 'result') {
  if (typeof expected === 'string' && (/^-?\d/.test(expected) || expected.startsWith('decimal:'))) return observation(actual, expected, path);
  if (Array.isArray(expected)) { assert(Array.isArray(actual), path); assert.equal(actual.length, expected.length, path); expected.forEach((v, i) => compare(actual[i], v, `${path}[${i}]`)); return; }
  if (expected && typeof expected === 'object') { assert(actual && typeof actual === 'object', path); for (const [key, value] of Object.entries(expected)) compare((actual as Record<string, unknown>)[key], value, `${path}.${key}`); return; }
  assert.equal(actual, expected, path);
}
function zeroCertificates(value: unknown) {
  if (value === null) return;
  if (isMasterObservation(value)) { observation(value, '0', 'exact certificate'); return; }
  assert(value && typeof value === 'object'); Object.values(value).forEach(zeroCertificates);
}
const input = (label: string) => { const r = references.cases.find(r => r.label === label); assert(r, label); return structuredClone(r.input); };

for (const reference of references.cases) test(`independent MPS reference: ${reference.label}`, () => {
  assert(valid(reference.input)); const result = model(reference.input);
  compare(result, reference.expected); zeroCertificates(result.exactResiduals); zeroCertificates(result.original.exactResiduals); zeroCertificates(result.gauged.exactResiduals);
  assert.deepEqual(result.original.state, result.gauged.state); assert.deepEqual(result.original.physical, result.gauged.physical);
  assert.doesNotThrow(() => JSON.stringify(result)); assert.deepEqual(result.parameters, reference.input);
});

test('269 owned bounded fields reject ambiguous dimensions, inactive tensor entries and singular gauges', () => {
  assert.equal(keys.length, 269); assert.equal(new Set(keys).size, 269); const p = { ...create() };
  for (const key of ['sites', 'tensorDenominator', 't0Real000']) for (const value of [NaN, Infinity, 1.5, 1000001, '1', true, null]) assert(!valid({ ...p, [key]: value }));
  for (const patch of [{ sites: 0 }, { sites: 5 }, { tensorDenominator: 0 }, { bond0: 0 }, { bond1: 5 }, { bond2: 1 }, { t0Real100: 1 }, { t2Real001: 1 }, { t3Real000: 1 }, { gaugeScaleNumerator: 0 }, { gaugeScaleDenominator: 0 }, { gaugeShearDenominator: 0 }, { gaugeBond: -2 }, { gaugeBond: 2 }, { gaugeRow: 2 }, { gaugeColumn: -1 }, { gaugeRow: 1, gaugeColumn: 1, gaugeShearReal: 1 }, { extra: 0 }]) assert(!valid({ ...p, ...patch }));
  assert(!valid(Object.assign(Object.create(null), p))); assert(!valid(Object.create(p))); assert(!valid({ ...p, [Symbol('extra')]: 0 }));
  let reads = 0; const getter = { ...p }; Object.defineProperty(getter, 'sites', { get() { reads++; return 3; } }); assert(!valid(getter)); assert.throws(() => snapshot(getter)); assert.equal(reads, 0);
  const proxy = new Proxy(p, { get() { reads++; throw Error('Caller read'); } }); assert(valid(proxy)); assert.deepEqual(snapshot(proxy), p); assert.equal(reads, 0);
  assert(!valid(new Proxy({}, { getPrototypeOf() { throw Error('Reflection'); } })));
  const missing = { ...p }; delete missing.tensorDenominator; assert(!valid(missing));
  p.t0Imag000 = -0; const saved = snapshot(p); p.t0Real000 = 7; assert(Object.isFrozen(saved)); assert.equal(saved.t0Real000, 1); assert(!Object.is(saved.t0Imag000, -0));
  for (const args of [[0, 2], [5, 2], [Infinity, 2], [NaN, 2], [1.5, 2], [3, 0], [3, 5]]) assert.throws(() => create(...args));
});

test('all85 open-boundary dimension profiles preserve an exactly known product state', () => {
  let count = 0;
  for (let sites = 1; sites <= 4; sites++) {
    const walk = (bonds: number[]) => {
      if (bonds.length < sites - 1) { for (let d = 1; d <= 4; d++) walk([...bonds, d]); return; }
      const p = { ...create(sites, 1) }; bonds.forEach((d, i) => { p[`bond${i}`] = d; });
      const r = model(p); assert.equal(r.original.normSquared.value, 1); assert.deepEqual(r.original.state, r.gauged.state);
      assert(r.original.cuts.every(c => c.coefficientRank === 1 && c.schmidtRank === 1));
      assert(r.original.physical!.localStates.every(s => s.bloch[2].value === 1 && s.purity.value === 1));
      zeroCertificates(r.exactResiduals); count++;
    }; walk([]);
  }
  assert.equal(count, 85);
});

test('nonzero tensors can cancel globally and must not be normalized or repaired', () => {
  const r = model(input('nonzero tensors cancel to global zero'));
  assert(valid(r.parameters)); assert(r.original.tensorParameterSquaredNorm.value! > 0);
  assert(r.original.tensors.every(T => T.flat(2).some(z => z.real.sign !== 0 || z.imaginary.sign !== 0)));
  assert.equal(r.original.normSquared.sign, 0); assert.equal(r.original.physical, null); assert.equal(r.gauged.physical, null);
  assert.equal(r.validity.reason, 'zero-contracted-vector-cannot-normalize'); assert(r.original.cuts.every(c => c.coefficientRank === 0 && c.schmidtRank === null));
  const zero = model(input('all tensors zero no normalized state')); assert.equal(zero.original.tensorParameterSquaredNorm.sign, 0); assert.equal(zero.original.physical, null);
});

test('GHZ raw norm and physical correlations stay fixed while tensor coordinate length changes', () => {
  const r = model(input('GHZ3 scale shear parameter norm changes'));
  assert.equal(r.original.normSquared.value, 2); assert.equal(r.gauged.normSquared.value, 2);
  assert.equal(r.original.tensorParameterSquaredNorm.value, 6); assert.equal(r.gauged.tensorParameterSquaredNorm.value, 19 / 2);
  assert(r.original.physical!.localStates.every(s => s.purity.value === 0.5));
  assert(r.original.physical!.correlations.every(pair => pair.values[2][2].connected.value === 1));
  assert(r.original.cuts.every(c => c.schmidtRank === 2));
  assert(r.original.canonicality[0].leftIsometry); assert(!r.gauged.canonicality[0].leftIsometry);
});

test('bra-ket environment orientation preserves a positive right Y phase', () => {
  const r = model(input('product right Y positive'));
  assert.equal(r.original.physical!.localStates[1].bloch[1].value, 1);
  assert.equal(r.original.physical!.localStates[1].density[0][1].imaginary.value, -0.5);
  const phase = model(input('complex Bell phase fixes environment orientation'));
  zeroCertificates(phase.exactResiduals); assert.notDeepEqual(phase.original.leftEnvironments[1], phase.gauged.leftEnvironments[1]);
  assert.deepEqual(phase.original.physical, phase.gauged.physical);
});

test('virtual dimensions are upper bounds rather than Schmidt ranks or guaranteed compression', () => {
  const r = model(input('redundant bond4 product rank1'));
  assert(r.original.cuts.every(c => c.bondDimension === 4 && c.schmidtRank === 1));
  assert(r.cost.activeComplexTensorEntries > r.cost.denseComplexStateAmplitudes);
  assert(r.cost.countsAreAlgorithmicSlotsNotRuntimeOrCompressionGuarantees);
  const tiny = model(input('smallest amplitude scale irrational normalization'));
  assert.equal(tiny.original.normSquared.sign, 1); assert(tiny.original.physical); assert(tiny.original.cuts.every(c => c.schmidtRank === 2));
});

test('one site and explicitly disabled gauges have no fabricated inverse or covariance', () => {
  const one = model(input('one site no virtual gauge')); assert.equal(one.gauge, null); assert.deepEqual(one.bondDimensions, []);
  assert.equal(one.exactResiduals.leftEnvironmentCovariance, null); assert.equal(one.exactResiduals.rightEnvironmentCovariance, null);
  const p = { ...create(), gaugeBond: -1, gaugeRow: 0, gaugeColumn: 0, gaugeScaleNumerator: 1, gaugeScaleDenominator: 1, gaugeShearReal: 0, gaugeShearImag: 0, gaugeShearDenominator: 1 };
  const r = model(p); assert.equal(r.gauge, null); assert.deepEqual(r.original, r.gauged);
  assert(!valid({ ...p, gaugeScaleNumerator: 2 })); assert(!valid({ ...p, gaugeShearImag: 1 }));
});

test('complex scale/shear gauges on every bond and pivot row preserve full physical information', () => {
  const p = input('general complex MPS profile12 sites4 bonds[4, 4, 4]');
  for (let bond = 0; bond < 3; bond++) for (let row = 0; row < 4; row++) {
    const r = model({ ...p, gaugeBond: bond, gaugeRow: row, gaugeColumn: (row + 1) % 4, gaugeScaleNumerator: -3, gaugeScaleDenominator: 2, gaugeShearReal: 2, gaugeShearImag: -3, gaugeShearDenominator: 5 });
    assert.deepEqual(r.original.state, r.gauged.state); assert.deepEqual(r.original.physical, r.gauged.physical);
    assert.deepEqual(r.original.cuts, r.gauged.cuts); zeroCertificates(r.exactResiduals);
  }
});

test('applying the exact inverse gauge restores the original tensors', () => {
  const p = input('GHZ3 scale shear parameter norm changes'), r = model(p);
  const changed: P = { ...p, tensorDenominator: 2, gaugeScaleNumerator: 1, gaugeScaleDenominator: 2, gaugeShearReal: -1, gaugeShearDenominator: 1 };
  r.gauged.tensors.forEach((T, site) => T.forEach((matrix, s) => matrix.forEach((row, left) => row.forEach((z, right) => {
    for (const [part, q] of [['Real', z.real], ['Imag', z.imaginary]] as const) { const v = exact(q); assert.equal((v.n * BigInt(2)) % v.d, BigInt(0)); changed[`t${site}${part}${left}${s}${right}`] = Number(v.n * BigInt(2) / v.d); }
  }))));
  assert.deepEqual(model(changed).gauged.tensors, r.original.tensors);
});

test('rescaling a tensor preserves normalized observables but changes the raw vector and norm', () => {
  const p = input('general complex MPS profile6 sites3 bonds[2, 2]'), changed = { ...p };
  for (const key of keys) if (key.startsWith('t1Real') || key.startsWith('t1Imag')) changed[key] *= -2;
  const a = model(p), b = model(changed); assert(a.original.physical && b.original.physical);
  assert.deepEqual(a.original.physical.localStates, b.original.physical.localStates); assert.deepEqual(a.original.physical.correlations, b.original.physical.correlations);
  assert(equal(exact(b.original.normSquared), multiply(frac(BigInt(4)), exact(a.original.normSquared))));
  assert.notDeepEqual(a.original.state, b.original.state);
});

test('complex conjugation preserves norms and reverses Y in local physical states', () => {
  const p = input('general complex MPS profile6 sites3 bonds[2, 2]'), changed = { ...p };
  for (const key of keys) if (/^t\dImag/.test(key) || key === 'gaugeShearImag') changed[key] *= -1;
  const a = model(p), b = model(changed); assert.deepEqual(a.original.normSquared, b.original.normSquared);
  for (let i = 0; i < p.sites; i++) {
    assert.deepEqual(a.original.physical!.localStates[i].purity, b.original.physical!.localStates[i].purity);
    assert(equal(exact(b.original.physical!.localStates[i].bloch[1]), multiply(frac(BigInt(-1)), exact(a.original.physical!.localStates[i].bloch[1]))));
  }
});

test('contracted GHZ local states and pair correlations agree with the previous dense spin model', () => {
  const r = model(input('GHZ3 scale shear parameter norm changes'));
  const p: SpinChainParameters = { ...createSpinChainParameters(3), stateReal7: 1 }, dense = spinChainModel(p);
  assert.deepEqual(r.original.physical!.localStates.map(s => ({ site: s.site, density: s.density, bloch: s.bloch, purity: s.purity })), dense.localStates.map(s => ({ site: s.site, density: s.density, bloch: s.bloch, purity: s.purity })));
  assert.deepEqual(r.original.physical!.correlations, dense.correlations);
});

after(() => writeFileSync('.runtime/learning/mps-foundation/reference-observations.json', JSON.stringify({ cases: references.cases.length, exactChecks, directedChecks }, null, 2) + '\n'));
