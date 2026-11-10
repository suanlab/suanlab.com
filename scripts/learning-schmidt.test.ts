import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { schmidtModel as model, createSchmidtParameters as create, snapshotSchmidtParameters as snapshot, isSchmidtParameters as valid, schmidtParameterKeys as keys, type SchmidtParameters as P } from '../src/lib/learning-schmidt';
import { mpsModel, createMpsParameters } from '../src/lib/learning-mps';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
ensureTestArtifactDirectory('.runtime/learning/schmidt-foundation', { recursive: true });


const references = JSON.parse(readFileSync('scripts/fixtures/schmidt-reference.json', 'utf8')) as { cases: { label: string; input: P; expected: unknown }[] };
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

let directedCertificateChecks = 0;
function directedZero(value: unknown) {
  if (value === null) return;
  if (isMasterObservation(value)) {
    const v = value as Q; assert(v.bounds);
    const lo = endpoint(v.bounds.lower), hi = endpoint(v.bounds.upper);
    assert(le(lo, frac(BigInt(0))) && le(frac(BigInt(0)), hi), 'directed certificate contains0');
    assert(le(frac(hi.n * lo.d - lo.n * hi.d, hi.d * lo.d), frac(BigInt(1), BigInt(10) ** BigInt(125))), 'directed certificate width');
    directedCertificateChecks++; return;
  }
  assert(value && typeof value === 'object'); Object.values(value).forEach(directedZero);
}
const numeric = (x: Q) => { assert(x.value !== null); return x.value; };
for (const reference of references.cases) test(`independent Schmidt reference: ${reference.label}`, () => {
  assert(valid(reference.input)); const result = model(reference.input);
  compare(result, reference.expected);
  for (const cut of result.cuts) { zeroCertificates(cut.exactResiduals); directedZero(cut.directedResiduals); if (cut.truncation.chosen) directedZero(cut.truncation.chosen.directedResiduals); }
  assert.doesNotThrow(() => JSON.stringify(result)); assert.deepEqual(result.parameters, reference.input);
});

test('35 owned bounded fields reject zero states, inactive coordinates and invalid cut/rank budgets', () => {
  assert.equal(keys.length, 35); assert.equal(new Set(keys).size, 35); const p = { ...create(2) };
  for (const key of ['sites', 'retainedRank', 'stateReal0']) for (const value of [NaN, Infinity, 1.5, 1000001, '1', true, null]) assert(!valid({ ...p, [key]: value }));
  for (const patch of [{ sites: 1 }, { sites: 5 }, { selectedCut: 0 }, { selectedCut: 2 }, { retainedRank: -1 }, { retainedRank: 5 }, { stateReal0: 0, stateReal3: 0 }, { stateReal4: 1 }, { extra: 0 }]) assert(!valid({ ...p, ...patch }));
  assert(!valid(Object.assign(Object.create(null), p))); assert(!valid(Object.create(p))); assert(!valid({ ...p, [Symbol('extra')]: 0 }));
  let reads = 0; const getter = { ...p }; Object.defineProperty(getter, 'sites', { get() { reads++; return 2; } }); assert(!valid(getter)); assert.throws(() => snapshot(getter)); assert.equal(reads, 0);
  const proxy = new Proxy(p, { get() { reads++; throw Error('Caller read'); } }); assert(valid(proxy)); assert.deepEqual(snapshot(proxy), p); assert.equal(reads, 0);
  assert(!valid(new Proxy({}, { getPrototypeOf() { throw Error('Reflection'); } })));
  const missing = { ...p }; delete missing.retainedRank; assert(!valid(missing));
  p.stateImag0 = -0; const saved = snapshot(p); p.stateReal0 = 3; assert(Object.isFrozen(saved)); assert.equal(saved.stateReal0, 1); assert(!Object.is(saved.stateImag0, -0));
  for (const args of [[1, 1], [5, 1], [Infinity, 1], [NaN, 1], [4, -1], [4, 5]]) assert.throws(() => create(...args));
});

test('all contiguous cuts identify the smaller physical side and full zero multiplicities', () => {
  const r = model(input('general complex 4 sites case0'));
  assert.deepEqual(r.cuts.map(c => c.smallerSide), ['left', 'left', 'right']);
  assert.deepEqual(r.cuts.map(c => c.smallerDimension), [2, 4, 2]);
  assert.deepEqual(r.cuts.map(c => c.schmidtRank), [2, 4, 2]);
  assert.deepEqual(r.cuts.map(c => [c.leftZeroMultiplicity, c.rightZeroMultiplicity]), [[0, 6], [0, 0], [6, 0]]);
  assert.equal(r.selectedCutIndex, r.selectedCut - 1);
});

test('diagonal9-4-1 weights separate the optimal raw error, normalized fidelity and normalized distance', () => {
  const r = model(input('diagonal weights9-4-1 rank budget2')).cuts[1];
  assert.equal(r.schmidtRank, 3); assert.equal(r.truncation.retainedPositiveRank, 2); assert(!r.truncation.boundaryDegenerate);
  assert(Math.abs(numeric(r.truncation.discardedWeight) - 1 / 14) < 1e-15);
  assert(Math.abs(numeric(r.truncation.optimalNormalizedFidelitySquared!) - 13 / 14) < 1e-15);
  assert(Math.abs(numeric(r.truncation.optimalNormalizedDistanceSquared!) - 2 * (1 - Math.sqrt(13 / 14))) < 1e-15);
  assert(r.truncation.chosen); assert.notEqual(r.truncation.optimalNormalizedDistanceSquared!.value, r.truncation.optimalUnnormalizedSquaredError.value);
  assert.deepEqual(r.truncation.optimalUnnormalizedSquaredError, r.truncation.discardedWeight);
});

test('a positive degenerate cutoff has a unique error value but no arbitrarily chosen state', () => {
  const ghz = model(input('GHZ4 degenerate rank1 optimum'));
  for (const c of ghz.cuts) {
    assert(c.truncation.boundaryDegenerate); assert(!c.truncation.optimalUnnormalizedStateUnique);
    assert.equal(c.truncation.chosen, null); assert(c.truncation.normalizedStateDefinedForEveryOptimum);
    assert.equal(c.truncation.discardedWeight.value, 0.5); assert.equal(c.truncation.optimalNormalizedFidelitySquared!.value, 0.5);
    assert.equal(c.entropyBits.value, 1);
  }
  const four = model(input('fourfold equal middle weights budget3')).cuts[1];
  assert.equal(four.spectrum[0].multiplicity, 4); assert.equal(four.truncation.discardedWeight.value, 0.25); assert.equal(four.truncation.chosen, null);
});

test('keeping an entire degenerate group restores a unique positive-support truncation', () => {
  const split = model(input('two positive degenerate groups budget1')).cuts[1]; assert(split.truncation.boundaryDegenerate);
  const whole = model(input('two positive degenerate groups budget2')).cuts[1]; assert(!whole.truncation.boundaryDegenerate); assert(whole.truncation.chosen);
  assert(Math.abs(numeric(whole.truncation.retainedWeight) - 0.8) < 1e-15);
  const splitLower = model(input('two positive degenerate groups budget3')).cuts[1]; assert(splitLower.truncation.boundaryDegenerate); assert.equal(splitLower.truncation.chosen, null);
});

test('rank zero chooses the zero raw approximation and never normalizes it', () => {
  const r = model(input('rank0 zero truncation no normalized optimum')).cuts[0].truncation;
  assert(r.chosen); assert(r.optimalUnnormalizedStateUnique); assert.equal(r.retainedPositiveRank, 0);
  assert.equal(r.optimalUnnormalizedSquaredError.value, 1); assert.equal(r.chosen.normalizedAmplitudes, null);
  assert.equal(r.optimalNormalizedFidelitySquared, null); assert.equal(r.optimalNormalizedDistanceSquared, null);
  assert(!r.normalizedStateDefinedForEveryOptimum);
});

test('right physical density and truncation preserve complex phase through a transpose', () => {
  const product = model(input('right physical density complex product transpose')).cuts[1];
  assert.equal(product.smallerSide, 'right'); assert.equal(product.rhoRight[0][1].imaginary.value, -0.5);
  assert.equal(product.truncation.discardedWeight.value, 0);
  const phase = model(input('right physical spectral truncation complex phase')).cuts[1];
  assert.equal(phase.smallerSide, 'right'); assert(phase.truncation.chosen);
  assert(Math.abs(numeric(phase.truncation.chosen.directSquaredError) - 0.2) < 1e-15);
  const retained = phase.truncation.chosen.truncatedRawCoefficientMatrix;
  assert(Math.abs(numeric(retained[0][0].real) - 2) < 1e-15); assert(Math.abs(numeric(retained[0][1].imaginary) - 2) < 1e-15);
  directedZero(retained[1][0].real);
});

test('tiny positive weights retain exact rank and near equal weights are not merged', () => {
  const near = model(input('near rank deficient exact nonzero determinant')).cuts[0];
  assert.equal(near.schmidtRank, 2); assert.equal(near.spectrum.length, 2);
  assert(numeric(near.spectrum[1].weight) > 0 && numeric(near.spectrum[1].weight) < 1e-24);
  assert(numeric(near.truncation.discardedWeight) > 0);
  const close = model(input('close positive Schmidt weights not merged')).cuts[1];
  assert.equal(close.schmidtRank, 3); assert(close.spectrum.filter(v => !v.zeroWeight).every(v => v.multiplicity === 1)); assert(!close.truncation.boundaryDegenerate);
});

test('budgets above physical rank preserve the original state without assigning nullspace Schmidt vectors', () => {
  const r = model(input('rank budget above smaller side dimension'));
  for (const c of r.cuts) {
    assert.equal(c.schmidtRank, 1); assert.equal(c.truncation.rankBudget, 4); assert.equal(c.truncation.retainedPositiveRank, 1);
    assert.equal(c.truncation.discardedWeight.value, 0); assert.equal(c.entropyBits.value, 0);
    assert(c.truncation.chosen); assert(!c.truncation.boundaryDegenerate);
  }
});

test('complex rescaling preserves density, entropy, spectrum and truncation values', () => {
  const p = input('general complex 3 sites case0'), changed = { ...p };
  for (let i = 0; i < 2 ** p.sites; i++) { const a = p[`stateReal${i}`], b = p[`stateImag${i}`]; changed[`stateReal${i}`] = 2 * a - 3 * b; changed[`stateImag${i}`] = 3 * a + 2 * b; }
  const a = model(p), b = model(changed);
  for (let i = 0; i < a.cuts.length; i++) {
    for (const key of ['rhoLeft', 'rhoRight', 'spectrum', 'purity', 'entropyBits', 'characteristicPolynomial'] as const) assert.deepEqual(a.cuts[i][key], b.cuts[i][key]);
    assert.deepEqual(a.cuts[i].truncation.discardedWeight, b.cuts[i].truncation.discardedWeight);
  }
  assert(equal(exact(b.rawNormSquared), multiply(frac(BigInt(13)), exact(a.rawNormSquared))));
});

test('a local Hadamard preserves every cut spectrum and error while doubling raw norm', () => {
  const p = input('general complex 3 sites case1'), changed = { ...p }, mask = 2;
  for (let i = 0; i < 8; i++) if (!(i & mask)) for (const part of ['Real', 'Imag']) {
    const a = p[`state${part}${i}`], b = p[`state${part}${i | mask}`]; changed[`state${part}${i}`] = a + b; changed[`state${part}${i | mask}`] = a - b;
  }
  const a = model(p), b = model(changed); assert(equal(exact(b.rawNormSquared), multiply(frac(BigInt(2)), exact(a.rawNormSquared))));
  a.cuts.forEach((v, i) => { assert.deepEqual(v.characteristicPolynomial, b.cuts[i].characteristicPolynomial); assert.deepEqual(v.entropyBits, b.cuts[i].entropyBits); assert.deepEqual(v.truncation.discardedWeight, b.cuts[i].truncation.discardedWeight); });
});

test('reversing qubit order exchanges cuts without changing positive weights or entropy', () => {
  const p = input('general complex 4 sites case1'), changed = { ...p };
  for (let i = 0; i < 16; i++) { const j = parseInt(i.toString(2).padStart(4, '0').split('').reverse().join(''), 2); for (const part of ['Real', 'Imag']) changed[`state${part}${j}`] = p[`state${part}${i}`]; }
  const a = model(p), b = model(changed);
  a.cuts.forEach((v, i) => { const w = b.cuts[2 - i]; assert.deepEqual(v.characteristicPolynomial, w.characteristicPolynomial); assert.equal(v.schmidtRank, w.schmidtRank); assert.deepEqual(v.entropyBits, w.entropyBits); });
});

test('Schmidt rank agrees with the independently contracted MPS and its bond lower bounds', () => {
  const mps = mpsModel({ ...createMpsParameters(3, 2), t0Real011: 1, t1Real111: 1, t2Real110: 1 });
  const schmidt = model({ ...create(3, 1), stateReal7: 1 });
  assert.deepEqual(schmidt.cuts.map(c => c.schmidtRank), mps.original.cuts.map(c => c.schmidtRank));
  assert(schmidt.cuts.every((c, i) => c.schmidtRank <= mps.bondDimensions[i]));
});

after(() => writeFileSync('.runtime/learning/schmidt-foundation/reference-observations.json', JSON.stringify({ cases: references.cases.length, exactChecks, directedChecks, directedCertificateChecks }, null, 2) + '\n'));
