import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { spinChainModel as model, createSpinChainParameters as create, snapshotSpinChainParameters as snapshot, isSpinChainParameters as valid, spinChainParameterKeys as keys, spinChainEdges as edges, type SpinChainParameters as P } from '../src/lib/learning-spin-chain';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';

const references = JSON.parse(readFileSync('scripts/fixtures/spin-chain-reference.json', 'utf8')) as { cases: { label: string; input: P; expected: unknown }[] };
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
  if (isMasterObservation(value)) { observation(value, '0', 'exact certificate'); return; }
  assert(value && typeof value === 'object'); Object.values(value).forEach(zeroCertificates);
}
const input = (label: string) => { const r = references.cases.find(r => r.label === label); assert(r, label); return structuredClone(r.input); };

for (const reference of references.cases) test(`independent spin-chain reference: ${reference.label}`, () => {
  assert(valid(reference.input)); const result = model(reference.input);
  compare(result, reference.expected); zeroCertificates(result.exactResiduals);
  assert.doesNotThrow(() => JSON.stringify(result));
  assert.deepEqual(result.parameters, reference.input);
});

test('owned102-field inputs reject ambiguous shapes, getters, inactive entries and invalid ranges', () => {
  assert.equal(keys.length, 102); assert.equal(new Set(keys).size, 102);
  const original = { ...create(2) };
  for (const key of ['sites', 'hDenominator', 'stateReal0']) for (const value of [NaN, Infinity, 1.5, 1000001, '1', true, null]) assert(!valid({ ...original, [key]: value }));
  for (const patch of [{ sites: 0 }, { sites: 5 }, { graph: -1 }, { graph: 3 }, { hDenominator: 0 }, { hDenominator: -1 }, { stateReal0: 0 }, { stateReal4: 1 }, { fieldX2: 1 }, { couplingXY02: 1 }, { extra: 0 }]) assert(!valid({ ...original, ...patch }));
  assert(!valid(Object.assign(Object.create(null), original))); assert(!valid(Object.create(original)));
  const symbol = { ...original, [Symbol('extra')]: 0 }; assert(!valid(symbol));
  let reads = 0; const getter = { ...original }; Object.defineProperty(getter, 'sites', { get() { reads++; return 2; } });
  assert(!valid(getter)); assert.equal(reads, 0); assert.throws(() => snapshot(getter)); assert.equal(reads, 0);
  assert.equal(valid(new Proxy({}, { getPrototypeOf() { throw new Error('Invalid reflection'); } })), false);
  const trapped = new Proxy(original, { get() { reads++; throw new Error('Caller property access'); } });
  assert(valid(trapped)); assert.deepEqual(snapshot(trapped), original); assert.equal(reads, 0);
  const missing = { ...original }; delete missing.offset; assert(!valid(missing));
});

test('snapshot ownership and every graph/dimension are explicit without truncation or duplicate ring bonds', () => {
  for (let n = 1; n <= 4; n++) for (let graph = 0; graph < 3; graph++) assert(valid(create(n, graph)));
  for (const [n, graph] of [[0, 0], [5, 0], [1.5, 0], [2, 3], [2, NaN]]) assert.throws(() => create(n, graph));
  assert.deepEqual(edges(1, 1), []); for (const graph of [0, 1, 2]) assert.deepEqual(edges(2, graph), [[0, 1]]);
  assert.equal(edges(4, 0).length, 3); assert.equal(edges(4, 1).length, 4); assert.equal(edges(4, 2).length, 6);
  const value = { ...create(3) }; value.stateImag0 = -0; const saved = snapshot(value); value.fieldX0 = 7;
  assert(Object.isFrozen(saved)); assert.equal(saved.fieldX0, 0); assert(!Object.is(saved.stateImag0, -0));
  assert(valid({ ...create(3, 1), couplingXY02: 1 })); assert(!valid({ ...create(3, 0), couplingXY02: 1 }));
});

test('zero variance does not create an uncomputed ground certificate', () => {
  const excited = model(input('zero variance excited ferromagnet')); assert(excited.classification.exactEigenstate); assert.equal(excited.classification.selectedStateGround, false);
  const singlet = model(input('XX YY singlet has no supplied ground certificate')); assert(singlet.classification.exactEigenstate); assert.equal(singlet.diagonalSpectrum, null); assert.equal(singlet.classification.selectedStateGround, null);
  const tiny = model(input('tiny positive near eigenstate variance')); assert.equal(tiny.classification.exactEigenstate, false); assert.equal(tiny.variance.sign, 1);
});

test('constant Hamiltonians retain whole ground spaces and absent excitation gaps', () => {
  for (let n = 1; n <= 4; n++) {
    const r = model({ ...create(n), offset: -2 }); assert(r.diagonalSpectrum);
    assert.equal(r.diagonalSpectrum.groundMultiplicity, 2 ** n); assert.equal(r.diagonalSpectrum.excitationGap, null);
    assert.equal(r.diagonalSpectrum.certifiedLeakageUpperBound, null); assert.equal(r.diagonalSpectrum.groundProbability.value, 1);
  }
  const ratio = model(input('gap ratio above probability range')).diagonalSpectrum!;
  assert.equal(ratio.excessEnergyToGapRatio!.value, 4); assert.equal(ratio.certifiedLeakageUpperBound!.value, 1); assert.equal(ratio.leakage.value, 1);
});

test('complex local Y orientation and correlations distinguish physical meanings', () => {
  const r = model(input('right local Y phase orientation')); assert.equal(r.localStates[1].bloch[1].value, 1);
  assert.equal(r.localStates[1].density[0][1].imaginary.value, -0.5); assert.equal(r.localStates[1].purity.value, 1);
  const ghz = model(input('GHZ4 diagonal ferromagnet')); assert(ghz.localStates.every(s => s.purity.value === 0.5)); assert(ghz.correlations.every(p => p.values[2][2].connected.value === 1));
  const w = model(input('W4 constant Hamiltonian')); assert(w.correlations.every(p => p.values[2][2].connected.value === -0.25));
  const product = model(input('maximum complex integer preparation'));
  assert(product.localStates.every(s => s.purity.value === 1));
  assert(product.correlations.every(p => p.values.flat().every(v => v.connected.sign === 0)), 'couplings do not automatically change the supplied product state');
});

test('Hamiltonian symmetries do not follow from a currently sharp state and account for cancellations', () => {
  const broken = model({ ...create(2), fieldX0: 1 }); assert(broken.totalZ.selectedStateSharp); assert(!broken.totalZ.conserved); assert(!broken.globalZParity.conserved);
  const xx = model({ ...create(2), couplingXX01: 1 }); assert(!xx.totalZ.conserved); assert(xx.globalZParity.conserved);
  const hopping = model(input('XX YY sharp total Z excited eigenstate')); assert(hopping.totalZ.conserved); assert(hopping.globalZParity.conserved);
});

for (const a of ['X', 'Y', 'Z']) for (const b of ['X', 'Y', 'Z']) test(`all nine two-site Pauli products are Hermitian involutions: ${a}${b}`, () => {
  const r = model({ ...create(2), [`coupling${a}${b}01`]: 1 });
  assert.equal(r.secondMoment.value, 1); assert.equal(r.hamiltonianFrobeniusSquared.value, 4); assert.equal(r.cost.nonzeroMatrixEntries, 4); zeroCertificates(r.exactResiduals);
});

test('complex rescaling preserves physical means, local densities and correlations', () => {
  const p = input('general complex 3 sites graph2 case0'), changed = { ...p };
  for (let i = 0; i < 2 ** p.sites; i++) { const a = p[`stateReal${i}`], b = p[`stateImag${i}`]; changed[`stateReal${i}`] = 2 * a - 3 * b; changed[`stateImag${i}`] = 3 * a + 2 * b; }
  const a = model(p), b = model(changed);
  for (const key of ['energy', 'variance', 'localStates', 'correlations', 'totalZ', 'globalZParity'] as const) assert.deepEqual(a[key], b[key]);
  assert(equal(exact(b.rawNormSquared), multiply(frac(BigInt(13)), exact(a.rawNormSquared))));
});

test('energy-origin changes and signed Hamiltonian scaling preserve the correct invariants', () => {
  const p = input('general complex 4 sites graph2 case0'), a = model(p), shifted = model({ ...p, offset: p.offset + 7 });
  assert(equal(exact(shifted.energy), add(exact(a.energy), frac(BigInt(7), BigInt(p.hDenominator))))); assert.deepEqual(shifted.variance, a.variance); assert.deepEqual(shifted.totalZ, a.totalZ);
  const scaled = { ...p };
  for (const key of keys) if (key === 'offset' || key.startsWith('field') || key.startsWith('coupling')) scaled[key] *= -2;
  const b = model(scaled); assert(equal(exact(b.energy), multiply(frac(BigInt(-2)), exact(a.energy)))); assert(equal(exact(b.variance), multiply(frac(BigInt(4)), exact(a.variance))));
  const ferromagnet = input('GHZ4 diagonal ferromagnet'), negative = { ...ferromagnet };
  for (const key of keys) if (key.startsWith('coupling')) negative[key] *= -1;
  assert.equal(model(ferromagnet).classification.selectedStateGround, true); assert.equal(model(negative).classification.selectedStateGround, false);
});

test('site permutations preserve energy and correctly relabel ordered-axis couplings', () => {
  const p = input('general complex 4 sites graph2 case0'), permutation = [2, 0, 3, 1];
  const changed: P = { ...create(4, 2), hDenominator: p.hDenominator, offset: p.offset };
  for (let i = 0; i < 4; i++) for (const a of ['X', 'Y', 'Z']) changed[`field${a}${i}`] = p[`field${a}${permutation[i]}`];
  for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) for (const a of ['X', 'Y', 'Z']) for (const b of ['X', 'Y', 'Z']) {
    const [u, v] = [permutation[i], permutation[j]]; changed[`coupling${a}${b}${i}${j}`] = p[u < v ? `coupling${a}${b}${u}${v}` : `coupling${b}${a}${v}${u}`];
  }
  for (let index = 0; index < 16; index++) {
    const bits = index.toString(2).padStart(4, '0'), originalBits = Array(4).fill('0'); permutation.forEach((old, i) => { originalBits[old] = bits[i]; }); const oldIndex = parseInt(originalBits.join(''), 2);
    for (const part of ['Real', 'Imag']) changed[`state${part}${index}`] = p[`state${part}${oldIndex}`];
  }
  const a = model(p), b = model(changed); assert.deepEqual(a.energy, b.energy); assert.deepEqual(a.variance, b.variance);
  for (let i = 0; i < 4; i++) { assert.deepEqual(b.localStates[i].bloch, a.localStates[permutation[i]].bloch); assert.deepEqual(b.localStates[i].density, a.localStates[permutation[i]].density); }
});

test('complex conjugation flips every odd-Y term and the local Y coordinates', () => {
  const p = input('general complex 4 sites graph2 case1'), changed = { ...p };
  for (let i = 0; i < 16; i++) changed[`stateImag${i}`] *= -1;
  for (const key of keys) if (key.startsWith('fieldY') || key.startsWith('coupling') && [...key.slice(8, 10)].filter(a => a === 'Y').length % 2 === 1) changed[key] *= -1;
  const a = model(p), b = model(changed); assert.deepEqual(a.energy, b.energy); assert.deepEqual(a.variance, b.variance);
  a.localStates.forEach((v, i) => { assert(equal(exact(b.localStates[i].bloch[1]), multiply(frac(BigInt(-1)), exact(v.bloch[1])))); assert.deepEqual(b.localStates[i].purity, v.purity); });
});

after(() => {
  writeFileSync('.runtime/learning/spin-chain-foundation/reference-observations.json', JSON.stringify({ cases: references.cases.length, exactChecks, directedChecks }, null, 2) + '\n');
});
