import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { fermionFockModel as model, createFermionFockParameters as create, snapshotFermionFockParameters as snapshot, isFermionFockParameters as valid, fermionFockParameterKeys as keys, type FermionFockParameters as P } from '../src/lib/learning-fermion-fock';
import { fermionOccupationCount } from '../src/lib/learning-fermion-fock-input';
import { spinChainModel, createSpinChainParameters, type SpinChainParameters } from '../src/lib/learning-spin-chain';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
ensureTestArtifactDirectory('.runtime/learning/fermion-fock-foundation', { recursive: true });


const references = JSON.parse(readFileSync('scripts/fixtures/fermion-fock-reference.json', 'utf8')) as { cases: { label: string; input: P; expected: unknown }[] };
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

for (const reference of references.cases) test(`independent fermion Fock reference: ${reference.label}`, () => {
  assert(valid(reference.input)); const result = model(reference.input);
  compare(result, reference.expected); zeroCertificates(result.exactResiduals);
  assert.doesNotThrow(() => JSON.stringify(result)); assert.deepEqual(result.parameters, reference.input);
});

test('74 owned bounded fields reject ambiguous inputs without evaluating caller getters', () => {
  assert.equal(keys.length, 74); assert.equal(new Set(keys).size, 74);
  const p = { ...create() };
  for (const key of ['modes', 'hDenominator', 'stateReal0']) for (const value of [NaN, Infinity, 1.5, 1000001, '1', true, null]) assert(!valid({ ...p, [key]: value }));
  for (const patch of [{ modes: 0 }, { modes: 5 }, { sector: 3 }, { sector: -1 }, { hDenominator: 0 }, { leftMode: -1 }, { rightMode: 2 }, { leftCreation: 2 }, { stateReal0: 0 }, { stateReal1: 1 }, { stateReal4: 1 }, { onsite2: 1 }, { hoppingImag02: 1 }, { pairingReal02: 1 }, { density02: 1 }, { extra: 0 }]) assert(!valid({ ...p, ...patch }));
  assert(!valid(Object.assign(Object.create(null), p))); assert(!valid(Object.create(p))); assert(!valid({ ...p, [Symbol('extra')]: 0 }));
  let reads = 0; const getter = { ...p }; Object.defineProperty(getter, 'modes', { get() { reads++; return 2; } });
  assert(!valid(getter)); assert.throws(() => snapshot(getter)); assert.equal(reads, 0);
  const proxy = new Proxy(p, { get() { reads++; throw Error('Caller property access'); } });
  assert(valid(proxy)); assert.deepEqual(snapshot(proxy), p); assert.equal(reads, 0);
  assert(!valid(new Proxy({}, { getPrototypeOf() { throw Error('Reflection'); } })));
  const missing = { ...p }; delete missing.offset; assert(!valid(missing));
  p.stateImag0 = -0; const saved = snapshot(p); p.onsite0 = 7;
  assert(Object.isFrozen(saved)); assert.equal(saved.onsite0, 0); assert(!Object.is(saved.stateImag0, -0));
  for (let n = 1; n <= 4; n++) for (let sector = 0; sector < 3; sector++) assert(valid(create(n, sector)));
  for (const [n, sector] of [[0, 1], [5, 1], [2, 3], [NaN, 1]]) assert.throws(() => create(n, sector));
  for (const index of [-1, 16, NaN, Infinity, 1.5]) assert.throws(() => fermionOccupationCount(index));
});

test('CAR holds as an operator identity for every mode pair and creation choice', () => {
  for (let n = 1; n <= 4; n++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (const a of [0, 1]) for (const b of [0, 1]) {
    const r = model({ ...create(n), leftMode: i, rightMode: j, leftCreation: a, rightCreation: b });
    assert.equal(r.probe.carResidualFrobeniusSquared.sign, 0);
    assert.equal(r.probe.expectedAnticommutatorScalar, Number(i === j && a !== b));
    assert.equal(r.probe.naiveCarResidualFrobeniusSquared.value, i === j ? 0 : 2 ** n);
    for (const entry of r.car) for (const key of ['mixedResidualFrobeniusSquared', 'annihilationResidualFrobeniusSquared', 'creationResidualFrobeniusSquared'] as const) assert.equal(entry[key].sign, 0);
  }
});

test('ordered occupation signs and Pauli exclusion are visible in actual actions', () => {
  const r = model({ ...create(3) });
  assert.deepEqual(r.operators[2].annihilationColumns[3], { row: 2, sign: -1 });
  assert.deepEqual(r.operators[0].annihilationColumns[4], { row: 0, sign: 1 });
  const creation = model(input('creation order on vacuum'));
  assert.equal(creation.probe.forward[3].real.value, 1); assert.equal(creation.probe.reverse[3].real.value, -1);
  const excluded = model(input('same mode exclusion on vacuum'));
  assert(excluded.probe.forward.every(z => z.real.sign === 0 && z.imaginary.sign === 0));
  assert.equal(excluded.probe.forwardSquaredNormOnNormalizedInput.sign, 0);
  const long = model(input('long range hopping through occupied middle mode'));
  assert.equal(long.energy.value, -1); assert.equal(long.naiveComparison.energy.value, 1);
  assert.equal(long.probe.naiveCarResidualFrobeniusSquared.value, 8);
});

test('pairing adjoints reverse order and conserve parity while breaking number', () => {
  const pair = model(input('even pair plus eigenstate'));
  assert.equal(pair.energy.value, 1); assert.equal(pair.variance.sign, 0); assert.equal(pair.totalNumber.expectation.value, 1); assert.equal(pair.totalNumber.variance.value, 1);
  assert(pair.fermionParity.conserved); assert(!pair.totalNumber.conserved); assert(pair.parityAssumption.pureStateCompatibleWithParitySuperselection);
  assert.equal(pair.classification.selectedStateGround, null);
  const complex = model(input('complex pairing adjoint order'));
  assert.equal(complex.energy.value, 1); assert.equal(complex.anomalous[0][1].imaginary.value, -0.5);
  assert.equal(complex.terms.find(t => t.type === 'pairing')!.expectation.imaginary.value, -0.5);
  const sharp = model(input('sharp number is not number conservation')); assert(sharp.totalNumber.selectedStateSharp); assert(!sharp.totalNumber.conserved);
  const spread = model(input('number conservation does not imply sharp number')); assert(!spread.totalNumber.selectedStateSharp); assert(spread.totalNumber.conserved);
});

test('parity coherence is labelled and twirling is an explicit mixed-state comparison', () => {
  const formal = model(input('formal even odd coherent one mode'));
  assert(!formal.parityAssumption.pureStateCompatibleWithParitySuperselection);
  assert.equal(formal.parityAssumption.coherenceFrobeniusSquared.value, 0.5); assert.equal(formal.oddMeans[0].real.value, 0.5);
  assert.equal(formal.parityDephased.purity.value, 0.5); assert.deepEqual(formal.parityDephased.energy, formal.energy);
  assert(formal.parityDephased.oddMeans.every(z => z.real.sign === 0 && z.imaginary.sign === 0));
  assert.equal(formal.density[0][1].real.value, 0.5); assert.equal(formal.parityDephased.density[0][1].real.sign, 0);
  const cat = model(input('formal cat has zero first odd moments'));
  assert(cat.oddMeans.every(z => z.real.sign === 0 && z.imaginary.sign === 0));
  assert(!cat.parityAssumption.pureStateCompatibleWithParitySuperselection);
  assert(model({ ...create(2, 0) }).parityAssumption.pureStateCompatibleWithParitySuperselection);
});

test('one-body and anomalous matrices do not determine the number distribution', () => {
  const fixed = model(input('fixed two particles fractional gamma occupations')), cat = model(input('even cat same gamma different number distribution'));
  assert.deepEqual(fixed.gamma, cat.gamma); assert.deepEqual(fixed.anomalous, cat.anomalous);
  assert.equal(fixed.totalNumber.expectation.value, 2); assert.equal(cat.totalNumber.expectation.value, 2);
  assert.equal(fixed.totalNumber.variance.sign, 0); assert.equal(cat.totalNumber.variance.value, 4);
  assert.notDeepEqual(fixed.numberDistribution, cat.numberDistribution);
});

test('full Fock minima, parity minima and unavailable spectra remain distinct', () => {
  const even = model(input('even sector minimum differs from full Fock minimum'));
  assert.equal(even.diagonalSpectrum!.groundEnergy.value, -1); assert.equal(even.classification.selectedStateGround, false);
  assert.equal(even.classification.selectedStateParitySectorGround, true); assert.equal(even.diagonalSpectrum!.parityMinima[0].excitationGap, null);
  for (let n = 1; n <= 4; n++) {
    const r = model({ ...create(n), offset: -3 }); assert(r.diagonalSpectrum);
    assert.equal(r.diagonalSpectrum.groundMultiplicity, 2 ** n); assert.equal(r.diagonalSpectrum.excitationGap, null);
    assert(r.diagonalSpectrum.parityMinima.every(m => m.multiplicity === 2 ** (n - 1)));
    assert.equal(r.diagonalSpectrum.certifiedLeakageUpperBound, null);
  }
  for (const label of ['tiny positive hopping variance', 'tiny positive near occupation eigenstate variance']) {
    const r = model(input(label)); assert.equal(r.variance.sign, 1); assert(!r.classification.exactEigenstate);
  }
  const pair = model(input('even pair minus eigenstate')); assert(pair.classification.exactEigenstate); assert.equal(pair.diagonalSpectrum, null); assert.equal(pair.classification.selectedStateGround, null);
});

test('complex rescaling preserves observables and operator-action squared norms', () => {
  const p = input('general complex 4 modes sector0 case0'), changed = { ...p };
  for (let i = 0; i < 16; i++) { const a = p[`stateReal${i}`], b = p[`stateImag${i}`]; changed[`stateReal${i}`] = 2 * a - 3 * b; changed[`stateImag${i}`] = 3 * a + 2 * b; }
  const a = model(p), b = model(changed);
  for (const key of ['energy', 'variance', 'gamma', 'anomalous', 'oddMeans', 'totalNumber', 'fermionParity', 'parityAssumption', 'parityDephased'] as const) assert.deepEqual(a[key], b[key]);
  assert.deepEqual(a.probe.forwardSquaredNormOnNormalizedInput, b.probe.forwardSquaredNormOnNormalizedInput);
  assert(equal(exact(b.rawNormSquared), multiply(frac(BigInt(13)), exact(a.rawNormSquared))));
});

test('energy origins and signed Hamiltonian scaling have exact physical effects', () => {
  const p = input('general complex 4 modes sector1 case0'), a = model(p), shifted = model({ ...p, offset: p.offset + 7 });
  assert(equal(exact(shifted.energy), add(exact(a.energy), frac(BigInt(7), BigInt(p.hDenominator))))); assert.deepEqual(shifted.variance, a.variance);
  const changed = { ...p }; for (const key of keys) if (/^(offset|onsite|hopping|pairing|density)/.test(key)) changed[key] *= -2;
  const b = model(changed); assert(equal(exact(b.energy), multiply(frac(BigInt(-2)), exact(a.energy)))); assert(equal(exact(b.variance), multiply(frac(BigInt(4)), exact(a.variance))));
  assert(equal(exact(b.totalNumber.commutatorFrobeniusSquared), multiply(frac(BigInt(4)), exact(a.totalNumber.commutatorFrobeniusSquared))));
});

test('complex conjugation reverses all imaginary preparation and Hamiltonian phases', () => {
  const p = input('general complex 3 modes sector0 case0'), changed = { ...p };
  for (const key of keys) if (/^(stateImag|hoppingImag|pairingImag)/.test(key)) changed[key] *= -1;
  const a = model(p), b = model(changed); assert.deepEqual(a.energy, b.energy); assert.deepEqual(a.variance, b.variance);
  for (let i = 0; i < p.modes; i++) for (let j = 0; j < p.modes; j++) for (const key of ['gamma', 'anomalous'] as const) {
    assert.deepEqual(a[key][i][j].real, b[key][i][j].real);
    assert(equal(exact(b[key][i][j].imaginary), multiply(frac(BigInt(-1)), exact(a[key][i][j].imaginary))));
  }
});

test('fermionic mode permutations include occupied permutation signs and antisymmetric pairing', () => {
  const p = input('general complex 4 modes sector0 case0'), perm = [2, 0, 3, 1];
  const changed: P = { ...create(4, 0), hDenominator: p.hDenominator, offset: p.offset };
  for (let i = 0; i < 4; i++) {
    changed[`onsite${i}`] = p[`onsite${perm[i]}`];
    for (let j = i + 1; j < 4; j++) {
      const a = perm[i], b = perm[j], suffix = `${Math.min(a, b)}${Math.max(a, b)}`;
      changed[`hoppingReal${i}${j}`] = p[`hoppingReal${suffix}`]; changed[`hoppingImag${i}${j}`] = (a < b ? 1 : -1) * p[`hoppingImag${suffix}`];
      for (const part of ['Real', 'Imag']) changed[`pairing${part}${i}${j}`] = (a < b ? 1 : -1) * p[`pairing${part}${suffix}`];
      changed[`density${i}${j}`] = p[`density${suffix}`];
    }
  }
  for (let index = 0; index < 16; index++) {
    const occupied = perm.filter((_, i) => Boolean(index & (1 << (3 - i))));
    const oldIndex = occupied.reduce((s, i) => s + (1 << (3 - i)), 0);
    const inversions = occupied.reduce((s, v, i) => s + occupied.slice(i + 1).filter(w => v > w).length, 0), sign = inversions % 2 ? -1 : 1;
    changed[`stateReal${index}`] = sign * p[`stateReal${oldIndex}`]; changed[`stateImag${index}`] = sign * p[`stateImag${oldIndex}`];
  }
  const a = model(p), b = model(changed); assert.deepEqual(a.energy, b.energy); assert.deepEqual(a.variance, b.variance);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { assert.deepEqual(b.gamma[i][j], a.gamma[perm[i]][perm[j]]); assert.deepEqual(b.anomalous[i][j], a.anomalous[perm[i]][perm[j]]); }
});

test('two-mode complex fermion Hamiltonians agree with an independently assembled Pauli Hamiltonian', () => {
  for (const sector of [0, 1, 2]) {
    const p = input(`general complex 2 modes sector${sector} case0`);
    const s: SpinChainParameters = { ...createSpinChainParameters(2), hDenominator: 4 * p.hDenominator };
    s.offset = 4 * p.offset + 2 * p.onsite0 + 2 * p.onsite1 + p.density01;
    s.fieldZ0 = -2 * p.onsite0 - p.density01; s.fieldZ1 = -2 * p.onsite1 - p.density01;
    s.couplingXX01 = 2 * (p.hoppingReal01 + p.pairingReal01); s.couplingYY01 = 2 * (p.hoppingReal01 - p.pairingReal01);
    s.couplingXY01 = 2 * (p.pairingImag01 - p.hoppingImag01); s.couplingYX01 = 2 * (p.pairingImag01 + p.hoppingImag01); s.couplingZZ01 = p.density01;
    for (let i = 0; i < 4; i++) { s[`stateReal${i}`] = p[`stateReal${i}`]; s[`stateImag${i}`] = p[`stateImag${i}`]; }
    const f = model(p), spin = spinChainModel(s);
    assert.deepEqual(f.hamiltonian, spin.hamiltonian); assert.deepEqual(f.energy, spin.energy); assert.deepEqual(f.variance, spin.variance);
  }
});

after(() => {
  writeFileSync('.runtime/learning/fermion-fock-foundation/reference-observations.json', JSON.stringify({ cases: references.cases.length, exactChecks, directedChecks }, null, 2) + '\n');
});
