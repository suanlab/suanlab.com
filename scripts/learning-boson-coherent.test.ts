import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { bosonCoherentModel as model, createBosonCoherentParameters as create, snapshotBosonCoherentParameters as snapshot, isBosonCoherentParameters as valid, bosonCoherentParameterKeys as keys, type BosonCoherentParameters as P } from '../src/lib/learning-boson-coherent';
import { bosonFockModel, createBosonFockParameters, evaluateBosonFockPreparedState } from '../src/lib/learning-boson-fock';
import { bComplex, bOne, bZero } from '../src/lib/learning-boson-algebraic';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
ensureTestArtifactDirectory('.runtime/learning/boson-coherent-foundation', { recursive: true });


const references = JSON.parse(readFileSync('scripts/fixtures/boson-coherent-reference.json', 'utf8')) as { cases: { label: string; input: P; expected: unknown }[] };
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
  if (typeof value === 'number') { assert.equal(value, 0); return; }
  assert(value && typeof value === 'object'); Object.values(value).forEach(zeroCertificates);
}
const input = (label: string) => { const r = references.cases.find(r => r.label === label); assert(r, label); return structuredClone(r.input); };

const numeric = (value: Q) => { assert(value.value !== null); return value.value; };
const containsZero = (value: Q) => { assert(value.bounds); assert(le(endpoint(value.bounds.lower), frac(BigInt(0)))); assert(le(frac(BigInt(0)), endpoint(value.bounds.upper))); };
for (const reference of references.cases) test(`independent coherent Fock reference: ${reference.label}`, () => {
  assert(valid(reference.input)); const result = model(reference.input);
  compare(result, reference.expected); zeroCertificates(result.exactResiduals); zeroCertificates(result.normalizedProjection.exactResiduals);
  result.perMode.forEach(v => zeroCertificates(v.exactResiduals));
  containsZero(result.truncation.jointMinusProductRetained);
  assert.doesNotThrow(() => JSON.stringify(result)); assert.deepEqual(result.parameters, reference.input);
  assert(!Object.hasOwn(result.normalizedProjection, 'parameters'), 'derived finite state must not advertise fabricated Gaussian controls');
});

test('26 owned bounded coherent fields enforce exact mu and inactive-mode conditions', () => {
  assert.equal(keys.length, 26); assert.equal(new Set(keys).size, 26); const p = { ...create() };
  for (const key of ['modes', 'alphaReal0', 'alphaDenominator0']) for (const value of [NaN, Infinity, 1.5, 1000001, '1', true, null]) assert(!valid({ ...p, [key]: value }));
  for (const patch of [{ modes: 0 }, { modes: 3 }, { cutoff0: -1 }, { cutoff0: 4 }, { cutoff1: 1 }, { hDenominator: 0 }, { alphaDenominator0: 0 }, { alphaReal0: 5 }, { alphaImag1: 1 }, { alphaDenominator1: 2 }, { driveReal1: 1 }, { leftMode: 1 }, { rightCreation: 2 }, { stateReal0: 1 }]) assert(!valid({ ...p, ...patch }));
  assert(valid({ ...p, alphaReal0: 1000000, alphaDenominator0: 250000 }));
  assert(!valid({ ...p, alphaReal0: 1000000, alphaImag0: 1, alphaDenominator0: 250000 }));
  assert(valid({ ...p, alphaReal0: 0, alphaImag0: 0 })); assert(valid(create(2, 0, 3)));
  assert(!valid(Object.assign(Object.create(null), p))); assert(!valid(Object.create(p))); assert(!valid({ ...p, [Symbol('extra')]: 0 }));
  let reads = 0; const getter = { ...p }; Object.defineProperty(getter, 'alphaReal0', { get() { reads++; return 1; } }); assert(!valid(getter)); assert.throws(() => snapshot(getter)); assert.equal(reads, 0);
  const proxy = new Proxy(p, { get() { reads++; throw Error('Caller property read'); } }); assert(valid(proxy)); assert.deepEqual(snapshot(proxy), p); assert.equal(reads, 0);
  assert(!valid(new Proxy({}, { getPrototypeOf() { throw Error('Reflection'); } })));
  const missing = { ...p }; delete missing.alphaReal0; assert(!valid(missing));
  p.alphaImag0 = -0; const saved = snapshot(p); p.alphaReal0 = 2;
  assert(Object.isFrozen(saved)); assert.equal(saved.alphaReal0, 1); assert(!Object.is(saved.alphaImag0, -0));
});

test('vacuum is unchanged by every cutoff with zero discarded probability', () => {
  for (let cutoff = 0; cutoff <= 3; cutoff++) {
    const r = model(input(`vacuum coherent cutoff${cutoff}`));
    assert.equal(r.truncation.retainedProbability.value, 1); assert.equal(r.truncation.discardedProbability.sign, 0);
    assert.equal(r.unnormalizedProjection.normSquared.value, 1); assert.equal(r.normalizedProjection.distanceSquaredToInfinite.sign, 0);
    assert.equal(r.perMode[0].normalizedProjection.annihilationEigenResidualSquared.exactSign, 0);
  }
});

test('the coherent finite coefficients keep factorial roots exactly', () => {
  const r = model(input('unit real alpha cutoff3'));
  assert.equal(r.rawSeriesNorm.value, 8 / 3);
  assert.equal(r.exactRawCoefficients[2].real.coefficients[1].value, 0.5);
  assert.equal(r.exactRawCoefficients[3].real.coefficients[3].value, 1 / 6);
  assert.equal(r.perMode[0].normalizedProjection.topProbability.value, 1 / 16);
  assert.equal(r.perMode[0].normalizedProjection.finiteCommutatorExpectation.value, 0.75);
  assert.equal(r.perMode[0].normalizedProjection.annihilationEigenResidualSquared.observation.value, 1 / 16);
  assert.equal(r.normalizedProjection.totalNumber.expectation.observation.value, 15 / 16);
});

test('cutoff zero is normalized vacuum but differs from nonzero infinite coherence', () => {
  const p = { ...create(1, 0), alphaReal0: 1 }, r = model(p);
  assert.equal(r.normalizedProjection.normalizedState[0].real.value, 1);
  assert.equal(r.normalizedProjection.totalNumber.expectation.exactSign, 0);
  assert.equal(r.infiniteState.meanTotalNumber.value, 1);
  assert.equal(r.perMode[0].normalizedProjection.annihilationEigenResidualSquared.observation.value, 1);
  assert(numeric(r.truncation.discardedProbability) > 0.63);
  assert.equal(r.perMode[0].normalizedProjection.finiteCommutatorExpectation.sign, 0);
});

test('projection error, normalized distance and fidelity have distinct meanings', () => {
  const r = model(input('unit real alpha cutoff3'));
  assert.deepEqual(r.unnormalizedProjection.distanceSquaredToInfinite, r.truncation.discardedProbability);
  assert.deepEqual(r.normalizedProjection.fidelitySquaredToInfinite, r.truncation.retainedProbability);
  assert.notEqual(r.normalizedProjection.distanceSquaredToInfinite.value, r.truncation.discardedProbability.value);
  assert.equal(r.unnormalizedProjection.normalizationApplied, false); assert.equal(r.normalizedProjection.normalizationApplied, true);
  assert.equal(r.normalizedProjection.rawNormSquared.observation.value, 8 / 3);
  assert(numeric(r.unnormalizedProjection.normSquared) < 1);
});

test('tiny positive Poisson tails stay positive and resolve the incomplete gamma reference', () => {
  for (const label of ['tiny mu positive Poisson tail', 'two tiny modes retain positive joint tail']) {
    const r = model(input(label)), tail = r.truncation.discardedProbability;
    assert.equal(tail.sign, 1); assert(tail.bounds); assert(endpoint(tail.bounds.lower).n > BigInt(0));
    assert(numeric(tail) > 0 && numeric(tail) < 1e-48);
  }
});

test('increasing a cutoff retains more probability without silently altering the infinite input', () => {
  let previous = 1;
  for (let cutoff = 0; cutoff <= 3; cutoff++) {
    const r = model({ ...create(1, cutoff), alphaReal0: 2, alphaImag0: 1, alphaDenominator0: 2 });
    assert.equal(r.totalMu.value, 1.25); assert(numeric(r.truncation.discardedProbability) < previous); previous = numeric(r.truncation.discardedProbability);
    const weights = r.perMode[0].normalizedProjection.probabilities;
    assert(equal(weights.map(exact).reduce(add), frac(BigInt(1))));
  }
});

test('two-mode retention is rectangular and uses a product, not an additive tail correction', () => {
  const r = model(input('maximum total mu32 unequal cutoffs'));
  assert.equal(r.totalMu.value, 32); assert.deepEqual(r.cutoffs, [3, 2]);
  const [a, b] = r.perMode.map(m => numeric(m.unnormalizedProjection.retainedProbability));
  assert(Math.abs(numeric(r.truncation.retainedProbability) - a * b) < 1e-18);
  assert(numeric(r.truncation.retainedProbability) > 0); assert(numeric(r.truncation.discardedProbability) > 0.999);
  assert(r.normalizedProjection.localStates.every(s => s.purity.observation.value === 1));
});

test('rational alpha rescaling changes its representation while preserving the physical state', () => {
  const p = input('general coherent 2 modes cutoff3-3'), changed = { ...p };
  for (let i = 0; i < 2; i++) for (const part of ['Real', 'Imag', 'Denominator']) changed[`alpha${part}${i}`] *= 2;
  const { parameters: original, ...a } = model(p), { parameters: rescaled, ...b } = model(changed);
  assert.notDeepEqual(original, rescaled); assert.deepEqual(a, b);
});

test('complex conjugation preserves energy and reverses both coherent phase and quadrature orientation', () => {
  const p = input('general coherent 2 modes cutoff2-3'), changed = { ...p };
  for (const key of keys) if (/^(alphaImag|driveImag|hoppingImag)/.test(key)) changed[key] *= -1;
  const a = model(p), b = model(changed); assert.deepEqual(a.normalizedProjection.energy, b.normalizedProjection.energy); assert.deepEqual(a.normalizedProjection.variance, b.normalizedProjection.variance);
  assert.deepEqual(a.infiniteState.energyForSameNormallyOrderedHamiltonian, b.infiniteState.energyForSameNormallyOrderedHamiltonian);
  assert.deepEqual(a.truncation, b.truncation);
  for (let i = 0; i < p.modes; i++) { assert.deepEqual(a.normalizedProjection.localStates[i].quadratures.xMean, b.normalizedProjection.localStates[i].quadratures.xMean); assert.equal(b.normalizedProjection.localStates[i].quadratures.pMean.exactSign, -a.normalizedProjection.localStates[i].quadratures.pMean.exactSign || 0); }
});

test('a one-level coherent projection agrees with an equivalent Gaussian-integer raw preparation', () => {
  const p: P = { ...create(1, 1), alphaReal0: 2, alphaImag0: -3, alphaDenominator0: 5, onsite0: 2, driveReal0: 3, driveImag0: -1, hDenominator: 7 };
  const coherent = model(p).normalizedProjection;
  const raw = bosonFockModel({ ...createBosonFockParameters(1, 1), stateReal0: 5, stateReal1: 2, stateImag1: -3, onsite0: 2, driveReal0: 3, driveImag0: -1, hDenominator: 7 });
  for (const key of ['hamiltonian', 'energy', 'variance', 'density', 'gamma', 'totalNumber', 'localStates'] as const) assert.deepEqual(coherent[key], raw[key]);
});

test('infinite normally ordered energy differs from cutoff energy without implying a ground certificate', () => {
  const r = model({ ...create(1, 0), onsite0: -1 });
  assert.equal(r.infiniteState.energyForSameNormallyOrderedHamiltonian.observation.value, -1);
  assert.equal(r.normalizedProjection.energy.exactSign, 0); assert(r.infiniteState.noGroundCertificate);
  assert.equal(r.normalizedProjection.diagonalSpectrum!.scope, 'finite-projected-Fock-space-only');
});

test('the internal prepared-state bridge rejects zero, mismatched, nonrational-norm and invalid coordinates', () => {
  const p = createBosonFockParameters(1, 0);
  assert.throws(() => evaluateBosonFockPreparedState(p, [])); assert.throws(() => evaluateBosonFockPreparedState(p, [bComplex(bZero)]));
  assert.throws(() => evaluateBosonFockPreparedState(p, [bComplex([bOne[0], bOne[0], bZero[0], bZero[0]])]));
  assert.throws(() => evaluateBosonFockPreparedState(p, [bComplex([{ n: BigInt(1), d: BigInt(0) }, bZero[0], bZero[0], bZero[0]])]));
  assert.throws(() => evaluateBosonFockPreparedState({ ...p, cutoff0: 4 }, [bComplex(bOne)]));
});

after(() => writeFileSync('.runtime/learning/boson-coherent-foundation/reference-observations.json', JSON.stringify({ cases: references.cases.length, exactChecks, directedChecks }, null, 2) + '\n'));
