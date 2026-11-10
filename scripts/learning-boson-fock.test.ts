import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { bosonFockModel as model, createBosonFockParameters as create, snapshotBosonFockParameters as snapshot, isBosonFockParameters as valid, bosonFockParameterKeys as keys, type BosonFockParameters as P } from '../src/lib/learning-boson-fock';
import { f } from '../src/lib/learning-quantum-hilbert-core';
import { bReal, bOne, bZero, bObserve, bSign, bAdd, bSub, bMul, bDiv, bEqual, bComplex, bNormalized, type BosonReal } from '../src/lib/learning-boson-algebraic';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
ensureTestArtifactDirectory('.runtime/learning/boson-fock-foundation', { recursive: true });


const references = JSON.parse(readFileSync('scripts/fixtures/boson-fock-reference.json', 'utf8')) as { cases: { label: string; input: P; expected: unknown }[] };
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

for (const reference of references.cases) test(`independent boson Fock reference: ${reference.label}`, () => {
  assert(valid(reference.input)); const result = model(reference.input);
  compare(result, reference.expected); zeroCertificates(result.exactResiduals);
  assert.doesNotThrow(() => JSON.stringify(result)); assert.deepEqual(result.parameters, reference.input);
});
const fields = JSON.parse(readFileSync('scripts/fixtures/boson-algebraic-reference.json', 'utf8')) as { cases: { label: string; a: string[]; b: string[]; expected: unknown }[] };
const field = (raw: string[]) => raw.map(v => { const x = parse(v); return f(x.n, x.d); }) as unknown as BosonReal;
for (const reference of fields.cases) test(`independent Q(sqrt2,sqrt3) reference: ${reference.label}`, () => {
  const a = field(reference.a), b = field(reference.b);
  compare({ a: bObserve(a), b: bObserve(b), add: bObserve(bAdd(a, b)), subtract: bObserve(bSub(a, b)), multiply: bObserve(bMul(a, b)), divide: bObserve(bDiv(a, b)) }, reference.expected);
  assert(bEqual(bMul(bDiv(a, b), b), a)); assert(bEqual(bSub(a, a), bZero));
});

test('52 owned bounded fields reject invalid cutoffs, inactive coordinates and caller getters', () => {
  assert.equal(keys.length, 52); assert.equal(new Set(keys).size, 52);
  const original = { ...create() };
  for (const key of ['modes', 'cutoff0', 'hDenominator', 'stateReal0']) for (const value of [NaN, Infinity, 1.5, 1000001, '1', true, null]) assert(!valid({ ...original, [key]: value }));
  for (const patch of [{ modes: 0 }, { modes: 3 }, { cutoff0: -1 }, { cutoff0: 4 }, { cutoff1: 1 }, { hDenominator: 0 }, { stateReal0: 0 }, { stateReal4: 1 }, { onsite1: 1 }, { hoppingReal: 1 }, { driveImag1: 1 }, { crossDensity: 1 }, { leftMode: -1 }, { rightMode: 1 }, { rightCreation: 2 }, { extra: 0 }]) assert(!valid({ ...original, ...patch }));
  assert(!valid(Object.assign(Object.create(null), original))); assert(!valid(Object.create(original))); assert(!valid({ ...original, [Symbol('extra')]: 0 }));
  let reads = 0; const getter = { ...original }; Object.defineProperty(getter, 'cutoff0', { get() { reads++; return 3; } }); assert(!valid(getter)); assert.throws(() => snapshot(getter)); assert.equal(reads, 0);
  const proxy = new Proxy(original, { get() { reads++; throw Error('Caller property read'); } }); assert(valid(proxy)); assert.deepEqual(snapshot(proxy), original); assert.equal(reads, 0);
  assert(!valid(new Proxy({}, { getPrototypeOf() { throw Error('Reflection'); } })));
  const missing = { ...original }; delete missing.offset; assert(!valid(missing));
  original.stateImag0 = -0; const saved = snapshot(original); original.onsite0 = 7;
  assert(Object.isFrozen(saved)); assert.equal(saved.onsite0, 0); assert(!Object.is(saved.stateImag0, -0));
  for (let c0 = 0; c0 <= 3; c0++) { assert(valid(create(1, c0))); for (let c1 = 0; c1 <= 3; c1++) assert(valid(create(2, c0, c1))); }
  for (const args of [[0, 1, 0], [3, 1, 0], [1, 4, 0], [1, 1, 1], [2, 1, -1]]) assert.throws(() => create(...args));
});

test('the exact field retains all four coordinates and resolves near cancellation signs', () => {
  const root2 = bReal(f(BigInt(0)), f(BigInt(1))), root3 = bReal(f(BigInt(0)), f(BigInt(0)), f(BigInt(1)));
  assert(bEqual(bMul(root2, root2), bReal(f(BigInt(2))))); assert(bEqual(bMul(root3, root3), bReal(f(BigInt(3)))));
  assert(bEqual(bMul(root2, root3), bReal(f(BigInt(0)), f(BigInt(0)), f(BigInt(0)), f(BigInt(1)))));
  assert.equal(bSign(bSub(root2, root3)), -1); assert.equal(bSign(bSub(root3, root2)), 1);
  assert.throws(() => bDiv(bOne, bZero)); assert.throws(() => bNormalized(bComplex(bOne), f(BigInt(0))));
  for (const r of fields.cases.filter(c => c.label.includes('cancellation'))) assert.equal(bSign(field(r.a)), (r.expected as { a: { exactSign: number } }).a.exactSign);
});

test('every cutoff has the projected boundary identity and zero commutator trace', () => {
  for (let n = 0; n <= 3; n++) {
    const r = model(create(1, n)), comm = r.commutation[0];
    assert.equal(comm.projectedIdentityResidualFrobeniusSquared.exactSign, 0);
    assert.equal(comm.infiniteCcrResidualFrobeniusSquared.observation.value, (n + 1) ** 2);
    assert.equal(comm.trace.real.exactSign, 0); assert.equal(comm.trace.imaginary.exactSign, 0);
    assert.equal(r.localStates[0].finiteCommutatorExpectation.observation.value, n === 0 ? 0 : 1);
    const p: P = { ...create(1, n), stateReal0: 0, [`stateReal${n}`]: 1 };
    const top = model(p); assert.equal(top.localStates[0].finiteCommutatorExpectation.observation.value, n === 0 ? 0 : -n);
    assert.equal(top.probe.reverseSquaredNormOnNormalizedInput.observation.value, n ** 2);
    assert.equal(top.probe.forwardSquaredNormOnNormalizedInput.exactSign, 0);
  }
});

test('ladder factors are sqrt2 and sqrt3, with bosonic hopping enhancement', () => {
  const two = model(input('number1 raises with sqrt2')), three = model(input('number2 raises with sqrt3'));
  assert.equal(two.operators[0].annihilation[1][2].real.coefficients[1].value, 1);
  assert.equal(three.operators[0].annihilation[2][3].real.coefficients[2].value, 1);
  const hopping = model(input('boson hopping ladder enhancement sqrt6'));
  assert.equal(hopping.secondMoment.observation.value, 6); assert.equal(hopping.variance.observation.value, 6);
  assert.equal(hopping.hamiltonianAction[10].real.coefficients[3].value, 1);
  const r = model({ ...create(2, 3, 3), leftMode: 0, rightMode: 1, leftCreation: 0, rightCreation: 1 });
  assert(r.probe.commutator.flat().every(v => v.real.exactSign === 0 && v.imaginary.exactSign === 0));
});

test('finite quadratures obey the actual Robertson bound rather than a forced infinite-space constant', () => {
  const zero = model(input('cutoff0 has vanishing quadratures')).localStates[0].quadratures;
  assert.equal(zero.xVariance.exactSign, 0); assert.equal(zero.pVariance.exactSign, 0); assert.equal(zero.robertsonBound.exactSign, 0);
  const vacuum = model(input('vacuum cutoff3 uncertainty')).localStates[0].quadratures;
  assert.equal(vacuum.xVariance.observation.value, 0.5); assert.equal(vacuum.pVariance.observation.value, 0.5); assert.equal(vacuum.uncertaintySlack.exactSign, 0);
  const top = model(input('top occupation reverses finite commutator')).localStates[0].quadratures;
  assert.equal(top.robertsonBound.observation.value, 2.25); assert.equal(top.uncertaintySlack.exactSign, 0);
  const cat = model(input('vacuum top superposition finite commutator minus1')).localStates[0]; assert.equal(cat.finiteCommutatorExpectation.observation.value, -1);
  const phase = model(input('complex drive and p orientation')).localStates[0].quadratures;
  assert.equal(phase.xMean.exactSign, 0); assert.equal(phase.pMean.exactSign, 1);
});

test('drive breaks number conservation without automatically evolving the supplied state', () => {
  const vacuum = model({ ...create(), driveReal0: 1 }); assert(vacuum.totalNumber.selectedStateSharp); assert(!vacuum.totalNumber.conserved); assert.equal(vacuum.totalNumber.expectation.exactSign, 0);
  const hop = model({ ...create(2, 3, 3), hoppingReal: 1 }); assert(hop.totalNumber.conserved);
  const frozen = model({ ...create(1, 0), driveReal0: 1 }); assert(frozen.totalNumber.conserved);
});

test('finite projected ground certificates do not claim infinite-space stability', () => {
  const falling = model(input('negative onsite finite ground has no infinite ground certificate'));
  assert.equal(falling.diagonalSpectrum!.groundEnergy.observation.value, -3); assert(falling.classification.selectedStateFiniteCutoffGround);
  assert.equal(falling.diagonalSpectrum!.scope, 'finite-projected-Fock-space-only');
  const constant = model({ ...create(2, 3, 2), offset: -2 });
  assert.equal(constant.diagonalSpectrum!.groundMultiplicity, 12); assert.equal(constant.diagonalSpectrum!.excitationGap, null);
  const phase = model(input('imaginary hopping phase orientation')); assert(phase.classification.exactEigenstate); assert.equal(phase.classification.selectedStateFiniteCutoffGround, null);
  const tiny = model(input('tiny positive number variance')); assert.equal(tiny.variance.exactSign, 1); assert(!tiny.classification.exactEigenstate);
});

test('complex preparation rescaling preserves finite physical observations', () => {
  const p = input('general complex 2 modes cutoff3-3'), changed = { ...p };
  for (let i = 0; i < 16; i++) { const a = p[`stateReal${i}`], b = p[`stateImag${i}`]; changed[`stateReal${i}`] = 2 * a - 3 * b; changed[`stateImag${i}`] = 3 * a + 2 * b; }
  const a = model(p), b = model(changed);
  for (const key of ['energy', 'variance', 'localStates', 'gamma', 'totalNumber', 'numberDistribution'] as const) assert.deepEqual(a[key], b[key]);
  assert.deepEqual(a.probe.forwardSquaredNormOnNormalizedInput, b.probe.forwardSquaredNormOnNormalizedInput);
});

test('energy-origin changes and complex conjugation preserve the appropriate observables', () => {
  const p = input('general complex 2 modes cutoff2-3'), a = model(p), shifted = model({ ...p, offset: p.offset + 7 });
  assert.deepEqual(shifted.variance, a.variance);
  assert(bEqual(field(shifted.energy.coefficients.map(v => { const x = exact(v); return `${x.n}/${x.d}`; })), bAdd(field(a.energy.coefficients.map(v => { const x = exact(v); return `${x.n}/${x.d}`; })), bReal(f(BigInt(7), BigInt(p.hDenominator))))));
  const changed = { ...p }; for (const key of keys) if (/^(stateImag|driveImag|hoppingImag)/.test(key)) changed[key] *= -1;
  const b = model(changed); assert.deepEqual(a.energy, b.energy); assert.deepEqual(a.variance, b.variance);
  for (let i = 0; i < p.modes; i++) { assert.deepEqual(a.localStates[i].quadratures.xMean, b.localStates[i].quadratures.xMean); assert.equal(a.localStates[i].quadratures.pMean.exactSign, -b.localStates[i].quadratures.pMean.exactSign || 0); }
});

test('swapping unequal-cutoff boson modes conjugates hopping and relabels ordinary tensor factors', () => {
  const p = input('general complex 2 modes cutoff2-3'), changed: P = { ...create(2, 3, 2), hDenominator: p.hDenominator, offset: p.offset, crossDensity: p.crossDensity, hoppingReal: p.hoppingReal, hoppingImag: -p.hoppingImag };
  for (const prefix of ['onsite', 'selfDensity', 'driveReal', 'driveImag']) { changed[`${prefix}0`] = p[`${prefix}1`]; changed[`${prefix}1`] = p[`${prefix}0`]; }
  for (let i = 0; i <= p.cutoff0; i++) for (let j = 0; j <= p.cutoff1; j++) for (const part of ['Real', 'Imag']) changed[`state${part}${j * (p.cutoff0 + 1) + i}`] = p[`state${part}${i * (p.cutoff1 + 1) + j}`];
  const a = model(p), b = model(changed); assert.deepEqual(a.energy, b.energy); assert.deepEqual(a.variance, b.variance);
  assert.deepEqual(a.localStates[0].density, b.localStates[1].density); assert.deepEqual(a.localStates[1].density, b.localStates[0].density);
});

after(() => writeFileSync('.runtime/learning/boson-fock-foundation/reference-observations.json', JSON.stringify({ cases: references.cases.length, algebraicCases: fields.cases.length, exactChecks, directedChecks }, null, 2) + '\n'));
