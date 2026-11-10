import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { finiteGibbsModel as model, type FiniteGibbsParameters, type GibbsKernelName } from '../src/lib/learning-finite-gibbs';
import type { PartitionObservation, PartitionRational } from '../src/lib/learning-partition-exact-core';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
interface Fixture { id: string; parameters: FiniteGibbsParameters; confirmationPrecisionDigits: number; reference: unknown }
const cases = (JSON.parse(readFileSync('scripts/fixtures/finite-gibbs-oracle.json', 'utf8')) as { cases: Fixture[] }).cases;
const names: GibbsKernelName[] = ['first', 'second', 'random', 'systematic', 'blocked'];
const Z = BigInt(0), O = BigInt(1);
interface Q { n: bigint; d: bigint }
const add = (a: Q, b: Q): Q => ({ n: a.n * b.d + b.n * a.d, d: a.d * b.d });
const cmp = (a: Q, b: Q) => a.n * b.d - b.n * a.d;
function decimal(s: string): Q {
  const m = /^([+-]?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s); assert.ok(m, s);
  const n = BigInt(`${m[1] === '-' ? '-' : ''}${m[2]}${m[3] ?? ''}`), e = Number(m[4] ?? 0) - (m[3]?.length ?? 0);
  return e >= 0 ? { n: n * BigInt(10) ** BigInt(e), d: O } : { n, d: BigInt(10) ** BigInt(-e) };
}
function decode(q: PartitionRational): Q {
  let n = BigInt(q.numerator.coefficient), d = BigInt(q.denominator.coefficient);
  const shift = q.numerator.exponent - q.denominator.exponent;
  if (shift >= 0) n <<= BigInt(shift); else d <<= BigInt(-shift);
  assert.ok(d > Z); return { n, d };
}
function check(q: PartitionObservation, reference: string, precision: number, label: string) {
  const truth = decimal(reference), sign = truth.n < Z ? -1 : truth.n > Z ? 1 : 0;
  assert.equal(q.sign, sign, label); assert.notEqual(q.resolution, 'unresolved-at-precision-ceiling', label);
  if (!truth.n) { assert.equal(q.value, 0, label); assert.equal(q.resolution, 'structural-zero', label); }
  else {
    const expected = Number(reference);
    if (!Number.isFinite(expected) || expected === 0) assert.equal(q.value, null, label);
    else { assert.notEqual(q.value, null, label); assert.ok(Math.abs(q.value! - expected) <= Math.max(2 * Number.MIN_VALUE, 1e-13 * Math.abs(expected)), `${label}: ${q.value} vs ${expected}`); }
  }
  assert.ok(q.bounds, label);
  const lo = decode(q.bounds.lower), hi = decode(q.bounds.upper); assert.ok(cmp(lo, hi) <= Z, label);
  const error = add({ n: truth.n < Z ? -truth.n : truth.n, d: truth.d * BigInt(10) ** BigInt(150) }, { n: O, d: BigInt(10) ** BigInt(precision - 110) });
  assert.ok(cmp(lo, add(truth, error)) <= Z && cmp(hi, add(truth, { n: -error.n, d: error.d })) >= Z, `${label}: independent Decimal reference outside directed enclosure`);
  if (q.resolution === 'interval-resolved') {
    assert.ok(lo.n > Z || hi.n < Z, label);
    const width = hi.n * lo.d - lo.n * hi.d, near = lo.n > Z ? lo.n * hi.d : -hi.n * lo.d;
    assert.ok((width << BigInt(48)) <= near, `${label}: relative width`);
  }
}
function compareReference(actual: unknown, reference: unknown, precision: number, label: string) {
  if (typeof reference === 'string') { check(actual as PartitionObservation, reference, precision, label); return; }
  if (Array.isArray(reference)) { assert.ok(Array.isArray(actual)); assert.equal(actual.length, reference.length, label); reference.forEach((r, i) => compareReference(actual[i], r, precision, `${label}[${i}]`)); return; }
  assert.ok(reference && typeof reference === 'object');
  for (const [key, value] of Object.entries(reference)) compareReference((actual as Record<string, unknown>)[key], value, precision, `${label}.${key}`);
}
for (const c of cases) test(`${c.id}: target, log law, five kernels, stationary flows and finite-step distributions match independent actual-input reference`, () => {
  const r = model(c.parameters); compareReference(r, c.reference, c.confirmationPrecisionDigits, c.id);
  assert.equal(r.fullyResolved, true);
  for (const name of names) {
    assert.equal(r.numericalChecks[name].rowSumsContainOne, true);
    assert.equal(r.numericalChecks[name].stationarityResidualsContainZero, true);
    assert.equal(r.diagnostics[name].targetStationary, true);
    assert.equal(r.diagnostics[name].rowStochastic, true);
    assert.equal(r.diagnostics[name].detailedBalance, name !== 'systematic' || c.parameters.couplingEnergy === 0 || c.parameters.support === 'fixed-zero-sum');
    if (c.parameters.support === 'fixed-zero-sum' && name !== 'blocked') {
      assert.equal(r.diagnostics[name].irreducible, false); assert.equal(r.diagnostics[name].uniqueStationaryDistribution, false);
    }
  }
  const visit = (v: unknown): void => {
    if (typeof v === 'number') assert.ok(Number.isFinite(v));
    else if (Array.isArray(v)) v.forEach(visit);
    else if (v && typeof v === 'object') Object.values(v).forEach(visit);
  };
  visit(r); assert.doesNotThrow(() => JSON.stringify(r));
});
const basic: FiniteGibbsParameters = { couplingEnergy: kB * Math.log(2), firstFieldEnergy: 0, secondFieldEnergy: 0, temperature: 1, energyOffset: 0, support: 'full-product' };
function close(a: number | null, b: number) { assert.notEqual(a, null); assert.ok(Math.abs(a! - b) <= 2e-12 * Math.max(Math.abs(b), Number.MIN_VALUE), `${a} vs ${b}`); }
const values = (matrix: PartitionObservation[][]) => matrix.map(row => row.map(q => q.value));
test('ideal-near input exposes stationarity without systematic detailed balance and respects row order K1 K2', () => {
  const r = model(basic); [2 / 5, 1 / 10, 1 / 10, 2 / 5].forEach((p, i) => close(r.target[i].value, p));
  close(r.kernels.systematic[0][1].value, 4 / 25); close(r.kernels.systematic[1][0].value, 4 / 25);
  close(r.flows.systematic[0][1].value, 8 / 125); close(r.flows.systematic[1][0].value, 2 / 125);
  close(r.systematicDetailedBalanceWitness.difference.value, 6 / 125);
  assert.equal(r.systematicDetailedBalanceWitness.provenSign, 1);
  close(r.flows.random[0][1].value, .04); close(r.flows.random[1][0].value, .04);
  assert.equal(r.diagnostics.systematic.detailedBalance, false); assert.equal(r.diagnostics.random.detailedBalance, true);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) close(r.kernels.systematic[i][j].value, r.kernels.first[i].reduce((s, q, k) => s + q.value! * r.kernels.second[k][j].value!, 0));
});
test('fixed-zero-sum single-coordinate kernels stay in isolated states while a joint block reaches the target', () => {
  const r = model({ ...basic, support: 'fixed-zero-sum' });
  for (const name of ['first', 'second', 'random', 'systematic'] as const) {
    assert.deepEqual(values(r.kernels[name]), [[1, 0], [0, 1]]);
    assert.deepEqual(values(r.distributions[name]), [[1, 0], [1, 0], [1, 0]]);
    assert.deepEqual(r.diagnostics[name].stronglyConnectedComponents, [[0], [1]]);
    assert.equal(r.diagnostics[name].convergesToTargetFromEveryInitialState, false);
    r.totalVariationToTarget[name].forEach(q => close(q.value, .5));
  }
  assert.deepEqual(values(r.kernels.blocked), [[.5, .5], [.5, .5]]);
  assert.equal(r.diagnostics.blocked.convergesToTargetFromEveryInitialState, true);
  r.totalVariationToTarget.blocked.forEach(q => assert.equal(q.resolution, 'structural-zero'));
});
test('independent sites make a systematic sweep the exact product refresh even with unequal fields', () => {
  const r = model({ ...basic, couplingEnergy: 0, firstFieldEnergy: .3 * kB, secondFieldEnergy: -.6 * kB });
  assert.deepEqual(values(r.kernels.systematic), values(r.kernels.blocked));
  r.totalVariationToTarget.systematic.forEach(q => assert.equal(q.resolution, 'structural-zero'));
  assert.equal(r.systematicDetailedBalanceWitness.difference.resolution, 'structural-zero');
  assert.equal(r.diagnostics.first.irreducible, false); assert.equal(r.diagnostics.random.irreducible, true);
});
test('random one-step and systematic two-coordinate sweep retain explicit different update counts', () => {
  const r = model(basic); assert.equal(r.diagnostics.random.coordinatesResampledPerStep, 1); assert.equal(r.diagnostics.systematic.coordinatesResampledPerStep, 2);
  assert.notDeepEqual(values(r.distributions.random), values(r.distributions.systematic));
});
test('fixed initial states and observation indices change actual distributions without changing the target', () => {
  const a = model(basic), b = model({ ...basic, initialStateIndex: 3, observationSteps: [1, 3, 5] });
  assert.deepEqual(a.target, b.target); assert.deepEqual(b.observationSteps, [1, 3, 5]);
  assert.deepEqual(b.distributions.systematic[0], b.kernels.systematic[3]);
  assert.notDeepEqual(a.distributions.random[0], b.distributions.random[0]);
});
test('common energy shifts and common energy/temperature scaling leave the update law unchanged', () => {
  const a = model(basic), b = model({ ...basic, energyOffset: 2 ** 60 }), c = model({ ...basic, couplingEnergy: 8 * basic.couplingEnergy, temperature: 8 });
  for (const r of [b, c]) {
    assert.deepEqual(a.target, r.target); for (const name of names) assert.deepEqual(a.kernels[name], r.kernels[name]);
  }
  assert.equal(new Set(b.states.map(s => s.energy.value)).size, 1); assert.notEqual(b.target[0].value, b.target[1].value);
});
test('nonzero rare edges preserve connectivity even if their displayed probability is null', () => {
  const r = model(cases.find(c => c.id === 'rare-conditional')!.parameters);
  assert.equal(r.kernels.first[0][2].value, null); assert.equal(r.kernels.first[0][2].sign, 1);
  assert.equal(r.kernels.first[0][2].resolution, 'interval-resolved');
  assert.equal(r.diagnostics.random.irreducible, true); assert.equal(r.diagnostics.systematic.irreducible, true);
  assert.equal(r.systematicDetailedBalanceWitness.difference.sign, 1);
});
test('precision ceiling reports unresolved rare transitions, with mathematical positivity and graph proofs intact', () => {
  const r = model(cases.find(c => c.id === 'rare-conditional')!.parameters, { maximumPrecision: 128 });
  assert.equal(r.fullyResolved, false); assert.equal(r.kernels.first[0][2].value, null);
  assert.equal(r.kernels.first[0][2].resolution, 'unresolved-at-precision-ceiling'); assert.equal(r.kernels.first[0][2].sign, 1);
  assert.equal(r.diagnostics.random.irreducible, true); assert.equal(r.diagnostics.systematic.detailedBalance, false);
});
test('a coupling invisible in displayed probabilities remains nonzero in the reversibility witness', () => {
  const r = model(cases.find(c => c.id === 'weak-interaction')!.parameters);
  assert.equal(new Set(r.target.map(q => q.value)).size, 1); assert.equal(r.diagnostics.systematic.detailedBalance, false);
  assert.equal(r.systematicDetailedBalanceWitness.difference.sign, 1); assert.ok(r.systematicDetailedBalanceWitness.difference.value! > 0);
});
test('extreme finite Hamiltonian sums preserve exact states outside the linear range without NaN energies', () => {
  const r = model({ ...basic, couplingEnergy: Number.MAX_VALUE, firstFieldEnergy: Number.MAX_VALUE, secondFieldEnergy: Number.MAX_VALUE, energyOffset: Number.MAX_VALUE }, { maximumPrecision: 128 });
  assert.ok(r.states.some(s => s.energy.value === null)); assert.doesNotMatch(JSON.stringify(r), /NaN|Infinity/);
});
test('zero and nonfinite temperatures, invalid supports, malformed observations and initial states are rejected', () => {
  const invalid = [{ ...basic, temperature: 0 }, { ...basic, temperature: -0 }, { ...basic, temperature: Infinity }, { ...basic, couplingEnergy: NaN },
    { ...basic, energyOffset: Infinity }, { ...basic, support: 'unknown' }, { ...basic, initialStateIndex: 4 }, { ...basic, initialStateIndex: .5 },
    { ...basic, support: 'fixed-zero-sum', initialStateIndex: 2 }, { ...basic, observationSteps: [] }, { ...basic, observationSteps: [1, 1] },
    { ...basic, observationSteps: [2, 1] }, { ...basic, observationSteps: [1, 65] }, { ...basic, observationSteps: Array(3) },
    { ...basic, observationSteps: [0] }, { ...basic, observationSteps: [1.5] }, { ...basic, observationSteps: { length: 1e9 } }];
  invalid.forEach(p => assert.throws(() => model(p as FiniteGibbsParameters), RangeError));
  assert.throws(() => model(basic, { maximumPrecision: 64 }), RangeError);
  assert.throws(() => model(basic, { minimumPrecision: 512, maximumPrecision: 128 }), RangeError);
});

test('systematic contraction preserves a positive distance far below both binary64 and the fixed-point quantum', () => {
  const r = model(cases.find(c => c.id === 'hot-subnormal-interaction')!.parameters);
  const q = r.totalVariationToTarget.systematic[2];
  assert.equal(q.value, null); assert.equal(q.sign, 1); assert.equal(q.resolution, 'interval-resolved');
  assert.ok(q.logMagnitude! < -9000); assert.equal(r.fullyResolved, true);
  assert.equal(r.systematicContraction.sign, 1); assert.equal(r.systematicContraction.resolution, 'interval-resolved');
});
test('ideal contraction and total variation match the directly iterated distributions', () => {
  const r = model(basic); close(r.systematicContraction.value, 9 / 25);
  for (const [i, t] of r.observationSteps.entries()) {
    const direct = r.distributions.systematic[i].reduce((s, q, j) => s + Math.abs(q.value! - r.target[j].value!), 0) / 2;
    close(r.totalVariationToTarget.systematic[i].value, direct);
    close(r.totalVariationToTarget.systematic[i].value, .3 * (9 / 25) ** (t - 1));
  }
});

test('long sweeps retain the exact contraction law and immediately forget the first starting coordinate', () => {
  const a = model({ ...basic, observationSteps: [1, 64] }), b = model({ ...basic, observationSteps: [1, 64], initialStateIndex: 2 });
  assert.deepEqual(values(a.distributions.systematic), values(b.distributions.systematic));
  close(a.totalVariationToTarget.systematic[1].value, .3 * (9 / 25) ** 63);
  assert.equal(a.totalVariationToTarget.systematic[1].sign, 1); assert.equal(a.fullyResolved, true);
  assert.equal(a.totalVariationToTarget.systematic[1].resolution, 'interval-resolved');
});
