import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { finiteCurieWeissModel as model, type CurieWeissParameters } from '../src/lib/learning-curie-weiss';
import type { PartitionObservation, PartitionRational } from '../src/lib/learning-partition-exact-core';
interface Fixture { id: string; parameters: CurieWeissParameters; confirmationPrecisionDigits: number; reference: Record<string, string | string[] | number[]> }
const cases = (JSON.parse(readFileSync('scripts/fixtures/curie-weiss-oracle.json', 'utf8')) as { cases: Fixture[] }).cases;
const saved = new Map<string, ReturnType<typeof model>>();
function evaluate(p: CurieWeissParameters) { const key = JSON.stringify(p), old = saved.get(key); if (old) return old; const r = model(p); saved.set(key, r); return r; }
const Z = BigInt(0), O = BigInt(1);
interface Q { n: bigint; d: bigint }
const add = (a: Q, b: Q): Q => ({ n: a.n * b.d + b.n * a.d, d: a.d * b.d });
const cmp = (a: Q, b: Q) => a.n * b.d - b.n * a.d;
function decimal(s: string): Q {
  const match = /^([+-]?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s); assert.ok(match, s);
  const n = BigInt(`${match[1] === '-' ? '-' : ''}${match[2]}${match[3] ?? ''}`), exponent = Number(match[4] ?? 0) - (match[3]?.length ?? 0);
  return exponent >= 0 ? { n: n * BigInt(10) ** BigInt(exponent), d: O } : { n, d: BigInt(10) ** BigInt(-exponent) };
}
function decode(q: PartitionRational): Q {
  let n = BigInt(q.numerator.coefficient), d = BigInt(q.denominator.coefficient);
  const shift = q.numerator.exponent - q.denominator.exponent;
  if (shift >= 0) n <<= BigInt(shift); else d <<= BigInt(-shift);
  assert.ok(d > Z); return { n, d };
}
function check(q: PartitionObservation, reference: string, precision: number, label: string) {
  if (reference.endsWith('infinity')) {
    assert.equal(q.resolution, 'mathematical-infinity', label); assert.equal(q.sign, reference[0] === '-' ? -1 : 1); return;
  }
  assert.notEqual(q.resolution, 'unresolved-at-precision-ceiling', label);
  const truth = decimal(reference), truthSign = truth.n < Z ? -1 : truth.n > Z ? 1 : 0;
  assert.equal(q.sign, truthSign, label);
  if (!truth.n) { assert.equal(q.value, 0, label); assert.equal(q.resolution, 'structural-zero', label); }
  else {
    const expected = Number(reference);
    if (expected === 0 || !Number.isFinite(expected)) assert.equal(q.value, null, label);
    else {
      assert.notEqual(q.value, null, label);
      assert.ok(Math.abs(q.value! - expected) <= Math.max(2 * Number.MIN_VALUE, 1e-13 * Math.abs(expected)), `${label}: ${q.value} vs ${expected}`);
    }
  }
  assert.ok(q.bounds, label);
  const lower = decode(q.bounds.lower), upper = decode(q.bounds.upper);
  assert.ok(cmp(lower, upper) <= Z, label);
  // The oracle is independently recomputed at 400/900 and +80 Decimal digits.
  // Allow 150 relative decimal digits plus a much smaller absolute rounding budget.
  const error = add({ n: truth.n < Z ? -truth.n : truth.n, d: truth.d * BigInt(10) ** BigInt(150) }, { n: O, d: BigInt(10) ** BigInt(precision - 110) });
  assert.ok(cmp(lower, add(truth, error)) <= Z && cmp(upper, add(truth, { n: -error.n, d: error.d })) >= Z, `${label}: independent reference must lie in directed bounds within its stated precision`);
  if (q.resolution === 'interval-resolved') {
    assert.ok(lower.n > Z || upper.n < Z, label);
    const width = upper.n * lower.d - lower.n * upper.d;
    const near = lower.n > Z ? lower.n * upper.d : -upper.n * lower.d;
    assert.ok((width << BigInt(48)) <= near, `${label}: relative width`);
  }
}
for (const c of cases) test(`${c.id}: actual binary64 Decimal reference, signs, modes and directed bounds`, () => {
  const r = evaluate(c.parameters); assert.equal(r.fullyResolved, true);
  assert.equal(r.modes.certified, true); assert.deepEqual(r.modes.candidateTotalMagnetizations, c.reference.modes);
  assert.deepEqual(r.states.map(s => s.multiplicity), c.reference.multiplicities);
  for (const [key, value] of Object.entries(c.reference)) {
    if (key === 'modes' || key === 'multiplicities') continue;
    if (key === 'probabilities' || key === 'logProbabilities') {
      for (const [i, reference] of (value as string[]).entries()) check(r.states[i][key === 'probabilities' ? 'probability' : 'logProbability'], reference, c.confirmationPrecisionDigits, `${c.id}.${key}[${i}]`);
    } else check((r as unknown as Record<string, PartitionObservation>)[key], value as string, c.confirmationPrecisionDigits, `${c.id}.${key}`);
  }
  const visit = (v: unknown): void => {
    if (typeof v === 'number') assert.ok(Number.isFinite(v));
    else if (Array.isArray(v)) v.forEach(visit);
    else if (v && typeof v === 'object') Object.values(v).forEach(visit);
  };
  visit(r); assert.doesNotThrow(() => JSON.stringify(r));
});
const near = (a: number | null, b: number, tolerance = 2e-12) => { assert.notEqual(a, null); assert.ok(Math.abs(a! - b) <= tolerance * Math.max(1, Math.abs(b)), `${a} vs ${b}`); };
test('binomial multiplicities retain exact integers beyond binary64 and sum to 2^N', () => {
  for (const N of [1, 2, 3, 7, 16, 64, 128]) {
    const r = evaluate({ spinCount: N, couplingK: 0, fieldB: 0 });
    assert.equal(r.microstateCount, (O << BigInt(N)).toString()); assert.equal(r.states.length, N + 1);
    assert.equal(r.states.reduce((a, s) => a + BigInt(s.multiplicity), Z), O << BigInt(N));
    r.states.forEach((s, i) => { assert.equal(s.multiplicity, r.states[N - i].multiplicity); assert.equal(s.totalMagnetization, 2 * i - N); });
    near(r.meanSquareMagnetization.value, 1 / N); near(r.magnetizationVariance.value, 1 / N); near(r.fieldSusceptibility.value, 1);
    assert.equal(r.meanMagnetization.resolution, 'structural-zero');
  }
  const central = BigInt(evaluate({ spinCount: 128, couplingK: 0, fieldB: 0 }).states[64].multiplicity);
  assert.ok(central > BigInt(Number.MAX_SAFE_INTEGER)); assert.notEqual(BigInt(Number(central)), central);
});
test('zero-field spin-flip symmetry is exact even for two separated modal peaks', () => {
  for (const N of [1, 5, 32, 128]) {
    const r = evaluate({ spinCount: N, couplingK: 2, fieldB: 0 });
    assert.equal(r.meanMagnetization.value, 0); assert.equal(r.meanMagnetization.sign, 0); assert.equal(r.spinFlipSymmetric, true);
    assert.ok(r.meanAbsoluteMagnetization.value! > 0);
    r.states.forEach((s, i) => assert.deepEqual(s.probability, r.states[N - i].probability));
  }
  assert.deepEqual(evaluate({ spinCount: 32, couplingK: 2, fieldB: 0 }).modes.candidateTotalMagnetizations, [-32, 32]);
});
test('self interaction changes normalization and energy but not one-spin probabilities or entropy', () => {
  for (const b of [0, .7, -4]) for (const K of [0, .5, 8]) {
    const r = evaluate({ spinCount: 1, couplingK: K, fieldB: b });
    near(r.meanMagnetization.value, Math.tanh(b)); near(r.magnetizationVariance.value, 1 / Math.cosh(b) ** 2);
    near(r.logPartition.value, K / 2 + Math.log(2 * Math.cosh(b)));
    near(r.logPartitionWithoutSelfInteraction.value, Math.log(2 * Math.cosh(b)));
    near(r.meanEnergyWithoutSelfInteractionInThermalUnits.value, -b * Math.tanh(b));
    assert.equal(r.meanAbsoluteMagnetization.value, 1); assert.equal(r.meanSquareMagnetization.value, 1);
    assert.equal(r.conditionalMicrostateEntropyNats.resolution, 'structural-zero');
  }
});
test('the K=0 law agrees with independent biased Bernoulli spins', () => {
  for (const N of [2, 7, 32]) for (const b of [-.6, .35]) {
    const r = evaluate({ spinCount: N, couplingK: 0, fieldB: b });
    near(r.meanMagnetization.value, Math.tanh(b)); near(r.magnetizationVariance.value, 1 / (N * Math.cosh(b) ** 2));
    near(r.fieldSusceptibility.value, 1 / Math.cosh(b) ** 2); near(r.logPartition.value, N * Math.log(2 * Math.cosh(b)));
  }
});
test('the linear-time positive variance agrees with a separate double pair sum', () => {
  for (const K of [0, .8, 2]) for (const b of [-.5, 0, .5]) {
    const r = evaluate({ spinCount: 7, couplingK: K, fieldB: b }); let pair = 0;
    for (let i = 0; i < r.states.length; i++) for (let j = i + 1; j < r.states.length; j++) pair += r.states[i].probability.value! * r.states[j].probability.value! * ((r.states[i].totalMagnetization - r.states[j].totalMagnetization) / 7) ** 2;
    near(r.magnetizationVariance.value, pair); near(r.fieldSusceptibility.value, 7 * pair);
  }
});
test('thermodynamic derivatives agree with independently perturbed finite sums', () => {
  const p = { spinCount: 7, couplingK: 1.3, fieldB: .35 }, r = evaluate(p);
  for (const h of [.001, .0001]) {
    const plus = evaluate({ ...p, fieldB: p.fieldB + h }), minus = evaluate({ ...p, fieldB: p.fieldB - h });
    near((plus.logPartition.value! - minus.logPartition.value!) / (2 * h), r.logZFieldDerivative.value!, 5 * h * h);
    near((plus.meanMagnetization.value! - minus.meanMagnetization.value!) / (2 * h), r.fieldSusceptibility.value!, 10 * h * h);
    const a = evaluate({ ...p, couplingK: p.couplingK + h }), b = evaluate({ ...p, couplingK: p.couplingK - h });
    near((a.logPartition.value! - b.logPartition.value!) / (2 * h), r.logZCouplingDerivative.value!, 5 * h * h);
  }
});
test('microstate entropy, macrostate entropy and the self-energy convention satisfy distinct identities', () => {
  const r = evaluate({ spinCount: 7, couplingK: .8, fieldB: .35 });
  near(r.microstateEntropyNats.value, r.macroEntropyNats.value! + r.conditionalMicrostateEntropyNats.value!);
  near(r.freeEnergyInThermalUnits.value, r.meanEnergyInThermalUnits.value! - r.microstateEntropyNats.value!);
  near(-r.logPartitionWithoutSelfInteraction.value!, r.meanEnergyWithoutSelfInteractionInThermalUnits.value! - r.microstateEntropyNats.value!);
});
test('field reversal reflects the distribution and magnetization but preserves variance, normalization and entropy', () => {
  const plus = evaluate({ spinCount: 7, couplingK: .8, fieldB: .35 }), minus = evaluate({ spinCount: 7, couplingK: .8, fieldB: -.35 });
  plus.states.forEach((s, i) => assert.deepEqual(s.probability, minus.states[7 - i].probability));
  near(plus.meanMagnetization.value, -minus.meanMagnetization.value!);
  for (const key of ['magnetizationVariance', 'fieldSusceptibility', 'logPartition', 'microstateEntropyNats'] as const) assert.deepEqual(plus[key], minus[key]);
});
test('rounded equal probabilities do not certify a modal tie at the actual log(2) input', () => {
  const low = evaluate(cases.find(c => c.id === 'two-actual-ln2')!.parameters), high = evaluate(cases.find(c => c.id === 'two-next-above-ln2')!.parameters);
  assert.deepEqual(low.states.map(s => s.probability.value), [1 / 3, 1 / 3, 1 / 3]);
  assert.deepEqual(low.modes.candidateTotalMagnetizations, [0]); assert.deepEqual(high.modes.candidateTotalMagnetizations, [-2, 2]);
});
test('subnormal field signs survive a capped unresolved mean and equal displayed probabilities', () => {
  const r = model({ spinCount: 3, couplingK: 0, fieldB: Number.MIN_VALUE }, { maximumPrecision: 128 });
  assert.equal(r.fullyResolved, false); assert.equal(r.meanMagnetization.value, null); assert.equal(r.meanMagnetization.sign, 1);
  assert.equal(r.meanMagnetization.resolution, 'unresolved-at-precision-ceiling');
  assert.equal(r.meanEnergyWithoutSelfInteractionInThermalUnits.sign, -1);
  assert.equal(r.modes.certified, true); assert.deepEqual(r.modes.candidateTotalMagnetizations, [1]);
});
test('outside-display probabilities and tiny nonzero energies retain signs, finite logarithms and directed bounds', () => {
  const r = evaluate({ spinCount: 128, couplingK: 8, fieldB: 4 });
  const outside = r.states.filter(s => s.probability.value === null); assert.ok(outside.length > 0);
  outside.forEach(s => { assert.equal(s.probability.sign, 1); assert.ok(Number.isFinite(s.probability.logMagnitude)); assert.notEqual(s.probability.resolution, 'structural-zero'); assert.ok(s.probability.bounds); });
  const tiny = evaluate({ spinCount: 4, couplingK: Number.MIN_VALUE, fieldB: 0 }).meanEnergyWithoutSelfInteractionInThermalUnits;
  assert.equal(tiny.value, null); assert.equal(tiny.sign, -1); assert.ok(Number.isFinite(tiny.logMagnitude)); assert.ok(tiny.bounds);
});
test('invalid finite-support, field, coupling and precision preparations fail explicitly', () => {
  const p = { spinCount: 2, couplingK: 1, fieldB: 0 };
  for (const patch of [{ spinCount: 0 }, { spinCount: 129 }, { spinCount: 2.5 }, { couplingK: -Number.MIN_VALUE }, { couplingK: 8.1 }, { couplingK: NaN }, { fieldB: Infinity }, { fieldB: 4.1 }]) assert.throws(() => model({ ...p, ...patch }), RangeError);
  for (const options of [{ minimumPrecision: 64 }, { maximumPrecision: 129 }, { minimumPrecision: 256, maximumPrecision: 128 }]) assert.throws(() => model(p, options), RangeError);
  const r = model(p); assert.ok(Object.isFrozen(r.parameters)); p.fieldB = 1; assert.equal(r.parameters.fieldB, 0);
});
