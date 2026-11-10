import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { binaryConnectedCorrelationsModel as model, type BinaryCorrelationsParameters } from '../src/lib/learning-binary-correlations';
import type { PartitionObservation, PartitionRational } from '../src/lib/learning-partition-exact-core';
interface Fixture { id: string; parameters: BinaryCorrelationsParameters; confirmationPrecisionDigits: number; reference: Record<string, string | string[] | string[][]> }
const cases = (JSON.parse(readFileSync('scripts/fixtures/binary-correlations-oracle.json', 'utf8')) as { cases: Fixture[] }).cases;
const saved = new Map<string, ReturnType<typeof model>>();
function evaluate(p: BinaryCorrelationsParameters) { const key = JSON.stringify(p), old = saved.get(key); if (old) return old; const r = model(p); saved.set(key, r); return r; }
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
for (const c of cases) test(`${c.id}: independent actual-binary centered moments and directed enclosures`, () => {
  const r = evaluate(c.parameters); assert.equal(r.fullyResolved, true);
  for (const [key, value] of Object.entries(c.reference)) {
    if (key === 'probabilities' || key === 'logProbabilities') {
      (value as string[]).forEach((v, i) => check(r.states[i][key === 'probabilities' ? 'probability' : 'logProbability'], v, c.confirmationPrecisionDigits, `${c.id}.${key}[${i}]`));
    } else if (key === 'means') (value as string[]).forEach((v, i) => check(r.means[i], v, c.confirmationPrecisionDigits, `${c.id}.mean[${i}]`));
    else if (key === 'covariance') (value as string[][]).forEach((row, i) => row.forEach((v, j) => check(r.covariance[i][j], v, c.confirmationPrecisionDigits, `${c.id}.C[${i}][${j}]`)));
    else check((r as unknown as Record<string, PartitionObservation>)[key], value as string, c.confirmationPrecisionDigits, `${c.id}.${key}`);
  }
  const visit = (v: unknown): void => {
    if (typeof v === 'number') assert.ok(Number.isFinite(v));
    else if (Array.isArray(v)) v.forEach(visit);
    else if (v && typeof v === 'object') Object.values(v).forEach(visit);
  };
  visit(r); assert.doesNotThrow(() => JSON.stringify(r));
});
const base = { couplingK: .7, fieldOne: -.4, fieldTwo: .9, probeOne: 2, probeTwo: -1 };
const near = (a: number | null, b: number, tolerance = 2e-12) => { assert.notEqual(a, null); assert.ok(Math.abs(a! - b) <= tolerance * Math.max(1, Math.abs(b)), `${a} vs ${b}`); };
test('uniform four-state law and arbitrary zero observable have exact structural results', () => {
  const r = evaluate({ ...base, couplingK: 0, fieldOne: 0, fieldTwo: 0 });
  assert.deepEqual(r.states.map(s => s.spins), [[-1,-1],[-1,1],[1,-1],[1,1]]);
  r.states.forEach(s => assert.equal(s.probability.value, .25));
  assert.deepEqual(r.covariance.map(row => row.map(v => v.value)), [[1,0],[0,1]]);
  assert.equal(r.covarianceDeterminant.value, 1); assert.equal(r.probeVariance.value, 5);
  const z = evaluate({ ...base, probeOne: 0, probeTwo: 0 });
  for (const v of [z.probeMean, z.probeVariance]) { assert.equal(v.resolution, 'structural-zero'); assert.equal(v.value, 0); }
});
test('negative cross correlation coexists with strict positive definiteness', () => {
  for (const K of [-8,-.5,0,.5,8]) {
    const r = evaluate({ ...base, couplingK: K });
    assert.equal(r.covariance[0][1].sign, Math.sign(K));
    assert.ok(r.covariance[0][0].value! > 0 && r.covariance[1][1].value! > 0 && r.covarianceDeterminant.value! > 0);
    const C = r.covariance.map(row => row.map(v => v.value!));
    near(r.probeVariance.value, 4*C[0][0]-4*C[0][1]+C[1][1]);
    near(r.commonFieldResponsePerSpin.value, (C[0][0]+2*C[0][1]+C[1][1])/2);
  }
});
test('all fields preserve independence at K=0 without rounded moment subtraction', () => {
  for (const h1 of [0,.2,-8,Number.MIN_VALUE]) for (const h2 of [0,1.3,8]) {
    const r = evaluate({ ...base, couplingK: 0, fieldOne: h1, fieldTwo: h2 });
    assert.equal(r.covariance[0][1].resolution, 'structural-zero');
    near(r.means[0].value, Math.tanh(h1)); near(r.means[1].value, Math.tanh(h2));
    near(r.covariance[0][0].value, 1/Math.cosh(h1)**2); near(r.covariance[1][1].value, 1/Math.cosh(h2)**2);
  }
});
test('simultaneous field reversal and site exchange transform observables consistently', () => {
  const a = evaluate(base), b = evaluate({ ...base, fieldOne: -base.fieldOne, fieldTwo: -base.fieldTwo }), c = evaluate({ ...base, fieldOne: base.fieldTwo, fieldTwo: base.fieldOne, probeOne: base.probeTwo, probeTwo: base.probeOne });
  a.means.forEach((v,i) => near(b.means[i].value, -v.value!)); near(b.probeMean.value,-a.probeMean.value!);
  near(b.probeVariance.value,a.probeVariance.value!); near(c.probeVariance.value,a.probeVariance.value!); near(c.probeMean.value,a.probeMean.value!);
  near(c.means[0].value,a.means[1].value!); near(c.covariance[0][0].value,a.covariance[1][1].value!);
});
test('source scaling transforms the mean linearly and the response quadratically', () => {
  const a = evaluate(base), b = evaluate({ ...base, probeOne: 4, probeTwo: -2 });
  near(b.probeMean.value,2*a.probeMean.value!); near(b.probeVariance.value,4*a.probeVariance.value!);
});
test('local-field response and directional Hessian agree with independent finite differences', () => {
  const d=2e-5, r=evaluate(base);
  for (const j of [0,1]) {
    const key=j===0?'fieldOne':'fieldTwo';
    const lo=evaluate({...base,[key]:base[key]-d}),hi=evaluate({...base,[key]:base[key]+d});
    for (const i of [0,1]) near((hi.means[i].value!-lo.means[i].value!)/(2*d),r.covariance[i][j].value!,1e-8);
  }
  const lo=evaluate({...base,fieldOne:base.fieldOne-2*d,fieldTwo:base.fieldTwo+d}), hi=evaluate({...base,fieldOne:base.fieldOne+2*d,fieldTwo:base.fieldTwo-d});
  near((hi.probeMean.value!-lo.probeMean.value!)/(2*d),r.probeVariance.value!,1e-8);
});
test('subnormal coupling retains its sign and a precision cap never manufactures independence', () => {
  for (const K of [Number.MIN_VALUE,-Number.MIN_VALUE]) {
    const p={...base,couplingK:K}, capped=model(p,{maximumPrecision:128}), full=evaluate(p);
    assert.equal(capped.fullyResolved,false);assert.equal(capped.covariance[0][1].resolution,'unresolved-at-precision-ceiling');assert.equal(capped.covariance[0][1].value,null);assert.equal(capped.covariance[0][1].sign,Math.sign(K));
    assert.equal(full.fullyResolved,true);assert.ok(full.precisionBits>=2048);assert.equal(full.covariance[0][1].sign,Math.sign(K));
  }
});
test('subnormal probe variance stays mathematically positive below linear display range', () => {
  const r=evaluate({...base,probeOne:Number.MIN_VALUE,probeTwo:-Number.MIN_VALUE});
  assert.equal(r.probeVariance.value,null);assert.equal(r.probeVariance.sign,1);assert.equal(r.probeVariance.resolution,'interval-resolved');assert.ok(r.probeVariance.logMagnitude! < -1400);
});
test('invalid nonfinite, oversized and inconsistent precision preparations are rejected', () => {
  for (const k of Object.keys(base)) for (const v of [NaN,Infinity,-Infinity,9,-9]) assert.throws(()=>model({...base,[k]:v}),RangeError);
  assert.throws(()=>model({...base,probeOne:4.01}),RangeError);
  assert.throws(()=>model(base,{minimumPrecision:64}),RangeError);
  assert.throws(()=>model(base,{minimumPrecision:256,maximumPrecision:128}),RangeError);
  assert.throws(()=>model(null as unknown as BinaryCorrelationsParameters),RangeError);
});
