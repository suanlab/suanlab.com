import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { quadraticGibbsModel as model, type QuadraticGibbsParameters } from '../src/lib/learning-quadratic-gibbs';
import { directedPi } from '../src/lib/learning-directed-pi';
import type { PartitionObservation, PartitionRational } from '../src/lib/learning-partition-exact-core';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
interface Fixture { id: string; parameters: QuadraticGibbsParameters; confirmationPrecisionDigits: number; reference: unknown }
const fixture = JSON.parse(readFileSync('scripts/fixtures/quadratic-gibbs-oracle.json', 'utf8')) as { cases: Fixture[]; piDecimalReference: string };
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
  if (typeof reference === 'string') {
    if (/^[+-]?\d/.test(reference)) check(actual as PartitionObservation, reference, precision, label); else assert.equal(actual, reference, label);
    return;
  }
  if (typeof reference === 'boolean' || typeof reference === 'number') { assert.equal(actual, reference, label); return; }
  if (Array.isArray(reference)) { assert.ok(Array.isArray(actual)); assert.equal(actual.length, reference.length, label); reference.forEach((r, i) => compareReference(actual[i], r, precision, `${label}[${i}]`)); return; }
  assert.ok(reference && typeof reference === 'object');
  for (const [key, value] of Object.entries(reference)) compareReference((actual as Record<string, unknown>)[key], value, precision, `${label}.${key}`);
}
for (const c of fixture.cases) test(`${c.id}: exact normalizability, moments, thermodynamics and transformed density match independent actual-input reference`, () => {
  const r = model(c.parameters); compareReference(r, c.reference, c.confirmationPrecisionDigits, c.id); assert.equal(r.fullyResolved, true);
  if (!r.normalizable) { assert.equal(r.logPartition.resolution, 'mathematical-infinity'); assert.ok(!('mean' in r)); }
  const visit = (v: unknown): void => {
    if (typeof v === 'number') assert.ok(Number.isFinite(v));
    else if (Array.isArray(v)) v.forEach(visit);
    else if (v && typeof v === 'object') Object.values(v).forEach(visit);
  };
  visit(r); assert.doesNotThrow(() => JSON.stringify(r));
});
const baseline: QuadraticGibbsParameters = { firstStiffness: 2 * kB, crossStiffness: kB, secondStiffness: 2 * kB, firstSourceEnergy: 0, secondSourceEnergy: 0, energyOffset: 0, temperature: 1 };
function proper(p: QuadraticGibbsParameters) { const r = model(p); assert.ok(r.normalizable); return r; }
function close(a: number | null, b: number) { assert.notEqual(a, null); assert.ok(Math.abs(a! - b) <= 3e-12 * Math.max(Math.abs(b), Number.MIN_VALUE), `${a} vs ${b}`); }
test('ideal covariance, source displacement, precision and energy identities use the stated base preparation', () => {
  const a = proper(baseline), b = proper({ ...baseline, firstSourceEnergy: kB });
  close(a.covariance[0][0].value, 2 / 3); close(a.covariance[0][1].value, -1 / 3); close(a.correlation.value, -.5);
  close(a.logPartition.value, Math.log(2 * Math.PI / Math.sqrt(3))); close(a.meanEnergy.value, kB);
  close(b.mean[0].value, 2 / 3); close(b.mean[1].value, -1 / 3);
  close(b.minimumEnergy.value, -kB / 3); close(b.meanEnergy.value, 2 * kB / 3);
  assert.deepEqual(a.covariance, b.covariance); assert.deepEqual(a.differentialEntropyNats, b.differentialEntropyNats);
});
test('common energy shift preserves the density, entropy and covariance while shifting U and F', () => {
  const a = proper({ ...baseline, firstSourceEnergy: kB }), b = proper({ ...baseline, firstSourceEnergy: kB, energyOffset: 2 ** 60 });
  assert.deepEqual(a.mean, b.mean); assert.deepEqual(a.covariance, b.covariance); assert.deepEqual(a.densityAtMean, b.densityAtMean);
  assert.deepEqual(a.differentialEntropyNats, b.differentialEntropyNats);
  close(b.meanEnergy.value! - a.meanEnergy.value!, 2 ** 60); close(b.freeEnergy.value! - a.freeEnergy.value!, 2 ** 60);
});
test('fixed A,h,c temperature response is one kB instead of holding A/tau fixed', () => {
  const a = proper(baseline), b = proper({ ...baseline, temperature: 2 });
  close(b.covariance[0][0].value!, 2 * a.covariance[0][0].value!); close(b.meanEnergy.value! - a.meanEnergy.value!, kB);
  assert.equal(a.heatCapacityInKB.value, 1); assert.equal(b.heatCapacityInKB.value, 1);
  close(b.differentialEntropyNats.value! - a.differentialEntropyNats.value!, Math.log(2));
});
test('common energy and temperature scaling preserves the whole normalized density', () => {
  const a = proper({ ...baseline, firstSourceEnergy: kB }), b = proper({ ...baseline, firstStiffness: 16 * kB, crossStiffness: 8 * kB, secondStiffness: 16 * kB, firstSourceEnergy: 8 * kB, temperature: 8 });
  assert.deepEqual(a.mean, b.mean); assert.deepEqual(a.covariance, b.covariance); close(a.densityAtMean.value, b.densityAtMean.value!);
});
test('invertible coordinate scales require an absolute Jacobian, including orientation reversal', () => {
  const a = proper({ ...baseline, firstSourceEnergy: kB }), b = proper({ ...baseline, firstSourceEnergy: kB, coordinateScale: [-2, 3], coordinateShift: [1, -2] });
  close(b.transformedCoordinates.mean[0].value, (a.mean[0].value! - 1) / -2);
  close(b.transformedCoordinates.covariance[0][1].value, a.covariance[0][1].value! / -6);
  close(b.transformedCoordinates.densityAtMean.value, 6 * a.densityAtMean.value!);
  close(b.transformedCoordinates.differentialEntropyNats.value! - a.differentialEntropyNats.value!, -Math.log(6));
});
test('a positive determinant outside the binary64 range still normalizes; null covariance is nonzero', () => {
  const r = proper(fixture.cases.find(c => c.id === 'overflowing-determinant')!.parameters);
  assert.equal(r.determinant.value, null); assert.equal(r.determinant.sign, 1); assert.equal(r.exactDeterminantSign, 1);
  assert.equal(r.covariance[0][0].value, null); assert.equal(r.covariance[0][0].sign, 1);
  assert.equal(r.meanEnergy.sign, 1); close(r.meanEnergy.value, kB); assert.equal(r.heatCapacityInKB.value, 1);
});
test('a subnormal positive temperature retains positive covariance and energy without claiming a zero-temperature density', () => {
  const r = proper({ ...baseline, temperature: Number.MIN_VALUE });
  assert.equal(r.thermalEnergy.value, null); assert.equal(r.thermalEnergy.sign, 1);
  assert.equal(r.meanEnergy.value, null); assert.equal(r.meanEnergy.sign, 1);
  assert.equal(r.heatCapacityInKB.value, 1); assert.ok(r.differentialEntropyNats.value! < -700);
});
test('density above one and negative differential entropy do not violate normalization', () => {
  const r = proper(fixture.cases.find(c => c.id === 'density-above-one')!.parameters);
  assert.ok(r.densityAtMean.value! > 1); assert.ok(r.differentialEntropyNats.value! < 0);
});
test('exact positive and negative determinant boundaries are not an eigenvalue tolerance', () => {
  const a = model({ ...baseline, firstStiffness: 1, crossStiffness: 1, secondStiffness: 1 + Number.EPSILON });
  const b = model({ ...baseline, firstStiffness: 1, crossStiffness: 1, secondStiffness: 1 });
  const c = model({ ...baseline, firstStiffness: 1, crossStiffness: 1, secondStiffness: 1 - Number.EPSILON / 2 });
  assert.equal(a.normalizable, true); assert.equal(a.exactDeterminantSign, 1); assert.equal(b.normalizable, false); assert.equal(b.exactDeterminantSign, 0);
  assert.equal(c.normalizable, false); assert.equal(c.exactDeterminantSign, -1);
});
test('malformed coordinates, zero/negative temperatures, nonfinite inputs and invalid precision are rejected', () => {
  const invalid = [{ ...baseline, temperature: 0 }, { ...baseline, temperature: -1 }, { ...baseline, temperature: Infinity }, { ...baseline, firstStiffness: NaN },
    { ...baseline, energyOffset: Infinity }, { ...baseline, coordinateScale: [0, 1] }, { ...baseline, coordinateScale: [1] }, { ...baseline, coordinateScale: [1, Infinity] },
    { ...baseline, coordinateScale: Array(2) }, { ...baseline, coordinateShift: [NaN, 0] }, { ...baseline, coordinateShift: null }, { ...baseline, coordinateScale: null }];
  invalid.forEach(p => assert.throws(() => model(p as unknown as QuadraticGibbsParameters), RangeError));
  assert.throws(() => model(baseline, { maximumPrecision: 64 }), RangeError);
  assert.throws(() => model(baseline, { minimumPrecision: 512, maximumPrecision: 128 }), RangeError);
});
test('directed pi encloses an independent 400-digit Decimal value at 128..1024 bits', () => {
  const truth = decimal(fixture.piDecimalReference), error = { n: O, d: BigInt(10) ** BigInt(350) };
  for (const P of [128, 256, 512, 1024]) {
    const r = directedPi(P), S = O << BigInt(P);
    assert.ok(cmp({ n: r.lower, d: S }, add(truth, error)) <= Z); assert.ok(cmp({ n: r.upper, d: S }, add(truth, { n: -error.n, d: error.d })) >= Z);
    assert.ok(r.upper - r.lower <= BigInt(2)); assert.ok(r.lower > BigInt(3) * S && BigInt(7) * r.upper < BigInt(22) * S);
  }
});
test('pi bounds at higher precision stay inside lower-precision bounds; cached results are isolated', () => {
  for (const P of [64, 128, 256, 512]) {
    const a = directedPi(P), b = directedPi(2 * P), scale = O << BigInt(P);
    assert.ok(b.lower >= a.lower * scale && b.upper <= a.upper * scale);
    const old = directedPi(P); a.lower = Z; assert.deepEqual(directedPi(P), old);
  }
  assert.throws(() => directedPi(63), RangeError); assert.throws(() => directedPi(8193), RangeError);
});

test('the maximum supported precision retains the same Gaussian observations with directed bounds', () => {
  const r = model(baseline, { minimumPrecision: 8192, maximumPrecision: 8192 });
  assert.ok(r.normalizable); assert.equal(r.precisionBits, 8192); assert.equal(r.fullyResolved, true);
  close(r.logPartition.value, Math.log(2 * Math.PI / Math.sqrt(3))); close(r.correlation.value, -.5);
});
