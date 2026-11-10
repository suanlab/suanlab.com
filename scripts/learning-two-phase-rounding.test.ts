import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { twoPhaseRoundingModel as model, isTwoPhaseRoundingParameters as valid, type TwoPhaseRoundingParameters as Params } from '../src/lib/learning-two-phase-rounding';
import type { PartitionObservation as Q, PartitionRational } from '../src/lib/learning-partition-exact-core';
const fixture = JSON.parse(readFileSync('scripts/fixtures/two-phase-rounding-oracle.json', 'utf8'));
const base: Params = { siteCount: 4, activityNumerator: 2, activityDenominator: 1, sizePath: 'fixed', approachSide: 1, offsetNumerator: 1, offsetDenominator: 2 };
const Z = BigInt(0), O = BigInt(1);
const at = (r: unknown, path: string): any => path.split('.').reduce((v: any, k) => v[k], r);
function decode(q: PartitionRational) { const e = q.numerator.exponent - q.denominator.exponent; return { n: BigInt(q.numerator.coefficient) * (e > 0 ? O << BigInt(e) : O), d: BigInt(q.denominator.coefficient) * (e < 0 ? O << BigInt(-e) : O) }; }
function equal(q: Q, n: bigint | number, d: bigint | number = 1) { assert.ok(q.bounds); const lo = decode(q.bounds.lower), hi = decode(q.bounds.upper); assert.equal(lo.n * BigInt(d), BigInt(n) * lo.d); assert.equal(hi.n * BigInt(d), BigInt(n) * hi.d); }
function decimal(s: string) { const match = /^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s); assert.ok(match, s); const decimals = match[3] ?? '', exponent = Number(match[4] ?? 0) - decimals.length, n = BigInt((match[1] ?? '') + match[2] + decimals); return { n: exponent >= 0 ? n * BigInt(10) ** BigInt(exponent) : n, d: exponent < 0 ? BigInt(10) ** BigInt(-exponent) : O }; }
function contains(q: Q, reference: string) { assert.ok(q.bounds); const lo = decode(q.bounds.lower), hi = decode(q.bounds.upper), r = decimal(reference); assert.ok(lo.n * r.d <= r.n * lo.d, `lower bound excludes ${reference}`); assert.ok(hi.n * r.d >= r.n * hi.d, `upper bound excludes ${reference}`); assert.notEqual(q.resolution, 'unresolved-at-precision-ceiling'); if (r.n !== Z) { assert.notEqual(q.value, 0); assert.equal(q.sign, r.n > Z ? 1 : -1); } }
const results = new Map<string, ReturnType<typeof model>>();
function evaluate(p: Params) { const key = JSON.stringify(p); if (!results.has(key)) results.set(key, model(p)); return results.get(key)!; }
for (const c of fixture.cases) test(`independent two-state and high-precision reference: ${c.id}`, () => {
 const r = evaluate(c.parameters);
 for (const [name, point] of Object.entries(c.observations) as [string, any][]) equal(at(r.current, name), BigInt(point.numerator), BigInt(point.denominator));
 for (const [name, reference] of Object.entries(c.logs)) contains(at(r.current, name), reference as string);
 for (const [name, reference] of Object.entries(c.limits)) contains(at(r.limits, name), reference as string);
 for (const [name, kind] of Object.entries(c.limitTags)) assert.equal(at(r.limits, name).kind, kind);
 for (const [name, value] of Object.entries(c.classification)) assert.equal(at(r.classification, name), value);
 for (const root of c.roots) { const actual = r.roots[root.index]; equal(actual.turns, BigInt(root.turns.numerator), BigInt(root.turns.denominator)); assert.ok(Math.abs(actual.plotReal - Number(root.real)) < 2e-14); assert.ok(Math.abs(actual.plotImaginary - Number(root.imaginary)) < 2e-14); }
 assert.equal(r.fullyResolved, true); assert.equal(r.history.at(-1), r.current); assert.deepEqual(structuredClone(r), r); assert.doesNotThrow(() => JSON.stringify(r));
});
test('the finite two-state ledger differs from an independent-site partition and preserves response identities', () => {
 const r = evaluate(base).current; equal(r.partitionFunction, 17); equal(r.occupiedProbability, 16, 17); equal(r.emptyProbability, 1, 17); equal(r.densityVariance, 16, 289); equal(r.meanCount, 64, 17); equal(r.countVariance, 256, 289); equal(r.densityResponseToField, 64, 289); equal(r.countResponseToField, 256, 289); equal(r.exponentialPressureCorrection, 17, 16); assert.equal(r.allowedConfigurationCount, 2); assert.equal(r.distribution.length, 2); assert.notEqual(r.partitionFunction.value, 81);
});
test('positive and reciprocal fields reflect the law and leave the correction and variance unchanged', () => {
 for (const sizePath of ['fixed', 'inverse-size', 'inverse-size-squared'] as const) {
  const a = evaluate({ ...base, sizePath }), b = evaluate({ ...base, sizePath, activityNumerator: 1, activityDenominator: 2, approachSide: -1 });
  assert.deepEqual(a.current.occupiedProbability, b.current.emptyProbability); assert.deepEqual(a.current.densityVariance, b.current.densityVariance); assert.deepEqual(a.current.pressureCorrection, b.current.pressureCorrection); assert.deepEqual(a.current.tenToNinetyFieldWidth, b.current.tenToNinetyFieldWidth); assert.deepEqual(a.limits.densityVariance, b.limits.densityVariance);
 }
});
test('coexistence keeps density variance one quarter and response grows while pressure correction shrinks', () => {
 for (const siteCount of [1, 2, 4, 64, 512]) { const r = evaluate({ ...base, siteCount, activityNumerator: 1 }); equal(r.current.densityVariance, 1, 4); equal(r.current.densityResponseToField, siteCount, 4); equal(r.current.partitionFunction, 2); equal(r.current.exponentialPressureCorrection, 2); equal(r.limits.meanDensity, 1, 2); assert.equal(r.classification.selfAveragesAlongPath, false); assert.equal(r.limits.densityResponseToField.kind, 'positive-infinity'); assert.ok(r.current.pressureCorrection.value! > 0); }
});
test('extreme positive corrections are certified nonzero even when activity and probability projections round', () => {
 for (const [activityNumerator, activityDenominator] of [[1000000, 1], [1, 1000000]]) {
  const r = evaluate({ ...base, siteCount: 512, activityNumerator, activityDenominator }).current;
  equal(r.minorityToMajorityWeight, O, BigInt(10) ** BigInt(3072)); assert.equal(r.minorityProbability.value, null); assert.equal(r.minorityProbability.sign, 1); assert.equal(r.pressureCorrection.value, null); assert.equal(r.pressureCorrection.sign, 1); assert.equal(r.pressureCorrection.resolution, 'interval-resolved'); assert.ok(r.pressureCorrection.logMagnitude! < -7000); assert.ok(BigInt(r.pressureCorrection.bounds!.lower.numerator.coefficient) > Z); assert.equal(r.exponentialPressureCorrection.value, 1); assert.notEqual(r.exponentialPressureCorrection.resolution, 'structural-zero');
 }
});
test('all exact root angles are distinct, paired and satisfy the defining polynomial phase identity', () => {
 for (const siteCount of [1, 2, 3, 4, 7, 64, 512]) {
  const r = evaluate({ ...base, siteCount }); assert.equal(r.roots.length, siteCount); const turns = new Set<string>();
  for (const root of r.roots) { const q = decode(root.turns.bounds!.lower), N = BigInt(siteCount); assert.ok(q.n > Z && q.n < q.d); assert.equal((TWO * N * q.n / q.d) % TWO, O); assert.equal(TWO * N * q.n % q.d, Z); equal(root.poweredTurnsModuloOne, 1, 2); turns.add(`${q.n}/${q.d}`); const other = r.roots[root.conjugateIndex], y = decode(other.turns.bounds!.lower); assert.equal(q.n * y.d + y.n * q.d, q.d * y.d); }
  assert.equal(turns.size, siteCount); assert.equal(r.roots.filter(root => root.isNegativeRealRoot).length, siteCount % 2); assert.equal(r.rootGeometry.positiveRealRootExists, false); equal(r.rootGeometry.nearestPositiveAxisTurns, 1, 2 * siteCount); equal(r.rootGeometry.nearestLogFieldImaginaryPiFactor, 1, siteCount);
 }
});
const TWO = BigInt(2);
test('history fixes the path controls, includes non-power endpoints and distinguishes ignored inputs', () => {
 const r = evaluate({ ...base, siteCount: 7, sizePath: 'inverse-size-squared' }); assert.deepEqual(r.history.map(row => row.siteCount), [1, 2, 4, 7]); for (const row of r.history) equal(row.activity, BigInt(2 * row.siteCount ** 2 + 1), BigInt(2 * row.siteCount ** 2));
 assert.deepEqual(evaluate({ ...base, sizePath: 'inverse-size', activityNumerator: 999999 }).current, evaluate({ ...base, sizePath: 'inverse-size' }).current);
 assert.deepEqual(evaluate({ ...base, activityNumerator: 4, activityDenominator: 2, offsetNumerator: 2, offsetDenominator: 4 }).current, evaluate(base).current);
});
test('invalid and unsupported parameters fail without mutation', () => {
 const invalid = [null, {}, [], { ...base, siteCount: 0 }, { ...base, siteCount: 513 }, { ...base, activityNumerator: 0 }, { ...base, activityDenominator: 0 }, { ...base, activityNumerator: 1000001 }, { ...base, offsetNumerator: -1 }, { ...base, offsetNumerator: 3 }, { ...base, offsetDenominator: 0 }, { ...base, approachSide: 0 }, { ...base, sizePath: 'time' }, { ...base, precision: 8192 }, { ...base, siteCount: NaN }, { ...base, offsetNumerator: Infinity }];
 for (const p of invalid) { assert.equal(valid(p), false); assert.throws(() => model(p as Params), RangeError); }
 const p = { ...base }; const before = structuredClone(p); model(p); assert.deepEqual(p, before);
});
