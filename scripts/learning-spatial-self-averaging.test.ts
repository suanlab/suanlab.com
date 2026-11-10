import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spatialDensitySelfAveragingModel as model, isSpatialSelfAveragingParameters as valid, type SpatialSelfAveragingParameters as Params } from '../src/lib/learning-spatial-self-averaging';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
const base: Params = { siteCount: 4, occupiedWeight: 1, emptyWeight: 1, commonNumerator: 1, commonDenominator: 2, commonScaling: 'fixed', epsilonNumerator: 1, epsilonDenominator: 2 };
const fixture = JSON.parse(readFileSync('scripts/fixtures/spatial-self-averaging-oracle.json', 'utf8'));
const at = (value: unknown, path: string): Q | null => path.split('.').reduce((v: any, k) => v[k], value);
function fraction(q: Q) { assert.ok(q.bounds); assert.deepEqual(q.bounds.lower, q.bounds.upper); assert.equal(q.bounds.lower.numerator.exponent, 0); assert.equal(q.bounds.lower.denominator.exponent, 0); return { n: BigInt(q.bounds.lower.numerator.coefficient), d: BigInt(q.bounds.lower.denominator.coefficient) }; }
function equal(q: Q, n: number | bigint, d: number | bigint = 1) { const a = fraction(q); assert.equal(a.n * BigInt(d), BigInt(n) * a.d); }
function compare(a: Q, b: Q) { const x = fraction(a), y = fraction(b); const delta = x.n * y.d - y.n * x.d; return delta > BigInt(0) ? 1 : delta < BigInt(0) ? -1 : 0; }
function check(actual: Q | null, expected: { numerator: string; denominator: string } | null) {
 if (expected === null) { assert.equal(actual, null); return; } assert.ok(actual); const n = BigInt(expected.numerator); equal(actual, n, BigInt(expected.denominator)); assert.equal(actual.sign, n === BigInt(0) ? 0 : n > BigInt(0) ? 1 : -1); assert.equal(actual.resolution, n === BigInt(0) ? 'structural-zero' : 'exact-rational');
 if (n === BigInt(0)) { assert.equal(actual.value, 0); assert.equal(actual.logMagnitude, null); } else { assert.notEqual(actual.value, 0); assert.ok(actual.value === null || Number.isFinite(actual.value)); assert.ok(Number.isFinite(actual.logMagnitude)); }
}
const cache = new Map<string, ReturnType<typeof model>>();
function evaluate(p: Params) { const k = JSON.stringify(p); if (!cache.has(k)) cache.set(k, model(p)); return cache.get(k)!; }
for (const c of fixture.cases) test(`independent spatial Fraction reference: ${c.id}`, () => {
 const r = evaluate(c.parameters);
 for (const [path, expected] of Object.entries(c.observations)) check(at(r, path), expected as any);
 for (const [path, expected] of Object.entries(c.limitObservations)) check(at(r, path), expected as any);
 for (const [name, kind] of Object.entries(c.limitTags)) assert.equal((r.limits as any)[name].kind,kind);
 assert.equal(r.limits.densityLawAtoms.length,c.limitAtoms.length);for(let i=0;i<c.limitAtoms.length;i++){check(r.limits.densityLawAtoms[i].density,c.limitAtoms[i].density);check(r.limits.densityLawAtoms[i].probability,c.limitAtoms[i].probability);}
 for(const [name,value] of Object.entries(c.limitClassifications)) assert.equal((r.classification as any)[name],value);
 assert.deepEqual(r.current.eventCounts, c.eventCounts);
 for (const [key, value] of Object.entries(c.flags)) assert.equal((r.current as any)[key], value, key);
 for (const row of c.selectedRows) { const actual = r.current.distribution[row.count]; assert.equal(actual.count, row.count); assert.equal(actual.inDeviationEvent, row.inDeviationEvent); for (const field of ['density', 'deviation', 'independentProbability', 'collectiveProbability', 'probability']) check((actual as any)[field], row[field]); }
 assert.equal(r.history.at(-1), r.current.summary); assert.equal(r.current.distribution.length, c.parameters.siteCount + 1); assert.deepEqual(structuredClone(r), r); assert.doesNotThrow(() => JSON.stringify(r));
});
test('the exact four-site ledger preserves ordered covariance pairs, moments and valid bounds', () => {
 const r = evaluate(base).current; equal(r.summary.densityVariance, 5, 32); equal(r.countVariance, 5, 2); equal(r.siteVariance, 1, 4); equal(r.distinctPairCovariance!, 1, 8); assert.equal(r.orderedCrossCovarianceMultiplicity, 12); equal(r.diagonalCovarianceSum, 1); equal(r.orderedCrossCovarianceSum, 3, 2); equal(r.covarianceSumDensityVariance, 5, 32);
 equal(r.summary.eventProbability, 9, 16); equal(r.summary.independentEventProbability, 1, 8); equal(r.summary.collectiveEventProbability, 1); equal(r.summary.chebyshevBound, 5, 8); equal(r.summary.mixtureAwareBound, 5, 8); equal(r.summary.independentReferenceBound, 1, 4); assert.equal(r.independentReferenceBoundFailsForMixture, true); equal(r.summary.varianceEquivalentCount!, 8, 5);
});
test('increasing variance can decrease a particular deviation tail', () => {
 const r = evaluate({ ...base, siteCount: 2, occupiedWeight: 1, emptyWeight: 9, epsilonNumerator: 3, epsilonDenominator: 20 }).current;
 equal(r.summary.densityVariance, 27, 400); equal(r.summary.independentDensityVariance, 9, 200); assert.equal(compare(r.summary.densityVariance, r.summary.independentDensityVariance), 1);
 equal(r.summary.eventProbability, 29, 200); equal(r.summary.independentEventProbability, 19, 100); equal(r.eventProbabilityMinusIndependent, -9, 200); equal(r.eventTailRatioToIndependent!, 29, 38); equal(r.summary.mixtureAwareBound, 11, 20); equal(r.summary.chebyshevBound, 1);
});
test('the event includes rational equality and does not silently clip epsilon', () => {
 const p = { ...base, siteCount: 3, occupiedWeight: 1, emptyWeight: 2, epsilonNumerator: 1, epsilonDenominator: 3 }; assert.deepEqual(evaluate(p).current.eventCounts, [0, 2, 3]); assert.deepEqual(evaluate({ ...p, epsilonNumerator: 333334, epsilonDenominator: 1000000 }).current.eventCounts, [3]);
 const empty = evaluate({ ...base, epsilonNumerator: 2, epsilonDenominator: 1 }).current; assert.deepEqual(empty.eventCounts, []); equal(empty.summary.eventProbability, 0); assert.equal(empty.eventTailRatioToIndependent, null); assert.ok(empty.summary.chebyshevBound.value! > 0);
 const all = evaluate({ ...base, siteCount: 2, occupiedWeight: 1, emptyWeight: 2, epsilonNumerator: 1, epsilonDenominator: 100 }).current; assert.equal(all.eventIsWholeGrid, true); equal(all.summary.eventProbability, 1);
});
test('singleton and deterministic preparations keep missing correlation distinct from zero covariance', () => {
 const one = evaluate({ ...base, siteCount: 1, commonNumerator: 1, commonDenominator: 1 }); assert.equal(one.current.distributionIsProductForThisSize, true); assert.equal(one.current.distinctPairCovariance, null); assert.equal(one.current.distinctPairCorrelation, null); equal(one.current.orderedCrossCovarianceSum, 0); equal(one.current.summary.varianceEquivalentCount!, 1); assert.equal(one.classification.selfAveragesInProbabilityAlongPath, false);
 for (const [occupiedWeight, emptyWeight] of [[0, 1], [1, 0]]) { const r = evaluate({ ...base, occupiedWeight, emptyWeight }); equal(r.current.distinctPairCovariance!, 0); assert.equal(r.current.distinctPairCorrelation, null); assert.equal(r.current.varianceRatioToIndependent, null); assert.equal(r.current.summary.varianceEquivalentCount, null); assert.equal(r.current.eventTailRatioToIndependent, null); assert.equal(r.limits.varianceEquivalentCount.kind, 'not-applicable'); assert.equal(r.limits.varianceEquivalentFraction.kind, 'not-applicable'); assert.equal(r.classification.selfAveragesInProbabilityAlongPath, true); assert.equal(r.limits.densityLawAtoms.length, 1); equal(r.limits.densityLawAtoms[0].probability, 1); }
});
test('analytic limiting laws merge atoms and preserve their exact masses', () => {
 const fixed = evaluate(base); equal(fixed.limits.densityVariance, 1, 8); equal(fixed.limits.deviationEventProbability, 1, 2); assert.equal(fixed.limits.scaledDensityVariance.kind, 'positive-infinity'); assert.equal(fixed.limits.varianceEquivalentCount.kind, 'finite'); if (fixed.limits.varianceEquivalentCount.kind === 'finite') equal(fixed.limits.varianceEquivalentCount.value, 2); assert.deepEqual(fixed.limits.densityLawAtoms.map(a => [a.density.value, a.probability.value]), [[0, .25], [.5, .5], [1, .25]]);
 const shared = evaluate({ ...base, commonNumerator: 1, commonDenominator: 1 }); assert.equal(shared.limits.densityLawAtoms.length, 2); assert.equal(shared.current.positiveMassCount, 2);
 for (const commonScaling of ['inverse-size', 'inverse-size-squared'] as const) { const r = evaluate({ ...base, commonScaling }); equal(r.limits.densityVariance, 0); equal(r.limits.deviationEventProbability, 0); assert.equal(r.limits.scaledDensityVariance.kind, 'finite'); if (r.limits.scaledDensityVariance.kind === 'finite') equal(r.limits.scaledDensityVariance.value, commonScaling === 'inverse-size' ? 3 : 1, commonScaling === 'inverse-size' ? 8 : 4); assert.equal(r.limits.varianceEquivalentFraction.kind, 'finite'); if (r.limits.varianceEquivalentFraction.kind === 'finite') equal(r.limits.varianceEquivalentFraction.value, commonScaling === 'inverse-size' ? 2 : 1, commonScaling === 'inverse-size' ? 3 : 1); assert.equal(r.classification.selfAveragesInProbabilityAlongPath, true); assert.equal(r.limits.densityLawAtoms.length, 1); equal(r.limits.densityLawAtoms[0].density, 1, 2); equal(r.limits.densityLawAtoms[0].probability, 1); }
});
test('the inverse-square path has IID leading variance but can amplify extremely rare tails beyond Number range', () => {
 const p: Params = { ...base, siteCount: 512, occupiedWeight: 1, emptyWeight: 999999, commonNumerator: 1, commonDenominator: 1000000, commonScaling: 'inverse-size-squared', epsilonNumerator: 999999, epsilonDenominator: 1000000 }, r = evaluate(p);
 assert.equal(r.current.summary.independentEventProbability.value, null); assert.equal(r.current.summary.independentEventProbability.sign, 1); assert.ok(r.current.summary.independentEventProbability.logMagnitude! < -7000);
 assert.ok(r.current.summary.eventProbability.value! > 0); assert.equal(r.current.eventTailRatioToIndependent!.value, null); assert.equal(r.current.eventTailRatioToIndependent!.sign, 1); assert.equal(r.current.eventTailRatioToIndependent!.resolution, 'exact-rational'); assert.ok(r.current.eventTailRatioToIndependent!.logMagnitude! > 7000);
 assert.equal(r.limits.scaledDensityVariance.kind, 'finite'); if (r.limits.scaledDensityVariance.kind === 'finite') equal(r.limits.scaledDensityVariance.value, 999999, BigInt(10) ** BigInt(12)); assert.equal(r.classification.selfAveragesInProbabilityAlongPath, true);
 const iid = evaluate({ ...p, commonNumerator: 0 }).current; equal(iid.eventTailRatioToIndependent!, 1); assert.equal(iid.summary.eventProbability.value, null); equal(iid.summary.eventProbability, BigInt(1), BigInt(10) ** BigInt(3072));
});
test('all records retain fixed parameters while the effective common weight changes with size', () => {
 for (const c of fixture.cases) { const r = evaluate(c.parameters), p = r.parameters, C = BigInt(p.commonNumerator), D = BigInt(p.commonDenominator), A = BigInt(p.occupiedWeight), B = BigInt(p.emptyWeight), W = A + B, power = p.commonScaling === 'fixed' ? 0 : p.commonScaling === 'inverse-size' ? 1 : 2;
  for (const s of r.history) { const N = BigInt(s.siteCount), DN = D * N ** BigInt(power); equal(s.commonWeight, C, DN); equal(s.meanDensity, A, W); equal(s.densityVariance, A * B * (DN + (N - BigInt(1)) * C), W * W * N * DN); assert.ok(compare(s.chebyshevBound, s.mixtureAwareBound) >= 0); assert.ok(compare(s.mixtureAwareBound, s.eventProbability) >= 0); }
  equal(r.current.varianceIdentityResidual, 0); equal(r.current.eventMixtureIdentityResidual, 0); assert.equal(r.current.distributionMomentsMatch, true);
 }
 assert.deepEqual(evaluate({ ...base, siteCount: 7 }).history.map(s => s.siteCount), [1, 2, 4, 7]);
});
test('reflection and rational re-expression preserve their exact law and event meanings', () => {
 const r = evaluate({ ...base, siteCount: 7, occupiedWeight: 1, emptyWeight: 9, epsilonNumerator: 3, epsilonDenominator: 20 }), mirror = evaluate({ ...base, siteCount: 7, occupiedWeight: 9, emptyWeight: 1, epsilonNumerator: 3, epsilonDenominator: 20 });
 assert.deepEqual(r.current.summary.densityVariance, mirror.current.summary.densityVariance); assert.deepEqual(r.current.summary.eventProbability, mirror.current.summary.eventProbability);
 for (const row of r.current.distribution) { const other = mirror.current.distribution.at(-1 - row.count)!; assert.deepEqual(row.probability, other.probability); assert.equal(row.inDeviationEvent, other.inDeviationEvent); }
 const scaled = model({ ...base, occupiedWeight: 2, emptyWeight: 2, commonNumerator: 2, commonDenominator: 4, epsilonNumerator: 2, epsilonDenominator: 4 }), original = evaluate(base); assert.deepEqual(scaled.current, original.current); assert.deepEqual(scaled.history, original.history); assert.deepEqual(scaled.limits, original.limits);
});
test('invalid weights, rational constraints and unsupported precision/settings are rejected without mutation', () => {
 const bad = [null, {}, { ...base, siteCount: 0 }, { ...base, siteCount: 513 }, { ...base, siteCount: .5 }, { ...base, occupiedWeight: 0, emptyWeight: 0 }, { ...base, occupiedWeight: -1 }, { ...base, commonNumerator: 3 }, { ...base, commonDenominator: 0 }, { ...base, commonScaling: 'time' }, { ...base, epsilonNumerator: 0 }, { ...base, epsilonDenominator: 0 }, { ...base, emptyWeight: NaN }, { ...base, options: {} }, { ...base, extra: 1 }];
 for (const p of bad) { assert.equal(valid(p), false); assert.throws(() => model(p as Params), RangeError); }
 const p = { ...base }, result = model(p); p.siteCount = 99; assert.equal(result.parameters.siteCount, 4); assert.ok(Object.isFrozen(result.parameters));
});
