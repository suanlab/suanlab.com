import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { spatialDensitySelfAveragingModel as model, type SpatialSelfAveragingParameters as Params } from '../src/lib/learning-spatial-self-averaging';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
const id = 'spatial-density-self-averaging-correlations';
const raw = (language: 'ko' | 'en') => readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
const base: Params = { siteCount: 4, occupiedWeight: 1, emptyWeight: 1, commonNumerator: 1, commonDenominator: 2, commonScaling: 'fixed', epsilonNumerator: 1, epsilonDenominator: 2 };
function fraction(q: Q) { assert.ok(q.bounds); assert.deepEqual(q.bounds.lower, q.bounds.upper); assert.equal(q.bounds.lower.numerator.exponent, 0); assert.equal(q.bounds.lower.denominator.exponent, 0); return { n: BigInt(q.bounds.lower.numerator.coefficient), d: BigInt(q.bounds.lower.denominator.coefficient) }; }
function equal(q: Q, n: number | bigint, d: number | bigint = 1) { const a = fraction(q); assert.equal(a.n * BigInt(d), BigInt(n) * a.d); }
const printed = (q: Q) => { const { n, d } = fraction(q); return `${n}${d === BigInt(1) ? '' : `/${d}`}`; };
function tables(source: string) { return (source.match(/(?:^\|.*\|\n)+/gm) ?? []).map(group => group.trim().split('\n').slice(2).map(line => line.split('|').slice(1, -1).map(cell => cell.trim()))); }
for (const language of ['ko', 'en'] as const) test(`${language}: spatial manuscript renders equations, keyboard-readable ledgers, a mode diagram and explained exercises`, async () => {
 const source = raw(language), content = await getLearningContent(id), html = content[language];
 assert.equal(content.outline[language].length, 16); assert.equal(new Set(content.outline[language].map(h => h.id)).size, 16);
 assert.equal((source.match(/^### [1-6]\. /gm) ?? []).length, 6); assert.equal((source.match(/\*\*(?:Explanation\.|해설\.)\*\*/g) ?? []).length, 6);
 assert.equal((html.match(/class="katex-display"/g) ?? []).length, 16); assert.equal((html.match(/aria-label="(?:Equation|수식) /g) ?? []).length, 16); assert.equal((html.match(/<table[^>]*tabindex="0"/g) ?? []).length, 4);
 assert.match(html, /<pre[^>]*tabindex="0"[^>]*role="region"/); assert.match(html, /ONE MODE CHOICE FOR THE WHOLE FIELD/); assert.doesNotMatch(html, /katex-error|\$\$|@@E\d/); assert.doesNotMatch(source, /^# |^---/m);
 if (language === 'en') { assert.ok(source.split(/\s+/).length > 2400); assert.doesNotMatch(html, /[가-힣]/); } else assert.ok((source.match(/[가-힣]/g) ?? []).length > 4200);
});
test('both translations preserve all displayed equations and the quantitative ledger cells', () => {
 const equations = (source: string) => [...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ''));
 assert.deepEqual(equations(raw('ko')), equations(raw('en')));
 for (const language of ['ko', 'en'] as const) {
  assert.equal(tables(raw(language)).length, 4);
  assert.deepEqual(tables(raw(language))[2].map(row => row.slice(1)), [['cv', '+∞', '0'], ['0', 'v(1+c)', '1/(1+c)'], ['0', 'v', '1']]);
 }
});
test('both printed count ledgers match exact IID, collective and mixed probabilities and inclusive events', () => {
 const r = model(base).current;
 for (const language of ['ko', 'en'] as const) {
  const rows = tables(raw(language))[1];
  assert.deepEqual(rows.map(row => row.slice(0, 4)), r.distribution.map(row => [String(row.count), printed(row.independentProbability), printed(row.collectiveProbability), printed(row.probability)]));
  assert.deepEqual(rows.map(row => row[4] === (language === 'ko' ? '예' : 'yes')), r.distribution.map(row => row.inDeviationEvent));
 }
 equal(r.diagonalCovarianceSum, 1); equal(r.orderedCrossCovarianceSum, 3, 2); assert.equal(r.orderedCrossCovarianceMultiplicity, 12); equal(r.summary.densityVariance, 5, 32); equal(r.summary.varianceEquivalentCount!, 8, 5);
 equal(r.summary.eventProbability, 9, 16); equal(r.summary.chebyshevBound, 5, 8); equal(r.summary.mixtureAwareBound, 5, 8); equal(r.summary.independentReferenceBound, 1, 4); assert.equal(r.independentReferenceBoundFailsForMixture, true);
});
test('the smaller-tail exercise and both finite-size path examples match their stated preparations', () => {
 const r = model({ ...base, siteCount: 2, occupiedWeight: 1, emptyWeight: 9, epsilonNumerator: 3, epsilonDenominator: 20 }).current;
 assert.deepEqual(r.eventCounts, [1, 2]); equal(r.summary.eventProbability, 29, 200); equal(r.summary.independentEventProbability, 19, 100); equal(r.summary.collectiveEventProbability, 1, 10); equal(r.summary.densityVariance, 27, 400); equal(r.summary.independentDensityVariance, 9, 200);
 for (const [commonScaling, n, d, coefficientN, coefficientD] of [['inverse-size', 11, 128, 3, 8], ['inverse-size-squared', 35, 512, 1, 4]] as const) {
  const m = model({ ...base, commonScaling }); equal(m.current.summary.densityVariance, n, d); assert.equal(m.limits.scaledDensityVariance.kind, 'finite'); if (m.limits.scaledDensityVariance.kind === 'finite') equal(m.limits.scaledDensityVariance.value, coefficientN, coefficientD);
 }
});
test('the event-boundary and deterministic exercises keep mass and missing pair information distinct', () => {
 const r = model({ ...base, siteCount: 3, occupiedWeight: 1, emptyWeight: 2, epsilonNumerator: 1, epsilonDenominator: 3 }); assert.deepEqual(r.current.eventCounts, [0, 2, 3]);
 assert.deepEqual(model({ ...r.parameters, epsilonNumerator: 333334, epsilonDenominator: 1000000 }).current.eventCounts, [3]);
 const deterministic = model({ ...base, occupiedWeight: 0 }).current; equal(deterministic.summary.eventProbability, 0); equal(deterministic.distinctPairCovariance!, 0); assert.equal(deterministic.distinctPairCorrelation, null); assert.equal(deterministic.summary.varianceEquivalentCount, null);
 const one = model({ ...base, siteCount: 1 }); assert.equal(one.current.distinctPairCovariance, null); assert.equal(one.classification.selfAveragesInProbabilityAlongPath, false);
});
test('the rare-event example keeps its exact tiny probability, finite huge ratio and IID leading variance coefficient', () => {
 const r = model({ ...base, siteCount: 512, occupiedWeight: 1, emptyWeight: 999999, commonNumerator: 1, commonDenominator: 1000000, commonScaling: 'inverse-size-squared', epsilonNumerator: 999999, epsilonDenominator: 1000000 });
 assert.deepEqual(r.current.eventCounts, [512]); equal(r.current.summary.independentEventProbability, BigInt(1), BigInt(10) ** BigInt(3072)); equal(r.current.collectiveEventContribution, BigInt(1), BigInt(1000000) ** BigInt(2) * BigInt(512) ** BigInt(2));
 assert.equal(r.current.summary.independentEventProbability.value, null); assert.equal(r.current.summary.independentEventProbability.sign, 1); assert.equal(r.current.eventTailRatioToIndependent!.value, null); assert.equal(r.current.eventTailRatioToIndependent!.resolution, 'exact-rational'); assert.ok(r.current.eventTailRatioToIndependent!.logMagnitude! > 7000);
 assert.equal(r.limits.scaledDensityVariance.kind, 'finite'); if (r.limits.scaledDensityVariance.kind === 'finite') equal(r.limits.scaledDensityVariance.value, 999999, BigInt(10) ** BigInt(12));
});
