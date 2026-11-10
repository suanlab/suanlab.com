import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { build } from 'esbuild';
import { finiteSizeKinds as kinds, finiteSizePresets as presets, finiteSizeGuides as guides, finiteSizePresetInput as preset, finiteSizeTitles, parseFiniteSizeInteger as parse, isFiniteSizeKind } from '../src/lib/learning-finite-size';
import { finiteSizeConceptSteps as steps, finiteSizeConceptQuestions as questions, finiteSizeConceptInputs as samples, finiteSizeConceptIntuition as intuition } from '../src/lib/learning-finite-size-concepts';
import { isFiniteSizeInput, finiteSizeInputKey as key, isFiniteSizeSuccessForInput } from '../src/lib/learning-finite-size-protocol';
import { evaluateFiniteSizeRequest as evaluate } from '../src/lib/learning-finite-size-evaluator';
import { useFiniteSizeSeries } from '../src/components/learning/use-finite-size-series';
import { finiteBoundaryExtensivityModel as boundary } from '../src/lib/learning-boundary-extensivity';
import { spatialDensitySelfAveragingModel as spatial } from '../src/lib/learning-spatial-self-averaging';
import { twoPhaseRoundingModel as two } from '../src/lib/learning-two-phase-rounding';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
const hangul = /[가-힣]/;
function equal(q: Q, n: number | bigint, d: number | bigint = 1) { assert.ok(q.bounds); assert.deepEqual(q.bounds.lower, q.bounds.upper); const a = q.bounds.lower; assert.equal(a.numerator.exponent, 0); assert.equal(a.denominator.exponent, 0); assert.equal(BigInt(a.numerator.coefficient) * BigInt(d), BigInt(n) * BigInt(a.denominator.coefficient)); }
for (const [family, kind] of kinds.entries()) test(`${kind}: bilingual teaching preparations and all fixed comparisons use valid real worker inputs`, () => {
 assert.equal(presets[kind].length, [12, 16, 16][family]); assert.equal(steps[kind].length, 6); assert.equal(guides[kind].length, 3); assert.equal(questions[kind].length, 3);
 const ids = new Set<string>(), inputKeys = new Set<string>(), texts = [finiteSizeTitles[kind], intuition[kind]];
 for (const [index, item] of presets[kind].entries()) {
  assert.ok(!ids.has(item.id)); ids.add(item.id); texts.push(item.label); const input = preset(kind, index); assert.equal(input.kind, kind); assert.ok(isFiniteSizeInput(input)); assert.ok(Object.isFrozen(input)); assert.ok(Object.isFrozen(input.parameters)); assert.ok(!inputKeys.has(key(input)), item.id); inputKeys.add(key(input));
  const response = evaluate({ version: 1, requestId: 1, input }); assert.ok(response.ok, item.id); assert.ok(isFiniteSizeSuccessForInput(response, input, 1), item.id);
 }
 const reachable = new Set<number>();
 for (const guide of guides[kind]) { texts.push(guide.title, guide.predict, guide.action, guide.explain); for (const index of guide.presetIndices) { assert.ok(presets[kind][index]); reachable.add(index); } }
 assert.equal(reachable.size, presets[kind].length, 'All presets are reachable from the guides');
 for (const [index, step] of steps[kind].entries()) {
  texts.push(step.title, step.explanation, step.diagram, step.check.question, step.check.explanation, ...step.check.options); assert.equal(step.check.options.length, 3); assert.ok(Number.isInteger(step.check.answer) && step.check.answer >= 0 && step.check.answer < 3); assert.ok(step.formula.length > 8); assert.ok(step.explanation.ko.length > 80); assert.ok(step.explanation.en.split(/\s+/).length > 35); assert.ok(!ids.has(step.check.id)); ids.add(step.check.id);
  const fixed = samples(kind, index); assert.ok(fixed.length > 0); for (const input of fixed) { assert.equal(input.kind, kind); assert.ok(isFiniteSizeInput(input)); assert.ok(inputKeys.has(key(input))); assert.ok(Object.isFrozen(input.parameters)); }
 }
 for (const question of questions[kind]) { assert.ok(!ids.has(question.id)); ids.add(question.id); const source = steps[kind].find(step => `${step.check.id}-quiz` === question.id)!.check; assert.equal(question.answer, source.answer); assert.deepEqual(question.question, source.question); assert.notEqual(question.options, source.options); assert.notEqual(question.options[0], source.options[0]); assert.notEqual(question.explanation, source.explanation); }
 for (const text of texts) { assert.ok(text.ko.trim().length, text.ko); assert.ok(text.en.trim().length); assert.equal(hangul.test(text.en), false, text.en); }
});
test('integer drafts and preset/step lookup reject truncation, underflow, invalid indices and unknown kinds', () => {
 for (const [text, value] of [['0', 0], ['+0001', 1], [' 512 ', 512], ['1000000', 1000000]] as const) assert.equal(parse(text), value);
 for (const text of ['', ' ', '-0', '-1', '1.0', '1.5', '1e-999', '1e6', 'Infinity', 'NaN', '0x10', '1000001', '999999999999999999999999', '1 2']) assert.equal(parse(text), null, text);
 for (const kind of kinds) { assert.ok(isFiniteSizeKind(kind)); for (const index of [-1, .5, NaN, Infinity, 100]) { assert.throws(() => preset(kind, index), RangeError); assert.throws(() => samples(kind, index), RangeError); } }
 assert.equal(isFiniteSizeKind('unknown'), false);
});
test('boundary teaching answers preserve geometry, norms, counterterms, cancellation and exact counts', () => {
 const m = (i: number) => { const input = preset(kinds[0], i); if (input.kind !== kinds[0]) throw Error(); return boundary(input.parameters); };
 assert.equal(m(0).current.geometry.siteCount, '12'); equal(m(0).current.geometry.interfaceFacesPerSite, 1, 4); equal(m(1).current.geometry.interfaceFacesPerSite, 1, 16); equal(m(3).current.geometry.interfaceFacesPerSite, 1, 4);
 assert.equal(m(1).classification.maximumEnergyGrowth['complete-kac'], 'linear-in-site-count'); assert.equal(m(1).classification.kacCutNegligiblePerSiteAlongPath, false); equal(m(1).current.modes['complete-kac'].crossMagnitudePerSite, 1, 4);
 for (const i of [4, 5]) { equal(m(i).current.separatelyNormalizedKac.combinedMinusSeparate, -1, 2); equal(m(i).current.modes['complete-kac'].crossEnergyPerSite, -1, 4); }
 equal(m(0).current.modes['complete-unscaled'].energyPerSite, 1, 2); assert.equal(m(0).classification.maximumEnergyGrowth['complete-unscaled'], 'quadratic-in-site-count'); equal(m(6).current.modes['complete-unscaled'].energyPerSite, 1, 5); equal(m(7).current.modes['complete-unscaled'].energyPerSite, -7, 10);
 assert.equal(m(8).current.modes['nearest-neighbor'].internalPairCountOne, '0'); assert.equal(m(8).current.modes['nearest-neighbor'].crossPairCount, '1'); assert.equal(m(9).classification.vanHoveGeometryAlongPath, false); equal(m(9).current.modes['nearest-neighbor'].combinedEnergy, 0);
 assert.equal(m(10).current.geometry.siteCount, '2000000000000000000000000000000000000'); assert.ok(m(10).current.modes['complete-kac'].energyPerSite.value! > 0);
});
test('spatial teaching answers preserve ordered covariances, scaling paths, inclusive events and rare tails', () => {
 const m = (i: number) => { const input = preset(kinds[1], i); if (input.kind !== kinds[1]) throw Error(); return spatial(input.parameters); };
 equal(m(0).current.diagonalCovarianceSum, 1); equal(m(0).current.orderedCrossCovarianceSum, 3, 2); assert.equal(m(0).current.orderedCrossCovarianceMultiplicity, 12); equal(m(0).current.summary.densityVariance, 5, 32); equal(m(1).current.summary.densityVariance, 1, 16); equal(m(2).current.summary.densityVariance, 1, 4);
 equal(m(3).current.summary.densityVariance, 11, 128); equal(m(4).current.summary.densityVariance, 35, 512); assert.equal(m(5).classification.selfAveragesInProbabilityAlongPath, false); for (const i of [6, 7]) assert.equal(m(i).classification.selfAveragesInProbabilityAlongPath, true);
 const low = m(8).current; equal(low.summary.densityVariance, 27, 400); equal(low.summary.independentDensityVariance, 9, 200); equal(low.summary.eventProbability, 29, 200); equal(low.summary.independentEventProbability, 19, 100);
 assert.deepEqual(m(9).current.eventCounts, [0, 2, 3]); assert.deepEqual(m(10).current.eventCounts, [3]); equal(m(11).current.distinctPairCovariance!, 0); assert.equal(m(11).current.distinctPairCorrelation, null); assert.equal(m(12).current.distinctPairCovariance, null); assert.deepEqual(m(13).current.eventCounts, []); equal(m(13).current.summary.eventProbability, 0); assert.ok(m(13).current.summary.chebyshevBound.value! > 0);
 const rare = m(14), iid = m(15); assert.equal(rare.current.summary.eventProbability.sign, 1); assert.equal(rare.current.eventTailRatioToIndependent!.value, null); assert.equal(rare.current.eventTailRatioToIndependent!.resolution, 'exact-rational'); assert.equal(iid.current.summary.eventProbability.value, null); equal(iid.current.eventTailRatioToIndependent!, 1); assert.deepEqual(rare.limits.scaledDensityVariance, iid.limits.scaledDensityVariance);
});
test('two-phase answers preserve state-space, coexistence response, path limits, root parity and nonzero logarithms', () => {
 const m = (i: number) => { const input = preset(kinds[2], i); if (input.kind !== kinds[2]) throw Error(); return two(input.parameters); };
 equal(m(0).current.partitionFunction, 17); equal(m(0).current.meanDensity, 16, 17); assert.deepEqual(m(0).current.densityVariance, m(1).current.densityVariance); assert.deepEqual(m(0).current.pressureCorrection, m(1).current.pressureCorrection);
 equal(m(2).current.densityVariance, 1, 4); equal(m(3).current.densityVariance, 1, 4); equal(m(2).current.densityResponseToField, 1); equal(m(3).current.densityResponseToField, 16); assert.equal(m(2).current.pressureCorrection.value, 16 * m(3).current.pressureCorrection.value!); assert.equal(m(2).current.tenToNinetyFieldWidth.value, 16 * m(3).current.tenToNinetyFieldWidth.value!);
 equal(m(4).current.meanDensity, 6561, 10657); equal(m(6).current.meanDensity, 1185921, 2234497); equal(m(6).limits.meanDensity, 1, 2); assert.ok(m(4).limits.meanDensity.value! > .62); assert.ok(m(5).limits.meanDensity.value! < .38); assert.equal(m(10).rootGeometry.negativeRealRootExists, true); assert.equal(m(11).rootGeometry.negativeRealRootExists, false);
 for (const i of [12, 13]) { const r = m(i); assert.equal(r.current.pressureCorrection.value, null); assert.equal(r.current.pressureCorrection.sign, 1); assert.equal(r.current.pressureCorrection.resolution, 'interval-resolved'); assert.equal(r.current.exponentialPressureCorrection.value, 1); }
 equal(m(14).current.activity, 1); equal(m(14).current.meanDensity, 1, 2); assert.equal(m(15).current.field.sign, 1);
});
test('fixed-series SSR reads no Worker and the teaching main graph excludes numerical implementations', async () => {
 const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let accesses = 0; Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { accesses++; throw Error('SSR Worker'); } });
 try { function Probe() { const series = useFiniteSizeSeries(samples(kinds[0], 1)); assert.deepEqual(series.responses, []); return React.createElement('span', null, series.status); } assert.equal(renderToStaticMarkup(React.createElement(Probe)), '<span>pending</span>'); assert.equal(accesses, 0); }
 finally { if (descriptor) Object.defineProperty(globalThis, 'Worker', descriptor); else Reflect.deleteProperty(globalThis, 'Worker'); }
 const bundle = await build({ entryPoints: ['src/lib/learning-finite-size-concepts.ts', 'src/components/learning/use-finite-size-series.ts'], bundle: true, write: false, outdir: '/tmp/finite-size-teaching-check', platform: 'browser', format: 'esm', metafile: true, external: ['react'], logLevel: 'silent' });
 for (const path of Object.keys(bundle.metafile!.inputs)) assert.doesNotMatch(path, /learning-(boundary-extensivity|spatial-self-averaging|two-phase-rounding|finite-markov-exact|rational-gcd|binary-products|interval-transcendentals|finite-size-evaluator)\./);
});
