import test from 'node:test';
import assert from 'node:assert/strict';
import { learningUnits, learningGuides } from '../src/data/interactive-book';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { build } from 'esbuild';
import { detailedBalanceKinds as kinds, detailedBalancePresets as presets, detailedBalanceGuides as guides, detailedBalancePresetInput as preset, detailedBalanceTitles, parseDetailedBalanceInteger as parse, isDetailedBalanceKind } from '../src/lib/learning-detailed-balance';
import { detailedBalanceConceptSteps as steps, detailedBalanceConceptQuestions as questions, detailedBalanceConceptInputs as samples, detailedBalanceConceptIntuition as intuition } from '../src/lib/learning-detailed-balance-concepts';
import { isDetailedBalanceInput, detailedBalanceInputKey as key, isDetailedBalanceSuccessForInput } from '../src/lib/learning-detailed-balance-protocol';
import { evaluateDetailedBalanceRequest as evaluate } from '../src/lib/learning-detailed-balance-evaluator';
import { useDetailedBalanceSeries } from '../src/components/learning/use-detailed-balance-series';
import { stationaryThreeStateCycleModel as cycle } from '../src/lib/learning-stationary-cycle';
import { metropolisHastingsProposalModel as hastings } from '../src/lib/learning-hastings-proposal';
import { reversibleTwoStateMixingModel as two } from '../src/lib/learning-two-state-mixing';
const hangul = /[가-힣]/;
for (const [family, kind] of kinds.entries()) test(`${kind}: bilingual fixed teaching preparations execute through the real evaluator`, () => {
 assert.equal(presets[kind].length, [12, 10, 14][family]);
 assert.equal(steps[kind].length, 6); assert.equal(guides[kind].length, 3); assert.equal(questions[kind].length, 3);
 const ids = new Set<string>();
 for (const item of presets[kind]) {
  assert.ok(!ids.has(item.id)); ids.add(item.id);
  const input = preset(kind, presets[kind].indexOf(item)); assert.equal(input.kind, kind); assert.ok(isDetailedBalanceInput(input));
  assert.ok(Object.isFrozen(input)); assert.ok(Object.isFrozen(input.parameters));
  const response = evaluate({ version: 1, requestId: 1, input }); assert.ok(response.ok, item.id); assert.ok(isDetailedBalanceSuccessForInput(response, input, 1), item.id);
 }
 const texts = [detailedBalanceTitles[kind], intuition[kind], ...presets[kind].map(p => p.label)];
 const used = new Set<number>();
 for (const guide of guides[kind]) {
  texts.push(guide.title, guide.predict, guide.action, guide.explain);
  for (const index of guide.presetIndices) { assert.ok(presets[kind][index]); used.add(index); }
 }
 assert.equal(used.size, presets[kind].length, 'Every preparation is reachable from a guide');
 for (const [index, step] of steps[kind].entries()) {
  texts.push(step.title, step.explanation, step.diagram, step.check.question, step.check.explanation, ...step.check.options);
  assert.equal(step.check.options.length, 3); assert.ok(step.check.answer >= 0 && step.check.answer < 3); assert.ok(step.formula.length > 8);
  assert.ok(!ids.has(step.check.id)); ids.add(step.check.id);
  assert.ok(step.explanation.ko.length > 70); assert.ok(step.explanation.en.split(/\s+/).length > 30);
  const inputs = samples(kind, index); assert.ok(inputs.length);
  for (const input of inputs) { assert.equal(input.kind, kind); assert.ok(isDetailedBalanceInput(input)); assert.ok(presets[kind].some(item => key(item.input) === key(input))); }
 }
 for (const question of questions[kind]) {
  assert.ok(!ids.has(question.id)); ids.add(question.id);
  const source = steps[kind].find(step => `${step.check.id}-quiz` === question.id)!.check;
  assert.equal(question.answer, source.answer); assert.notEqual(question.options, source.options); assert.notEqual(question.options[0], source.options[0]);
 }
 for (const text of texts) { assert.ok(hangul.test(text.ko) || /^[\d/]+$/.test(text.ko), text.ko); assert.ok(text.en.trim().length); assert.equal(hangul.test(text.en), false, text.en); }
});
test('integer drafts cannot silently become zero, truncate, overflow or accept non-integers', () => {
 for (const [draft, expected] of [['0', 0], ['+0001', 1], [' 512 ', 512], ['1000000', 1000000]] as const) assert.equal(parse(draft), expected);
 for (const draft of ['', ' ', '-1', '-0', '1.5', '1.0', '1e-999', '1e6', 'NaN', 'Infinity', '0x10', '1000001', '999999999999999999999999', '1 2']) assert.equal(parse(draft), null, draft);
 for (const kind of kinds) { assert.ok(isDetailedBalanceKind(kind)); for (const index of [-1, .5, NaN, Infinity, 100]) { assert.throws(() => preset(kind, index), RangeError); assert.throws(() => samples(kind, index), RangeError); } }
 assert.equal(isDetailedBalanceKind('unknown'), false);
});
test('cycle examples preserve stationarity versus currents, period and reference distinctions', () => {
 const model = (index: number) => { const input = preset(kinds[0], index); assert.equal(input.kind, kinds[0]); if (input.kind !== kinds[0]) throw Error(); return cycle(input.parameters); };
 const driven = model(1); assert.equal(driven.classification.currentDistributionIsStationary, true); assert.equal(driven.classification.detailedBalanceAtUniform, false); assert.equal(driven.clockwiseCurrent.value, 1 / 12);
 assert.equal(model(2).classification.uniqueClassPeriod, 1); assert.equal(model(3).classification.uniqueClassPeriod, 3); assert.equal(model(4).classification.uniqueClassPeriod, 1);
 assert.equal(model(5).classification.stationaryReferenceIsUnique, false); assert.equal(model(5).classification.currentDistributionIsStationary, true); assert.ok(model(5).current.totalVariationFromUniform.value! > 0);
 assert.ok(model(8).current.contractionUpperBound.value! > model(8).current.totalVariationFromUniform.value!);
 const tiny = model(9); assert.equal(tiny.current.totalVariationFromUniform.value, null); assert.equal(tiny.current.totalVariationFromUniform.sign, 1); assert.equal(tiny.classification.currentDistributionIsStationary, false);
});
test('Hastings examples distinguish accidental invariance, missing proposals, moves and jump law', () => {
 const model = (index: number) => { const input = preset(kinds[1], index); if (input.kind !== kinds[1]) throw Error(); return hastings(input.parameters); };
 const base = model(0); assert.equal(base.corrected.acceptance[2][0]!.value, 1 / 8); assert.equal(base.uncorrected.targetInvariant, false); assert.equal(base.corrected.targetDetailedBalance, true);
 assert.deepEqual(base.corrected.jumpStationaryDistribution!.map(v => v.value), [3 / 14, 3 / 7, 5 / 14]);
 const accidental = model(2); assert.equal(accidental.uncorrected.targetInvariant, true); assert.equal(accidental.uncorrected.targetDetailedBalance, false); assert.equal(accidental.uncorrected.stationaryCurrent![0][1].value, 1 / 24);
 const missing = model(4); assert.equal(missing.corrected.acceptance[0][2], null); assert.equal(missing.corrected.acceptance[0][1]!.value, 0); assert.equal(missing.corrected.stationaryDistribution, null);
 assert.equal(model(6).corrected.meanProposalAcceptanceAtTarget.value, 1); assert.equal(model(6).corrected.meanMovementAtTarget.value, 0); assert.equal(model(6).corrected.jumpStationaryDistribution, null);
 const high = model(7); assert.ok(high.corrected.meanProposalAcceptanceAtTarget.value! > .999); assert.ok(high.corrected.meanMovementAtTarget.value! < .001);
 const drift = model(9).uncorrected.targetResidual[0]; assert.equal(drift.sign, 1); assert.ok(drift.value! > 0); assert.equal(drift.value!.toFixed(6), '0.000000');
});
test('two-state examples preserve periodicity, Cesaro counting, support and finite-range mixing semantics', () => {
 const model = (index: number) => { const input = preset(kinds[2], index); if (input.kind !== kinds[2]) throw Error(); return two(input.parameters); };
 assert.equal(model(0).stationaryDistribution![1].value, 1 / 3); assert.equal(model(1).secondEigenvalue.sign, -1);
 assert.equal(model(2).current.cesaroProbabilityOne.value, 1 / 3); assert.equal(model(3).current.cesaroProbabilityOne.value, .5); assert.equal(model(3).current.probabilities[1].value, 1);
 assert.equal(model(4).classification.currentDistributionIsStationary, true); assert.equal(model(4).mixingAtQuarter.status, 'never-periodic'); assert.equal(model(4).signedSpectralGap.value, 2); assert.equal(model(4).absoluteSpectralGap.value, 0);
 const identity = model(5); assert.equal(identity.stationaryDistribution, null); assert.equal(identity.limitDistribution![1].value, 1 / 3); assert.equal(identity.mixingAtQuarter.status, 'not-applicable-nonunique-stationary');
 const absorb = model(6); assert.equal(absorb.classification.irreducible, false); assert.equal(absorb.classification.convergesToUniqueStationaryFromEveryInitial, true); assert.equal(absorb.stationaryTimeReversal![1], null); assert.equal(absorb.stationaryLagOneCorrelation, null);
 assert.equal(model(7).current.totalVariationFromStationary!.value, 0); assert.equal(model(8).mixingAtQuarter.firstStep, 1);
 for (const index of [9, 10]) { assert.equal(model(index).mixingAtQuarter.status, 'not-reached-within-recorded-range'); assert.equal(model(index).classification.convergesToUniqueStationaryFromEveryInitial, true); }
 for (const index of [11, 12]) { assert.equal(model(index).current.totalVariationFromStationary!.value, null); assert.equal(model(index).current.totalVariationFromStationary!.sign, 1); }
 assert.equal(model(13).current.totalVariationFromStationary!.value, 0); assert.equal(model(13).mixingAtQuarter.status, 'not-reached-within-recorded-range');
});
test('fixed-series SSR creates no Worker and main bundle excludes numerical models', async () => {
 const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let accesses = 0;
 Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { accesses++; throw Error('SSR Worker access'); } });
 try {
  function Probe() { const series = useDetailedBalanceSeries(samples(kinds[0], 1)); return React.createElement('span', null, series.status); }
  assert.equal(renderToStaticMarkup(React.createElement(Probe)), '<span>pending</span>'); assert.equal(accesses, 0);
 } finally { if (descriptor) Object.defineProperty(globalThis, 'Worker', descriptor); else Reflect.deleteProperty(globalThis, 'Worker'); }
 const bundle = await build({ entryPoints: ['src/lib/learning-detailed-balance-concepts.ts', 'src/components/learning/use-detailed-balance-series.ts'], bundle: true, write: false, outdir: '/tmp/detailed-balance-teaching-check', platform: 'browser', format: 'esm', metafile: true, external: ['react'] });
 for (const path of Object.keys(bundle.metafile!.inputs)) assert.doesNotMatch(path, /learning-(stationary-cycle|hastings-proposal|two-state-mixing|finite-markov-exact|rational-gcd|binary-products|detailed-balance-evaluator)\./);
});
test('registered readers retain their prepared questions and six-step guides', () => {
 for (const kind of kinds) { assert.deepEqual(learningUnits.find(unit => unit.id === kind)?.questions, questions[kind]); assert.equal(learningGuides[kind].steps.length, 6); assert.equal(learningGuides[kind].experiments.length, 3); }
});
