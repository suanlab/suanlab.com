import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { build } from 'esbuild';
import { detailedBalanceKinds as kinds, detailedBalancePresetInput as preset } from '../src/lib/learning-detailed-balance';
import { detailedBalanceInputKey as key } from '../src/lib/learning-detailed-balance-protocol';
import { detailedBalanceFields as fields, editDetailedBalanceDraft as edit, advanceDetailedBalanceTime as advance, rewindDetailedBalanceTime as rewind } from '../src/lib/learning-detailed-balance-controls';
import Lab from '../src/components/learning/detailed-balance-lab';
import Cover from '../src/components/learning/detailed-balance-cover';
import { LanguageProvider } from '../src/components/language-provider';
import { stationaryThreeStateCycleModel as cycle } from '../src/lib/learning-stationary-cycle';
import { metropolisHastingsProposalModel as hastings } from '../src/lib/learning-hastings-proposal';
import { reversibleTwoStateMixingModel as two } from '../src/lib/learning-two-state-mixing';
(globalThis as typeof globalThis & { React?: typeof React }).React = React;
for (const [family, kind] of kinds.entries()) test(`${kind}: controls edit owned exact preparations and reject invalid field paths`, () => {
 const input = preset(kind, 0), original = key(input), descriptors = fields(input);
 assert.equal(descriptors.length, [7, 6, 6][family]); assert.equal(new Set(descriptors.map(f => f.id)).size, descriptors.length);
 for (const field of descriptors) {
  assert.match(field.label.ko, /[가-힣]/); assert.doesNotMatch(field.label.en, /[가-힣]/);
  const result = edit(input, field.id, String(field.value)); assert.ok(result); assert.equal(key(result), original); assert.ok(Object.isFrozen(result)); assert.notEqual(result.parameters, input.parameters);
  for (const value of ['1.2', '1e-999', '-1', 'NaN', '1000001', '']) assert.equal(edit(input, field.id, value), null);
 }
 for (const path of ['options', '__proto__', 'initialWeights.3', 'targetWeights.-1', 'stepCount.extra']) assert.equal(edit(input, path, '1'), null);
 assert.equal(key(input), original);
});
test('denominator edits never clip existing numerators or substitute zero for invalid probabilities', () => {
 const input = preset(kinds[2], 0); assert.equal(edit(input, 'transitionDenominator', '1'), null); assert.equal(edit(input, 'transitionDenominator', '0'), null); assert.equal(edit(input, 'initialDenominator', '0'), null);
 const upper = edit(input, 'forwardNumerator', '4'); assert.ok(upper); assert.equal(edit(upper, 'transitionDenominator', '3'), null); assert.equal(edit(input, 'forwardNumerator', '5'), null);
 const initialOne = edit(input, 'initialOneNumerator', '1'); assert.ok(initialOne); assert.equal(edit(initialOne, 'initialDenominator', '0'), null);
 assert.equal(edit(preset(kinds[1], 0), 'targetWeights.0', '0'), null);
 const identity = preset(kinds[0], 5); assert.equal(edit(identity, 'stayWeight', '0'), null);
 const zero = edit(preset(kinds[0], 0), 'initialWeights.0', '0'); assert.equal(zero, null);
});
test('manual time changes preserve kernel and initial law while advancing actual exact distributions', () => {
 const input = preset(kinds[0], 10), one = advance(input); assert.ok(one); assert.equal(one.kind, kinds[0]); if (one.kind !== kinds[0]) throw Error();
 assert.equal(one.parameters.stepCount, 1); assert.deepEqual(one.parameters.initialWeights, [1, 0, 0]); assert.deepEqual(cycle(one.parameters).current.probabilities.map(q => q.value), [.25, .5, .25]);
 const binary = preset(kinds[2], 0), next = advance(binary); assert.ok(next); if (next.kind !== kinds[2] || binary.kind !== kinds[2]) throw Error();
 assert.equal(next.parameters.stepCount, binary.parameters.stepCount + 1); assert.equal(next.parameters.initialOneNumerator, binary.parameters.initialOneNumerator); assert.equal(next.parameters.transitionDenominator, binary.parameters.transitionDenominator);
 const beginning = rewind(next); assert.equal(beginning.parameters.stepCount, 0); assert.deepEqual(two(beginning.parameters as typeof next.parameters).current.probabilities.map(q => q.value), [1, 0]);
 const end = edit(binary, 'stepCount', '512'); assert.ok(end); assert.equal(advance(end), null); assert.equal(advance(preset(kinds[1], 0)), null);
});
test('fixed covers use the same reference currents, accepted flow and alternating error as the models', () => {
 const cycleInput = preset(kinds[0], 1); if (cycleInput.kind !== kinds[0]) throw Error(); assert.equal(cycle(cycleInput.parameters).clockwiseCurrent.value, 1 / 12);
 const hInput = preset(kinds[1], 0); if (hInput.kind !== kinds[1]) throw Error(); const h = hastings(hInput.parameters); assert.equal(h.corrected.targetFlux[2][0].value, 1 / 28); assert.equal(h.corrected.acceptance[2][0]!.value, 1 / 8);
 const history = two({ forwardNumerator: 3, backwardNumerator: 3, transitionDenominator: 4, initialOneNumerator: 1, initialDenominator: 1, stepCount: 6 }).history;
 assert.deepEqual(history.map(s => 125 - 120 * s.deviationFromStationary![1].value!), [65, 155, 110, 132.5, 121.25, 126.875, 124.0625]);
 for (const kind of kinds) { const markup = renderToStaticMarkup(React.createElement(Cover, { kind })); assert.match(markup, /aria-hidden="true"/); assert.ok(markup.includes(`data-detailed-balance-cover="${kind}"`)); assert.doesNotMatch(markup, /NaN|undefined/); }
});
test('saved laboratory SSR waits for browser state, does not execute Workers and has no precision selector', async () => {
 const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let reads = 0;
 Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { reads++; throw Error('Unexpected SSR Worker'); } });
 try { for (const kind of kinds) { const html = renderToStaticMarkup(React.createElement(LanguageProvider, null, React.createElement(Lab, { kind }))); assert.match(html, /data-detailed-balance-state="loading"/); assert.match(html, /data-detailed-balance-ready="false"/); assert.doesNotMatch(html, /data-detailed-balance-result=|<select/); } assert.equal(reads, 0); }
 finally { if (descriptor) Object.defineProperty(globalThis, 'Worker', descriptor); else Reflect.deleteProperty(globalThis, 'Worker'); }
 const result = await build({ entryPoints: ['src/components/learning/detailed-balance-lab.tsx', 'src/components/learning/detailed-balance-cover.tsx'], bundle: true, write: false, outdir: '/tmp/detailed-balance-lab-check', platform: 'browser', format: 'esm', metafile: true, external: ['react', 'next/navigation'] });
 for (const path of Object.keys(result.metafile!.inputs)) assert.doesNotMatch(path, /learning-(stationary-cycle|hastings-proposal|two-state-mixing|finite-markov-exact|binary-products|rational-gcd|detailed-balance-evaluator)\./);
});
