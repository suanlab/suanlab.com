import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { build } from 'esbuild';
import { detailedBalanceKinds as kinds, detailedBalancePresets as presets, detailedBalancePresetInput as preset } from '../src/lib/learning-detailed-balance';
import { evaluateDetailedBalanceRequest as evaluate } from '../src/lib/learning-detailed-balance-evaluator';
import { DetailedBalanceResultView, detailedBalanceLog } from '../src/components/learning/detailed-balance-view';
import Concept from '../src/components/learning/detailed-balance-concept';
import type { DetailedBalanceSuccess } from '../src/lib/learning-detailed-balance-protocol';
const responses = new Map<string, DetailedBalanceSuccess>();
function response(family: number, index: number) {
 const id = `${family}:${index}`, cached = responses.get(id); if (cached) return cached;
 const result = evaluate({ version: 1, requestId: 1, input: preset(kinds[family], index) }); assert.ok(result.ok); responses.set(id, result); return result;
}
const view = (family: number, index: number, language: 'ko' | 'en' = 'en') => renderToStaticMarkup(React.createElement(DetailedBalanceResultView, { response: response(family, index), language }));
for (const [family, kind] of kinds.entries()) test(`${kind}: all preparations render bilingual results without Worker access or object-valued fractions`, () => {
 const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let reads = 0;
 Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { reads++; throw Error('Unexpected SSR Worker'); } });
 try {
  for (let index = 0; index < presets[kind].length; index++) for (const language of ['ko', 'en'] as const) {
   const html = view(family, index, language); assert.ok(html.includes('data-detailed-balance-result=')); assert.doesNotMatch(html, /\[object Object\]|NaN|Infinity|>undefined</);
   assert.ok(html.includes('role="img"')); assert.ok(html.includes('<caption')); assert.ok(html.includes('scope="row"')); assert.ok(html.includes('Inspect exact rational') || language === 'ko');
   if (language === 'en') assert.doesNotMatch(html, /[가-힣]/); else assert.match(html, /[가-힣]/);
  }
  const concept = renderToStaticMarkup(React.createElement(Concept, { kind, language: 'en' })); assert.match(concept, /Calculating fixed examples/); assert.doesNotMatch(concept, /[가-힣]/); assert.equal(reads, 0);
 } finally { if (descriptor) Object.defineProperty(globalThis, 'Worker', descriptor); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
test('display logs retain nonzero tails while exact zero and missing references remain absent', () => {
 const r = response(0, 9); assert.equal(r.kind, kinds[0]); if (r.kind !== kinds[0]) throw Error();
 const tiny = r.result.current.totalVariationFromUniform; assert.equal(tiny.value, null); assert.ok(Number.isFinite(detailedBalanceLog(tiny))); assert.ok(detailedBalanceLog(tiny)! < -700);
 assert.equal(detailedBalanceLog(null), null); assert.equal(detailedBalanceLog(r.result.stationaryResidual[0]), null);
 assert.ok(view(0, 9).includes('Nonzero · outside display range')); assert.ok(view(0, 9).includes('outside display range'));
});
test('identity, periodicity and absorption display distinct reference, limit and mixing meanings', () => {
 const identity = view(2, 5), flip = view(2, 2), absorbing = view(2, 6), slow = view(2, 9);
 assert.match(identity, /Inapplicable: nonunique stationary law/); assert.match(identity, /Every initial law is stationary/); assert.match(identity, /Not provided for this preparation/);
 assert.match(flip, /Never reached: exact alternation/); assert.match(flip, /Not provided for this preparation/);
 assert.match(absorbing, /normalized correlation/); assert.match(absorbing, /Not provided for this preparation/); assert.match(absorbing, /Reached within the recorded range/);
 assert.match(slow, /Not reached within the recorded range/); assert.doesNotMatch(slow, /Never reached: exact alternation/);
});
test('flow captions keep final-step current preparation separate from stationary reference and history', () => {
 const html = view(0, 0); assert.match(html, /One-step flow from requested t=3/); assert.match(html, /does not change with table pages/); assert.match(html, /Stationary net current J/); assert.match(html, /Time reversal P\*/);
 assert.match(html, /Blue follows 0/); assert.match(html, /Line width is not a probability scale/);
});
test('exact inspector shows integer coefficients and main graph excludes numerical runtime modules', async () => {
 const html = view(0, 1); assert.match(html, />1 \/ 12</); assert.doesNotMatch(html, /\[object Object\]/);
 const result = await build({ entryPoints: ['src/components/learning/detailed-balance-view.tsx', 'src/components/learning/detailed-balance-concept.tsx'], bundle: true, write: false, outdir: '/tmp/detailed-balance-views-check', platform: 'browser', format: 'esm', metafile: true, external: ['react'] });
 for (const file of Object.keys(result.metafile!.inputs)) assert.doesNotMatch(file, /learning-(stationary-cycle|hastings-proposal|two-state-mixing|finite-markov-exact|binary-products|rational-gcd|detailed-balance-evaluator)\./);
});
