import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup as render } from 'react-dom/server';
import { build } from 'esbuild';
import { finiteSizeKinds as kinds, finiteSizePresets as presets, finiteSizePresetInput as preset } from '../src/lib/learning-finite-size';
import { finiteSizeConceptSteps as steps } from '../src/lib/learning-finite-size-concepts';
import { evaluateFiniteSizeRequest as evaluate } from '../src/lib/learning-finite-size-evaluator';
import { FiniteSizeResultView, finiteSizeQuantity, finiteSizeLog, finiteSizeScope } from '../src/components/learning/finite-size-view';
import Concept from '../src/components/learning/finite-size-concept';
const result = (family: number, index: number) => { const response = evaluate({ version: 1, requestId: 1, input: preset(kinds[family], index) }); assert.ok(response.ok); return response; };
for (const language of ['ko', 'en'] as const) for (const [family, kind] of kinds.entries()) test(`${kind}/${language}: every prepared result renders labeled figures, table alternatives and certificates`, () => {
 for (let i = 0; i < presets[kind].length; i++) {
  const response = result(family, i), html = render(React.createElement(FiniteSizeResultView, { response, language })); assert.match(html, /data-finite-size-result-key=/); assert.match(html, /role="img" aria-labelledby=/); assert.match(html, /<table>/); assert.match(html, /<caption/); assert.match(html, /<th scope="col"/); assert.match(html, /data-finite-size-certificate/); assert.doesNotMatch(html, /NaN|\[object Object\]|Infinity/);
  if (language === 'en') assert.doesNotMatch(html, /[가-힣]/);
  if (family === 0) assert.match(html, /E0/); if (family === 1) assert.match(html, /data-finite-size-event-count="0"/); if (family === 2) assert.match(html, /1\+z/);
 }
});
test('concept panels preserve all six localized steps, fixed-input separation and SSR without worker access', () => {
 const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let reads = 0; Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { reads++; throw Error('SSR Worker'); } });
 try { for (const language of ['ko', 'en'] as const) for (const kind of kinds) { const html = render(React.createElement(Concept, { kind, language })); for (const step of steps[kind]) assert.ok(html.includes(step.title[language])); assert.match(html, /data-finite-size-concept-state="pending"/); assert.doesNotMatch(html, /data-finite-size-result-key=/); if (language === 'en') assert.doesNotMatch(html, /[가-힣]/); } assert.equal(reads, 0); }
 finally { if (descriptor) Object.defineProperty(globalThis, 'Worker', descriptor); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
test('exact zero, undefined pair, huge counts, finite rare ratios and positive logarithm intervals retain distinct display meanings', () => {
 const t = (_ko: string, en: string) => en, deterministic = result(1, 11), rare = result(1, 14), tiny = result(2, 12), huge = result(0, 10);
 assert.ok(deterministic.kind === kinds[1] && rare.kind === kinds[1] && tiny.kind === kinds[2] && huge.kind === kinds[0]);
 assert.equal(finiteSizeLog(deterministic.result.current.summary.densityVariance), null); assert.equal(finiteSizeQuantity(deterministic.result.current.distinctPairCorrelation, t), 'Unavailable: defining conditions are not met'); assert.equal(finiteSizeQuantity(deterministic.result.current.summary.densityVariance, t), '0');
 const rareHtml = render(React.createElement(FiniteSizeResultView, { response: rare, language: 'en' })); assert.match(rareHtml, /Nonzero, outside display range/); assert.doesNotMatch(finiteSizeQuantity(rare.result.current.eventTailRatioToIndependent, t), /∞|Infinity/); assert.ok(finiteSizeLog(rare.result.current.eventTailRatioToIndependent)! > 7000);
 const tinyHtml = render(React.createElement(FiniteSizeResultView, { response: tiny, language: 'en' })); assert.match(tinyHtml, /Certified interval/); assert.match(tinyHtml, /Lower bound/); assert.match(tinyHtml, /Upper bound/); assert.ok(finiteSizeLog(tiny.result.current.pressureCorrection)! < -7000);
 assert.match(render(React.createElement(FiniteSizeResultView, { response: huge, language: 'en' })), /2000000000000000000000000000000000000/);
 const scope = finiteSizeScope(preset(kinds[2], 4), t); assert.doesNotMatch(scope, /z=2\/1/); assert.match(scope, /zN=/);
});
test('the concept and result bundle contains no numerical evaluator, models or BigInt implementations', async () => {
 const bundle = await build({ entryPoints: ['src/components/learning/finite-size-concept.tsx'], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true, external: ['react'], logLevel: 'silent' });
 for (const path of Object.keys(bundle.metafile!.inputs)) assert.doesNotMatch(path, /learning-(boundary-extensivity|spatial-self-averaging|two-phase-rounding|finite-size-evaluator|finite-markov-exact|binary-products|rational-gcd|interval-transcendentals)\./);
});
