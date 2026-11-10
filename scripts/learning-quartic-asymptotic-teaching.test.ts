import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createElement } from 'react';
import { renderToString } from 'react-dom/server';
import { quarticAsymptoticSteps, quarticAsymptoticExperiments, quarticAsymptoticQuestions, quarticAsymptoticLabels } from '../src/data/quartic-asymptotic-teaching';
import { quarticAsymptoticDefault, quarticAsymptoticExamples } from '../src/lib/learning-quartic-asymptotic-input';
import { calculateQuarticAsymptotic } from '../src/lib/learning-quartic-asymptotic';
import { quarticAsymptoticView, quarticSurfaceDepth } from '../src/lib/learning-quartic-asymptotic-view';
import { parseStoredQuarticAsymptotic } from '../src/components/learning/use-saved-quartic-asymptotic';
import Lab from '../src/components/learning/quartic-asymptotic-lab';
import Result from '../src/components/learning/quartic-asymptotic-result';

test('eight concepts,six guides,eight explained questions and fourteen examples cover the declared scope bilingually', () => {
  assert.equal(quarticAsymptoticSteps.length, 8); assert.equal(quarticAsymptoticExperiments.length, 6); assert.equal(quarticAsymptoticQuestions.length, 8);
  const covered = new Set(quarticAsymptoticExperiments.flatMap(guide => guide.examples));
  for (const example of quarticAsymptoticExamples) { assert(covered.has(example.id)); assert(quarticAsymptoticLabels[example.id].ko && quarticAsymptoticLabels[example.id].en); }
  for (const guide of quarticAsymptoticExperiments) { assert(guide.explanation.en.length > 80); for (const id of guide.examples) assert(quarticAsymptoticExamples.some(example => example.id === id)); }
  assert.equal(new Set(quarticAsymptoticQuestions.map(q => q.correct)).size, 3);
  const English = [...quarticAsymptoticSteps.map(s => [s.title.en, s.body.en]), ...quarticAsymptoticExperiments.map(g => [g.title.en, g.predict.en, g.explanation.en]), ...quarticAsymptoticQuestions.map(q => [q.prompt.en, q.explanation.en, ...q.options.map(o => o.en)])];
  assert(!/[가-힣]/.test(JSON.stringify(English)));
});

test('integrand surfaces and every order remain finite,with exact log magnitudes and explicit excluded points', () => {
  for (const example of quarticAsymptoticExamples) {
    const result = calculateQuarticAsymptotic(example.input);
    for (const width of [2, 4, 8]) for (const scale of ['physical', 'all'] as const) {
      const view = quarticAsymptoticView(result, result.input.order, width, scale);
      assert(view.samples.every(p => Number.isFinite(p.x) && p.value >= 0 && p.value <= p.gaussian && Number.isFinite(p.logValue)));
      assert(view.surface.flat().every(p => Number.isFinite(p.x) && p.value >= 0 && p.value <= 1));
      assert(view.rows.every(row => Number.isFinite(row.partial) && [row.term, row.bound, row.distance, row.error].every(x => x === null || Number.isFinite(x))));
      assert(view.rows.every(row => Number.isFinite(view.partialTransform(row.partial))));
      assert(view.logMin < view.logMax && view.yMin < view.yMax);
    }
  }
});

test('tiny positive coupling remains distinct from the exact Gaussian branch', () => {
  const tiny = calculateQuarticAsymptotic(quarticAsymptoticExamples.find(x => x.id === 'tiny-positive')!.input);
  const view = quarticAsymptoticView(tiny, 32, 4, 'physical');
  assert(view.rows[32].bound !== null && view.rows[32].bound! < -300);
  assert.equal(tiny.series, 'divergent-asymptotic'); assert.equal(tiny.approximations[32].quadratureDistance.lower, '0');
  const gaussian = calculateQuarticAsymptotic({ coupling: '0', order: 32 });
  assert(quarticAsymptoticView(gaussian, 32, 4, 'physical').rows.every(row => row.bound === null && row.error === null));
});

test('zero coupling collapses the surface depth while positive coupling keeps the same normalized mapping', () => {
  const fractions = Array.from({ length: 21 }, (_, i) => i / 20);
  assert.deepEqual(new Set(fractions.map(f => quarticSurfaceDepth(0, f))), new Set([0]));
  for (const g of [1 / 999999999989, .1, 16]) {
    const depths = fractions.map(f => quarticSurfaceDepth(g, f));
    assert.equal(new Set(depths).size, fractions.length);
    assert.equal(Math.max(...depths) - Math.min(...depths), 4);
    for (let i = 0; i < depths.length; i++) {
      if (i) assert(depths[i] > depths[i - 1]);
      assert(Math.abs(depths[i] + depths[depths.length - 1 - i]) < 1e-12);
    }
  }
});

test('storage requires a bounded versioned physical input and owns the restored object', () => {
  const raw = JSON.stringify({ version: 1, value: quarticAsymptoticDefault });
  assert.deepEqual(parseStoredQuarticAsymptotic(raw), quarticAsymptoticDefault); assert(Object.isFrozen(parseStoredQuarticAsymptotic(raw)));
  for (const invalid of ['{', 'null', '[]', JSON.stringify({ version: 2, value: quarticAsymptoticDefault }), JSON.stringify({ version: 1, value: { coupling: '-1', order: 12 } }), JSON.stringify({ version: 1, value: quarticAsymptoticDefault, extra: true }), ' '.repeat(4097)]) assert.equal(parseStoredQuarticAsymptotic(invalid), null);
});

test('full and embedded labs server-render without touching Worker', () => {
  const before = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let calls = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { calls++; throw Error('SSR Worker'); } });
  try { for (const language of ['ko', 'en'] as const) { const full = renderToString(createElement(Lab, { language })), embedded = renderToString(createElement(Lab, { language, embedded: true })); assert(full.includes('information-question') && full.includes('information-step')); assert(!embedded.includes('information-question') && !embedded.includes('information-step')); if (language === 'en') assert(!/[가-힣]/.test(full)); } assert.equal(calls, 0); }
  finally { if (before) Object.defineProperty(globalThis, 'Worker', before); else Reflect.deleteProperty(globalThis, 'Worker'); }
});

test('result preserves provenance,budget limits,unresolved distances and full rational values', () => {
  for (const id of ['tiny-positive', 'maximum-coprime']) {
    const result = calculateQuarticAsymptotic(quarticAsymptoticExamples.find(x => x.id === id)!.input);
    for (const language of ['ko', 'en'] as const) {
      const html = renderToString(createElement(Result, { language, result }));
      assert(html.includes(result.quadratureInterval.lower)); assert(html.includes(result.approximations.at(-1)!.firstOmittedMagnitude));
      assert(html.includes('data-quartic-budget') && html.includes('log₁₀') && !html.includes('NaN') && !html.includes('Infinity'));
      if (id === 'tiny-positive') assert(html.includes('data-quartic-unresolved'));
      if (language === 'en') assert(!/[가-힣]/.test(html));
    }
  }
});

test('the teaching bundle excludes producer,quadrature and independent arithmetic', async () => {
  const bundle = await build({ entryPoints: ['src/components/learning/quartic-asymptotic-lab.tsx'], bundle: true, write: false, format: 'esm', platform: 'browser', metafile: true, logLevel: 'silent' });
  for (const name of ['learning-quartic-asymptotic.ts', 'learning-quartic-asymptotic-integral.ts', 'learning-quartic-asymptotic-validation.ts', 'learning-quartic-asymptotic-validation-integral.ts', 'learning-interval-transcendentals.ts', 'learning-quantum-hilbert-core.ts']) assert(!Object.keys(bundle.metafile!.inputs).includes('src/lib/' + name));
  assert(!bundle.outputFiles[0].text.includes('BigInt('));
});
