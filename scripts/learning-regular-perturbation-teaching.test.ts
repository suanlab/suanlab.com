import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createElement } from 'react';
import { renderToString } from 'react-dom/server';
import { regularPerturbationSteps, regularPerturbationExperiments, regularPerturbationQuestions, regularPerturbationLabels } from '../src/data/regular-perturbation-teaching';
import { regularPerturbationExamples, regularPerturbationDefault } from '../src/lib/learning-regular-perturbation-input';
import { calculateRegularPerturbation } from '../src/lib/learning-regular-perturbation';
import { perturbationLogMagnitude, regularPerturbationView } from '../src/lib/learning-regular-perturbation-view';
import { parseStoredRegularPerturbation } from '../src/components/learning/use-saved-regular-perturbation';
import Lab from '../src/components/learning/regular-perturbation-lab';
import Result from '../src/components/learning/regular-perturbation-result';

test('eight concepts, six experiments, seven explained questions and fourteen labelled examples are bilingual', () => {
  assert.equal(regularPerturbationSteps.length, 8); assert.equal(regularPerturbationExperiments.length, 6); assert.equal(regularPerturbationQuestions.length, 7);
  for (const example of regularPerturbationExamples) assert(regularPerturbationLabels[example.id].ko && regularPerturbationLabels[example.id].en);
  for (const guide of regularPerturbationExperiments) {
    assert(guide.explanation.ko.length > 40 && guide.explanation.en.length > 80);
    for (const id of guide.examples) assert(regularPerturbationExamples.some(example => example.id === id));
  }
  for (const question of regularPerturbationQuestions) assert(question.correct >= 0 && question.correct < question.options.length);
  const English = [...regularPerturbationSteps.map(x => [x.title.en, x.body.en]), ...regularPerturbationExperiments.map(x => [x.title.en, x.predict.en, x.explanation.en]), ...regularPerturbationQuestions.map(x => [x.prompt.en, x.explanation.en, ...x.options.map(o => o.en)])];
  assert(!/[가-힣]/.test(JSON.stringify(English)));
});

test('logarithmic magnitudes preserve huge and tiny nonzero fractions without BigInt conversion', () => {
  assert.equal(perturbationLogMagnitude('0'), null); assert.equal(perturbationLogMagnitude('-100/10'), 1);
  assert.equal(perturbationLogMagnitude('1' + '0'.repeat(5000)), 5000);
  assert.equal(perturbationLogMagnitude('1/' + '1' + '0'.repeat(5000)), -5000);
  assert.equal(perturbationLogMagnitude('not-a-fraction'), null);
});

test('all example windows and surface coordinates remain finite while off-window values are explicit', () => {
  for (const example of regularPerturbationExamples) {
    const result = calculateRegularPerturbation(example.input);
    for (const window of ['equilibrium', 'linear'] as const) for (const order of [0, result.input.order]) {
      const view = regularPerturbationView(result, window, order);
      assert(view.width > 0 && view.forceMax > 0);
      assert(view.samples.every(p => Number.isFinite(p.x) && Number.isFinite(p.force) && Number.isFinite(p.energy)));
      assert(view.surface.flat().every(p => Number.isFinite(p.x) && Number.isFinite(p.coupling) && Number.isFinite(p.energy)));
      assert(view.orderRows.every(row => [row.term, row.residual, row.bound, row.distance].every(n => n === null || Number.isFinite(n))));
      assert(view.energyMin <= view.energyMax && view.logMin < view.logMax);
      assert(view.lower <= view.root && view.root <= view.upper);
    }
  }
  const huge = calculateRegularPerturbation(regularPerturbationExamples.find(x => x.id === 'maximum-soft')!.input);
  assert.equal(regularPerturbationView(huge, 'equilibrium', 16).partialInside, false);
});

test('stored input is bounded, versioned, owned and validates physical assumptions', () => {
  const stored = JSON.stringify({ version: 1, value: regularPerturbationDefault });
  assert.deepEqual(parseStoredRegularPerturbation(stored), regularPerturbationDefault);
  assert(Object.isFrozen(parseStoredRegularPerturbation(stored)));
  for (const raw of ['{', 'null', '[]', JSON.stringify({ version: 2, value: regularPerturbationDefault }), JSON.stringify({ version: 1, value: { ...regularPerturbationDefault, stiffness: '0' } }), JSON.stringify({ version: 1, value: regularPerturbationDefault, extra: 1 }), ' '.repeat(4097)]) assert.equal(parseStoredRegularPerturbation(raw), null);
});

test('lab and embedded mode server-render without Worker construction', () => {
  const before = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let count = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { count++; throw Error('SSR Worker'); } });
  try {
    for (const language of ['ko', 'en'] as const) {
      const full = renderToString(createElement(Lab, { language })), embedded = renderToString(createElement(Lab, { language, embedded: true }));
      assert(full.includes('information-question') && full.includes('information-step'));
      assert(!embedded.includes('information-question') && !embedded.includes('information-step'));
      if (language === 'en') assert(!/[가-힣]/.test(full));
    }
    assert.equal(count, 0);
  } finally { if (before) Object.defineProperty(globalThis, 'Worker', before); else Reflect.deleteProperty(globalThis, 'Worker'); }
});

test('result rendering keeps exact fractions, log labels and explicit off-window messages', () => {
  const result = calculateRegularPerturbation(regularPerturbationExamples.find(x => x.id === 'maximum-soft')!.input);
  for (const language of ['ko', 'en'] as const) {
    const html = renderToString(createElement(Result, { language, result }));
    assert(html.includes('data-perturbation-off-window'));
    assert(html.includes(result.approximations.at(-1)!.errorBound));
    assert(html.includes('log₁₀') && !html.includes('NaN') && !html.includes('Infinity'));
    if (language === 'en') assert(!/[가-힣]/.test(html));
  }
});

test('the teaching screen excludes both exact calculation engines and the independent validator', async () => {
  const bundle = await build({ entryPoints: ['src/components/learning/regular-perturbation-lab.tsx'], bundle: true, write: false, format: 'esm', platform: 'browser', metafile: true, logLevel: 'silent' });
  for (const name of ['learning-regular-perturbation.ts', 'learning-regular-perturbation-validation.ts', 'learning-quantum-hilbert-core.ts', 'learning-graph-gauge-validation-math.ts']) assert(!Object.keys(bundle.metafile!.inputs).includes('src/lib/' + name));
  assert(!bundle.outputFiles[0].text.includes('BigInt('));
});
