import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { learningUnits, learningGuides } from '../src/data/interactive-book';
import { curriculumDomains } from '../src/data/learning-curriculum';
import { boltzmannGibbsKinds as kinds, boltzmannGibbsGuides as guides } from '../src/lib/learning-boltzmann-gibbs';
import { boltzmannConceptSteps, boltzmannConceptQuestions } from '../src/lib/learning-boltzmann-gibbs-concepts';
import { getLearningContent } from '../src/lib/learning-content';
import { LanguageProvider } from '../src/components/language-provider';
(globalThis as typeof globalThis & { React?: typeof React }).React = React;
// Node SSR checks markup; exported CSS is exercised by the actual browser suite.
createRequire(`${process.cwd()}/package.json`).extensions['.css'] = () => {};
const baseline = JSON.parse(readFileSync('scripts/fixtures/pre-boltzmann-gibbs-registry-hashes.json', 'utf8'));
const hash = (v: unknown) => createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('all 518 previous units and guides retain their complete values and iteration order', () => {
  assert.ok(learningUnits.length >= 521); assert.equal(baseline.ids.length, 518);
  assert.deepEqual(learningUnits.slice(0, 518).map(u => u.id), baseline.ids);
  assert.deepEqual(Object.keys(learningGuides).slice(0, 518), baseline.guideOrder);
  for (const u of learningUnits.slice(0, 518)) { assert.equal(hash(u), baseline.unitHashes[u.id], u.id); assert.equal(hash(learningGuides[u.id]), baseline.guideHashes[u.id], u.id); }
  assert.deepEqual(learningUnits.slice(518, 521).map(u => u.id), kinds);
  assert.equal(Object.values(learningGuides).slice(0, 521).reduce((n, g) => n + g.experiments.length, 0), 1563);
});
test('curriculum preserves definitions and the original statistical-mechanics membership prefix', () => {
  for (const domain of curriculumDomains) {
    const { unitIds, ...rest } = domain;
    assert.equal(hash(rest), baseline.curriculumDomainHashesWithoutUnitIds[domain.id], domain.id);
    const expected = domain.id === 'thermo-statmech' ? [...baseline.curriculumDomainUnitIds[domain.id], ...kinds] : baseline.curriculumDomainUnitIds[domain.id];
    assert.deepEqual(unitIds.slice(0, expected.length), expected);
  }
});
for (const kind of kinds) test(`${kind}: actual registered reader renders detailed bilingual content and dedicated teaching components`, async () => {
  const unit = learningUnits.find(u => u.id === kind)!; assert.ok(unit);
  assert.ok(unit.prerequisiteIds.every(id => { const i = learningUnits.findIndex(u => u.id === id); return i >= 0 && i < learningUnits.indexOf(unit); }));
  assert.deepEqual(unit.questions, boltzmannConceptQuestions[kind]);
  assert.deepEqual(learningGuides[kind].steps, boltzmannConceptSteps[kind]);
  assert.deepEqual(learningGuides[kind].experiments, guides[kind].map(({ presetIndices, ...g }) => ({ ...g, preset: presetIndices[0] })));
  const html = await getLearningContent(kind);
  for (const language of ['ko', 'en'] as const) {
    assert.ok(html.outline[language].length >= 10); assert.ok(html[language].length > 10000);
    assert.doesNotMatch(html[language], /katex-error|NaN|undefined/);
    assert.ok(html[language].includes('class="katex"'));
  }
  const original = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let reads = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { reads++; throw Error('SSR must not create a worker'); } });
  try {
    const { InteractiveBookReader } = await import('../src/components/learning/interactive-book');
    const rendered = renderToStaticMarkup(React.createElement(LanguageProvider, null, React.createElement(InteractiveBookReader, { unit, html })));
    assert.ok(rendered.includes(`data-boltzmann-lab="${kind}"`));
    assert.ok(rendered.includes(`data-boltzmann-concept="${kind}"`));
    assert.ok(rendered.includes('/book/interactive'));
    assert.ok(rendered.includes(`/book/online/physics-math-for-ai/${unit.prerequisiteIds[0]}`)); assert.equal(reads, 0);
  } finally { if (original) Object.defineProperty(globalThis, 'Worker', original); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
