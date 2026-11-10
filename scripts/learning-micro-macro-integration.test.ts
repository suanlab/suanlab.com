import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import prior from './fixtures/pre-micro-macro-registry-hashes.json';
import { learningUnits, learningGuides, microMacroTeaching } from '../src/data/interactive-book';
import { curriculumDomains } from '../src/data/learning-curriculum';
import { microMacroGuides, microMacroConceptInput, microMacroConceptPresetIndices, type MicroMacroKind } from '../src/lib/learning-micro-macro';
import { evaluateMicroMacroRequest } from '../src/lib/learning-micro-macro-evaluator';
import { MicroMacroConceptResult, MicroMacroResultView } from '../src/components/learning/micro-macro-view';
import { MicroMacroConceptView } from '../src/components/learning/micro-macro-concept';
import BookArt from '../src/components/learning/book-art';
(globalThis as typeof globalThis & { React?: typeof React }).React = React;
const ids = Object.keys(microMacroTeaching) as MicroMacroKind[];
const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
test('registration preserves all 509 original units and guides in their exact prior order', () => {
  assert.ok(learningUnits.length >= 512); assert.equal(prior.units, 509); assert.equal(prior.experiments, 1527);
  assert.deepEqual(learningUnits.slice(0, 509).map(u => u.id), prior.ids);
  assert.deepEqual(learningUnits.slice(509, 512).map(u => u.id), ids);
  assert.deepEqual(Object.keys(learningGuides).slice(0, 509), prior.guideIds);
  for (const unit of learningUnits.slice(0, 509)) {
    assert.equal(hash(unit), (prior.unitHashes as Record<string, string>)[unit.id], unit.id);
    assert.equal(hash(learningGuides[unit.id]), (prior.guideHashes as Record<string, string>)[unit.id], unit.id);
  }
  assert.equal(learningUnits.slice(0, 512).reduce((n, u) => n + learningGuides[u.id].experiments.length, 0), 1536);
});
test('twelve concepts, nine guides, six explained quizzes and earlier prerequisites agree with real examples', () => {
  const order = new Map(learningUnits.map((u, i) => [u.id, i])), domain = curriculumDomains.find(d => d.id === 'thermo-statmech')!;
  let concepts = 0, guides = 0, quizzes = 0;
  for (const id of ids) {
    const teaching = microMacroTeaching[id];
    assert.ok(domain.unitIds.includes(id)); assert.equal(teaching.guide.steps.length, 4); assert.equal(teaching.unit.questions.length, 2);
    assert.deepEqual(teaching.guide.experiments, microMacroGuides[id]); assert.equal(microMacroConceptPresetIndices[id].length, 4);
    for (const p of teaching.unit.prerequisiteIds) assert.ok(order.get(p)! < order.get(id)!);
    for (const step of teaching.guide.steps) for (const key of ['title', 'explanation', 'diagram'] as const) {
      assert.ok(step[key].ko.length > 3); assert.ok(step[key].en.length > 3); assert.doesNotMatch(step[key].en, /[가-힣]/);
    }
    for (const q of teaching.unit.questions) { assert.equal(q.options.length, 3); assert.ok(Number.isInteger(q.answer) && q.answer >= 0 && q.answer < q.options.length); assert.ok(q.explanation.en.length > 60); assert.doesNotMatch(q.explanation.en, /[가-힣]/); }
    concepts += teaching.guide.steps.length; guides += teaching.guide.experiments.length; quizzes += teaching.unit.questions.length;
  }
  assert.equal(concepts, 12); assert.equal(guides, 9); assert.equal(quizzes, 6);
});
for (const id of ids) for (let step = 0; step < 4; step++) for (const language of ['ko', 'en'] as const)
  test(`${id} fixed concept ${step + 1} ${language} renders a finite calculated figure`, () => {
    const input = microMacroConceptInput(id, step), response = evaluateMicroMacroRequest({ version: 1, requestId: 1, input }); assert.ok(response.ok);
    const html = renderToStaticMarkup(React.createElement(MicroMacroConceptResult, { response, step, language }));
    const paired = renderToStaticMarkup(React.createElement(React.Fragment, null, React.createElement(MicroMacroConceptResult, { response, step, language }), React.createElement(MicroMacroResultView, { response, language })));
    const regionNames = [...paired.matchAll(/<[^>]*role="region"[^>]*aria-label="([^"]+)"/g)].map(m => m[1]);
    assert.equal(new Set(regionNames).size, regionNames.length, 'Concept and experiment regions need distinct names');
    assert.equal((html.match(/<svg\b/g) ?? []).length, 1); assert.match(html, /role="region"/); assert.match(html, /scope="row"/);
    assert.doesNotMatch(html, /NaN|Infinity|="undefined"/); if (language === 'en') assert.doesNotMatch(html, /[가-힣]/);
  });
test('registered covers dispatch and concepts remain safe to render without a browser Worker', () => {
  for (const kind of ids) {
    const cover = renderToStaticMarkup(React.createElement(BookArt, { kind })); assert.match(cover, /aria-hidden="true"/); assert.doesNotMatch(cover, /NaN|Infinity/);
    const pending = renderToStaticMarkup(React.createElement(MicroMacroConceptView, { kind, step: 0, language: 'en' }));
    assert.match(pending, /Calculating the fixed example/); assert.doesNotMatch(pending, /[가-힣]/);
  }
});
