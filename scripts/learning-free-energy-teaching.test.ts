import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { freeEnergyTeaching, learningUnits, learningGuides } from '../src/data/interactive-book';
import { freeEnergyPresets, validFreeEnergyValues, freeEnergyComparisonValues, type FreeEnergyKind } from '../src/lib/learning-free-energy';
import { FreeEnergyConceptView } from '../src/components/learning/free-energy-lab';
import FreeEnergyCover from '../src/components/learning/free-energy-cover';
(globalThis as typeof globalThis & { React?: typeof React }).React = React;
const kinds = Object.keys(freeEnergyTeaching) as FreeEnergyKind[];
for (const kind of kinds) for (let step = 0; step < 4; step++) for (const language of ['ko', 'en'] as const)
  test(`${kind} fixed concept ${step + 1} ${language}`, () => {
    const values = freeEnergyTeaching[kind].conceptInputs[step];
    assert.ok(validFreeEnergyValues(kind, values));
    const html = renderToStaticMarkup(React.createElement(FreeEnergyConceptView, { kind, step, language }));
    assert.doesNotMatch(html, /NaN|Infinity|="undefined"/);
    if (language === 'en') assert.doesNotMatch(html, /[가-힣]/);
    assert.equal((html.match(/<svg\b/g) ?? []).length, 1);
    assert.match(html, /role="img"/); assert.match(html, /scope="row"/);
    assert.match(html, /role="region"/);
    if (kind === kinds[1] && step === 3 && language === 'en') assert.match(html, /Approximate cancellation; zero unresolved/);
    if (kind === kinds[0] && step === 2 && language === 'en') assert.match(html, /Structural or algebraic zero/);
  });
test('drafts contain validated guides, prerequisites and explained quizzes with registration', () => {
  assert.ok(learningUnits.length >= 512);
  const existing = new Set(learningUnits.map(u => u.id as string));
  let concepts = 0, experiments = 0, questions = 0;
  for (const kind of kinds) {
    const d = freeEnergyTeaching[kind];
    assert.equal(existing.has(kind), true); assert.equal(kind in learningGuides, true);
    assert.equal(d.guide.steps.length, 4); assert.equal(d.guide.experiments.length, 3); assert.equal(d.unit.questions.length, 2);
    for (const p of d.unit.prerequisiteIds) assert.ok(existing.has(p));
    for (const g of d.guide.experiments) {
      assert.ok(Number.isInteger(g.preset)); assert.ok(freeEnergyPresets[kind][g.preset]);
      assert.ok(validFreeEnergyValues(kind, freeEnergyPresets[kind][g.preset].values));
      for (const key of ['title', 'predict', 'action', 'explain'] as const) { assert.ok(g[key].en.length > 10); assert.doesNotMatch(g[key].en, /[가-힣]/); }
    }
    for (const q of d.unit.questions) {
      assert.equal(q.options.length, 3); assert.ok(Number.isInteger(q.answer) && q.answer >= 0 && q.answer < q.options.length);
      assert.ok(q.explanation.ko.length > 20); assert.ok(q.explanation.en.length > 40);
    }
    concepts += d.guide.steps.length; experiments += d.guide.experiments.length; questions += d.unit.questions.length;
  }
  assert.equal(concepts, 8); assert.equal(experiments, 6); assert.equal(questions, 4);
});
test('all 33 comparison samples preserve the selected physical constraints', () => {
  for (const kind of kinds) {
    const base = freeEnergyPresets[kind][0].values;
    let last = -Infinity;
    for (let i = 0; i <= 32; i++) {
      const next = freeEnergyComparisonValues(kind, base, i);
      assert.ok(validFreeEnergyValues(kind, next)); assert.notEqual(next, base);
      if (kind === kinds[0]) {
        for (const j of [0, 1, 5, 6]) assert.equal(next[j], base[j]);
        assert.equal(next[2], next[3]); assert.equal(next[4] / 2, i / 32);
      } else {
        for (const j of [0, 1, 2, 4, 5]) assert.equal(next[j], base[j]);
        assert.ok(next[3] > last); last = next[3];
      }
    }
    assert.deepEqual(base, freeEnergyPresets[kind][0].values);
    for (const index of [-1, 33, .5, NaN]) assert.throws(() => freeEnergyComparisonValues(kind, base, index), RangeError);
  }
});
test('covers are decorative theme-native vectors, independent from calculated figures', () => {
  for (const kind of kinds) {
    const html = renderToStaticMarkup(React.createElement(FreeEnergyCover, { kind }));
    assert.match(html, /aria-hidden="true"/); assert.match(html, /var\(--learning-/);
    assert.doesNotMatch(html, /<text|NaN|Infinity/);
  }
});
test('concept boundaries reject unsupported step indices', () => {
  assert.throws(() => renderToStaticMarkup(React.createElement(FreeEnergyConceptView, { kind: kinds[0], step: 4, language: 'en' })), RangeError);
});
