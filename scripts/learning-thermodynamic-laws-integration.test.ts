import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import prior from './fixtures/pre-thermodynamic-laws-registry-hashes.json';
import { learningUnits, learningGuides, thermodynamicLawLabs } from '../src/data/interactive-book';
import { curriculumDomains } from '../src/data/learning-curriculum';
import { thermodynamicLawKinds as ids, thermodynamicLawConceptInputs as cases, evaluateCarnotLaw,
  evaluateReservoirLaw, evaluateSpectrumLaw } from '../src/lib/learning-thermodynamic-laws';
import { ThermodynamicLawConceptView, ThermodynamicLawView } from '../src/components/learning/thermodynamic-laws-lab';
(globalThis as typeof globalThis & { React?: typeof React }).React = React;
const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
test('three new units preserve all 504 previous metadata/guide hashes and their exact order', () => {
  assert.equal(prior.ids.length, 504); assert.ok(learningUnits.length >= 512);
  assert.deepEqual(learningUnits.slice(0, 504).map(u => u.id), prior.ids);
  for (const unit of learningUnits.slice(0, 504)) {
    assert.equal(hash(unit), (prior.units as Record<string, string>)[unit.id], unit.id);
    assert.equal(hash(learningGuides[unit.id]), (prior.guides as Record<string, string>)[unit.id], unit.id);
  }
  assert.deepEqual(learningUnits.slice(504, 507).map(u => u.id), ids);
  assert.equal(learningUnits.slice(0, 512).reduce((n, u) => n + learningGuides[u.id].experiments.length, 0), 1536);
});
test('new units have real earlier prerequisites, paired teaching content and curriculum links', () => {
  const order = new Map(learningUnits.map((unit, i) => [unit.id, i]));
  const domain = curriculumDomains.find(d => d.id === 'thermo-statmech')!;
  for (const id of ids) {
    const unit = learningUnits.find(u => u.id === id)!, guide = learningGuides[id];
    assert.ok(domain.unitIds.includes(id)); assert.equal(unit.questions.length, 2);
    assert.equal(guide.steps.length, 4); assert.equal(guide.experiments.length, 3);
    for (const p of unit.prerequisiteIds) assert.ok(order.get(p)! < order.get(id)!, `${p} before ${id}`);
    for (const experiment of guide.experiments) {
      assert.ok(thermodynamicLawLabs[id].presets[experiment.preset]);
      for (const field of [experiment.title, experiment.predict, experiment.action, experiment.explain]) {
        assert.ok(field.ko.length >= 3 && field.en.length >= 3); assert.doesNotMatch(field.en, /[가-힣]/);
      }
    }
    for (const question of unit.questions) {
      assert.equal(question.options.length, 3); assert.ok(question.answer >= 0 && question.answer < 3);
      assert.ok(question.explanation.en.length > 70); assert.doesNotMatch(question.explanation.en, /[가-힣]/);
    }
    // Concept and live readouts coexist in the actual reader. Their named
    // regions must remain distinct for landmark navigation by screen readers.
    const combined = renderToStaticMarkup(React.createElement(React.Fragment, null,
      React.createElement(ThermodynamicLawConceptView, { kind: id, step: 0, language: 'en' }),
      React.createElement(ThermodynamicLawView, { kind: id, values: thermodynamicLawLabs[id].controls.map(c => c.initial), language: 'en' })));
    const regions = [...combined.matchAll(/role="region" aria-label="([^"]+)"/g)].map(match => match[1]);
    assert.equal(new Set(regions).size, regions.length, `${id}: duplicate named regions`);
  }
});
for (const id of ids) for (let step = 0; step < 4; step++) for (const language of ['ko', 'en'] as const) {
  test(`${id} concept ${step + 1} ${language}: fixed calculated figure, units and availability statuses`, () => {
    const html = renderToStaticMarkup(React.createElement(ThermodynamicLawConceptView, { kind: id, step, language }));
    assert.equal((html.match(/<svg /g) ?? []).length, 1);
    assert.match(html, /role="img"/); assert.match(html, /scope="row"/);
    assert.doesNotMatch(html, /NaN|Infinity|="undefined"/);
    if (language === 'en') assert.doesNotMatch(html, /[가-힣]/);
    for (const [, axis] of html.matchAll(/data-wave-[xy]-(?:min|max)="([^"]+)"/g)) assert.ok(Number.isFinite(Number(axis)));
    const guide = learningGuides[id].steps[step];
    assert.ok(guide.formula.length > 5); assert.ok(guide.explanation.en.length > 80);
  });
}
test('Carnot teaching statements agree with actual fixed cycle, reverse COP and degenerate ratio', () => {
  const [forward, entropy, reverse, degenerate] = cases[ids[0]].map(evaluateCarnotLaw);
  assert.ok(Math.abs(forward.cycle!.complete.workByGas.value! - 1728.9438964931937) < 1e-10);
  assert.equal(forward.cycle!.totalEnergyChange.value, 0); assert.equal(entropy.cycle!.engineEfficiency!.value, .5);
  assert.equal(reverse.cycle!.complete.workByGas.sign, -1); assert.equal(reverse.cycle!.refrigeratorCOP!.value, 1);
  assert.equal(reverse.cycle!.heatingCOP!.value, 2); assert.equal(degenerate.cycle!.workPerHotHeat, null);
  assert.equal(degenerate.cycle!.carnotTemperatureBound.value, .5);
});
test('reservoir teaching examples separate reversible, dissipative and counterfactual budgets', () => {
  const [reversible, positive, negative, input] = cases[ids[1]].map(v => evaluateReservoirLaw(v).query!);
  assert.equal(reversible.workByDevice.value, 500); assert.equal(reversible.entropyGeneration.value, 0);
  assert.equal(positive.workByDevice.value, 375); assert.equal(positive.workDeficitFromReversible.value, 125);
  assert.equal(positive.entropyGeneration.value, 5 / 12); assert.equal(negative.entropyGeneration.value, -5 / 12);
  assert.equal(negative.workByDevice.value, 625); assert.equal(negative.engineEfficiency, null);
  assert.equal(input.workByDevice.value, -500); assert.equal(input.heatColdIntoDevice.value, -1500);
  assert.equal(input.regime, 'work-driven-dissipation');
});
test('spectrum statements preserve degeneracy, ordered limits and nonzero out-of-range tails', () => {
  const [transition, positiveGap, zeroGap, tail] = cases[ids[2]].map(v => evaluateSpectrumLaw(v).query);
  assert.ok(Math.abs(transition.upperGroupProbability.value! - .2887654057724061) < 1e-14);
  assert.equal(positiveGap.entropyPerCopyInKB.value, Math.log(4)); assert.equal(zeroGap.entropyPerCopyInKB.value, Math.log(7));
  assert.equal(zeroGap.upperGroupProbability.value, 3 / 7); assert.equal(tail.upperGroupProbability.value, null);
  assert.equal(tail.upperGroupProbability.sign, 1); assert.equal(tail.upperGroupProbability.logMagnitudeStatus, 'outside-number-range');
  assert.equal(tail.upperGroupProbability.logNegativeLogMagnitudeStatus, 'finite');
});
