import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { learningUnits, learningGuides } from '../src/data/interactive-book';
import { energyLadderModel as model } from '../src/lib/learning-energy-ladder';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
const id = 'infinite-energy-ladder-normalization-cutoff';
const source = (language: 'en' | 'ko') => readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
for (const language of ['en', 'ko'] as const) test(`${language}: ladder manuscript renders strict accessible mathematics, tables and explained exercises`, async () => {
  const text = source(language), rendered = await getLearningContent(id), html = rendered[language];
  assert.equal(rendered.outline[language].length, 15);
  assert.equal(new Set(rendered.outline[language].map(h => h.id)).size, 15);
  assert.doesNotMatch(text, /^# |^---/m); assert.doesNotMatch(html, /katex-error|\$\$|\{\{equation/);
  assert.equal((html.match(/class="katex-display"/g) ?? []).length, 13);
  assert.equal((html.match(/aria-label="(?:Equation|수식) /g) ?? []).length, 13);
  assert.equal((html.match(/<table[^>]*tabindex="0"/g) ?? []).length, 3);
  assert.equal((text.match(/^### /gm) ?? []).length, 6);
  assert.equal((text.match(/^\*\*(?:Answer|풀이)\.\*\*/gm) ?? []).length, 6);
  if (language === 'en') { assert.ok(text.split(/\s+/).length >= 2300); assert.doesNotMatch(html, /[가-힣]/); }
  else assert.ok((text.match(/[가-힣]/g) ?? []).length >= 5300);
});
test('bilingual equations preserve cutoff, KL direction, thermodynamics and cold entropy terms', () => {
  const equations = (language: 'en' | 'ko') => [...source(language).matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ''));
  assert.deepEqual(equations('en'), equations('ko'));
  assert.match(source('en'), /not the thermodynamic limit/);
  assert.match(source('en'), /\(1\+x\)q/);
  assert.match(source('ko'), /최소 비영 수의 정확한 절반/);
});
test('hand probabilities, energy reference, error criterion and heat-capacity limits match the model', () => {
  const gap = kB * Math.log(2), r = model({ gapEnergy: gap, temperature: 1, cutoff: 2 });
  assert.ok(Math.abs(r.infinite.quantities!.meanEnergy.value! / gap - 3 / 2) < 1e-13);
  assert.ok(Math.abs(r.finite.meanEnergy.value! / gap - 15 / 14) < 1e-13);
  assert.ok(Math.abs(r.comparison!.forwardKL.value! - Math.log(8 / 7)) < 1e-13);
  const adequate = model({ gapEnergy: gap, temperature: 1, cutoff: 6 }), insufficient = model({ gapEnergy: gap, temperature: 1, cutoff: 5 });
  assert.ok(adequate.comparison!.missingMass.value! <= .01); assert.ok(insufficient.comparison!.missingMass.value! > .01);
  const hot = model({ gapEnergy: kB * 1e-12, temperature: 1, cutoff: 2 });
  assert.ok(hot.finite.heatCapacityInKB.value! < 1e-20); assert.ok(hot.infinite.quantities!.heatCapacityInKB.value! > .999999);
});
test('all three prepared families are appended after the prior reader prefix', () => {
  assert.ok(learningUnits.length >= 518);
  assert.equal(learningUnits.slice(0, 518).reduce((n, u) => n + learningGuides[u.id].experiments.length, 0), 1554);
  for (const pending of [id, 'coupled-binary-sites-partition-factorization', 'finite-spectrum-partition-cumulant-generators'])
    assert.ok(learningUnits.slice(515, 518).some(u => (u.id as string) === pending));
});
