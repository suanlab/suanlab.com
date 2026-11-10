import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { learningUnits, learningGuides } from '../src/data/interactive-book';
import { coupledPartitionModel as model } from '../src/lib/learning-coupled-partition';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
const id = 'coupled-binary-sites-partition-factorization';
const source = (language: 'en' | 'ko') => readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
for (const language of ['en', 'ko'] as const) test(`${language}: detailed coupled-site manuscript renders with strict accessible equations and tables`, async () => {
  const text = source(language), rendered = await getLearningContent(id), html = rendered[language];
  assert.equal(rendered.outline[language].length, 15);
  assert.equal(new Set(rendered.outline[language].map(h => h.id)).size, 15);
  assert.doesNotMatch(text, /^# |^---/m); assert.doesNotMatch(html, /katex-error|\$\$/);
  assert.equal((html.match(/class="katex-display"/g) ?? []).length, 14);
  assert.equal((html.match(/aria-label="(?:Equation|수식) /g) ?? []).length, 14);
  assert.equal((html.match(/<table[^>]*tabindex="0"/g) ?? []).length, 4);
  assert.equal((text.match(/^### /gm) ?? []).length, 6);
  assert.equal((text.match(/^\*\*(?:Answer|풀이)\.\*\*/gm) ?? []).length, 6);
  if (language === 'en') { assert.ok(text.split(/\s+/).length >= 2400); assert.doesNotMatch(html, /[가-힣]/); }
  else assert.ok((text.match(/[가-힣]/g) ?? []).length >= 5300);
});
test('bilingual derivations correspond, including normalization, information, response and pair covariance', () => {
  const equations = (language: 'en' | 'ko') => [...source(language).matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ''));
  assert.deepEqual(equations('en'), equations('ko'));
  assert.match(source('en'), /zero covariance by an infinite/);
  assert.match(source('en'), /changing J changes a common energy/);
  assert.match(source('ko'), /국소적으로 일정한 0 응답/);
});
test('hand example Hessian, information and signed log-normalizer difference have distinct verified values', () => {
  const r = model({ couplingEnergy: kB * Math.log(2), fieldEnergy: 0, temperature: 1, support: 'full-product' });
  const expected = [[16 / 25, 0], [0, 16 / 5]];
  r.sourceHessian.forEach((row, i) => row.forEach((q, j) => assert.ok(Math.abs(q.value! - expected[i][j]) < 1e-13)));
  assert.ok(Math.abs(r.mutualInformation.value! - .1927447570217575) < 1e-13);
  assert.ok(Math.abs(r.logFactorizationDefect.value! - .22314355131420976) < 1e-13);
  assert.ok(Math.abs(r.responses.magnetizationToField.observation!.value! * kB - 16 / 5) < 1e-13);
});
test('the prepared coupled family is registered with the expanded reader scope', () => {
  assert.ok(learningUnits.length >= 518);
  assert.equal(learningUnits.slice(0, 518).reduce((n, u) => n + learningGuides[u.id].experiments.length, 0), 1554);
  assert.ok(learningUnits.slice(515).some(u => (u.id as string) === id));
});
