import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { learningUnits, learningGuides } from '../src/data/interactive-book';
import { partitionCumulantModel as model } from '../src/lib/learning-partition-cumulants';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
const id = 'finite-spectrum-partition-cumulant-generators';
const source = (lang: 'ko' | 'en') => readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
const near = (a: number | null, b: number) => { assert.notEqual(a, null); assert.ok(Math.abs(a! - b) <= 2e-10 * Math.abs(b), `${a} versus ${b}`); };
for (const lang of ['ko', 'en'] as const) test(`${lang}: detailed finite-partition theory and six explained exercises render with strict accessible mathematics`, async () => {
  const text = source(lang), rendered = await getLearningContent(id), html = rendered[lang];
  assert.equal((text.match(/^## /gm) ?? []).length, 17); assert.equal((text.match(/^### /gm) ?? []).length, 10);
  assert.equal((text.match(/^\*\*(?:Answer|풀이)\.\*\*/gm) ?? []).length, 6);
  assert.equal((text.match(/\$\$/g) ?? []).length, 52);
  if (lang === 'en') { assert.ok(text.split(/\s+/).length >= 2600); assert.doesNotMatch(html, /[가-힣]/); }
  else assert.ok((text.match(/[가-힣]/g) ?? []).length >= 5000);
  assert.doesNotMatch(text, /^# |^---/m); assert.doesNotMatch(html, /katex-error|\$\$/);
  assert.equal(rendered.outline[lang].length, 17); assert.equal(new Set(rendered.outline[lang].map(h => h.id)).size, 17);
  assert.equal((html.match(/class="katex-display"/g) ?? []).length, 26);
  assert.equal((html.match(/<table\b/g) ?? []).length, 5);
  assert.equal((html.match(/aria-label="(?:Equation|수식) /g) ?? []).length, 26);
  assert.equal((html.match(/<table[^>]*tabindex="0"/g) ?? []).length, 5);
});
test('paired bilingual derivations match, including moving-Hamiltonian and nonlinear-control terms', () => {
  const equations = (s: string) => [...s.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ''));
  assert.deepEqual(equations(source('ko')), equations(source('en')));
  assert.match(source('en'), /E-T\\partial_TE/); assert.match(source('en'), /\\langle\\partial_\\eta\\partial_\\lambda\s*E\\rangle/);
  assert.match(source('en'), /noncommuting quantum response/); assert.match(source('en'), /not automatically a physical energy/);
  assert.match(source('ko'), /비가환/); assert.match(source('ko'), /구간이 이미 확인한 부호는 보존/);
});
test('the manuscripts hand calculations agree with the independently checked numerical preparation', () => {
  const gap = kB * Math.log(2), a = model({ energyLevels: [0, gap, 2 * gap], energyOffset: 0, temperature: 1 });
  near(a.meanEnergy.value! / gap, 4 / 7); near(a.energyVariance.value! / gap ** 2, 26 / 49); near(a.thirdEnergyCumulant.value! / gap ** 3, 114 / 343);
  near(a.logPartition.value, Math.log(7 / 4));
  const b = model({ energyLevels: [0, gap, gap], energyOffset: 0, temperature: 1 });
  near(b.partitionFunction.value, 2); near(b.states[0].probability.value, .5); near(b.states[1].probability.value, .25); near(b.states[2].probability.value, .25);
});
test('moving the gap with T demonstrates the stated fixed-Hamiltonian counterexample', () => {
  const expected = kB * Math.log(2) / 3;
  const evaluate = (T: number) => model({ energyLevels: [0, kB * T * Math.log(2)], energyOffset: 0, temperature: T });
  for (const h of [.1, .01, .001]) near((evaluate(1 + h).meanEnergy.value! - evaluate(1 - h).meanEnergy.value!) / (2 * h), expected);
  const fixedLevelResponse = evaluate(1).heatCapacity.value!;
  near(fixedLevelResponse, kB * Math.log(2) ** 2 * 2 / 9);
  assert.ok(Math.abs(expected - fixedLevelResponse) > .1 * kB);
});
test('the prepared spectrum is appended after the 515 prior registered readers', () => {
  assert.ok(learningUnits.length >= 518); assert.ok(learningUnits.slice(515).some(u => (u.id as string) === id));
  assert.equal(learningUnits.slice(0, 518).reduce((n, u) => n + learningGuides[u.id].experiments.length, 0), 1554);
});
