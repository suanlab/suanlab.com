import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { weightedBoltzmannModel as model } from '../src/lib/learning-weighted-boltzmann';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
const id = 'reference-measure-boltzmann-softmax-identifiability';
const source = (lang: 'ko' | 'en') => readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
for (const lang of ['ko', 'en'] as const) test(`${lang}: substantial reference-measure theory, accessible math, tables and six explained exercises`, async () => {
  const text = source(lang), rendered = await getLearningContent(id), html = rendered[lang];
  assert.equal(rendered.outline[lang].length, 15);
  assert.equal(new Set(rendered.outline[lang].map(h => h.id)).size, 15);
  assert.equal((text.match(/\$\$/g) ?? []).length, 42);
  assert.equal((text.match(/^\d\. \*\*/gm) ?? []).length, 6);
  assert.equal((text.match(/\*\*(?:Answer|해설):\*\*/g) ?? []).length, 6);
  assert.equal((html.match(/class="katex-display"/g) ?? []).length, 21);
  assert.equal((html.match(/aria-label="(?:Equation|수식) /g) ?? []).length, 21);
  assert.equal((html.match(/<table[^>]*tabindex="0"/g) ?? []).length, 4);
  assert.doesNotMatch(text, /^# |^---/m); assert.doesNotMatch(html, /katex-error|\$\$/);
  if (lang === 'en') { assert.ok(text.split(/\s+/).length >= 2400); assert.doesNotMatch(html, /[가-힣]/); }
  else assert.ok((text.match(/[가-힣]/g) ?? []).length >= 5000);
});
test('bilingual derivations agree and distinguish counting, prior entropy, fixed-Hamiltonian response and energy-scale ambiguity', () => {
  const equations = (s: string) => [...s.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, '').replace(/[.,]$/, ''));
  assert.deepEqual(equations(source('ko')), equations(source('en')));
  for (const pattern of [/positive real reference/i, /−KL\(p\|\|q\)/, /effective energy depends on temperature/, /absolute energy scale and temperature/, /an energy scale E_\*/, /positive-side zero-temperature/, /relative width at most 2\^-48/]) assert.match(source('en'), pattern);
});
const near = (a: number | null, b: number) => { assert.notEqual(a, null); assert.ok(Math.abs(a! - b) <= 2e-12 * Math.abs(b), `${a} vs ${b}`); };
test('worked level probabilities, entropy decomposition and heat capacity agree with checked actual-input model', () => {
  const gap = kB * Math.log(2), input = { energyLevels: [0, gap, 2 * gap], referenceWeights: [1, 2, 1], energyOffset: 0, temperature: 1 };
  const a = model(input), b = model({ ...input, referenceWeights: [.25, .5, .25] });
  [4 / 9, 4 / 9, 1 / 9].forEach((p, i) => near(a.states[i].probability.value, p));
  near(a.meanEnergy.value! / gap, 2 / 3); near(a.energyVariance.value! / gap ** 2, 4 / 9);
  near(a.heatCapacityInKB.value, 4 * Math.log(2) ** 2 / 9);
  near(a.labelEntropyNats.value, Math.log(9) - 8 / 9 * Math.log(4));
  near(a.relativeEntropyInKB.value, Math.log(9) - 4 / 3 * Math.log(2));
  near(a.meanLogReferenceWeight.value, 4 / 9 * Math.log(2));
  near(b.logPartition.value, Math.log(9 / 16));
  near(b.relativeEntropyInKB.value, a.relativeEntropyInKB.value! - Math.log(4));
});
test('exercise two reference advantage beats an energy penalty; exercise four inversion recovers only a gap', () => {
  const gap = kB * Math.log(2), r = model({ energyLevels: [0, gap], referenceWeights: [1, 3], energyOffset: 0, temperature: 1 });
  near(r.states[0].probability.value, .4); near(r.states[1].probability.value, .6);
  const recovered = -kB * Math.log(r.states[1].probability.value! / (3 * r.states[0].probability.value!)); near(recovered, gap);
});
test('a softmax-to-physical-energy conversion needs the stated separate energy scale', () => {
  const scale = 8 * kB, alpha = .5, logits = [0, 1, 2];
  const r = model({ energyLevels: logits.map(s => -scale * s), referenceWeights: [1, 2, 1], energyOffset: 0, temperature: alpha * scale / kB });
  const raw = logits.map((s, i) => [1, 2, 1][i] * Math.exp(s / alpha)), W = raw.reduce((a, b) => a + b);
  r.states.forEach((s, i) => near(s.probability.value, raw[i] / W));
});
