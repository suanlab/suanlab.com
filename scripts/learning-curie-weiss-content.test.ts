import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { finiteCurieWeissModel as model } from '../src/lib/learning-curie-weiss';
const id = 'finite-curie-weiss-magnetization-symmetry';
const source = (language: 'ko' | 'en') => readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
for (const language of ['ko', 'en'] as const) test(`${language}: finite symmetry, counting, responses and modal distinctions render as accessible detailed theory`, async () => {
  const raw = source(language), rendered = await getLearningContent(id), html = rendered[language];
  assert.equal(rendered.outline[language].length, 15); assert.equal(new Set(rendered.outline[language].map(h => h.id)).size, 15);
  assert.equal((raw.match(/^### [1-6]\. /gm) ?? []).length, 6);
  assert.equal((raw.match(/\*\*(?:Explanation\.|해설\.)\*\*/g) ?? []).length, 6);
  assert.equal((html.match(/class="katex-display"/g) ?? []).length, 17);
  assert.equal((html.match(/aria-label="(?:Equation|수식) /g) ?? []).length, 17);
  assert.equal((html.match(/<table[^>]*tabindex="0"/g) ?? []).length, 3);
  assert.doesNotMatch(raw, /^# |^---/m); assert.doesNotMatch(html, /katex-error|\$\$/);
  if (language === 'en') { assert.ok(raw.split(/\s+/).length > 2400); assert.doesNotMatch(html, /[가-힣]/); }
  else assert.ok((raw.match(/[가-힣]/g) ?? []).length > 4500);
});
test('both manuscripts keep identical equations and explicitly separate finite observables, limits and actual binary input', () => {
  const equations = (s: string) => [...s.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ''));
  assert.deepEqual(equations(source('ko')), equations(source('en')));
  const en = source('en');
  for (const pattern of [/positive finite sum of analytic exponentials/, /mean magnitude/, /self-interaction/, /conditional entropy/, /order of a vanishing field/, /physical J and h fixed/, /not a uniform bound/, /binary64.*Math\.log\(2\)/, /unresolved value is distinct from zero/]) assert.match(en, pattern);
});
const near = (a: number | null, b: number) => { assert.notEqual(a, null); assert.ok(Math.abs(a! - b) < 2e-12 * Math.max(1, Math.abs(b)), `${a} vs ${b}`); };
test('exercise one has no zero-magnetization configuration but an exactly zero ensemble mean', () => {
  const r = model({ spinCount: 3, couplingK: 0, fieldB: 0 });
  assert.deepEqual(r.states.map(s => s.totalMagnetization), [-3, -1, 1, 3]);
  assert.deepEqual(r.states.map(s => s.probability.value), [1 / 8, 3 / 8, 3 / 8, 1 / 8]);
  assert.equal(r.meanMagnetization.resolution, 'structural-zero'); near(r.meanAbsoluteMagnetization.value, .5); near(r.meanSquareMagnetization.value, 1 / 3); near(r.fieldSusceptibility.value, 1);
});
test('the two-spin hand calculation agrees numerically while the actual input retains its nonideal mode', () => {
  const r = model({ spinCount: 2, couplingK: Math.log(2), fieldB: 0 });
  r.states.forEach(s => near(s.probability.value, 1 / 3));
  near(r.microstateEntropyNats.value, Math.log(3) + Math.log(2) / 3); near(r.macroEntropyNats.value, Math.log(3));
  near(r.fieldSusceptibility.value, 4 / 3); near(r.meanEnergyInThermalUnits.value, -2 * Math.log(2) / 3);
  assert.deepEqual(r.modes.candidateTotalMagnetizations, [0]);
});
test('the exercise self-energy shift changes thermodynamics without manufacturing one-spin order', () => {
  const r = model({ spinCount: 1, couplingK: 6, fieldB: 0 });
  near(r.logPartition.value, 3 + Math.log(2)); near(r.logPartitionWithoutSelfInteraction.value, Math.log(2));
  assert.equal(r.meanEnergyInThermalUnits.value, -3); assert.equal(r.meanEnergyWithoutSelfInteractionInThermalUnits.value, 0);
  assert.equal(r.meanAbsoluteMagnetization.value, 1); assert.equal(r.meanMagnetization.value, 0); near(r.microstateEntropyNats.value, Math.log(2));
});
test('the announced ordered limiting branch is distinct from the finite zero-field expectation', () => {
  const finite = model({ spinCount: 32, couplingK: Math.log(3), fieldB: 0 }); assert.equal(finite.meanMagnetization.value, 0);
  // Stipulated exp(log 3)=3 gives tanh(log 3/2)=(3-1)/(3+1).
  assert.equal((3 - 1) / (3 + 1), .5); assert.ok(3 ** 5 < 2 ** 8);
  assert.match(source('en'), /next chapter establishes.*variational limit/i);
});
