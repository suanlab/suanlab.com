import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { finiteGibbsModel as model } from '../src/lib/learning-finite-gibbs';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
const id = 'finite-gibbs-conditionals-scan-support';
const source = (lang: 'ko' | 'en') => readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
for (const lang of ['ko', 'en'] as const) test(`${lang}: detailed Gibbs conditioning, flows, contraction, support and six explained exercises render accessible mathematics`, async () => {
  const text = source(lang), r = await getLearningContent(id), html = r[lang];
  assert.equal(r.outline[lang].length, 15); assert.equal(new Set(r.outline[lang].map(h => h.id)).size, 15);
  assert.equal((text.match(/\$\$/g) ?? []).length, 30);
  assert.equal((text.match(/^\d\. \*\*/gm) ?? []).length, 6); assert.equal((text.match(/\*\*(?:Answer|해설):\*\*/g) ?? []).length, 6);
  assert.equal((html.match(/class="katex-display"/g) ?? []).length, 15);
  assert.equal((html.match(/aria-label="(?:Equation|수식) /g) ?? []).length, 15);
  assert.equal((html.match(/<table[^>]*tabindex="0"/g) ?? []).length, 2);
  assert.doesNotMatch(text, /^# |^---/m); assert.doesNotMatch(html, /katex-error|\$\$/);
  if (lang === 'en') { assert.ok(text.split(/\s+/).length >= 2500); assert.doesNotMatch(html, /[가-힣]/); }
  else assert.ok((text.match(/[가-힣]/g) ?? []).length >= 4800);
});
test('bilingual derivations agree and distinguish stationarity, reversibility, support, physical time and bit-grid sampling', () => {
  const equations = (s: string) => [...s.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, '').replace(/[.,]$/, ''));
  assert.deepEqual(equations(source('ko')), equations(source('en')));
  for (const p of [/preservation is weaker than convergence/i, /source-row transition matrices/i, /not bare transition entries/, /different coordinate-update counts/, /physical spin-flip rate/, /precision ceiling remains unresolved/, /reader provides a separate finite-bit path generator/]) assert.match(source('en'), p);
});
const basic = { couplingEnergy: kB * Math.log(2), firstFieldEnergy: 0, secondFieldEnergy: 0, temperature: 1, energyOffset: 0, support: 'full-product' as const };
const near = (a: number | null, b: number) => { assert.notEqual(a, null); assert.ok(Math.abs(a! - b) <= 3e-12 * Math.abs(b), `${a} vs ${b}`); };
test('worked flow counterexample and contraction calculations agree with the independently referenced model', () => {
  const r = model(basic);
  [16 / 25, 4 / 25, 1 / 25, 4 / 25].forEach((p, i) => near(r.distributions.systematic[0][i].value, p));
  near(r.flows.systematic[0][1].value, 8 / 125); near(r.flows.systematic[1][0].value, 2 / 125);
  near(r.flows.random[0][1].value, 1 / 25); near(r.flows.random[1][0].value, 1 / 25);
  near(r.systematicContraction.value, 9 / 25);
  [.3, .108, .0139968].forEach((d, i) => near(r.totalVariationToTarget.systematic[i].value, d));
  // Reversing the order gives the separately stated example, not K1 K2.
  const reverseRow = r.kernels.second[0].map((_, j) => r.kernels.second[0].reduce((s, q, k) => s + q.value! * r.kernels.first[k][j].value!, 0));
  [16 / 25, 1 / 25, 4 / 25, 4 / 25].forEach((p, i) => near(reverseRow[i], p));
});
test('the zero-sum exercise has infinitely many stationary mixtures while a block refresh reaches the target', () => {
  const r = model({ ...basic, support: 'fixed-zero-sum' });
  for (const name of ['first', 'second', 'random', 'systematic'] as const) {
    assert.equal(r.diagnostics[name].uniqueStationaryDistribution, false);
    r.totalVariationToTarget[name].forEach(q => near(q.value, .5));
  }
  r.totalVariationToTarget.blocked.forEach(q => assert.equal(q.resolution, 'structural-zero'));
});
test('the bit-grid exercise counts the event rather than assuming an exact real uniform variate', () => {
  const eventCount = Array.from({ length: 256 }, (_, r) => r / 256 < .001).filter(Boolean).length;
  assert.equal(eventCount, 1); assert.equal(eventCount / 256, 1 / 256); assert.notEqual(eventCount / 256, .001);
  assert.equal(Math.ceil(256 * .001) / 256, eventCount / 256);
});
