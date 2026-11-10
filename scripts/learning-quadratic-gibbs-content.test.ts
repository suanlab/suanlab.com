import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { quadraticGibbsModel as model } from '../src/lib/learning-quadratic-gibbs';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
const id = 'quadratic-gaussian-gibbs-normalizability';
const source = (lang: 'ko' | 'en') => readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
for (const lang of ['ko', 'en'] as const) test(`${lang}: detailed Gaussian normalization, reference volume, response and six explained exercises render accessible math`, async () => {
  const text = source(lang), r = await getLearningContent(id), html = r[lang];
  assert.equal(r.outline[lang].length, 14); assert.equal(new Set(r.outline[lang].map(h => h.id)).size, 14);
  assert.equal((text.match(/\$\$/g) ?? []).length, 26);
  assert.equal((text.match(/^\d\. \*\*/gm) ?? []).length, 6); assert.equal((text.match(/\*\*(?:Answer|해설):\*\*/g) ?? []).length, 6);
  assert.equal((html.match(/class="katex-display"/g) ?? []).length, 13);
  assert.equal((html.match(/aria-label="(?:Equation|수식) /g) ?? []).length, 13);
  assert.equal((html.match(/<table[^>]*tabindex="0"/g) ?? []).length, 2);
  assert.doesNotMatch(text, /^# |^---/m); assert.doesNotMatch(html, /katex-error|\$\$/);
  if (lang === 'en') { assert.ok(text.split(/\s+/).length >= 2500); assert.doesNotMatch(html, /[가-힣]/); }
  else assert.ok((text.match(/[가-힣]/g) ?? []).length >= 4600);
});
test('bilingual derivations agree and distinguish dimensional precision, densities, coordinates and classical low-temperature limits', () => {
  const equations = (s: string) => [...s.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, '').replace(/[.,]$/, ''));
  assert.deepEqual(equations(source('ko')), equations(source('en')));
  for (const p of [/relative to our area measure/, /Precision is inverse covariance/, /eigenvalue tolerance/, /same measure with a different reference volume/, /positive-temperature slope remains kB/, /A local Gaussian can approximate the neighborhood/, /relative mantissa precision/]) assert.match(source('en'), p);
});
const basic = { firstStiffness: 2 * kB, crossStiffness: kB, secondStiffness: 2 * kB, firstSourceEnergy: 0, secondSourceEnergy: 0, energyOffset: 0, temperature: 1 };
const near = (a: number | null, b: number) => { assert.notEqual(a, null); assert.ok(Math.abs(a! - b) <= 3e-12 * Math.abs(b), `${a} vs ${b}`); };
test('the worked inverse matrix, source shift and energy identities agree with independently checked preparations', () => {
  const a = model(basic), b = model({ ...basic, firstSourceEnergy: kB }); assert.ok(a.normalizable && b.normalizable);
  near(a.covariance[0][0].value, 2 / 3); near(a.covariance[0][1].value, -1 / 3); near(a.correlation.value, -.5);
  near(a.logPartition.value, Math.log(2 * Math.PI / Math.sqrt(3)));
  near(b.mean[0].value, 2 / 3); near(b.mean[1].value, -1 / 3); near(b.minimumEnergy.value, -kB / 3); near(b.meanEnergy.value, 2 * kB / 3);
  near(b.meanEnergy.value! - b.freeEnergy.value!, kB * b.differentialEntropyNats.value!);
});
test('the small-cell example bounds a numerical area integral instead of treating peak density as probability', () => {
  const r = model({ ...basic, firstStiffness: 100 * kB, crossStiffness: 0, secondStiffness: 100 * kB }); assert.ok(r.normalizable);
  const peak = r.densityAtMean.value!, side = .01, N = 100, step = side / N;
  let total = 0;
  for (let i = 0; i <= N; i++) total += (i === 0 || i === N ? 1 : i % 2 ? 4 : 2) * Math.exp(-50 * (-side / 2 + i * step) ** 2);
  const integral1D = total * step / 3, cellMass = peak * integral1D ** 2;
  assert.ok(peak > 15); assert.ok(cellMass < .002);
  assert.ok(cellMass >= peak * side ** 2 * Math.exp(-.0025)); assert.ok(cellMass <= peak * side ** 2);
});
test('dimensional and dimensionless source responses differ by the stated thermal energy', () => {
  const delta = .01 * kB, minus = model({ ...basic, firstSourceEnergy: -delta }), plus = model({ ...basic, firstSourceEnergy: delta }), base = model(basic);
  assert.ok(minus.normalizable && plus.normalizable && base.normalizable);
  for (let i = 0; i < 2; i++) near((plus.mean[i].value! - minus.mean[i].value!) / (2 * delta), base.covariance[i][0].value! / kB);
});
