import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { binaryConnectedCorrelationsModel as model } from '../src/lib/learning-binary-correlations';
const id = 'binary-connected-correlations-field-hessian';
const source = (language: 'ko' | 'en') => readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
const near = (a: number | null, b: number) => { assert.notEqual(a, null); assert.ok(Math.abs(a! - b) <= 2e-12 * Math.max(1, Math.abs(b)), `${a} vs ${b}`); };
for (const language of ['ko', 'en'] as const) test(`${language}: detailed correlation theory, accessible equations and explained exercises render`, async () => {
  const raw = source(language), rendered = await getLearningContent(id), html = rendered[language];
  assert.equal(rendered.outline[language].length, 15); assert.equal(new Set(rendered.outline[language].map(h => h.id)).size, 15);
  assert.equal((raw.match(/^### [1-6]\. /gm) ?? []).length, 6);
  assert.equal((raw.match(/\*\*(?:Explanation\.|해설\.)\*\*/g) ?? []).length, 6);
  assert.equal((html.match(/class="katex-display"/g) ?? []).length, 21);
  assert.equal((html.match(/aria-label="(?:Equation|수식) /g) ?? []).length, 21);
  assert.equal((html.match(/<table[^>]*tabindex="0"/g) ?? []).length, 3);
  assert.doesNotMatch(raw, /^# |^---|%%/m); assert.doesNotMatch(html, /katex-error|\$\$/);
  if (language === 'en') { assert.ok(raw.split(/\s+/).length > 2400); assert.doesNotMatch(html, /[가-힣]/); }
  else assert.ok((raw.match(/[가-힣]/g) ?? []).length > 4500);
});
test('translations share all mathematical contracts and separate source coordinates, dependence and display limits', () => {
  const equations = (s: string) => [...s.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ''));
  assert.deepEqual(equations(source('ko')), equations(source('en')));
  for (const phrase of [/two binary variables specifically/, /not a general theorem/, /physical temperature/, /negative cross response/, /strictly positive definite/, /fixed linear coordinate map/, /additional second-derivative terms/, /not automatically an intervention/, /not replaced by zero/, /actual input/, /those require integration evidence/]) assert.match(source('en'), phrase);
});
test('aligned and opposing hand preparations have exchanged total and difference fluctuations', () => {
  for (const sign of [-1, 1]) {
    const p = { couplingK: sign * Math.log(3) / 2, fieldOne: 0, fieldTwo: 0, probeOne: 1, probeTwo: 1 };
    const r = model(p), difference = model({ ...p, probeTwo: -1 });
    near(r.covariance[0][1].value, sign / 2); near(r.covarianceDeterminant.value, .75);
    near(r.probeVariance.value, 2 + sign); near(difference.probeVariance.value, 2 - sign);
    near(r.commonFieldResponsePerSpin.value, 1 + sign / 2);
    assert.equal(r.means[0].resolution, 'structural-zero'); assert.equal(r.means[1].resolution, 'structural-zero');
  }
});
test('opposing biases create a negative raw product while connected covariance remains exactly zero', () => {
  const r = model({ couplingK: 0, fieldOne: Math.log(3)/2, fieldTwo: -Math.log(3)/2, probeOne: 1, probeTwo: 1 });
  near(r.means[0].value,.5);near(r.means[1].value,-.5);near(r.pairMean.value,-.25);
  assert.equal(r.covariance[0][1].resolution,'structural-zero');near(r.covariance[0][0].value,.75);near(r.covariance[1][1].value,.75);
});
test('scalar common-field Fisher information and per-spin response differ by their observable factor', () => {
  const r = model({ couplingK: -.5, fieldOne: 0, fieldTwo: 0, probeOne: 1, probeTwo: 1 });
  near(r.probeVariance.value,2*r.commonFieldResponsePerSpin.value!);
  near(r.probeVariance.value,2+2*Math.tanh(-.5));
});
test('the special binary independence implication cannot be extended to three parity bits', () => {
  const states=[[0,0,0],[0,1,1],[1,0,1],[1,1,0]];
  for (const [i,j] of [[0,1],[0,2],[1,2]]) {
    assert.equal(states.reduce((s,x)=>s+x[i]*x[j],0)/4,.25);
    assert.equal(states.reduce((s,x)=>s+x[i],0)/4,.5);
  }
  assert.equal(states.filter(s=>s[0]===1&&s[1]===1&&s[2]===1).length,0);
  assert.notEqual(0,.5**3);
});
