import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getLearningContent } from '../src/lib/learning-content';
import { noisyMacroObservationModel as evaluate } from '../src/lib/learning-noisy-macro-observation';
function near(actual: number | null, expected: number) {
  assert.notEqual(actual, null); assert.ok(Math.abs(actual! - expected) < 2e-12, `${actual} versus ${expected}`);
}
const h = (p: number) => -p * Math.log(p) - (1 - p) * Math.log(1 - p);
test('noisy-observation bilingual manuscripts render shared math, tables, unique anchors and six solutions', async () => {
  const id = 'finite-noisy-macro-observation-channel', result = await getLearningContent(id);
  const equations: string[][] = [];
  for (const language of ['en', 'ko'] as const) {
    const source = await readFile(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8'), html = result[language];
    assert.doesNotMatch(source, /@@E\d+@@/);
    for (const link of source.matchAll(/\]\((https?:[^)]+)\)/g)) { assert.doesNotMatch(link[1], /\s/); assert.ok(new URL(link[1]).hostname); }
    assert.doesNotMatch(html, /katex-error|<h1|href="\.\.\//);
    assert.match(html, /katex-display/); assert.match(html, /role="region"/); assert.match(html, /tabindex="0"/);
    assert.equal((html.match(/<table\b/g) ?? []).length, 4);
    assert.equal((html.match(/<table\b(?=[^>]*tabindex="0")(?=[^>]*aria-label="[^"]+")[^>]*>/g) ?? []).length, 4);
    assert.equal((source.match(/^## /gm) ?? []).length, 16);
    assert.equal((source.match(/^### (?:연습|Exercise) [1-6]:/gm) ?? []).length, 6);
    assert.equal((source.match(/\*\*(?:해설|Answer):\*\*/g) ?? []).length, 6);
    assert.equal(new Set(result.outline[language].map(x => x.id)).size, result.outline[language].length);
    for (const heading of result.outline[language]) assert.ok(html.includes(`id="${heading.id}"`));
    if (language === 'en') { assert.doesNotMatch(source, /[가-힣]/); assert.ok(source.split(/\s+/).length > 2000); }
    else assert.ok((source.match(/[가-힣]/g) ?? []).length > 3500);
    equations.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ' ').trim()));
  }
  assert.equal(equations[0].length, 26); assert.deepEqual(equations[0], equations[1]);
  assert.equal(result.outline.en.length, result.outline.ko.length);
});
test('worked joint tables and biased Bayes posteriors distinguish both conditional directions', () => {
  const fair = evaluate([1, 1], [[9, 1], [1, 9]]);
  assert.deepEqual(fair.jointProbabilities.map(row => row.map(p => p.value)), [[.45, .05], [.05, .45]]);
  assert.deepEqual(fair.outputProbabilities.map(p => p.value), [.5, .5]);
  assert.deepEqual(fair.posteriorProbabilities[0]!.map(p => p.value), [.9, .1]);
  const biased = evaluate([3, 1], [[9, 1], [1, 9]]);
  assert.deepEqual(biased.outputProbabilities.map(p => p.value), [.7, .3]);
  near(biased.posteriorProbabilities[0]![0].value, 27 / 28); near(biased.posteriorProbabilities[0]![1].value, 1 / 28);
  assert.deepEqual(biased.posteriorProbabilities[1]!.map(p => p.value), [.25, .75]);
  near(biased.mutualInformation.value, h(.3) - h(.1));
  assert.equal(Number(biased.mutualInformation.value!.toFixed(9)), .285781329);
});
test('worked noise and reversal examples separate output uncertainty from state information', () => {
  const noise = evaluate([1, 0], [[1, 1], [1, 1]]);
  assert.equal(noise.inputEntropy.value, 0); assert.equal(noise.hiddenConditionalEntropy.value, 0);
  assert.equal(noise.mutualInformation.value, 0); near(noise.outputEntropy.value, Math.LN2); near(noise.noiseConditionalEntropy.value, Math.LN2);
  const error = evaluate([1, 1], [[9, 1], [1, 9]]), independent = evaluate([1, 1], [[1, 1], [1, 1]]), flipped = evaluate([1, 1], [[0, 1], [1, 0]]);
  near(error.noiseConditionalEntropy.value, h(.1)); near(error.mutualInformation.value, Math.LN2 - h(.1));
  assert.equal(Number(error.mutualInformation.value!.toFixed(9)), .368064207);
  assert.equal(independent.mutualInformation.value, 0); near(flipped.mutualInformation.value, Math.LN2);
  assert.equal(flipped.hiddenConditionalEntropy.value, 0);
});
test('worked erasure table recovers a known missing reading and both entropy chains', () => {
  const r = evaluate([1, 1], [[3, 0, 1], [0, 3, 1]]);
  assert.deepEqual(r.jointProbabilities.map(row => row.map(p => p.value)), [[3 / 8, 0, 1 / 8], [0, 3 / 8, 1 / 8]]);
  assert.deepEqual(r.outputProbabilities.map(p => p.value), [3 / 8, 3 / 8, 1 / 4]);
  assert.deepEqual(r.posteriorProbabilities.map(row => row!.map(p => p.value)), [[1, 0], [0, 1], [.5, .5]]);
  near(r.outputEntropy.value, h(.25) + .75 * Math.LN2); near(r.hiddenConditionalEntropy.value, .25 * Math.LN2);
  near(r.mutualInformation.value, .75 * Math.LN2);
  assert.ok(r.outputEntropy.value! > r.inputEntropy.value!);
  near(r.jointEntropy.value, r.inputEntropy.value! + r.noiseConditionalEntropy.value!);
  near(r.jointEntropy.value, r.outputEntropy.value! + r.hiddenConditionalEntropy.value!);
});
test('unused-rule and range examples preserve undefined posteriors, exact zeros and positive subnormal information', () => {
  const unused = evaluate([1, 0], [[1, 0, 0], [1, 1, 0]]);
  assert.equal(unused.posteriorProbabilities[2], null); assert.equal(unused.hiddenEntropyContributions[2].value, 0);
  near(unused.rowEntropies[1].value, Math.LN2); assert.equal(unused.noiseEntropyContributions[1].value, 0);
  const rare = evaluate([Number.MIN_VALUE, 1024], [[1, 0], [0, 1]]);
  assert.equal(rare.inputProbabilities[0].value, null); assert.equal(rare.inputProbabilities[0].sign, 1);
  assert.equal(rare.inputEntropy.value, Number.MIN_VALUE); assert.equal(rare.mutualInformation.value, Number.MIN_VALUE);
  assert.deepEqual(rare.posteriorProbabilities[0]!.map(p => p.value), [1, 0]);
});
