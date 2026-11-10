import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getLearningContent } from '../src/lib/learning-content';
import { relativeEntropyFromWeights as evaluate } from '../src/lib/learning-relative-entropy';
function near(actual: number, expected: number) { assert.ok(Math.abs(actual - expected) < 2e-12); }
test('Helmholtz–KL bilingual manuscript renders matching math, outlines and worked solutions', async () => {
  const id = 'finite-spectrum-helmholtz-kl-variational', result = await getLearningContent(id);
  const equations: string[][] = [];
  for (const language of ['en', 'ko'] as const) {
    const source = await readFile(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
    const html = result[language];
    assert.doesNotMatch(html, /katex-error|<h1|href="\.\.\//);
    assert.match(html, /katex-display/); assert.match(html, /role="region"/); assert.match(html, /tabindex="0"/);
    assert.equal((source.match(/^### (?:연습|Exercise) [1-6]:/gm) ?? []).length, 6);
    assert.equal((source.match(/\*\*(?:해설|Answer):\*\*/g) ?? []).length, 6);
    assert.equal(new Set(result.outline[language].map(h => h.id)).size, result.outline[language].length);
    for (const heading of result.outline[language]) assert.ok(html.includes(`id="${heading.id}"`));
    if (language === 'en') { assert.doesNotMatch(source, /[가-힣]/); assert.ok(source.split(/\s+/).length > 2000); }
    else assert.ok((source.match(/[가-힣]/g) ?? []).length > 3500);
    equations.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ' ').trim()));
  }
  assert.ok(equations[0].length >= 12); assert.deepEqual(equations[0], equations[1]);
  assert.equal(result.outline.en.length, result.outline.ko.length);
});
test('worked reference-temperature examples recover the free-energy gap and hidden conditional KL', () => {
  const uniform = evaluate([1, 1], [3, 1]);
  near(uniform.divergence.value!, .5 * Math.log(3) - Math.log(2) + Math.log(4 / 3));
  near(evaluate([1, 0], [3, 1]).divergence.value!, Math.log(4 / 3));
  const hidden = evaluate([.8, 0, .2], [.4, .4, .2]);
  near(hidden.divergence.value!, .8 * Math.log(2));
  near(hidden.targetEntropy.value! - hidden.trialEntropy.value!, .8 * Math.log(2));
  assert.equal(evaluate([.8, .2], [.8, .2]).divergence.value, 0);
});
test('fixed-gap temperature change differs from changing Hamiltonian to preserve weights', () => {
  const ground = Math.sqrt(3) / (1 + Math.sqrt(3));
  near(ground, .6339745962155613); assert.notEqual(ground, .75);
  const ratioAtTwiceTemperature = Math.exp(-Math.log(3) / 2);
  near((1 - ground) / ground, ratioAtTwiceTemperature);
});
