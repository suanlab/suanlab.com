import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getLearningContent } from '../src/lib/learning-content';
import { macroPartitionModel as evaluate } from '../src/lib/learning-macro-partition';
function near(actual: number | null, expected: number) {
  assert.notEqual(actual, null); assert.ok(Math.abs(actual! - expected) < 2e-12, `${actual} versus ${expected}`);
}
test('weighted-partition bilingual manuscripts render shared math, unique anchors, tables and answered exercises', async () => {
  const id = 'finite-macro-partition-conditional-entropy', result = await getLearningContent(id);
  const equations: string[][] = [];
  for (const language of ['en', 'ko'] as const) {
    const source = await readFile(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8'), html = result[language];
    assert.doesNotMatch(source, /@@E\d+@@/);
    for (const link of source.matchAll(/\]\((https?:[^)]+)\)/g)) { assert.doesNotMatch(link[1], /\s/); assert.ok(new URL(link[1]).hostname); }
    assert.doesNotMatch(html, /katex-error|<h1|href="\.\.\//);
    assert.match(html, /katex-display/); assert.match(html, /role="region"/); assert.match(html, /tabindex="0"/);
    assert.equal((html.match(/<table\b/g) ?? []).length, 3);
    assert.equal((html.match(/<table\b(?=[^>]*tabindex="0")(?=[^>]*aria-label="[^"]+")[^>]*>/g) ?? []).length, 3);
    assert.equal((source.match(/^## /gm) ?? []).length, 15);
    assert.equal((source.match(/^### (?:연습|Exercise) [1-6]:/gm) ?? []).length, 6);
    assert.equal((source.match(/\*\*(?:해설|Answer):\*\*/g) ?? []).length, 6);
    assert.equal(new Set(result.outline[language].map(h => h.id)).size, result.outline[language].length);
    for (const heading of result.outline[language]) assert.ok(html.includes(`id="${heading.id}"`));
    if (language === 'en') { assert.doesNotMatch(source, /[가-힣]/); assert.ok(source.split(/\s+/).length > 2000); }
    else assert.ok((source.match(/[가-힣]/g) ?? []).length > 3500);
    equations.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ' ').trim()));
  }
  assert.equal(equations[0].length, 16); assert.deepEqual(equations[0], equations[1]);
  assert.equal(result.outline.en.length, result.outline.ko.length);
});
test('four-box worked solutions reproduce conditioning, entropy chain and reconstruction gain', () => {
  const a = evaluate([6, 2, 1, 1], [4, 4, 1, 1], [0, 0, 1, 1], 2);
  assert.deepEqual(a.groups.map(g => g.probability.value), [.8, .2]);
  assert.deepEqual(a.groups.map(g => g.conditionalProbabilities!.map(p => p.value)), [[.75, .25], [.5, .5]]);
  assert.deepEqual(a.reconstructedProbabilities.map(p => p.value), [.4, .4, .1, .1]);
  near(a.microEntropy.value, 1.0888999753452235); near(a.macroEntropy.value, .5004024235381879);
  near(a.conditionalEntropy.value, .5884975518070358); near(a.reconstructedEntropy.value, 1.1935496040981333);
  near(a.reconstructionGain.value, .10464962875290956); assert.equal(a.macroDivergence.value, 0);
  const b = evaluate([8, 0, 1, 1], [4, 4, 1, 1], [0, 0, 1, 1], 2);
  near(b.microEntropy.value, .639031859650177); near(b.conditionalEntropy.value, .13862943611198905);
  near(b.reconstructionGain.value, .5545177444479562);
});
test('preserving a comparison differs from identifying states and refinement reveals hidden information', () => {
  const p = [4, 4, 1, 1], q = [1, 1, 4, 4];
  const coarse = evaluate(p, q, [0, 0, 1, 1], 2), fine = evaluate(p, q, [0, 1, 2, 2], 3);
  near(coarse.microDivergence.value, .6 * Math.log(4)); near(coarse.macroDivergence.value, .6 * Math.log(4));
  assert.equal(coarse.hiddenDivergence!.value, 0); near(coarse.conditionalEntropy.value, Math.log(2));
  assert.ok(coarse.macroEntropy.value! < coarse.microEntropy.value!);
  near(fine.microEntropy.value, coarse.microEntropy.value!);
  const revealedInsideBlue = .8 * Math.log(2);
  near(fine.macroEntropy.value! - coarse.macroEntropy.value!, revealedInsideBlue);
  near(coarse.conditionalEntropy.value! - fine.conditionalEntropy.value!, revealedInsideBlue);
});
test('zero-mass and support-mismatch worked solutions distinguish undefined and infinite observations', () => {
  const zero = evaluate([1, 0, 0], [1, 0, 0], [0, 1, 1], 3);
  assert.equal(zero.groups[1].count, 2); assert.equal(zero.groups[2].count, 0);
  for (const group of zero.groups.slice(1)) { assert.equal(group.conditionalProbabilities, null); assert.equal(group.conditionalEntropyContribution.value, 0); }
  const hidden = evaluate([1, 1], [1, 0], [0, 0], 1), visible = evaluate([1, 1], [1, 0], [0, 1], 2);
  assert.equal(hidden.microDivergence.status, 'infinite'); assert.equal(hidden.macroDivergence.value, 0);
  assert.equal(hidden.hiddenDivergence!.status, 'infinite'); assert.equal(visible.macroDivergence.status, 'infinite');
  assert.equal(visible.groups[1].targetConditionalProbabilities, null); assert.equal(visible.hiddenDivergence, null);
});
