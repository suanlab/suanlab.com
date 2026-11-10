import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getLearningContent } from '../src/lib/learning-content';
import { binaryMacrostateModel as evaluate } from '../src/lib/learning-binary-macrostates';
function near(actual: number | null, expected: number) {
  assert.notEqual(actual, null);
  assert.ok(Math.abs(actual! - expected) < 2e-12, `${actual} versus ${expected}`);
}
test('binary-shell bilingual manuscripts render matching equations, navigable outlines and six solutions', async () => {
  const id = 'binary-shell-multiplicity-block-counts', result = await getLearningContent(id);
  const equations: string[][] = [];
  for (const language of ['en', 'ko'] as const) {
    const source = await readFile(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
    const html = result[language];
    assert.doesNotMatch(source, /@@E\d+@@/);
    // A paired render can succeed while both languages share a damaged probability label.
    assert.doesNotMatch(source, /[pP]_\{\s*m full\}/);
    assert.equal(source.split(String.raw`P_{\mathrm{full}}`).length - 1, 5);
    for (const link of source.matchAll(/\]\((https?:[^)]+)\)/g)) {
      assert.doesNotMatch(link[1], /\s/); assert.ok(new URL(link[1]).hostname);
    }
    assert.doesNotMatch(html, /katex-error|<h1|href="\.\.\//);
    assert.match(html, /katex-display/); assert.match(html, /role="region"/); assert.match(html, /tabindex="0"/);
    assert.equal((source.match(/^## /gm) ?? []).length, 15);
    assert.equal((source.match(/^### (?:연습|Exercise) [1-6]:/gm) ?? []).length, 6);
    assert.equal((source.match(/\*\*(?:해설|Answer):\*\*/g) ?? []).length, 6);
    assert.equal(new Set(result.outline[language].map(h => h.id)).size, result.outline[language].length);
    for (const heading of result.outline[language]) assert.ok(html.includes(`id="${heading.id}"`));
    if (language === 'en') { assert.doesNotMatch(source, /[가-힣]/); assert.ok(source.split(/\s+/).length > 2000); }
    else assert.ok((source.match(/[가-힣]/g) ?? []).length > 3500);
    equations.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ' ').trim()));
  }
  assert.equal(equations[0].length, 15); assert.deepEqual(equations[0], equations[1]);
  assert.equal(result.outline.en.length, result.outline.ko.length);
});
test('worked shell and block examples distinguish the three probabilities and finite-population moments', () => {
  const four = evaluate(4, 2, 2);
  near(four.uniformFullSpaceMicrostateProbability.value, 1 / 16);
  near(four.selectedShell.probabilityUnderUniformFullSpace.value, 6 / 16);
  near(four.uniformWithinSelectedShellProbability.value, 1 / 6);
  assert.deepEqual(four.blockDistribution.map(b => b.compatibleMicrostates), ['1', '4', '1']);
  near(four.conditionalBlockMean.value, 1); near(four.conditionalBlockVariance.value, 1 / 3);
  const eight = evaluate(8, 4, 4);
  assert.deepEqual(eight.blockDistribution.map(b => b.compatibleMicrostates), ['1', '16', '36', '16', '1']);
  assert.equal(eight.fixedShellCountSum, '70'); near(eight.conditionalBlockVariance.value, 4 / 7);
  const nine = evaluate(9, 4, 4);
  assert.deepEqual(nine.macroModes, [4, 5]);
  assert.equal(nine.shells[4].multiplicity, '126'); assert.equal(nine.shells[5].multiplicity, '126');
});
test('worked entropy decomposition and rare-shell interpretation remain distinct', () => {
  const four = evaluate(4, 2, 2);
  near(four.macroInformationEntropy.value, 1.4075317407193153);
  near(four.meanWithinShellInformationEntropy.value, 1.3650569815204658);
  near(four.fullSpaceInformationEntropy.value, 4 * Math.log(2));
  near(four.selectedShell.logMultiplicity.value, Math.log(6));
  const rare = evaluate(1080, 0, 540);
  assert.equal(rare.selectedShell.probabilityUnderUniformFullSpace.value, null);
  assert.equal(rare.selectedShell.probabilityUnderUniformFullSpace.sign, 1);
  assert.ok(rare.selectedShell.macroEntropyContribution.value! > 0);
  assert.equal(rare.selectedShell.boltzmannEntropy.value, 0);
  assert.equal(rare.uniformWithinSelectedShellProbability.value, 1);
});
