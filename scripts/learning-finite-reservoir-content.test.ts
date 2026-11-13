import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getLearningContent } from '../src/lib/learning-content';
import { finiteReservoirModel as evaluate } from '../src/lib/learning-finite-reservoir';
import { binaryMultiplicityRow } from '../src/lib/learning-binary-macrostates';
function near(actual: number | null, expected: number) {
  assert.notEqual(actual, null);
  assert.ok(Math.abs(actual! - expected) <= 3e-12 * Math.max(1, Math.abs(expected)), `${actual} vs ${expected}`);
}
test('finite-reservoir bilingual manuscripts render matching equations, complete outlines and six solutions', async () => {
  const id = 'finite-binary-reservoir-canonical-comparison', result = await getLearningContent(id);
  const equations: string[][] = [];
  for (const language of ['en', 'ko'] as const) {
    const source = await readFile(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
    const html = result[language];
    assert.doesNotMatch(source, /@@E\d+@@/);
    assert.doesNotMatch(html, /katex-error|<h1|href="\.\.\//);
    assert.match(html, /katex-display/); assert.match(html, /role="region"/); assert.match(html, /tabindex="0"/);
    assert.equal((source.match(/^## /gm) ?? []).length, 17);
    assert.equal((source.match(/^### (?:연습|Exercise) [1-6]:/gm) ?? []).length, 6);
    assert.equal((source.match(/\*\*(?:해설|Answer):\*\*/g) ?? []).length, 6);
    assert.equal((source.match(/^\| ---/gm) ?? []).length, 4);
    assert.equal(new Set(result.outline[language].map(h => h.id)).size, result.outline[language].length);
    for (const heading of result.outline[language]) assert.ok(html.includes(`id="${heading.id}"`));
    for (const link of source.matchAll(/\]\((https?:[^)]+)\)/g)) {
      assert.doesNotMatch(link[1], /\s/); assert.ok(new URL(link[1]).hostname);
    }
    if (language === 'en') { assert.doesNotMatch(source, /[가-힣]/); assert.ok(source.split(/\s+/).length > 2400); }
    else assert.ok((source.match(/[가-힣]/g) ?? []).length > 5000);
    equations.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ' ').trim()));
  }
  assert.equal(equations[0].length, 26); assert.deepEqual(equations[0], equations[1]);
  assert.equal(result.outline.en.length, result.outline.ko.length);
});
test('worked distributions distinguish mean, variance, event error and individual-pattern ratios', () => {
  const four = evaluate(2, 2, 2);
  assert.deepEqual(four.distribution.map(r => r.compatibleMicrostates), ['1', '4', '1']);
  near(four.meanExcitations.value, 1); near(four.varianceExcitations.value, 1 / 3);
  near(four.canonicalVariance.value, .5); near(four.totalVariation.value, 1 / 6);
  const six = evaluate(2, 4, 2), p = six.distribution.map(r => r.microcanonical.value!);
  p.forEach((value, a) => near(value, [6, 8, 1][a] / 15));
  six.distribution.forEach((r, a) => near(r.canonical.value, [4, 4, 1][a] / 9));
  near(six.meanExcitations.value, 2 / 3); near(six.varianceExcitations.value, 16 / 45);
  near(six.canonicalVariance.value, 4 / 9); near(six.totalVariation.value, 4 / 45);
  const patternP = p.map((value, a) => value / [1, 2, 1][a]);
  near(patternP[1] / patternP[0], 2 / 3); near(patternP[2] / patternP[1], 1 / 4);
  near(Math.exp(-Math.log(2)), .5);
});
test('conditioning independent preparations gives the same fixed-shell law and a dependent pair', () => {
  const configurations = Array.from({ length: 16 }, (_, i) => Array.from({ length: 4 }, (_, j) => (i >> j) & 1))
    .filter(bits => bits.reduce((a, b) => a + b, 0) === 2);
  for (const r of [.2, .5, .8]) {
    const weights = configurations.map(bits => bits.reduce((w, x) => w * (x ? r : 1 - r), 1));
    const total = weights.reduce((a, b) => a + b, 0), probabilities = weights.map(w => w / total);
    probabilities.forEach(p => near(p, 1 / 6));
    const firstOn = configurations.reduce((s, x, i) => s + x[0] * probabilities[i], 0);
    const bothOn = configurations.reduce((s, x, i) => s + x[0] * x[1] * probabilities[i], 0);
    near(firstOn, .5); near(bothOn / firstOn, 1 / 3);
  }
});
test('entropy identity compares the same pattern spaces and restores hidden multiplicity', () => {
  for (const [n, b, k] of [[2, 2, 2], [2, 4, 2], [3, 5, 3], [2, 0, 1]]) {
    const result = evaluate(n, b, k), counts = binaryMultiplicityRow(n), rho = k / (n + b);
    const within = result.distribution.reduce((s, row, a) => s + row.microcanonical.value! * Math.log(Number(counts[a])), 0);
    const patternEntropy = result.entropyNats.value! + within;
    const canonicalPatternEntropy = -n * (rho * Math.log(rho) + (1 - rho) * Math.log1p(-rho));
    near(canonicalPatternEntropy - patternEntropy, result.microToCanonicalKlNats.value!);
  }
  const four = evaluate(2, 2, 2);
  near(four.entropyNats.value, Math.log(6) - 4 / 3 * Math.LN2);
  near(four.microToCanonicalKlNats.value, 8 / 3 * Math.LN2 - Math.log(6));
  const fixed = evaluate(2, 0, 1);
  near(fixed.microToCanonicalKlNats.value, Math.LN2);
  assert.equal(fixed.canonicalToMicroKlNats.status, 'infinite');
});
test('accuracy target fixes density and block size; a growing block keeps a variance discrepancy', () => {
  for (const [b, tv, variance] of [[2, 1 / 6, 1 / 3], [6, 1 / 14, 3 / 7], [18, 1 / 38, 9 / 19], [98, 1 / 198, 49 / 99]]) {
    const result = evaluate(2, b, (b + 2) / 2);
    near(result.totalVariation.value, tv); near(result.varianceExcitations.value, variance);
  }
  assert.ok(evaluate(2, 48, 25).totalVariation.value! > .01);
  assert.ok(evaluate(2, 50, 26).totalVariation.value! < .01);
  const growing = evaluate(128, 128, 128);
  near(growing.varianceExcitations.value! / growing.canonicalVariance.value!, 128 / 255);
  assert.ok(growing.varianceExcitations.value! / growing.canonicalVariance.value! < .51);
});
