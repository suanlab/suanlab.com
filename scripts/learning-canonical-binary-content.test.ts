import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getLearningContent } from '../src/lib/learning-content';
import { canonicalBinaryModel as evaluate } from '../src/lib/learning-canonical-binary';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
function near(actual: number | null, expected: number) {
  assert.notEqual(actual, null);
  assert.ok(Math.abs(actual! - expected) <= 8e-12 * Math.max(1, Math.abs(expected)), `${actual} vs ${expected}`);
}
test('canonical fluctuation manuscripts render matched equations, complete outlines and six solutions', async () => {
  const id = 'canonical-binary-energy-fluctuations', result = await getLearningContent(id);
  const equations: string[][] = [];
  for (const language of ['en', 'ko'] as const) {
    const source = await readFile(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
    const html = result[language];
    assert.doesNotMatch(source, /@@E\d+@@/); assert.doesNotMatch(html, /katex-error|<h1|href="\.\.\//);
    assert.match(html, /katex-display/); assert.match(html, /role="region"/); assert.match(html, /tabindex="0"/);
    assert.equal((source.match(/^## /gm) ?? []).length, 16);
    assert.equal((source.match(/^### (?:연습|Exercise) [1-6]:/gm) ?? []).length, 6);
    assert.equal((source.match(/\*\*(?:해설|Answer):\*\*/g) ?? []).length, 6);
    assert.equal((source.match(/^\| ---/gm) ?? []).length, 3);
    assert.equal(new Set(result.outline[language].map(h => h.id)).size, result.outline[language].length);
    for (const heading of result.outline[language]) assert.ok(html.includes(`id="${heading.id}"`));
    for (const link of source.matchAll(/\]\((https?:[^)]+)\)/g)) {
      assert.doesNotMatch(link[1], /\s/); assert.ok(new URL(link[1]).hostname);
    }
    if (language === 'en') { assert.doesNotMatch(source, /[가-힣]/); assert.ok(source.split(/\s+/).length > 2400); }
    else assert.ok((source.match(/[가-힣]/g) ?? []).length > 5000);
    equations.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ' ').trim()));
  }
  assert.equal(equations[0].length, 16); assert.deepEqual(equations[0], equations[1]);
  assert.equal(result.outline.en.length, result.outline.ko.length);
});
test('worked pattern/count law and four-site physical normalizations agree with the model', () => {
  const two = evaluate({ sites: 2, energyGap: kB * Math.log(2), temperature: 1 });
  near(two.oneProbability.value, 1 / 3);
  two.distribution.forEach((r, k) => near(r.probability.value, [4, 4, 1][k] / 9));
  near(two.distribution[1].probability.value! / 2, 2 / 9);
  near(two.meanExcitations.value, 2 / 3); near(two.excitationVariance.value, 4 / 9);
  near(two.heatCapacityInKB.value, 4 * Math.log(2) ** 2 / 9);
  const four = evaluate({ sites: 4, energyGap: kB, temperature: 1 });
  near(four.meanExcitations.value, 1.07576568548);
  near(four.excitationVariance.value, .78644773297);
  near(four.entropyInKB.value, 2.32881243555);
  near(four.heatCapacityInKB.value, .78644773297);
});
test('count entropy restores hidden labels while absolute and relative fluctuations scale differently', () => {
  for (const temperature of [1, -1]) {
    const p = { sites: 4, energyGap: kB, temperature }, a = evaluate(p), b = evaluate({ ...p, sites: 16 });
    for (const key of ['meanEnergy', 'energyVariance', 'entropy', 'heatCapacity'] as const)
      near(b[key].value! / a[key].value!, 4);
    near(b.relativeEnergyFluctuation!.value! / a.relativeEnergyFluctuation!.value!, .5);
    for (const r of [a, b]) {
      const H = -r.distribution.reduce((s, row) => s + row.probability.value! * Math.log(row.probability.value!), 0);
      const hidden = r.distribution.reduce((s, row) => s + row.probability.value! * Math.log(Number(row.multiplicity)), 0);
      near(r.entropyInKB.value, H + hidden);
    }
  }
});
test('independent-repetition worked solution separates standard error from ensemble variance', () => {
  const p = { sites: 2, energyGap: kB * Math.log(2), temperature: 1 }, r = evaluate(p);
  near(Math.sqrt(r.excitationVariance.value! / 400), 1 / 30);
  assert.ok(Math.sqrt(r.excitationVariance.value! / 399) > 1 / 30);
  assert.ok(r.excitationVariance.value! / (400 * .1 ** 2) > .05);
  const combined = evaluate({ ...p, sites: 6 });
  near(combined.distribution[0].probability.value, 64 / 729);
  near(evaluate(p).excitationVariance.value, r.excitationVariance.value!);
});
test('heat capacity peak is smooth and a changing Hamiltonian produces a different derivative', () => {
  // Independent 110-digit Decimal bisection of t*tanh(t/2)=2 supplies these constants.
  const t = 2.3993572805154676678, T = 1 / t;
  near(t * Math.tanh(t / 2), 2);
  const p = { sites: 1, energyGap: kB, temperature: T }, peak = evaluate(p);
  near(peak.heatCapacityInKB.value, .43922883989064515078);
  assert.ok(evaluate({ ...p, temperature: T * .99 }).heatCapacityInKB.value! < peak.heatCapacityInKB.value!);
  assert.ok(evaluate({ ...p, temperature: T * 1.01 }).heatCapacityInKB.value! < peak.heatCapacityInKB.value!);
  const h = 1e-4, n = 4;
  const along = (temperature: number) => evaluate({ sites: n, temperature, energyGap: kB * temperature });
  const pathDerivative = (along(1 + h).meanEnergy.value! - along(1 - h).meanEnergy.value!) / (2 * h);
  near(pathDerivative / kB, n / (1 + Math.E));
  assert.ok(pathDerivative > along(1).heatCapacity.value! * 1.3);
});
test('inverted example, limiting preparations and range-rescued moments remain distinct', () => {
  const p = { sites: 2, energyGap: kB * Math.log(2), temperature: 1 }, a = evaluate(p), b = evaluate({ ...p, temperature: -1 });
  b.distribution.forEach((r, k) => near(r.probability.value, [1, 4, 4][k] / 9));
  near(b.meanExcitations.value, 4 / 3); near(b.excitationVariance.value, 4 / 9);
  near(a.relativeCountFluctuation!.value, 1); near(b.relativeCountFluctuation!.value, .5);
  const tail = evaluate({ sites: 2, energyGap: 1e280, temperature: 1e280 / (kB * 800) });
  assert.equal(tail.oneProbability.value, null); assert.equal(tail.oneProbability.sign, 1);
  assert.ok(tail.meanEnergy.value! > 0); assert.ok(tail.energyVariance.value! > 0);
  const cold = evaluate({ ...p, temperature: 0 }), degenerate = evaluate({ ...p, temperature: 0, energyGap: 0 });
  assert.equal(cold.oneProbability.sign, 0); assert.equal(cold.entropy.sign, 0);
  near(degenerate.entropyInKB.value, 2 * Math.LN2); near(degenerate.excitationVariance.value, .5);
  assert.equal(degenerate.energyVariance.sign, 0); assert.equal(degenerate.relativeEnergyFluctuation, null);
  assert.equal(evaluate({ ...p, sites: 0 }).relativeEnergyFluctuation, null);
});
