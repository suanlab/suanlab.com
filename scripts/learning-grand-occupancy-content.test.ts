import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getLearningContent } from '../src/lib/learning-content';
import { grandCanonicalOccupancyModel as evaluate } from '../src/lib/learning-grand-canonical-occupancy';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
function near(actual: number | null, expected: number, tolerance = 8e-12) {
  assert.notEqual(actual, null);
  assert.ok(Math.abs(actual! - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} vs ${expected}`);
}
test('grand occupancy manuscripts render matched equations, complete outlines and six solutions', async () => {
  const id = 'grand-canonical-finite-occupancy-fluctuations', result = await getLearningContent(id);
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
test('balanced and biased worked laws retain pattern, response and potential normalizations', () => {
  const p = { sites: 2, energyGap: kB, chemicalPotential: kB, temperature: 1 }, balanced = evaluate(p);
  near(balanced.distribution[1].probability.value, .5);
  near(balanced.distribution[1].probability.value! / 2, .25);
  near(balanced.distribution[1].conditionalPatternProbability!.value, .5);
  const biased = evaluate({ ...p, chemicalPotential: kB * (1 - Math.log(2)) });
  biased.distribution.forEach((row, k) => near(row.probability.value, [4, 4, 1][k] / 9));
  near(biased.meanParticles.value, 2 / 3); near(biased.numberVariance.value, 4 / 9);
  near(biased.numberSusceptibility!.value! * kB, 4 / 9);
  near(biased.energyNumberCovariance.value! / kB, 4 / 9);
  near(biased.grandPotential.value! / kB, -2 * Math.log(1.5));
  const four = evaluate({ ...p, sites: 4 }), changed = evaluate({ ...p, sites: 4, chemicalPotential: kB * 1.1 });
  near(four.numberSusceptibility!.value! * .1 * kB, .1);
  near(changed.meanParticles.value! - four.meanParticles.value!, 4 / (1 + Math.exp(-.1)) - 2);
  assert.ok(Math.abs(changed.meanParticles.value! - four.meanParticles.value! - .1) > 5e-5);
});
test('fixed-alpha and fixed-mu derivatives differ and an open-system energy slope can be negative', () => {
  const M = 4, alpha = .25, h = 1e-4;
  const fixedAlpha = (b: number) => evaluate({ sites: M, energyGap: kB, chemicalPotential: alpha * kB / b, temperature: 1 / b });
  const fixedMu = (b: number) => evaluate({ sites: M, energyGap: kB, chemicalPotential: alpha * kB, temperature: 1 / b });
  const at = fixedMu(1);
  const alphaSlope = (fixedAlpha(1 + h).logGrandPartition!.value! - fixedAlpha(1 - h).logGrandPartition!.value!) / (2 * h);
  const muSlope = (fixedMu(1 + h).logGrandPartition!.value! - fixedMu(1 - h).logGrandPartition!.value!) / (2 * h);
  near(-alphaSlope, at.meanEnergy.value! / kB, 1e-7);
  near(-muSlope, at.meanEnergy.value! / kB - alpha * at.meanParticles.value!, 1e-7);
  assert.ok(Math.abs(alphaSlope - muSlope) > .2);
  const p = { sites: M, energyGap: kB, chemicalPotential: 2 * kB, temperature: 1 }, r = evaluate(p);
  const energySlope = (evaluate({ ...p, temperature: 1 + h }).meanEnergy.value! - evaluate({ ...p, temperature: 1 - h }).meanEnergy.value!) / (2 * h * kB);
  near(energySlope, -r.numberVariance.value!, 1e-7); assert.ok(energySlope < 0);
});
test('gauge shifts preserve preparation, and entropy restores the hidden count multiplicity', () => {
  const p = { sites: 4, energyGap: 2, chemicalPotential: 1, temperature: 1 / kB }, a = evaluate(p), b = evaluate({ ...p, energyGap: 10, chemicalPotential: 9 });
  near(a.meanParticles.value, b.meanParticles.value!); near(a.logGrandPartition!.value, b.logGrandPartition!.value!);
  near(a.grandPotential.value, b.grandPotential.value!); near(a.entropyInKB.value, b.entropyInKB.value!);
  near(b.meanEnergy.value! - a.meanEnergy.value!, 8 * a.meanParticles.value!);
  near(b.energyVariance.value! / a.energyVariance.value!, 25);
  for (const temperature of [1, -1]) {
    const r = evaluate({ sites: 4, energyGap: kB, chemicalPotential: .25 * kB, temperature });
    const H = -r.distribution.reduce((s, row) => s + row.probability.value! * Math.log(row.probability.value!), 0);
    const hidden = r.distribution.reduce((s, row) => s + row.probability.value! * Math.log(Number(row.multiplicity)), 0);
    near(r.entropyInKB.value, H + hidden);
  }
});
test('joint threshold paths retain their bias and differ from the literal fixed-threshold prescription', () => {
  for (const temperature of [1, 1e-4, 1e-50, 1e-200]) {
    // Epsilon=0 keeps the shrinking chemical offset representable in binary64.
    const r = evaluate({ sites: 2, energyGap: 0, chemicalPotential: Math.log(3) * kB * temperature, temperature });
    near(r.oneProbability.value, .75); near(r.meanParticles.value, 1.5); near(r.numberVariance.value, 3 / 8);
    r.distribution.forEach((row, k) => near(row.probability.value, [1, 6, 9][k] / 16));
  }
  const fixed = evaluate({ sites: 2, energyGap: 0, chemicalPotential: 0, temperature: 0 });
  near(fixed.oneProbability.value, .5); near(fixed.meanParticles.value, 1); near(fixed.numberVariance.value, .5);
  assert.equal(fixed.numberSusceptibility, null);
});
test('zero-temperature potential limits, forbidden conditionals and rescued finite-temperature response differ', () => {
  const p = { sites: 2, energyGap: 1, chemicalPotential: 2, temperature: 0 }, full = evaluate(p);
  near(full.meanParticles.value, 2); near(full.meanEnergy.value, 2); near(full.grandPotential.value, -2);
  assert.equal(full.entropy.sign, 0); assert.equal(full.logGrandPartition, null);
  assert.equal(full.logGrandPartitionStatus, 'infinite-zero-temperature-limit');
  assert.equal(full.numberSusceptibility!.sign, 0); assert.equal(full.distribution[1].conditionalPatternProbability, null);
  const empty = evaluate({ ...p, chemicalPotential: 0 }), threshold = evaluate({ ...p, chemicalPotential: 1 });
  assert.equal(empty.meanParticles.sign, 0); near(threshold.numberVariance.value, .5);
  near(threshold.entropyInKB.value, 2 * Math.LN2); assert.equal(threshold.numberSusceptibility, null);
  const temperature = 1e-297 / (800 * kB), rare = evaluate({ sites: 2, energyGap: 1e-297, chemicalPotential: 0, temperature });
  assert.equal(rare.oneProbability.value, null); assert.equal(rare.oneProbability.sign, 1);
  assert.ok(rare.numberSusceptibility!.value! > 0);
  near(Math.log(rare.numberSusceptibility!.value!), Math.log(2) - 800 - Math.log(kB * temperature), 1e-10);
});
