import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/noisy-macro-observation-independent.json';
import { noisyMacroObservationModel as evaluate } from '../src/lib/learning-noisy-macro-observation';
import { macroPartitionModel } from '../src/lib/learning-macro-partition';
import type { BinaryObservation } from '../src/lib/learning-binary-products';
function near(a: number, b: number) {
  assert.ok(Number.isFinite(a)); if (b !== 0) assert.notEqual(a, 0);
  assert.ok(Math.abs(a - b) <= Math.max(2 * Number.MIN_VALUE, 9e-12 * Math.abs(b)), `${a} vs ${b}`);
}
function observation(a: BinaryObservation | null, b: BinaryObservation | null) {
  if (b === null) { assert.equal(a, null); return; }
  assert.notEqual(a, null); assert.equal(a!.sign, b.sign);
  if (b.value === null) assert.equal(a!.value, null);
  else { assert.notEqual(a!.value, null); near(a!.value!, b.value); }
  if (b.logMagnitude === null) assert.equal(a!.logMagnitude, null);
  else { assert.notEqual(a!.logMagnitude, null); assert.ok(Math.abs(a!.logMagnitude! - b.logMagnitude) < 3e-9, `${a!.logMagnitude} vs ${b.logMagnitude}`); }
}
for (const rows of [refs.ordinary, refs.limits]) for (const row of rows) {
  test(`independent exact channel/adaptive-precision reference: ${row.name}`, () => {
    const r = evaluate(row.preparationWeights, row.channelWeights);
    for (const key of ['inputEntropy', 'outputEntropy', 'jointEntropy', 'noiseConditionalEntropy', 'hiddenConditionalEntropy', 'mutualInformation'] as const)
      observation(r[key], row[key]);
    for (const key of ['inputProbabilities', 'outputProbabilities', 'rowEntropies', 'posteriorEntropies', 'noiseEntropyContributions', 'hiddenEntropyContributions'] as const)
      r[key].forEach((value, i) => observation(value, row[key][i]));
    for (const key of ['channelProbabilities', 'jointProbabilities', 'posteriorProbabilities'] as const)
      r[key].forEach((values, i) => {
        const expected = row[key][i];
        if (expected === null) assert.equal(values, null);
        else { assert.notEqual(values, null); values!.forEach((value, j) => observation(value, expected[j])); }
      });
    assert.equal(r.independentExactly, row.independentExactly);
    assert.equal(r.deterministicOnPreparationSupport, row.deterministicOnPreparationSupport);
    assert.ok(Object.values(r.normalization).every(Boolean));
  });
}
const binaryEntropy = (p: number) => -p * Math.log(p) - (1 - p) * Math.log(1 - p);
test('binary error and erasure examples match independent closed information formulas', () => {
  const error = evaluate([1, 1], [[9, 1], [1, 9]]);
  near(error.inputEntropy.value!, Math.LN2); near(error.outputEntropy.value!, Math.LN2);
  near(error.noiseConditionalEntropy.value!, binaryEntropy(.1)); near(error.hiddenConditionalEntropy.value!, binaryEntropy(.1));
  near(error.mutualInformation.value!, Math.LN2 - binaryEntropy(.1));
  assert.deepEqual(error.posteriorProbabilities.map(p => p!.map(x => x.value)), [[.9, .1], [.1, .9]]);
  const erased = evaluate([1, 1], [[3, 0, 1], [0, 3, 1]]);
  assert.deepEqual(erased.outputProbabilities.map(p => p.value), [3 / 8, 3 / 8, 1 / 4]);
  near(erased.outputEntropy.value!, binaryEntropy(.25) + .75 * Math.LN2);
  near(erased.noiseConditionalEntropy.value!, binaryEntropy(.25)); near(erased.hiddenConditionalEntropy.value!, .25 * Math.LN2);
  near(erased.mutualInformation.value!, .75 * Math.LN2);
  assert.deepEqual(erased.posteriorProbabilities[2]!.map(p => p.value), [.5, .5]);
});
test('deterministic channel reproduces the separately implemented macro partition', () => {
  const w = [6, 2, 1, 1], mapping = [0, 0, 1, 1];
  const channel = evaluate(w, [[1, 0, 0], [7, 0, 0], [0, 2, 0], [0, 9, 0]]);
  const partition = macroPartitionModel(w, w, mapping, 3);
  observation(channel.inputEntropy, partition.microEntropy); observation(channel.outputEntropy, partition.macroEntropy);
  observation(channel.hiddenConditionalEntropy, partition.conditionalEntropy);
  observation(channel.mutualInformation, partition.macroEntropy);
  assert.equal(channel.noiseConditionalEntropy.value, 0); assert.equal(channel.deterministicOnPreparationSupport, true);
});
test('independent output noise may exceed input entropy while supplying exactly zero information', () => {
  for (const p of [[1, 0], [3, 1], [Number.MIN_VALUE, Number.MAX_VALUE]]) {
    const r = evaluate(p, [[1, 1], [7, 7]]);
    assert.equal(r.independentExactly, true); assert.equal(r.mutualInformation.value, 0);
    observation(r.hiddenConditionalEntropy, r.inputEntropy); near(r.outputEntropy.value!, Math.LN2);
    if (p[1] === 0) { assert.equal(r.inputEntropy.value, 0); assert.ok(r.outputEntropy.value! > r.inputEntropy.value!); }
    r.posteriorProbabilities.forEach(row => row!.forEach((v, i) => observation(v, r.inputProbabilities[i])));
  }
});
test('an input with zero preparation probability cannot change information via its channel row', () => {
  const a = evaluate([1, 0], [[1, 0], [1, 1]]), b = evaluate([1, 0], [[1, 0], [0, 7]]);
  for (const key of ['inputEntropy', 'outputEntropy', 'jointEntropy', 'noiseConditionalEntropy', 'hiddenConditionalEntropy', 'mutualInformation'] as const)
    assert.deepEqual(a[key], b[key]);
  assert.equal(a.deterministicOnPreparationSupport, true); assert.equal(a.rowEntropies[1].value, Math.LN2);
  assert.equal(a.noiseEntropyContributions[1].value, 0);
});
test('an unused output is undefined posterior, while an underflowed positive output remains conditionable', () => {
  const empty = evaluate([1, 1], [[1, 0, 0], [0, 1, 0]]);
  assert.equal(empty.posteriorProbabilities[2], null); assert.equal(empty.posteriorEntropies[2], null);
  assert.equal(empty.hiddenEntropyContributions[2].value, 0);
  const rare = evaluate([Number.MIN_VALUE, Number.MAX_VALUE], [[1, 0], [0, 1]]);
  assert.equal(rare.outputProbabilities[0].value, null); assert.equal(rare.outputProbabilities[0].sign, 1);
  assert.deepEqual(rare.posteriorProbabilities[0]!.map(p => p.value), [1, 0]);
});
test('a nonrepresentable probability may retain a representable positive entropy and mutual information', () => {
  const r = evaluate([Number.MIN_VALUE, 1024], [[1, 0], [0, 1]]);
  assert.equal(r.inputProbabilities[0].value, null); assert.equal(r.inputProbabilities[0].sign, 1);
  assert.ok(r.inputEntropy.value! > 0); assert.ok(r.mutualInformation.value! > 0);
  assert.equal(r.noiseConditionalEntropy.value, 0); assert.equal(r.hiddenConditionalEntropy.value, 0);
});
test('one-ULP channel differences preserve positive MI rather than subtracting entropy displays', () => {
  const r = evaluate([1, 1], [[1, 1 + Number.EPSILON], [1, 1]]);
  assert.equal(r.independentExactly, false); assert.ok(r.mutualInformation.value! > 0);
  near(r.mutualInformation.value!, Number.EPSILON ** 2 / 32);
  assert.notEqual(r.outputEntropy.value! - r.noiseConditionalEntropy.value!, r.mutualInformation.value);
});
test('error probability one is invertible and equal error complements preserve information', () => {
  const a = evaluate([1, 1], [[9, 1], [1, 9]]), b = evaluate([1, 1], [[1, 9], [9, 1]]);
  observation(a.mutualInformation, b.mutualInformation);
  const flipped = evaluate([3, 1], [[0, 1], [1, 0]]);
  observation(flipped.mutualInformation, flipped.inputEntropy); assert.equal(flipped.hiddenConditionalEntropy.value, 0);
});
test('both joint-entropy chains and information bounds hold over nonuniform channels', () => {
  for (let j = 1; j <= 24; j++) {
    const r = evaluate([j, 0, 3], [[j, 1, 3], [0, 1, 1], [1, 2, j]]);
    near(r.jointEntropy.value!, r.inputEntropy.value! + r.noiseConditionalEntropy.value!);
    near(r.jointEntropy.value!, r.outputEntropy.value! + r.hiddenConditionalEntropy.value!);
    near(r.mutualInformation.value!, r.inputEntropy.value! - r.hiddenConditionalEntropy.value!);
    near(r.mutualInformation.value!, r.outputEntropy.value! - r.noiseConditionalEntropy.value!);
    assert.ok(r.mutualInformation.value! >= 0); assert.ok(r.mutualInformation.value! <= Math.min(r.inputEntropy.value!, r.outputEntropy.value!));
  }
});
test('state and output relabeling and separate exact row rescaling preserve information', () => {
  const a = evaluate([3, 1], [[9, 1], [1, 9]]);
  const alternatives = [evaluate([3 * 2 ** 800, 2 ** 800], [[9 * 2 ** 900, 2 ** 900], [2 ** -900, 9 * 2 ** -900]]),
    evaluate([1, 3], [[1, 9], [9, 1]]), evaluate([3, 1], [[1, 9], [9, 1]])];
  for (const r of alternatives) for (const key of ['inputEntropy', 'outputEntropy', 'jointEntropy', 'noiseConditionalEntropy', 'hiddenConditionalEntropy', 'mutualInformation'] as const)
    observation(r[key], a[key]);
});
test('16 by 16 resource boundary stays normalized and serializable without input mutation', () => {
  const p = Object.freeze(Array.from({ length: 16 }, (_, i) => i + 1));
  const rows = Object.freeze(Array.from({ length: 16 }, (_, i) => Object.freeze(Array.from({ length: 16 }, (_, j) => i + j + 1))));
  const r = evaluate(p, rows);
  assert.doesNotThrow(() => JSON.stringify(r)); assert.ok(Object.values(r.normalization).every(Boolean));
  assert.equal(r.jointProbabilities.length, 16); assert.equal(r.jointProbabilities[0].length, 16);
  assert.equal('temperature' in r, false); assert.equal('time' in r, false);
});
test('malformed, sparse, nonfinite and zero-total inputs fail without invented rows', () => {
  for (const p of [[], [0, 0], [-1, 2], [Infinity, 1], [NaN, 1], Array(2), Array(17).fill(1)])
    assert.throws(() => evaluate(p as number[], [[1, 0], [0, 1]]), RangeError);
  for (const rows of [[], [[0, 0], [1, 0]], [[1], [1, 0]], [[1, -1], [1, 1]], [[Infinity], [1]], [Array(2), [1, 0]], Array(2), [Array(17).fill(1), Array(17).fill(1)]])
    assert.throws(() => evaluate([1, 0], rows as number[][]), RangeError);
});

test('an unprepared extreme channel row still has a positive conditional entropy of its own', () => {
  const r = evaluate([1, 0], [[1, 0], [Number.MIN_VALUE, Number.MAX_VALUE]]);
  const logDelta = Math.log(Number.MIN_VALUE) - Math.log(Number.MAX_VALUE);
  assert.equal(r.rowEntropies[1].value, null); assert.equal(r.rowEntropies[1].sign, 1);
  assert.ok(Math.abs(r.rowEntropies[1].logMagnitude! - (logDelta + Math.log(1 - logDelta))) < 3e-9);
  assert.equal(r.noiseEntropyContributions[1].value, 0); assert.equal(r.noiseConditionalEntropy.value, 0);
});
