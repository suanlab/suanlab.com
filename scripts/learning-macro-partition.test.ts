import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/macro-partition-independent.json';
import { macroPartitionModel as evaluate } from '../src/lib/learning-macro-partition';
import type { BinaryObservation } from '../src/lib/learning-binary-products';
function near(a: number, b: number) {
  assert.ok(Number.isFinite(a));
  if (b !== 0) assert.notEqual(a, 0);
  assert.ok(Math.abs(a - b) <= Math.max(2 * Number.MIN_VALUE, 8e-12 * Math.abs(b)), `${a} vs ${b}`);
}
function observation(a: BinaryObservation | null, b: BinaryObservation | null) {
  if (b === null) { assert.equal(a, null); return; }
  assert.notEqual(a, null); assert.equal(a!.sign, b.sign);
  if (b.value === null) assert.equal(a!.value, null);
  else { assert.notEqual(a!.value, null); near(a!.value!, b.value); }
  if (b.logMagnitude === null) assert.equal(a!.logMagnitude, null);
  else { assert.notEqual(a!.logMagnitude, null); assert.ok(Math.abs(a!.logMagnitude! - b.logMagnitude) < 2e-9, `${a!.logMagnitude} vs ${b.logMagnitude}`); }
}
for (const rows of [refs.ordinary, refs.limits]) for (const row of rows) {
  test(`independent Fraction/4096-digit reference: ${row.name}`, () => {
    const r = evaluate(row.weights, row.targetWeights, row.mapping, row.groupCount);
    for (const key of ['microEntropy', 'macroEntropy', 'conditionalEntropy', 'reconstructedEntropy', 'reconstructionGain'] as const)
      observation(r[key], row[key]);
    for (const key of ['microDivergence', 'macroDivergence', 'hiddenDivergence'] as const) {
      observation(r[key], row[key]); assert.equal(r[key]?.status, row[key]?.status);
    }
    for (const key of ['probabilities', 'reconstructedProbabilities'] as const)
      r[key].forEach((a, i) => observation(a, row[key][i]));
    assert.equal(r.groups.length, row.groups.length);
    r.groups.forEach((group, j) => {
      const expected = row.groups[j];
      assert.equal(group.count, expected.count); assert.deepEqual(group.members, expected.members);
      for (const key of ['probability', 'targetProbability', 'conditionalEntropy', 'conditionalEntropyContribution', 'reconstructionGainContribution', 'conditionalDivergenceContribution'] as const)
        observation(group[key], expected[key]);
      assert.equal(group.conditionalDivergenceContribution?.status, expected.conditionalDivergenceContribution?.status);
      for (const key of ['conditionalProbabilities', 'targetConditionalProbabilities'] as const) {
        if (expected[key] === null) assert.equal(group[key], null);
        else { assert.notEqual(group[key], null); group[key]!.forEach((a, i) => observation(a, expected[key]![i])); }
      }
    });
  });
}
test('same observed groups do not imply the same hidden distribution or entropy', () => {
  const a = evaluate([6, 2, 1, 1], [4, 4, 1, 1], [0, 0, 1, 1], 2);
  const b = evaluate([8, 0, 1, 1], [4, 4, 1, 1], [0, 0, 1, 1], 2);
  a.groups.forEach((g, i) => {
    assert.equal(g.probability.value, b.groups[i].probability.value); observation(g.probability, b.groups[i].probability);
  });
  a.reconstructedProbabilities.forEach((p, i) => observation(p, b.reconstructedProbabilities[i]));
  assert.notEqual(a.microEntropy.value, b.microEntropy.value);
  assert.equal(a.macroDivergence.value, 0); assert.equal(b.macroDivergence.value, 0);
  assert.ok(a.microDivergence.value! > 0); assert.ok(b.microDivergence.value! > 0);
});
test('identity observations and a single group realize opposite information limits', () => {
  const w = [1, 2, 3, 4], v = [4, 3, 2, 1];
  const identity = evaluate(w, v, [0, 1, 2, 3], 4), erased = evaluate(w, v, [0, 0, 0, 0], 1);
  identity.probabilities.forEach((p, i) => { assert.equal(p.value, identity.reconstructedProbabilities[i].value); observation(p, identity.reconstructedProbabilities[i]); });
  assert.equal(identity.conditionalEntropy.value, 0); assert.equal(identity.reconstructionGain.value, 0);
  near(identity.microDivergence.value!, identity.macroDivergence.value!);
  assert.equal(erased.macroEntropy.value, 0); assert.equal(erased.macroDivergence.value, 0);
  near(erased.microEntropy.value!, erased.conditionalEntropy.value!);
  near(erased.microDivergence.value!, erased.hiddenDivergence!.value!);
});
test('zero-mass and empty groups have undefined conditionals and zero weighted entropy', () => {
  const r = evaluate([1, 0, 0], [1, 0, 0], [0, 1, 1], 3);
  assert.equal(r.groups[1].count, 2); assert.equal(r.groups[2].count, 0);
  for (const g of r.groups.slice(1)) {
    assert.equal(g.conditionalProbabilities, null); assert.equal(g.targetConditionalProbabilities, null);
    assert.equal(g.conditionalEntropy, null); assert.equal(g.conditionalEntropyContribution.value, 0);
    assert.equal(g.conditionalDivergenceContribution!.value, 0);
  }
});
test('support violations separate infinite KL, undefined decomposition and numerical underflow', () => {
  const hidden = evaluate([1, 1], [1, 0], [0, 0], 1);
  assert.equal(hidden.microDivergence.status, 'infinite'); assert.equal(hidden.macroDivergence.value, 0);
  assert.equal(hidden.hiddenDivergence!.status, 'infinite');
  const observed = evaluate([1, 1], [1, 0], [0, 1], 2);
  assert.equal(observed.macroDivergence.status, 'infinite'); assert.equal(observed.hiddenDivergence, null);
  assert.equal(observed.groups[1].targetConditionalProbabilities, null);
  const rare = evaluate([Number.MIN_VALUE, Number.MAX_VALUE], [2 * Number.MIN_VALUE, Number.MAX_VALUE], [0, 1], 2);
  assert.equal(rare.probabilities[0].value, null); assert.equal(rare.probabilities[0].sign, 1);
  assert.equal(rare.microDivergence.status, 'finite'); assert.equal(rare.microDivergence.value, null);
  assert.equal(rare.microDivergence.sign, 1); assert.ok(Number.isFinite(rare.microDivergence.logMagnitude));
});
test('one-ULP biases retain positive reconstruction gain unresolved by entropy subtraction', () => {
  const r = evaluate([1, 1 + Number.EPSILON], [1, 1], [0, 0], 1);
  assert.ok(r.reconstructionGain.value! > 0); near(r.reconstructionGain.value!, Number.EPSILON ** 2 / 8);
  assert.ok(Math.abs(r.reconstructedEntropy.value! - r.microEntropy.value!) <= Number.EPSILON);
  assert.notEqual(r.reconstructedEntropy.value! - r.microEntropy.value!, r.reconstructionGain.value);
});
test('group relabeling, state reordering and exact weight rescaling preserve measures', () => {
  const w = [6, 2, 1, 1], v = [4, 4, 1, 1], mapping = [0, 0, 1, 1];
  const r = evaluate(w, v, mapping, 2);
  const alternatives = [evaluate(w.map(x => x * 2 ** 900), v.map(x => x * 2 ** -900), mapping, 2),
    evaluate([...w].reverse(), [...v].reverse(), [...mapping].reverse(), 2), evaluate(w, v, [1, 1, 0, 0], 2)];
  for (const other of alternatives) for (const key of ['microEntropy', 'macroEntropy', 'conditionalEntropy', 'reconstructedEntropy', 'reconstructionGain', 'microDivergence', 'macroDivergence'] as const)
    observation(other[key], r[key]);
});
test('finite entropy and divergence chains hold over a grid of nonuniform preparations', () => {
  for (let j = 1; j <= 40; j++) {
    const w = [j, 1, 0, j + 1, 3], v = [1, j + 1, 4, 2, 3], map = [0, 0, 1, 2, 2];
    const r = evaluate(w, v, map, 4);
    near(r.microEntropy.value!, r.macroEntropy.value! + r.conditionalEntropy.value!);
    near(r.reconstructedEntropy.value!, r.microEntropy.value! + r.reconstructionGain.value!);
    near(r.microDivergence.value!, r.macroDivergence.value! + r.hiddenDivergence!.value!);
    assert.ok(r.macroEntropy.value! <= r.microEntropy.value!);
    assert.ok(r.macroDivergence.value! <= r.microDivergence.value!);
    r.groups.forEach(g => { if (g.conditionalProbabilities) near(g.conditionalProbabilities.reduce((s, p) => s + p.value!, 0), 1); });
  }
});
test('32-state resource boundary is JSON-safe and inputs are not mutated', () => {
  const w = Object.freeze(Array.from({ length: 32 }, (_, i) => i)), v = Object.freeze(Array(32).fill(Number.MAX_VALUE));
  const map = Object.freeze(Array.from({ length: 32 }, (_, i) => i % 7));
  const r = evaluate(w, v, map, 32);
  assert.doesNotThrow(() => JSON.stringify(r)); assert.equal(r.groups.length, 32);
  assert.equal('time' in r, false); assert.equal('temperature' in r, false);
  assert.equal(r.parameters.trialWeights[31], 31);
});
test('invalid weights, sparse inputs and partitions fail without silent repair', () => {
  for (const w of [[], [0, 0], [-1, 1], [NaN, 1], [Infinity, 1], Array(33).fill(1), Array(2), [1, , 1]])
    assert.throws(() => evaluate(w as number[], [1, 1], [0, 0], 1), RangeError);
  assert.throws(() => evaluate([1, 1], [1], [0, 0], 1), RangeError);
  for (const count of [0, -1, 1.5, NaN, Infinity, 33]) assert.throws(() => evaluate([1], [1], [0], count), RangeError);
  for (const map of [[], [-1, 0], [0, 2], [0, .5], [NaN, 0], Array(2)])
    assert.throws(() => evaluate([1, 1], [1, 1], map, 2), RangeError);
});

test('grouping can expose a finite KL at the cube of an underflowed probability', () => {
  const r = evaluate([0, Number.MAX_VALUE, Number.MIN_VALUE], [Number.MIN_VALUE, Number.MAX_VALUE, Number.MIN_VALUE], [0, 0, 1], 3);
  const logDelta = Math.log(Number.MIN_VALUE) - Math.log(Number.MAX_VALUE);
  assert.equal(r.macroDivergence.status, 'finite'); assert.equal(r.macroDivergence.value, null);
  assert.equal(r.macroDivergence.sign, 1);
  assert.ok(Math.abs(r.macroDivergence.logMagnitude! - (3 * logDelta - Math.LN2)) < 2e-9);
  assert.ok(r.microDivergence.logMagnitude! > r.macroDivergence.logMagnitude!);
});
