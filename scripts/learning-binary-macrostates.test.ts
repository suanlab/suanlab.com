import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/binary-macrostates-independent.json';
import { binaryMacrostateModel as evaluate, binaryMultiplicityRow } from '../src/lib/learning-binary-macrostates';
import type { BinaryObservation } from '../src/lib/learning-binary-products';
function near(a: number, b: number) {
  assert.ok(Number.isFinite(a)); if (b !== 0) assert.notEqual(a, 0);
  assert.ok(Math.abs(a - b) <= Math.max(2 * Number.MIN_VALUE, 6e-12 * Math.abs(b)), `${a} vs ${b}`);
}
function observation(a: BinaryObservation, b: BinaryObservation) {
  assert.equal(a.sign, b.sign);
  if (b.value === null) assert.equal(a.value, null);
  else { assert.notEqual(a.value, null); near(a.value!, b.value); }
  if (b.logMagnitude === null) assert.equal(a.logMagnitude, null);
  else { assert.notEqual(a.logMagnitude, null); assert.ok(Math.abs(a.logMagnitude! - b.logMagnitude) < 2e-9); }
}
for (const [family, rows] of Object.entries({ ordinary: refs.ordinary, limits: refs.limits })) {
  for (const [i, row] of rows.entries()) test(`${family} independent factorial/enumeration reference ${i + 1}`, () => {
    const r = evaluate(row.sites, row.excitations, row.blockASites, row.energyGap);
    assert.equal(r.totalMicrostates, row.totalMicrostates); assert.equal(r.selectedShell.multiplicity, row.multiplicity);
    assert.deepEqual(r.macroModes, row.macroModes); assert.deepEqual(r.conditionalBlockModes, row.conditionalBlockModes);
    observation(r.selectedShell.probabilityUnderUniformFullSpace, row.selectedProbability);
    observation(r.uniformWithinSelectedShellProbability, row.withinShellProbability);
    observation(r.selectedShell.logMultiplicity, row.logMultiplicity);
    observation(r.selectedShell.boltzmannEntropy, row.boltzmannEntropy);
    observation(r.selectedShell.macroEntropyContribution, row.selectedMacroEntropyContribution);
    for (const key of ['selectedEnergy', 'fullSpaceInformationEntropy', 'macroInformationEntropy', 'meanWithinShellInformationEntropy', 'conditionalBlockMean', 'conditionalBlockVariance'] as const)
      observation(r[key], row[key]);
    for (const probe of row.blockProbes) {
      const actual = r.blockDistribution.find(x => x.excitationsA === probe.excitationsA)!;
      assert.equal(actual.compatibleMicrostates, probe.count); observation(actual.probabilityGivenFixedTotal, probe.probability);
    }
    assert.equal(r.fullSpaceNormalizedExactly, true); assert.equal(r.fixedShellNormalizedExactly, true);
    assert.equal(r.shells.reduce((sum, s) => sum + BigInt(s.multiplicity), BigInt(0)), BigInt(r.totalMicrostates));
    assert.equal(r.blockDistribution.reduce((sum, s) => sum + BigInt(s.compatibleMicrostates), BigInt(0)), BigInt(r.selectedShell.multiplicity));
  });
}
test('four distinguishable sites expose three different probabilities', () => {
  const r = evaluate(4, 2, 2);
  assert.deepEqual(r.shells.map(s => s.multiplicity), ['1', '4', '6', '4', '1']);
  assert.equal(r.uniformFullSpaceMicrostateProbability.value, 1 / 16);
  assert.equal(r.selectedShell.probabilityUnderUniformFullSpace.value, 6 / 16);
  near(r.uniformWithinSelectedShellProbability.value!, 1 / 6);
  assert.deepEqual(r.blockDistribution.map(s => s.compatibleMicrostates), ['1', '4', '1']);
  assert.equal(r.conditionalBlockMean.value, 1); near(r.conditionalBlockVariance.value!, 1 / 3);
});
test('explicit bit-string enumeration independently checks fixed-shell block counts', () => {
  for (let n = 0; n <= 10; n++) for (const k of [...new Set([0, Math.floor(n / 2), n])]) {
    const a = Math.floor(n / 2), counts = new Map<number, number>();
    for (let mask = 0; mask < 2 ** n; mask++) {
      if (mask.toString(2).split('1').length - 1 !== k) continue;
      const count = (mask & (2 ** a - 1)).toString(2).split('1').length - 1;
      counts.set(count, (counts.get(count) ?? 0) + 1);
    }
    const r = evaluate(n, k, a);
    for (const s of r.blockDistribution) assert.equal(BigInt(s.compatibleMicrostates), BigInt(counts.get(s.excitationsA)!));
  }
});
test('entropy chain rule distinguishes a full-space preparation from one fixed shell', () => {
  for (const n of [0, 1, 4, 32, 256, 4096]) {
    const r = evaluate(n, Math.floor(n / 2), Math.floor(n / 2));
    near(r.macroInformationEntropy.value! + r.meanWithinShellInformationEntropy.value!, r.fullSpaceInformationEntropy.value!);
    if (n > 0) assert.ok(r.selectedShell.logMultiplicity.value! < r.fullSpaceInformationEntropy.value!);
  }
});
test('an undisplayable probability can still contribute a representable entropy', () => {
  const r = evaluate(1080, 0, 540);
  assert.equal(r.selectedShell.probabilityUnderUniformFullSpace.value, null);
  assert.equal(r.selectedShell.probabilityUnderUniformFullSpace.sign, 1);
  assert.ok(r.selectedShell.macroEntropyContribution.value! > 0);
  assert.equal(r.selectedShell.boltzmannEntropy.value, 0);
  assert.equal(r.uniformWithinSelectedShellProbability.value, 1);
});
test('exact multiplicities preserve peak ties, spin reversal and block complement symmetry', () => {
  const odd = evaluate(9, 4, 4); assert.deepEqual(odd.macroModes, [4, 5]);
  const r = evaluate(8, 3, 3), reversed = evaluate(8, 5, 3), complement = evaluate(8, 3, 5);
  assert.equal(r.selectedShell.multiplicity, reversed.selectedShell.multiplicity);
  for (const a of r.blockDistribution) {
    const opposite = reversed.blockDistribution.find(x => x.excitationsA === 3 - a.excitationsA)!;
    const otherBlock = complement.blockDistribution.find(x => x.excitationsA === a.excitationsB)!;
    assert.equal(a.compatibleMicrostates, opposite.compatibleMicrostates);
    assert.equal(a.compatibleMicrostates, otherBlock.compatibleMicrostates);
  }
});
test('energy-gap changes preserve counting and probabilities but change shell energy', () => {
  const a = evaluate(12, 4, 6, 1), b = evaluate(12, 4, 6, 2);
  assert.equal(b.selectedEnergy.value, 2 * a.selectedEnergy.value!);
  assert.deepEqual(a.selectedShell, b.selectedShell); assert.deepEqual(a.blockDistribution, b.blockDistribution);
});
test('degenerate block/vacuum boundaries stay exact and no dynamics is inferred', () => {
  for (const [n, k, a] of [[0, 0, 0], [1, 1, 0], [32, 12, 0], [32, 12, 32]]) {
    const r = evaluate(n, k, a);
    assert.equal(r.conditionalBlockVariance.value, 0); assert.equal(r.blockDistribution.length, 1);
    assert.equal(r.blockDistribution[0].probabilityGivenFixedTotal.value, 1);
    assert.equal('time' in r, false); assert.equal('temperature' in r, false);
  }
});
test('count/probability overflow and serialization do not require floating factorials', () => {
  const r = evaluate(4096, 2048, 2048, Number.MAX_VALUE);
  assert.equal(r.totalMicrostatesObservation.value, null); assert.equal(r.totalMicrostatesObservation.sign, 1);
  assert.equal(r.selectedEnergy.value, null); assert.equal(r.selectedEnergy.sign, 1);
  assert.equal(r.uniformWithinSelectedShellProbability.value, null); assert.equal(r.uniformWithinSelectedShellProbability.sign, 1);
  assert.doesNotThrow(() => JSON.stringify(r));
});
test('invalid counts and energy inputs fail rather than being rounded or clamped', () => {
  for (const n of [-1, .5, 4097, NaN, Infinity]) assert.throws(() => binaryMultiplicityRow(n), RangeError);
  for (const [n, k, a] of [[4, 5, 2], [4, -1, 2], [4, 2, 5], [4, 2.5, 2]]) assert.throws(() => evaluate(n, k, a), RangeError);
  for (const gap of [0, -1, NaN, Infinity]) assert.throws(() => evaluate(4, 2, 2, gap), RangeError);
});
