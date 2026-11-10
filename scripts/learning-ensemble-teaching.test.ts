import test from 'node:test';
import assert from 'node:assert/strict';
import { ensemblePresets, ensembleGuides, ensemblePresetInput, ensembleSavedValidators, ensemblePlotValue, isEnsembleKind, type EnsembleKind } from '../src/lib/learning-ensembles';
import { ensembleInputKey, ensembleResultMatchesInput } from '../src/lib/learning-ensemble-protocol';
import { evaluateEnsembleRequest } from '../src/lib/learning-ensemble-evaluator';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
function near(a: number | null, b: number) { assert.notEqual(a, null); assert.ok(Math.abs(a! - b) < 8e-12 * Math.max(1, Math.abs(b)), `${a} vs ${b}`); }
function result(kind: EnsembleKind, index: number) {
  const input = ensemblePresetInput(kind, index), response = evaluateEnsembleRequest({ version: 1, requestId: 1, input });
  assert.ok(response.ok); assert.ok(ensembleResultMatchesInput(input, response.result)); return response;
}
test('24 immutable bilingual presets produce matching serializable worker responses', () => {
  let count = 0;
  for (const kind of Object.keys(ensemblePresets) as EnsembleKind[]) {
    assert.equal(isEnsembleKind(kind), true);
    for (const [index, preset] of ensemblePresets[kind].entries()) {
      const input = ensemblePresetInput(kind, index); assert.equal(input.kind, kind);
      assert.ok(Object.isFrozen(input)); assert.ok(Object.isFrozen(input.parameters));
      assert.notEqual(input, preset.input); assert.equal(ensembleInputKey(input), ensembleInputKey(preset.input));
      assert.ok(preset.label.ko.length > 0); assert.ok(preset.label.en.length > 0); assert.doesNotMatch(preset.label.en, /[가-힣]/);
      assert.doesNotThrow(() => structuredClone(result(kind, index))); count++;
    }
  }
  assert.equal(count, 24);
  const preset = ensemblePresetInput('canonical-binary-energy-fluctuations', 0); assert.ok(preset.kind === 'canonical-binary-energy-fluctuations');
  assert.equal(preset.parameters.energyGap, kB);
});
test('nine bilingual guides reference valid presets and retain prediction, action and explanation', () => {
  let count = 0;
  for (const kind of Object.keys(ensembleGuides) as EnsembleKind[]) {
    assert.equal(ensembleGuides[kind].length, 3);
    for (const guide of ensembleGuides[kind]) {
      for (const key of ['title', 'predict', 'action', 'explain'] as const) {
        assert.ok(guide[key].ko.length > 0); assert.ok(guide[key].en.length > 0); assert.doesNotMatch(guide[key].en, /[가-힣]/);
      }
      assert.equal(ensemblePresetInput(kind, guide.preset).kind, kind); count++;
    }
  }
  assert.equal(count, 9);
});
test('finite-reservoir guide predictions match mean/variance, TV and directed support', () => {
  const kind = 'finite-binary-reservoir-canonical-comparison';
  const a = result(kind, ensembleGuides[kind][0].preset), b = result(kind, ensembleGuides[kind][1].preset), c = result(kind, ensembleGuides[kind][2].preset);
  assert.ok(a.kind === kind && b.kind === kind && c.kind === kind);
  near(a.result.meanExcitations.value, 1); near(a.result.varianceExcitations.value, 1 / 3); near(a.result.canonicalVariance.value, .5); near(a.result.totalVariation.value, 1 / 6);
  near(b.result.varianceExcitations.value, 49 / 99); near(b.result.totalVariation.value, 1 / 198);
  near(c.result.microToCanonicalKlNats.value, Math.LN2); assert.equal(c.result.canonicalToMicroKlNats.status, 'infinite');
});
test('canonical guide predictions distinguish response, scaling and range-rescued energy', () => {
  const kind = 'canonical-binary-energy-fluctuations';
  const a = result(kind, ensembleGuides[kind][0].preset), b = result(kind, ensembleGuides[kind][1].preset), c = result(kind, ensembleGuides[kind][2].preset), base = result(kind, 0);
  assert.ok(a.kind === kind && b.kind === kind && c.kind === kind && base.kind === kind);
  near(a.result.excitationVariance.value, 4 / 9); near(a.result.heatCapacityInKB.value, 4 * Math.log(2) ** 2 / 9);
  near(b.result.energyVariance.value! / base.result.energyVariance.value!, 4);
  near(b.result.relativeCountFluctuation!.value! / base.result.relativeCountFluctuation!.value!, .5);
  assert.equal(c.result.oneProbability.value, null); assert.equal(c.result.oneProbability.sign, 1);
  assert.ok(c.result.meanEnergy.value! > 0); assert.ok(c.result.energyVariance.value! > 0);
});
test('grand guides preserve conditional direction, threshold response and dilute susceptibility', () => {
  const kind = 'grand-canonical-finite-occupancy-fluctuations';
  const a = result(kind, ensembleGuides[kind][0].preset), b = result(kind, ensembleGuides[kind][1].preset), c = result(kind, ensembleGuides[kind][2].preset), full = result(kind, 3);
  assert.ok(a.kind === kind && b.kind === kind && c.kind === kind && full.kind === kind);
  near(a.result.distribution[1].probability.value, .5); near(a.result.distribution[1].conditionalPatternProbability!.value, .5);
  near(a.result.numberSusceptibility!.value! * kB, .5);
  near(b.result.numberVariance.value, .5); near(b.result.entropyInKB.value, 2 * Math.LN2); assert.equal(b.result.numberSusceptibility, null);
  assert.equal(full.result.logGrandPartitionStatus, 'infinite-zero-temperature-limit'); near(full.result.grandPotential.value, -2);
  assert.equal(c.result.oneProbability.value, null); assert.equal(c.result.oneProbability.sign, 1); assert.ok(c.result.numberSusceptibility!.value! > 0);
});
test('saved validators reject cross-family/invalid data and unavailable plot values stay distinct from zero', () => {
  const kinds = Object.keys(ensemblePresets) as EnsembleKind[];
  for (const kind of kinds) {
    const input = ensemblePresetInput(kind, 0), restored = JSON.parse(JSON.stringify(input)); assert.ok(ensembleSavedValidators[kind](restored));
    assert.equal(ensembleInputKey(restored), ensembleInputKey(input));
    for (const other of kinds.filter(v => v !== kind)) assert.equal(ensembleSavedValidators[other](restored), false);
    for (const bad of [null, 'bad', [], { ...restored, parameters: {} }]) assert.equal(ensembleSavedValidators[kind](bad), false);
    for (const index of [-1, .5, 999]) assert.throws(() => ensemblePresetInput(kind, index), RangeError);
  }
  assert.equal(isEnsembleKind('other'), false); assert.throws(() => ensemblePresetInput('other' as EnsembleKind, 0), RangeError);
  assert.equal(ensemblePlotValue(null), null); assert.equal(ensemblePlotValue({ value: null, sign: 1, logMagnitude: -800 }), null);
  assert.equal(ensemblePlotValue({ value: Infinity, sign: 1, logMagnitude: 800 }), null);
  assert.equal(ensemblePlotValue({ value: 0, sign: 0, logMagnitude: null }), 0);
});
