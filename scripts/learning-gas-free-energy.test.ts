import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/gas-free-energy-independent.json';
import { gasVolumeFreeEnergy as evaluate, type GasFreeEnergyParameters, type AnchoredFreeEnergy } from '../src/lib/learning-gas-free-energy';
import type { BinaryObservation } from '../src/lib/learning-binary-products';
import { idealGasConstant as R } from '../src/lib/learning-ideal-gas-paths';
const base: GasFreeEnergyParameters = { moles: 1, temperature: 256, externalPressure: R * 8192,
  volume: .03125, referenceVolume: .03125, referenceHelmholtz: 1000 };
const anchored = new Set(['helmholtzFreeEnergy', 'gasGibbsFreeEnergy', 'constrainedPotential', 'gibbsAtExternalPressureEquilibrium']);
interface Reference { value: string; sign: number; logMagnitude: string | null }
function near(actual: number, target: number, tolerance = 4e-12) {
  assert.ok(Number.isFinite(actual));
  assert.ok(Math.abs(actual - target) <= Math.max(2 * Number.MIN_VALUE, tolerance * Math.abs(target)), `${actual} vs ${target}`);
}
function observation(actual: BinaryObservation, expected: Reference) {
  assert.equal(actual.sign, expected.sign);
  if (expected.sign === 0) { assert.equal(actual.value, 0); assert.equal(actual.logMagnitude, null); }
  else {
    assert.notEqual(actual.value, null); assert.notEqual(actual.value, 0);
    near(actual.value!, Number(expected.value));
    assert.ok(Math.abs(actual.logMagnitude! - Number(expected.logMagnitude)) < 3e-10);
  }
}
for (const [family, cases] of Object.entries({ ordinary: refs.ordinary, limits: refs.limits })) {
  for (const [i, row] of cases.entries()) test(`${family} independent force/pressure quadrature ${i + 1}`, () => {
    const result = evaluate(row.parameters);
    assert.equal(result.regime, row.regime); assert.equal(result.equilibriumConditionSatisfied, row.equilibriumConditionSatisfied);
    for (const [key, expected] of Object.entries(row.expected)) {
      const value = (result as unknown as Record<string, BinaryObservation>)[key];
      if (anchored.has(key)) {
        const energy = value as AnchoredFreeEnergy, target = Number(expected.value);
        // Anchored absolute energies use approximate transcendental values.
        // Their small cancellations are not sign/exact-zero certificates.
        assert.notEqual(energy.value, null);
        const tolerance = Math.max(5e-11, 4e-13 * Math.abs(target));
        assert.ok(Math.abs(energy.value! - target) <= tolerance, `${key}: ${energy.value} vs ${target}`);
        if (Math.abs(target) > 4 * tolerance) assert.equal(energy.sign, expected.sign);
        if (energy.sign === 0 && expected.sign !== 0) assert.equal(energy.zeroStatus, 'unresolved-transcendental-cancellation');
      } else observation(value, expected);
    }
    for (const [key, expected] of Object.entries(row.conditionalEquilibrationBudget))
      observation((result.conditionalEquilibrationBudget as unknown as Record<string, BinaryObservation>)[key], expected);
  });
}
test('reference scope includes 24 ordinary, 20 limiting and four independently differentiated examples', () => {
  assert.equal(refs.ordinary.length, 24); assert.equal(refs.limits.length, 20);
  assert.equal(refs.ordinary.filter(row => row.derivativeCheck).length, 4);
  assert.equal(refs.serializedSignificantDigits, 80);
});
test('binary-compatible equilibrium is structural while its isothermal F need not be minimal', () => {
  const q = evaluate(base);
  assert.equal(q.regime, 'equilibrium'); assert.equal(q.equilibriumVolume.value, .03125);
  assert.equal(q.availability.value, 0); assert.equal(q.externalMinusGasPressure.value, 0);
  assert.equal(q.constrainedPotential.value, q.gasGibbsFreeEnergy.value);
  assert.equal(q.constrainedPotential.value, q.gibbsAtExternalPressureEquilibrium.value);
  assert.equal(q.helmholtzFreeEnergy.value, 1000);
  const expanded = evaluate({ ...base, volume: .0625 });
  assert.ok(expanded.helmholtzFreeEnergy.value! < q.helmholtzFreeEnergy.value!);
  assert.ok(expanded.availability.value! > 0);
});
test('half/double-volume examples recover the dimensionless convex function and endpoint signs', () => {
  for (const x of [.5, 2]) {
    const q = evaluate({ ...base, volume: base.volume * x });
    near(q.availabilityPerThermalEnergy.value!, x - 1 - Math.log(x));
    assert.equal(q.relativeVolumeDisplacement.sign, Math.sign(x - 1));
    assert.equal(q.externalMinusGasPressure.sign, Math.sign(x - 1));
    assert.equal(q.constrainedPotentialCurvature.sign, 1);
    const b = q.conditionalEquilibrationBudget;
    assert.equal(b.workByGas.sign, -Math.sign(x - 1));
    assert.equal(b.heatIntoGas.value, b.workByGas.value); assert.equal(b.energyChangeGas.value, 0);
    assert.equal(b.entropyProduction.sign, 1); near(b.entropyProduction.value!, q.availability.value! / base.temperature);
    near(b.maximumReversibleWorkByGas.value! - b.workByGas.value!, q.availability.value!);
    assert.deepEqual(b.workDeficitFromReversible, q.availability);
  }
});
test('a rounded equilibrium coordinate and displayed ratio one do not erase nonzero balance and availability', () => {
  const candidates = refs.limits.slice(8, 12).map(row => evaluate(row.parameters));
  assert.ok(candidates.some(q => q.volumeRatioToEquilibrium.value === 1 && q.relativeVolumeDisplacement.sign !== 0));
  for (const q of candidates) {
    assert.equal(q.equilibriumConditionSatisfied, false); assert.equal(q.availability.sign, 1);
    assert.ok(q.availability.value! > 0); assert.notEqual(q.externalMinusGasPressure.sign, 0);
  }
});
test('direct positive gap survives identical rounded constrained potentials and cancelling displayed entropies', () => {
  const candidates = refs.limits.slice(0, 12).map(row => evaluate(row.parameters));
  assert.ok(candidates.some(q => q.constrainedPotential.value === q.gibbsAtExternalPressureEquilibrium.value && q.availability.value! > 0));
  assert.ok(candidates.some(q => q.conditionalEquilibrationBudget.entropyChangeGas.value! + q.conditionalEquilibrationBudget.entropyChangeThermalReservoir.value! === 0
    && q.conditionalEquilibrationBudget.entropyProduction.value! > 0));
});
test('an energy-reference shift affects all four anchored energies and no pressure, gap or entropy budget', () => {
  const a = evaluate({ ...base, volume: .0625 }), b = evaluate({ ...base, volume: .0625, referenceHelmholtz: 5096 });
  for (const key of anchored) near((b as unknown as Record<string, BinaryObservation>)[key].value! - (a as unknown as Record<string, BinaryObservation>)[key].value!, 4096);
  assert.deepEqual(a.availability, b.availability); assert.deepEqual(a.gasPressure, b.gasPressure);
  assert.deepEqual(a.conditionalEquilibrationBudget, b.conditionalEquilibrationBudget);
});
test('changing the reference volume changes an additive anchor, not equilibrium or the availability gap', () => {
  const a = evaluate({ ...base, volume: .0625 }), b = evaluate({ ...base, volume: .0625, referenceVolume: .0625 });
  const change = R * base.temperature * Math.log(2);
  for (const key of anchored) near((b as unknown as Record<string, BinaryObservation>)[key].value! - (a as unknown as Record<string, BinaryObservation>)[key].value!, change);
  assert.deepEqual(a.availability, b.availability); assert.deepEqual(a.equilibriumVolume, b.equilibriumVolume);
});
test('extensive enlargement scales amount, both volumes and anchor while preserving intensive pressure', () => {
  const a = evaluate({ ...base, volume: .0625 }), b = evaluate({ ...base, volume: .125, moles: 2,
    referenceVolume: .0625, referenceHelmholtz: 2000 });
  assert.deepEqual(a.availabilityPerThermalEnergy, b.availabilityPerThermalEnergy);
  near(b.availability.value!, 2 * a.availability.value!); near(b.helmholtzFreeEnergy.value!, 2 * a.helmholtzFreeEnergy.value!);
  assert.deepEqual(b.gasPressure, a.gasPressure);
  assert.deepEqual(b.externalMinusGasPressure, a.externalMinusGasPressure);
  near(b.constrainedPotentialCurvature.value!, a.constrainedPotentialCurvature.value! / 2);
});
test('inverse pressure and volume-coordinate scaling leaves energy curves unchanged', () => {
  const a = evaluate({ ...base, volume: .0625 }), b = evaluate({ ...base, volume: .125,
    referenceVolume: .0625, externalPressure: base.externalPressure / 2 });
  assert.deepEqual(a.availability, b.availability); assert.deepEqual(a.helmholtzFreeEnergy, b.helmholtzFreeEnergy);
  assert.deepEqual(a.constrainedPotential, b.constrainedPotential);
  near(b.equilibriumVolume.value!, 2 * a.equilibriumVolume.value!);
  near(b.gasPressure.value!, a.gasPressure.value! / 2);
  near(b.constrainedPotentialCurvature.value!, a.constrainedPotentialCurvature.value! / 4);
});
test('an approximate ln(e)=1 cancellation is labelled unresolved rather than structural zero', () => {
  const f = evaluate({ ...base, volume: Math.E / 32, referenceHelmholtz: R * 256 });
  assert.equal(f.helmholtzFreeEnergy.value, 0);
  assert.equal(f.helmholtzFreeEnergy.zeroStatus, 'unresolved-transcendental-cancellation');
  const g = evaluate({ ...base, volume: Math.E / 32, referenceHelmholtz: 0 });
  assert.equal(g.gasGibbsFreeEnergy.value, 0); assert.equal(g.gasGibbsFreeEnergy.zeroStatus, 'unresolved-transcendental-cancellation');
  assert.equal(refs.limits[18].expected.helmholtzFreeEnergy.sign, 1);
  assert.equal(refs.limits[19].expected.gasGibbsFreeEnergy.sign, 1);
});
test('a true anchored zero from an unchanged reference volume remains structural', () => {
  const f = evaluate({ ...base, referenceHelmholtz: 0 });
  assert.equal(f.helmholtzFreeEnergy.zeroStatus, 'structural-zero');
  const g = evaluate({ ...base, referenceHelmholtz: -R * 256 });
  assert.equal(g.gasGibbsFreeEnergy.zeroStatus, 'structural-zero');
  assert.equal(g.constrainedPotential.zeroStatus, 'structural-zero');
  assert.equal(g.gibbsAtExternalPressureEquilibrium.zeroStatus, 'structural-zero');
});
test('the supplied isothermal anchor specifies no absolute U/S, temperature derivative or piston history', () => {
  const q = evaluate(base);
  assert.equal(q.scope, 'fixed-temperature-branch-with-supplied-reference-anchor');
  for (const name of ['absoluteEntropy', 'absoluteInternalEnergy', 'time', 'history', 'relaxationRate', 'power']) assert.equal(name in q, false);
});
test('every input is finite and bounded, and malformed/missing numeric fields are rejected', () => {
  const bounds = { moles: [.25, 2], temperature: [250, 800], externalPressure: [25000, 200000],
    volume: [.001, 1], referenceVolume: [.001, 1], referenceHelmholtz: [-10000, 10000] };
  for (const key of Object.keys(bounds) as (keyof GasFreeEnergyParameters)[]) {
    for (const v of [NaN, Infinity, -Infinity, bounds[key][0] - 1, bounds[key][1] + 1, undefined, '1'])
      assert.throws(() => evaluate({ ...base, [key]: v } as GasFreeEnergyParameters), RangeError);
  }
});
test('inputs remain unchanged and all returned observations serialize without BigInt or nonfinite numbers', () => {
  const before = JSON.stringify(base), q = evaluate(base), serialized = JSON.stringify(q);
  assert.equal(JSON.stringify(base), before); assert.doesNotMatch(serialized, /NaN|Infinity/);
  assert.deepEqual(JSON.parse(serialized), q);
});
