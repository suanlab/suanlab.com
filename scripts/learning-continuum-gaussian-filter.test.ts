import test from 'node:test';
import assert from 'node:assert/strict';
import { gaussianBurgersAt as at, gaussianBurgersBudgets as budgets, type GaussianBurgersParameters } from '../src/lib/learning-continuum-gaussian-filter';
import type { TurbulenceObservation } from '../src/lib/learning-continuum-turbulence';
import refs from './fixtures/continuum-gaussian-independent.json';
const p: GaussianBurgersParameters = { wavenumber: 1, filterWidth: .5, kinematicViscosity: .1, initialFirstAmplitude: 1, initialSecondAmplitude: -.5, density: 1, area: 1, frameVelocity: 0 };
const near = (a: number, b: number, tol = 3e-10) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} versus ${b}`);
function expected(a: TurbulenceObservation, e: { value: string; sign: number; logMagnitude: string | null }, name: string) {
  assert.equal(a.sign, e.sign, name);
  if (!e.sign) { assert.equal(a.value, 0, name); assert.equal(a.logMagnitude, null, name); return; }
  near(a.logMagnitude!, Number(e.logMagnitude), 3e-11);
  const n = Number(e.value);
  if (!n || !Number.isFinite(n)) assert.equal(a.value, null, name);
  else { assert.notEqual(a.value, null, name); assert.ok(Math.abs(a.value! - n) <= 5e-10 * Math.abs(n) + Number.MIN_VALUE, `${name}: ${a.value} versus ${n}`); }
}
for (const [name, cases] of [['ordinary', refs.ordinaryCases], ['limit', refs.limitCases]] as const) for (const [i, r] of cases.entries()) {
  test(`${name} Gaussian polynomial/convolution/spatial-budget reference ${i + 1}`, () => {
    const q = at(r.parameters, r.time, r.xCycles, r.phaseCycles), b = budgets(r.parameters, r.time);
    for (const [key, e] of Object.entries(r.expected)) expected(q[key as keyof typeof q] as TurbulenceObservation, e, key);
    for (const [key, e] of Object.entries(r.budgets)) expected(b[key as keyof typeof b] as TurbulenceObservation, e, key);
  });
}
test('worked whole-cell energies and independently forced energy decay match the declared numerical example', () => {
  const b = budgets(p, .3); near(b.totalKineticEnergy.value!, 1.788228308739314); near(b.resolvedKineticEnergy.value!, 1.26573670137878); near(b.residualKineticEnergy.value!, .5224916073605345);
  near(b.resolvedMolecularLoss.value!, .3213318902614171); near(b.resolvedToResidualTransfer.value!, .2695724913206165); near(b.resolvedEnergyRate.value!, -.3213318902614171);
  near(b.resolvedForcingWork.value! - b.resolvedMolecularLoss.value! - b.resolvedToResidualTransfer.value!, b.resolvedEnergyRate.value!);
});
test('positive variance factors remain nonnegative over positions, frames, widths, signs and single modes', () => {
  for (const width of [0, Number.MIN_VALUE, 1e-20, .1, .5, 4]) for (const B of [-.5, 0, .5]) for (const phase of [0, .125, .25, .375, .5, .9]) {
    const q = at({ ...p, filterWidth: width, initialSecondAmplitude: B, frameVelocity: -.7 }, .3, phase);
    assert.ok(q.variance.sign >= 0); assert.ok(q.positiveVarianceTerms.every(t => t.sign >= 0));
    if (q.variance.value !== null) near(q.positiveVarianceTerms.reduce((sum, t) => sum + (t.value ?? 0), 0), q.variance.value);
  }
});
test('zero width is the identity while a finite Gaussian is not an idempotent Reynolds projection', () => {
  for (const x of [0, .125, .25, .75]) {
    const q = at({ ...p, filterWidth: 0 }, .3, x); assert.deepEqual(q.velocity, q.filteredVelocity); assert.equal(q.variance.value, 0); assert.equal(q.varianceGradient.value, 0); assert.equal(q.residualVelocity.value, 0); assert.equal(q.filteredResidualVelocity.value, 0);
  }
  const q = at(p, .3, .125); assert.notEqual(q.filteredResidualVelocity.value, 0); near(q.filteredResidualVelocity.value!, q.filteredVelocity.value! - q.twiceFilteredVelocity.value!);
});
test('actual physical spatial differences verify stress gradient, filtered force and local energy flux divergence', () => {
  const time = .3, x = .13, h = 1e-5, L = 2 * Math.PI / p.wavenumber, q = at(p, time, x), lo = at(p, time, x - h / L), hi = at(p, time, x + h / L);
  near((hi.filteredVelocity.value! - lo.filteredVelocity.value!) / (2 * h), q.filteredGradient.value!, 2e-8);
  near((hi.variance.value! - lo.variance.value!) / (2 * h), q.varianceGradient.value!, 2e-8);
  near((hi.resolvedEnergyFlux.value! - lo.resolvedEnergyFlux.value!) / (2 * h), q.resolvedEnergyFluxGradient.value!, 2e-8);
  near(q.filteredTimeRate.value! + q.resolvedNonlinearFluxGradient.value!, p.kinematicViscosity * q.filteredCurvature.value! + q.filteredForce.value! + q.subgridForce.value!);
  near(q.resolvedEnergyRatePerMass.value! + q.resolvedEnergyFluxGradient.value!, -q.resolvedMolecularLossPerMass.value! + q.filteredForcingWorkPerMass.value! - q.transferPerMass.value!);
});
test('actual time differences include both molecular amplitude decay and the moving-coordinate derivative', () => {
  const q = { ...p, frameVelocity: 1.3 }, t = .3, h = 1e-5, x = .17, a = at(q, t, x), lo = at(q, t - h, x), hi = at(q, t + h, x);
  near((hi.velocity.value! - lo.velocity.value!) / (2 * h), a.velocityTimeRate.value!, 2e-8); near((hi.filteredVelocity.value! - lo.filteredVelocity.value!) / (2 * h), a.filteredTimeRate.value!, 2e-8);
  const early = budgets(q, t - h), late = budgets(q, t + h), b = budgets(q, t);
  for (const [energy, rate] of [['totalKineticEnergy', 'totalEnergyRate'], ['resolvedKineticEnergy', 'resolvedEnergyRate'], ['residualKineticEnergy', 'residualEnergyRate']] as const) near((late[energy].value! - early[energy].value!) / (2 * h), b[rate].value!, 2e-8);
});
test('Galilean stress and transfer agree at corresponding moving positions while velocity and local energy flux change', () => {
  const t = .3, x = .17, C = 1.3, shiftedX = x + C * t * p.wavenumber / (2 * Math.PI), a = at(p, t, x), b = at({ ...p, frameVelocity: C }, t, shiftedX);
  near(b.velocity.value! - a.velocity.value!, C); near(b.filteredVelocity.value! - a.filteredVelocity.value!, C);
  for (const key of ['variance', 'varianceGradient', 'filteredGradient', 'force', 'filteredForce', 'transferPerMass'] as const) near(a[key].value!, b[key].value!);
  assert.notEqual(a.resolvedEnergyFlux.value, b.resolvedEnergyFlux.value);
  const ba = budgets(p, t), bb = budgets({ ...p, frameVelocity: C }, t); near(bb.totalKineticEnergy.value! - ba.totalKineticEnergy.value!, .5 * p.density * p.area * ba.cellLength * C * C);
  near(ba.resolvedToResidualTransfer.value!, bb.resolvedToResidualTransfer.value!);
});
test('positive variance permits both local transfer signs and either whole-cell transfer direction without negative molecular loss', () => {
  const values = Array.from({ length: 64 }, (_, i) => at(p, .3, i / 64)); assert.ok(values.some(q => q.transferPerMass.sign > 0)); assert.ok(values.some(q => q.transferPerMass.sign < 0));
  const a = budgets(p, .3), b = budgets({ ...p, initialSecondAmplitude: .5 }, .3); near(a.resolvedToResidualTransfer.value!, -b.resolvedToResidualTransfer.value!); near(a.resolvedMolecularLoss.value!, b.resolvedMolecularLoss.value!); near(a.resolvedKineticEnergy.value!, b.resolvedKineticEnergy.value!);
  for (const q of values) assert.ok(q.resolvedMolecularLossPerMass.sign >= 0);
});
test('single-mode extrema retain fourth-order variance, and a cancelled gradient retains the sixth-order Gaussian moment coefficient', () => {
  for (const width of [1e-20, 1e-200, Number.MIN_VALUE]) {
    const single = at({ ...p, filterWidth: width, initialSecondAmplitude: 0 }, 0, .25);
    near(single.variance.logMagnitude!, -Math.log(2) + 4 * Math.log(width)); assert.equal(single.variance.sign, 1);
    const cancelled = at({ ...p, filterWidth: width }, 0, 0);
    near(cancelled.variance.logMagnitude!, Math.log(15 / 4) + 6 * Math.log(width)); assert.equal(cancelled.variance.sign, 1);
  }
});
test('tiny differential diffusion, tiny moving phase and subnormal amplitudes survive exact initial cancellation', () => {
  const q = at({ ...p, filterWidth: Number.MIN_VALUE, frameVelocity: 2 }, Number.MIN_VALUE, 0);
  assert.notEqual(q.velocityDeparture.sign, 0); assert.notEqual(q.firstAmplitudeChange.sign, 0); assert.notEqual(q.filteredResidualVelocity.sign, 0);
  assert.ok(q.firstAmplitudeChange.logMagnitude! < -740); assert.ok(q.variance.logMagnitude! < -1400);
  const small = at({ ...p, filterWidth: Number.MIN_VALUE, initialFirstAmplitude: 1e-200, initialSecondAmplitude: -.5e-200 }, 0, 0);
  near(small.variance.logMagnitude!, Math.log(15 / 4) + 2 * Math.log(1e-200) + 6 * Math.log(Number.MIN_VALUE));
});
test('broad-filter tails retain positive variance gradients and actual forcing even after the visible variance has rounded flat', () => {
  const q = at({ ...p, wavenumber: 4, filterWidth: 4, initialSecondAmplitude: 0 }, 0, .125);
  near(q.variance.value!, .5); assert.notEqual(q.varianceGradient.sign, 0); assert.notEqual(q.filteredForce.sign, 0);
  near(q.filteredForce.logMagnitude!, Math.log(2) - 512); near(q.varianceGradient.logMagnitude!, Math.log(4) - 256);
  const second = at({ ...p, wavenumber: 4, filterWidth: 4, initialFirstAmplitude: 0, initialSecondAmplitude: 1 }, 0, .0625);
  assert.notEqual(second.filteredDeparture.sign, 0); assert.notEqual(second.twiceFilteredVelocity.sign, 0); assert.equal(second.twiceFilteredVelocity.value, null);
});
test('zero amplitudes retain a constant moving background but no variance, forcing, transfer or molecular loss', () => {
  const q = { ...p, initialFirstAmplitude: 0, initialSecondAmplitude: 0, frameVelocity: 2 }, a = at(q, 4, .13), b = budgets(q, 4);
  assert.equal(a.velocity.value, 2); assert.equal(a.filteredVelocity.value, 2); assert.equal(a.variance.value, 0); assert.equal(a.filteredForce.value, 0); assert.equal(a.transferPerMass.value, 0); assert.equal(b.totalMolecularLoss.value, 0);
});
test('density and cross-sectional area scale quadratic energies and powers but leave local filtered fields unchanged', () => {
  const a = budgets(p, .3), b = budgets({ ...p, density: 2, area: 3 }, .3);
  for (const key of ['totalKineticEnergy', 'resolvedKineticEnergy', 'residualKineticEnergy', 'resolvedMolecularLoss', 'resolvedToResidualTransfer'] as const) near(b[key].value!, 6 * a[key].value!);
  assert.deepEqual(at(p, .3, .125).variance, at({ ...p, density: 2, area: 3 }, .3, .125).variance);
});
test('invalid parameter domains and missing/nonfinite inputs throw rather than clamping into a different filter experiment', () => {
  for (const key of Object.keys(p) as (keyof typeof p)[]) for (const bad of [NaN, Infinity, -Infinity, undefined]) {
    assert.throws(() => at({ ...p, [key]: bad } as GaussianBurgersParameters, .3, .125), RangeError); assert.throws(() => budgets({ ...p, [key]: bad } as GaussianBurgersParameters, .3), RangeError);
  }
  for (const time of [-1, 5, NaN, Infinity]) { assert.throws(() => at(p, time, .125), RangeError); assert.throws(() => budgets(p, time), RangeError); }
  for (const x of [-.1, 1.1, NaN, Infinity]) { assert.throws(() => at(p, .3, x), RangeError); assert.throws(() => at(p, .3, 0, x), RangeError); }
  assert.throws(() => at({ ...p, wavenumber: 0 }, .3, .125), RangeError); assert.throws(() => at({ ...p, filterWidth: -1 }, .3, .125), RangeError);
});
test('all supported extreme trees contain finite numbers or explicit unavailable observations', () => {
  function finite(v: unknown): void { if (typeof v === 'number') assert.ok(Number.isFinite(v)); else if (v && typeof v === 'object') Object.values(v).forEach(finite); }
  for (const width of [0, Number.MIN_VALUE, 1e-200, .5, 4]) for (const A of [0, Number.MIN_VALUE, -1e-200, 2]) for (const time of [0, Number.MIN_VALUE, 4]) {
    const q = { ...p, filterWidth: width, initialFirstAmplitude: A, initialSecondAmplitude: A, frameVelocity: -2, wavenumber: 4, kinematicViscosity: 2 }; finite(at(q, time, .125)); finite(budgets(q, time));
  }
});
