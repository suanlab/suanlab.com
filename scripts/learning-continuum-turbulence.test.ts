import test from 'node:test';
import assert from 'node:assert/strict';
import { reynoldsEnsembleAt as at, reynoldsInstantaneousAt as instant, reynoldsEnsembleBudgets as budgets, type ReynoldsEnsembleParameters, type TurbulenceObservation } from '../src/lib/learning-continuum-turbulence';
import refs from './fixtures/continuum-reynolds-independent.json';
import cancellationRefs from './fixtures/continuum-reynolds-cancellation-independent.json';
const p: ReynoldsEnsembleParameters = { height: 2, wavenumber: 1.5, spanWidth: 1, density: 1, dynamicViscosity: .1, bulkVelocity: .8, meanAsymmetry: .25, firstStreamfunctionAmplitude: 1, secondStreamfunctionAmplitude: .7, candidateEddyViscosity: .01 };
const near = (a: number, b: number, tol = 2e-10) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} versus ${b}`);
function expected(a: TurbulenceObservation, e: { value: string; sign: number; logMagnitude: string | null }, name: string) {
  assert.equal(a.sign, e.sign, name);
  if (e.sign === 0) { assert.equal(a.value, 0, name); assert.equal(a.logMagnitude, null, name); return; }
  near(a.logMagnitude!, Number(e.logMagnitude), 3e-11);
  const value = Number(e.value);
  if (value === 0 || !Number.isFinite(value)) assert.equal(a.value, null, name);
  else { assert.notEqual(a.value, null, name); assert.ok(Math.abs(a.value! - value) <= 4e-10 * Math.abs(value) + Number.MIN_VALUE, `${name}: ${a.value} versus ${value}`); }
}
for (const [label, cases] of [['ordinary', refs.ordinaryCases], ['limit', refs.limitCases], ['cancellation', cancellationRefs.limitCases]] as const) {
  for (const [i, r] of cases.entries()) test(`${label} independent phase/force/physical-space reference ${i + 1}`, () => {
    const q = at(r.parameters, r.xi), s = instant(r.parameters, r.xi, r.xCycles, r.phaseCycles), b = budgets(r.parameters);
    for (const [key, e] of Object.entries(r.expected)) expected(q[key as keyof typeof q] as TurbulenceObservation, e, key);
    for (const [key, e] of Object.entries(r.instantaneous)) expected(s[key as keyof typeof s] as TurbulenceObservation, e, key);
    for (const [key, e] of Object.entries(r.budgets)) expected(b[key as keyof typeof b] as TurbulenceObservation, e, key);
    assert.equal(q.inferredEddyViscosityStatus, r.inferredStatus);
    if (r.inferredEddyViscosity) expected(q.inferredEddyViscosity!, r.inferredEddyViscosity, 'inferred viscosity');
    else assert.equal(q.inferredEddyViscosity, null);
  });
}
test('eight actual phase samples reproduce the mean, all covariance entries and both force components', () => {
  for (const xi of [0, .13, .5, .81, 1]) {
    const q = at(p, xi), states = Array.from({ length: 8 }, (_, j) => instant(p, xi, 0, j / 8));
    const avg = (f: (r: typeof states[number]) => number) => states.reduce((s, r) => s + f(r), 0) / 8;
    near(avg(r => r.u.value!), q.meanU.value!); near(avg(r => r.v.value!), 0);
    near(avg(r => r.uPrime.value! ** 2), q.covarianceUU.value!); near(avg(r => r.v.value! ** 2), q.covarianceVV.value!);
    near(avg(r => r.uPrime.value! * r.v.value!), q.covarianceUV.value!); near(avg(r => r.forceX.value!), q.meanForceX.value!); near(avg(r => r.forceY.value!), q.meanForceY.value!);
  }
  assert.notEqual(at(p, .35).meanForceY.value, 0);
});
test('actual spatial differences check the declared derivatives, divergence and complete steady forced momentum equations', () => {
  const xi = .35, cycles = .13, h = 1e-5, L = 2 * Math.PI / p.wavenumber;
  const q = instant(p, xi, cycles, .25), xlo = instant(p, xi, cycles - h / L, .25), xhi = instant(p, xi, cycles + h / L, .25);
  const ylo = instant(p, xi - h / p.height, cycles, .25), yhi = instant(p, xi + h / p.height, cycles, .25);
  near((xhi.u.value! - xlo.u.value!) / (2 * h), q.derivativeUX.value!, 2e-9);
  near((yhi.u.value! - ylo.u.value!) / (2 * h), q.derivativeUY.value!, 2e-9);
  near((xhi.v.value! - xlo.v.value!) / (2 * h), q.derivativeVX.value!, 2e-9);
  near((yhi.v.value! - ylo.v.value!) / (2 * h), q.derivativeVY.value!, 2e-9);
  near(q.derivativeUX.value! + q.derivativeVY.value!, 0);
  near(p.density * (q.u.value! * q.derivativeUX.value! + q.v.value! * q.derivativeUY.value!) - p.dynamicViscosity * q.laplacianU.value!, q.forceX.value!);
  near(p.density * (q.u.value! * q.derivativeVX.value! + q.v.value! * q.derivativeVY.value!) - p.dynamicViscosity * q.laplacianV.value!, q.forceY.value!);
});
test('zero wall velocities coexist with nonzero wall shear and satisfy the actual no-slip condition', () => {
  for (const xi of [0, 1]) for (let j = 0; j < 8; j++) { const q = instant(p, xi, .13, j / 8); assert.equal(q.u.value, 0); assert.equal(q.v.value, 0); }
  assert.ok(at(p, 0).meanShear.value! > 0); assert.ok(at(p, 1).meanShear.value! < 0);
});
test('same mean can accompany different covariance and different actual forcing', () => {
  const a = at(p, .35), b = at({ ...p, firstStreamfunctionAmplitude: 0, secondStreamfunctionAmplitude: 0 }, .35);
  assert.deepEqual(a.meanU, b.meanU); assert.notDeepEqual(a.covarianceUV, b.covarianceUV); assert.notDeepEqual(a.meanForceX, b.meanForceX); assert.notDeepEqual(a.meanForceY, b.meanForceY);
});
test('scalar viscosity is undefined at zero shear and can be negative even for a positive covariance matrix', () => {
  const symmetric = { ...p, meanAsymmetry: 0 }, center = at(symmetric, .5);
  assert.equal(center.meanShear.value, 0); assert.ok(center.covarianceUV.value! < 0); assert.equal(center.inferredEddyViscosityStatus, 'nonzero-stress-at-zero-shear'); assert.equal(center.inferredEddyViscosity, null);
  const quiet = at({ ...symmetric, firstStreamfunctionAmplitude: 0, secondStreamfunctionAmplitude: 0 }, .5);
  assert.equal(quiet.inferredEddyViscosityStatus, 'indeterminate'); assert.equal(quiet.inferredEddyViscosity, null);
  assert.ok(at(symmetric, .25).inferredEddyViscosity!.value! > 0); assert.ok(at(symmetric, .75).inferredEddyViscosity!.value! < 0);
});
test('positive Gram determinant and exact rank loss are preserved independently of covariance-product cancellation', () => {
  for (let i = 0; i <= 200; i++) {
    const q = at(p, i / 200); assert.ok(q.covarianceUU.sign >= 0); assert.ok(q.covarianceVV.sign >= 0); assert.ok(q.covarianceDeterminant.sign >= 0);
    near(q.covarianceUU.value! * q.covarianceVV.value! - q.covarianceUV.value! ** 2, q.covarianceDeterminant.value!, 1e-12);
  }
  assert.equal(at(p, .5).covarianceDeterminant.value, 0); assert.ok(at(p, .5).covarianceVV.value! > 0);
});
test('local and whole-cell work distinguish molecular loss, signed transfer and modeled closure defect', () => {
  const b = budgets(p), negative = budgets({ ...p, secondStreamfunctionAmplitude: -.7 });
  near(b.meanForcingWork.value!, b.molecularMeanLoss.value! + b.meanToFluctuationTransfer.value!);
  near(b.forcingWorkDefect.value!, b.meanForcingWork.value! - b.modeledForcingWork.value!);
  near(negative.meanToFluctuationTransfer.value!, -b.meanToFluctuationTransfer.value!);
  assert.ok(at({ ...p, meanAsymmetry: 0 }, .25).meanToFluctuationTransferDensity.value! > 0);
  assert.ok(at({ ...p, meanAsymmetry: 0 }, .75).meanToFluctuationTransferDensity.value! < 0);
  assert.equal(budgets({ ...p, meanAsymmetry: 0 }).meanToFluctuationTransfer.value, 0);
  const q = at(p, .35); near(q.forceDefect.value!, q.meanForceX.value! - q.modeledForceX.value!); near(q.covarianceDefect.value!, q.covarianceUV.value! - q.modeledCovarianceUV.value!);
});
test('a tiny loss or transfer remains nonzero in signed logs even when an ordinary number cannot represent it', () => {
  const tiny = { ...p, firstStreamfunctionAmplitude: Number.MIN_VALUE, secondStreamfunctionAmplitude: Number.MIN_VALUE };
  assert.equal(at(tiny, .5).covarianceUV.value, null); assert.equal(at(tiny, .5).covarianceUV.sign, -1); assert.ok(at(tiny, .5).covarianceUV.logMagnitude! < -1400);
  const wall = at(p, Number.MIN_VALUE); assert.equal(wall.covarianceVV.value, null); assert.equal(wall.covarianceVV.sign, 1); assert.equal(wall.position.sign, 1);
  const q = at({ ...p, meanAsymmetry: Number.MIN_VALUE }, .5); assert.equal(q.meanShear.sign, 1); assert.equal(q.inferredEddyViscosityStatus, 'defined'); assert.equal(q.inferredEddyViscosity!.value, null); assert.ok(q.inferredEddyViscosity!.logMagnitude! > 700);
});
test('quarter-cycle coordinates have exact zeros while tiny independent phase offsets remain nonzero', () => {
  const q = { ...p, firstStreamfunctionAmplitude: 0 };
  assert.equal(instant(q, .35, 0, .25).uPrime.value, 0);
  const shifted = instant(q, .35, Number.MIN_VALUE, .25);
  assert.notEqual(shifted.uPrime.sign, 0); assert.notEqual(shifted.uPrime.logMagnitude, null);
  near(instant(p, .35, 0, 1).u.value!, instant(p, .35, 0, 0).u.value!);
});
test('geometry scales extensive powers and energies; inferred viscosity is kinematic with units m²/s', () => {
  const a = budgets(p), b = budgets({ ...p, spanWidth: 4 });
  for (const key of ['meanKineticEnergy', 'fluctuationKineticEnergy', 'meanForcingWork', 'meanToFluctuationTransfer'] as const) near(b[key].value!, 4 * a[key].value!);
  const r1 = at(p, .35), r2 = at({ ...p, density: 4 }, .35);
  assert.deepEqual(r1.inferredEddyViscosity, r2.inferredEddyViscosity); near(r2.reynoldsShearStress.value!, 4 * r1.reynoldsShearStress.value!);
});
test('Reynolds-number conditions and all input bounds reject missing, nonfinite and unsupported data', () => {
  assert.equal(at({ ...p, dynamicViscosity: 0 }, .5).reynoldsNumberStatus, 'inviscid-nonzero-bulk');
  assert.equal(at({ ...p, dynamicViscosity: 0, bulkVelocity: 0 }, .5).reynoldsNumberStatus, 'indeterminate');
  assert.equal(at({ ...p, bulkVelocity: 0 }, .5).reynoldsNumber!.value, 0);
  const small = at({ ...p, dynamicViscosity: Number.MIN_VALUE }, .5); assert.equal(small.reynoldsNumber!.value, null); assert.ok(Number.isFinite(small.reynoldsNumber!.logMagnitude));
  for (const key of Object.keys(p) as (keyof typeof p)[]) for (const bad of [NaN, Infinity, -Infinity, undefined]) {
    assert.throws(() => at({ ...p, [key]: bad } as ReynoldsEnsembleParameters, .5), RangeError);
    assert.throws(() => budgets({ ...p, [key]: bad } as ReynoldsEnsembleParameters), RangeError);
    assert.throws(() => instant({ ...p, [key]: bad } as ReynoldsEnsembleParameters, .5), RangeError);
  }
  for (const bad of [-.1, 1.1, NaN, Infinity]) { assert.throws(() => at(p, bad), RangeError); assert.throws(() => instant(p, .5, bad), RangeError); assert.throws(() => instant(p, .5, 0, bad), RangeError); }
  assert.throws(() => at(p, undefined as unknown as number), RangeError);
});
test('supported extremes expose only finite numbers or explicitly unavailable observations', () => {
  function finite(v: unknown): void { if (typeof v === 'number') assert.ok(Number.isFinite(v)); else if (v && typeof v === 'object') Object.values(v).forEach(finite); }
  for (const xi of [0, Number.MIN_VALUE, 1e-200, .5, 1]) for (const amp of [0, Number.MIN_VALUE, 1e-200, -2, 2]) {
    const q = { ...p, firstStreamfunctionAmplitude: amp, secondStreamfunctionAmplitude: amp, meanAsymmetry: Number.MIN_VALUE, bulkVelocity: Number.MIN_VALUE, dynamicViscosity: Number.MIN_VALUE, candidateEddyViscosity: Number.MIN_VALUE };
    finite(at(q, xi)); finite(instant(q, xi, Number.MIN_VALUE, .25)); finite(budgets(q));
  }
});
test('maximal mean asymmetry retains the quadratic near-wall velocity and linear shear instead of rounding them to exact zero', () => {
  for (const xi of [1e-20, 1e-200, Number.MIN_VALUE]) {
    const q = at({ ...p, meanAsymmetry: 1 }, xi);
    // U=12 Ub xi²(1-xi), Uy=12 Ub xi(2-3xi)/H for c=1.
    near(q.meanU.logMagnitude!, Math.log(12 * p.bulkVelocity) + 2 * Math.log(xi) + Math.log1p(-xi));
    near(q.meanShear.logMagnitude!, Math.log(12 * p.bulkVelocity / p.height) + Math.log(xi) + Math.log(2 - 3 * xi));
    assert.equal(q.meanU.sign, 1); assert.equal(q.meanShear.sign, 1);
  }
});
test('equal streamfunction amplitudes at an eighth-cycle retain the next near-wall term after their leading derivatives cancel', () => {
  const q = { ...p, firstStreamfunctionAmplitude: 1, secondStreamfunctionAmplitude: 1 };
  for (const xi of [1e-20, 1e-200, Number.MIN_VALUE]) {
    const field = instant(q, xi, 0, .125);
    // u'=2 xi²(1-xi)(3-5xi)/(sqrt(2) H), exactly for this phase.
    assert.equal(field.uPrime.sign, 1);
    near(field.uPrime.logMagnitude!, Math.log(Math.SQRT2 / p.height) + 2 * Math.log(xi) + Math.log1p(-xi) + Math.log(3 - 5 * xi));
  }
  // At the centre the first basis derivative is exactly zero; a tiny second amplitude survives.
  const center = instant({ ...q, secondStreamfunctionAmplitude: Number.MIN_VALUE }, .5, 0, .125);
  assert.equal(center.uPrime.sign, 1); near(center.uPrime.logMagnitude!, Math.log(Number.MIN_VALUE) + Math.log(Math.SQRT1_2 / (8 * p.height)));
});
