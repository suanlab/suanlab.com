import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/continuum-conservation-independent.json';
import {
  transportedDensityState as density, movingMassWindowState as mass, movingMassWindowBalance as massBalance,
  affineGasState as gas, affineGasWindowState as windowState, affineGasWindowBalance as gasBalance,
  type DensityTransport, type TranslatingMassWindow, type AffineGas, type AffineControlWindow,
} from '../src/lib/learning-continuum-conservation';
const S: DensityTransport = {meanDensity: 1, contrast: .6, wavenumber: 1, velocity: 1, area: 1};
const W: TranslatingMassWindow = {center: .3, halfWidth: .6, speed: -.3};
const G: AffineGas = {initialDensity: 1, initialPressure: 1, expansionRate: .5, drift: 0, gamma: 1.4, area: 1};
const C: AffineControlWindow = {center: 0, halfWidth: .5, speed: .3, mode: 'material'};
const close = (a: number | null, b: number, tol = 2e-10) => assert.ok(a !== null && Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} != ${b}`);
const relative = (a: number | null, b: number, tol = 1e-12) => assert.ok(a !== null && Math.abs(a / b - 1) <= tol, `${a} != ${b}`);
const finite = (x: unknown): void => {if (typeof x === 'number') assert.ok(Number.isFinite(x)); else if (x && typeof x === 'object') Object.values(x).forEach(finite);};

test('72 independently integrated moving-window mass histories and rates match', () => {
  assert.equal(refs.transportCases.length, 72);
  for (const r of refs.transportCases) {
    const q = mass(r.system, r.window, r.time), b = massBalance(r.system, r.window, -.4, .7);
    close(q.mass, r.mass); close(q.massRate, r.massRate); close(b.massChange, r.change);
    close(b.integratedNetInflow, r.integral); close(b.residual, 0); close(b.quadratureOrderDifference, 0);
  }
});
test('a fixed camera can gain or lose mass without creating material', () => {
  const q = mass(S, {...W, speed: 0}, .2);
  assert.notEqual(q.massRate, 0); close(q.massRate, q.ordinaryNetInflow!);
  assert.equal(q.wrongFluxResidual, 0);
});
test('material windows have exact zero relative flux while the laboratory flux can be nonzero', () => {
  const q = mass(S, {...W, speed: S.velocity}, .2), b = massBalance(S, {...W, speed: S.velocity}, -.8, .9);
  assert.equal(q.materialWindow, true); assert.equal(q.leftOutward, 0); assert.equal(q.rightOutward, 0);
  assert.equal(q.massRate, 0); assert.notEqual(q.wrongLabFrameNetInflow, 0);
  assert.equal(b.massChange, 0); assert.equal(b.integratedNetInflow, 0);
});
test('moving the window through a stationary density pattern changes its observed mass', () => {
  const q = mass({...S, velocity: 0}, W, .2);
  assert.notEqual(q.massRate, 0); assert.equal(q.wrongLabFrameNetInflow, 0);
  assert.notEqual(q.wrongFluxResidual, 0);
});
test('zero contrast and zero wavenumber are spatially uniform families with known zero net inflow', () => {
  for (const system of [{...S, contrast: 0}, {...S, wavenumber: 0}]) {
    const q = mass(system, W, .2); assert.equal(q.massRate, 0);
    close(q.densityLeft, q.densityRight!); assert.equal(massBalance(system, W, -.4, .7).massChange, 0);
  }
});
test('a zero-volume window does not define observed average density or actual oriented faces', () => {
  const q = mass(S, {...W, halfWidth: 0}, .2);
  assert.equal(q.mass, 0); assert.equal(q.observedMeanDensity, null); assert.ok(q.analyticMeanDensity! > 0);
  assert.equal(q.leftOutward, null); assert.equal(q.rightOutward, null); assert.equal(q.geometryStatus, 'zero-length');
});
test('positive width below coordinate resolution is recorded without silently widening it', () => {
  const q = mass(S, {...W, halfWidth: 1e-20}, .2);
  assert.equal(q.left, q.right); assert.ok(q.mass! > 0); assert.equal(q.geometryStatus, 'width-below-coordinate-resolution');
});
test('the density family has positive mass, correct physical dimensions and pointwise continuity', () => {
  for (const x of [-4, 0, 4]) for (const t of [-1, 0, 1]) {
    const q = density(S, x, t); assert.ok(q.density! >= .4); close(q.massFlux, q.density! * S.velocity);
    close(q.densityT, -S.velocity * q.densityX!); assert.equal(q.continuityResidual, 0);
  }
  const q = mass(S, W, .2), doubled = mass({...S, area: 2}, W, .2);
  close(doubled.mass, 2 * q.mass!); close(doubled.massRate, 2 * q.massRate!);
  close(doubled.observedMeanDensity, q.observedMeanDensity!);
});
test('Galilean changes preserve density and relative moving-window observations', () => {
  const boost = .3, t = .2, q = mass(S, W, t), moved = mass({...S, velocity: S.velocity + boost}, {...W, speed: W.speed + boost}, t);
  close(q.mass, moved.mass!); close(q.massRate, moved.massRate!);
  close(density(S, .4, t).density, density({...S, velocity: S.velocity + boost}, .4 + boost * t, t).density!);
});
test('tiny transport changes survive subtraction of masses rounded to equality', () => {
  const b = massBalance({...S, wavenumber: refs.tinyTransportChange.wavenumber}, W, -.4, .7);
  assert.equal(b.roundedMassChange, 0); relative(b.massChange, refs.tinyTransportChange.change);
});
test('reversed time swaps signs and zero time gives exact zero balance', () => {
  const a = massBalance(S, W, -.4, .7), b = massBalance(S, W, .7, -.4), z = massBalance(S, W, .2, .2);
  close(a.massChange, -b.massChange!); close(a.integratedNetInflow, -b.integratedNetInflow!);
  assert.equal(z.massChange, 0); assert.equal(z.integratedNetInflow, 0); assert.equal(z.residual, 0);
});
test('small nonzero derivatives lost to arithmetic are unavailable, not reported as physical zero', () => {
  const q = density({...S, contrast: Number.MIN_VALUE, wavenumber: Number.MIN_VALUE}, 1, 0);
  assert.equal(q.densityX, null); assert.equal(q.densityT, null);
  const zero = density({...S, contrast: 0, wavenumber: Number.MIN_VALUE}, 1, 0);
  assert.equal(zero.densityX, 0); assert.equal(zero.densityT, 0);
});
test('144 independent gas-volume budgets match all three conserved quantities and rates', () => {
  assert.equal(refs.gasCases.length, 144);
  for (const r of refs.gasCases) {
    const w = r.window as AffineControlWindow, q = windowState(r.system, w, r.time), b = gasBalance(r.system, w, -.4, .7);
    for (let i = 0; i < 3; i++) {
      close(q.storage[i], r.storage[i]); close(q.storageRates[i], r.rate[i]);
      close(q.ordinaryNetInflow[i], q.storageRates[i]!); close(b.changes[i], r.change[i]);
      close(b.integratedNetInflows[i], r.integrated[i]); close(b.residuals[i], 0);
    }
    close(q.missingPressureWorkNetInflow, r.withoutPressureWork[2]);
  }
});
test('independent field differences verify local derivatives rather than only built-in residuals', () => {
  for (const r of refs.gasCases.filter(r => r.window.mode === 'fixed')) {
    const q = gas(r.system, .23, .2);
    for (let i = 0; i < 3; i++) {
      close(q.conserved[i], r.localConserved[i]); close(q.flux[i], r.localFlux[i]);
      close(q.timeDerivatives[i], r.localTimeDerivative[i]); close(q.fluxDerivatives[i], r.localFluxDerivative[i]);
    }
  }
});
test('48 particle IVPs match the affine material trajectories', () => {
  assert.equal(refs.particleCases.length, 48);
  for (const r of refs.particleCases) {
    const x = (1 + r.system.expansionRate * r.time) * r.initialPosition + r.system.drift * r.time;
    close(x, r.position); close(gas(r.system, x, r.time).velocity, r.system.drift + r.system.expansionRate * r.initialPosition);
  }
});
test('material expansion conserves mass and momentum while pressure work lowers energy', () => {
  const q = windowState(G, C, .5), b = gasBalance(G, C, 0, .5);
  assert.equal(q.storageRates[0], 0); assert.equal(q.storageRates[1], 0); assert.ok(q.storageRates[2]! < 0);
  close(q.storageRates[2], q.pressureWorkNet!); assert.equal(q.missingPressureWorkNetInflow, 0);
  assert.equal(b.changes[0], 0); assert.equal(b.changes[1], 0); assert.equal(b.kineticChange, 0);
  assert.ok(b.internalChange! < 0); close(b.changes[2], b.internalChange!);
});
test('compression raises material internal energy without kinetic-energy production', () => {
  const b = gasBalance({...G, expansionRate: -.5}, C, 0, .5);
  assert.ok(b.internalChange! > 0); assert.equal(b.kineticChange, 0); close(b.residuals[2], 0);
});
test('material pressure forces need not vanish on individual faces even when net momentum change is zero', () => {
  const q = windowState(G, C, .2);
  assert.ok(q.leftOutward[1]! < 0); assert.ok(q.rightOutward[1]! > 0);
  assert.equal(q.ordinaryNetInflow[1], 0); assert.equal(q.storageRates[1], 0);
});
test('the gas flux carries pressure work in addition to stored total energy', () => {
  const q = gas(G, 1, .2);
  close(q.flux[2], (q.energyDensity! + q.pressure!) * q.velocity!);
  close(q.missingPressureWorkResidual, -q.pressure! * q.velocityGradient);
  assert.notEqual(q.flux[2], q.energyDensity! * q.velocity!);
});
test('zero strain permits a nonzero static pressure without transport or energy change', () => {
  const s = {...G, expansionRate: 0, drift: 0}, q = gas(s, 1, .2), b = gasBalance(s, C, -.4, .7);
  assert.ok(q.pressure! > 0); assert.ok(q.internalEnergyDensity! > 0); assert.equal(q.flux[2], 0);
  assert.deepEqual(b.changes, [0, 0, 0]); assert.deepEqual(b.integratedNetInflows, [0, 0, 0]);
});
test('zero pressure is a pressureless positive-density limit, not a vacuum replacement', () => {
  const s = {...G, initialPressure: 0}, q = gas(s, 1, .2), b = gasBalance(s, C, -.4, .7);
  assert.ok(q.density > 0); assert.equal(q.internalEnergyDensity, 0); assert.ok(q.kineticEnergyDensity! > 0);
  assert.equal(b.internalChange, 0); assert.equal(b.changes[2], 0);
});
test('100-digit references retain tiny material energy changes under rounded endpoint equality', () => {
  for (const r of refs.tinyMaterialEnergyChanges) {
    const b = gasBalance({...G, expansionRate: r.expansionRate}, C, 0, r.time);
    relative(b.internalChange, r.materialInternalChange); close(b.residuals[2], 0);
    if (Math.abs(r.expansionRate) < 1e-15) assert.equal(b.roundedChanges[2], 0);
  }
});
test('zero length and zero duration retain their separate meanings in gas observations', () => {
  const q = windowState(G, {...C, halfWidth: 0}, .2), b = gasBalance(G, C, .2, .2);
  assert.deepEqual(q.storage, [0, 0, 0]); assert.deepEqual(q.leftOutward, [null, null, null]);
  assert.equal(q.geometryStatus, 'zero-length'); assert.deepEqual(b.changes, [0, 0, 0]);
  assert.deepEqual(b.integratedNetInflows, [0, 0, 0]);
});
test('gas relative budgets transform consistently with a constant observation-frame boost', () => {
  const t = .2, boost = .4, w = {...C, mode: 'translating' as const}, s = {...G, drift: .2};
  const q = windowState(s, w, t), b = windowState({...s, drift: s.drift + boost}, {...w, speed: w.speed + boost}, t);
  close(b.storage[0], q.storage[0]!); close(b.storage[1], q.storage[1]! + boost * q.storage[0]!);
  close(b.storage[2], q.storage[2]! + boost * q.storage[1]! + .5 * boost * boost * q.storage[0]!);
});
test('small pressure work underflow remains unavailable rather than being replaced by zero', () => {
  const q = gas({...G, initialPressure: Number.MIN_VALUE, expansionRate: Number.MIN_VALUE}, 1, 0);
  assert.equal(q.missingPressureWorkResidual, null);
});
test('malformed,missing,out-of-domain inputs and unknown control modes fail explicitly', () => {
  for (const system of [{...S, meanDensity: 0}, {...S, contrast: 1}, {...S, wavenumber: NaN}, {...S, area: undefined}]) assert.throws(() => mass(system as DensityTransport, W, 0), RangeError);
  assert.throws(() => mass(S, {...W, halfWidth: -1}, 0), RangeError); assert.throws(() => density(S, Infinity, 0), RangeError);
  assert.throws(() => density(S, 0, 2), RangeError); assert.throws(() => gas({...G, gamma: 1}, 0, 0), RangeError);
  assert.throws(() => gas({...G, expansionRate: -1}, 0, 1), RangeError);
  assert.throws(() => windowState(G, {...C, mode: 'unknown'} as unknown as AffineControlWindow, 0), RangeError);
});
test('available observations never leak nonfinite numbers or mutate caller systems', () => {
  const before = JSON.stringify([S, W, G, C]);
  for (const t of [-1, 0, 1]) {
    finite(massBalance(S, W, 0, t)); finite(gasBalance(G, C, 0, t));
    finite(gasBalance({...G, initialPressure: Number.MIN_VALUE}, {...C, halfWidth: Number.MIN_VALUE}, 0, t));
  }
  assert.equal(JSON.stringify([S, W, G, C]), before);
});
test('the manuscript mass example has2kg storage,1kg/s initial influx and sqrt2/pi finite change', () => {
  const s = {...S, contrast: .5, wavenumber: Math.PI / 2}, w = {center: 1, halfWidth: 1, speed: 0};
  const q = mass(s, w, 0), b = massBalance(s, w, 0, .5);
  close(q.mass, 2); close(q.massRate, 1); close(q.leftOutward, -1.5); close(q.rightOutward, .5);
  close(b.massChange, Math.sqrt(2) / Math.PI); close(b.integratedNetInflow, Math.sqrt(2) / Math.PI);
});
test('the manuscript gamma2 expansion transfers2/3J while conserving material kinetic energy', () => {
  const s = {...G, gamma: 2}, w = {...C, halfWidth: 1}, b = gasBalance(s, w, 0, 1);
  close(b.initial.storage[0], 2); close(b.initial.internalEnergy, 2); close(b.initial.kineticEnergy, 1 / 12);
  close(b.initial.storage[2], 25 / 12); close(b.selected.storage[2], 17 / 12);
  close(b.changes[2], -2 / 3); close(b.integratedNetInflows[2], -2 / 3);
  close(b.initial.pressureWorkNet, -1); close(b.selected.pressureWorkNet, -4 / 9);
});
