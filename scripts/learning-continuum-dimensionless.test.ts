import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { pecletAt, damkohlerAt, type PecletParameters, type DamkohlerParameters, type TransportObservation } from '../src/lib/learning-continuum-dimensionless';

interface Reference { value: string; sign: number; logMagnitude: string | null }
interface Case<P> { parameters: P; xi: number; expected: Record<string, Reference>; physicalFluxIntegral?: Reference; physicalMassIntegral?: Reference; physicalConsumptionIntegral?: Reference }
const refs = JSON.parse(readFileSync('scripts/fixtures/continuum-dimensionless-independent.json', 'utf8')) as {
  pecletCases: Case<PecletParameters>[]; pecletLimits: Case<PecletParameters>[];
  damkohlerCases: Case<DamkohlerParameters>[]; damkohlerLimits: Case<DamkohlerParameters>[];
};
const p: PecletParameters = { diffusivity: .1, length: 2, area: 1.5, velocity: .5, leftConcentration: 1, rightConcentration: .2 };
const d: DamkohlerParameters = { diffusivity: .25, length: 1, area: 1.5, reactionRate: 1, surfaceConcentration: 2 };
function near(a: number | null, b: number, tol = 2e-10) {
  assert.notEqual(a, null); assert.ok(Math.abs(a! - b) <= tol * Math.max(Math.abs(b), 1e-290), `${a} != ${b}`);
}
function observation(actual: TransportObservation, reference: Reference, label: string) {
  assert.equal(actual.sign, reference.sign, label + ' sign');
  if (reference.sign === 0) { assert.equal(actual.value, 0, label); assert.equal(actual.logMagnitude, null); return; }
  assert.ok(actual.logMagnitude !== null, label);
  assert.ok(Math.abs(actual.logMagnitude! - Number(reference.logMagnitude)) < 2e-8, label + ' log');
  const n = Number(reference.value);
  if (n === 0 || !Number.isFinite(n)) assert.equal(actual.value, null, label + ' unavailable');
  else if (Math.abs(n) < 1e-300) {
    assert.notEqual(actual.value, null, label); assert.ok(Math.abs(actual.value! - n) <= 8 * Number.MIN_VALUE + Math.abs(n) * 2e-9, label);
  } else near(actual.value, n, 2e-9);
}
function compare<P>(cases: Case<P>[], fn: (p: P, xi: number) => object) {
  for (const [i, row] of cases.entries()) {
    const actual = fn(row.parameters, row.xi) as Record<string, TransportObservation>;
    for (const [key, expected] of Object.entries(row.expected)) observation(actual[key], expected, `${i}:${key}`);
  }
}
test('20 independently differentiated advection-diffusion profiles verify fields, derivatives, flux and clocks', () => {
  assert.equal(refs.pecletCases.length, 20); compare(refs.pecletCases, pecletAt);
  for (const row of refs.pecletCases) near(pecletAt(row.parameters, row.xi).totalFlux.value! * row.parameters.length, Number(row.physicalFluxIntegral!.value));
});
test('20 independently differentiated slab profiles and physical-coordinate integrals verify storage and consumption', () => {
  assert.equal(refs.damkohlerCases.length, 20); compare(refs.damkohlerCases, damkohlerAt);
  for (const row of refs.damkohlerCases) {
    const actual = damkohlerAt(row.parameters, row.xi);
    observation(actual.storedMass, row.physicalMassIntegral!, 'physical storage');
    observation(actual.totalConsumption, row.physicalConsumptionIntegral!, 'physical consumption');
  }
});
test('12 high-precision Peclet limits distinguish tiny transport, exact zeros, constant fields and underflow', () => {
  assert.equal(refs.pecletLimits.length, 12); compare(refs.pecletLimits, pecletAt);
});
test('12 high-precision Damkohler limits preserve tiny depletion, consumption and large-phi sealed-end tails', () => {
  assert.equal(refs.damkohlerLimits.length, 12); compare(refs.damkohlerLimits, damkohlerAt);
});
test('Peclet worked example has Pe10, diffusion40s, advection4s and spatially constant flux', () => {
  for (const xi of [0, .2, .5, .9, 1]) {
    const q = pecletAt(p, xi); near(q.peclet.value, 10); near(q.diffusionTime.value, 40); near(q.advectionTime!.value, 4);
    near(q.totalFlux.value, .5000181607964039); near(q.roundedTotalFlux, q.totalFlux.value!);
  }
  near(pecletAt(p, 0).concentration.value, 1); near(pecletAt(p, 1).concentration.value, .2);
});
test('Damkohler worked example uses half-thickness, a sealed nonzero concentration and actual area', () => {
  const q = damkohlerAt(d, 1); near(q.damkohler.value, 4); near(q.thiele.value, 2);
  near(q.concentration.value, 2 / Math.cosh(2)); near(q.analyticEffectiveness.value, Math.tanh(2) / 2);
  near(q.inletFlux.value, Math.tanh(2)); near(q.totalConsumption.value, 1.5 * Math.tanh(2));
  near(q.storedMass.value, 1.5 * Math.tanh(2)); assert.equal(q.diffusiveFlux.value, 0);
  near(damkohlerAt(d, 0).concentration.value, 2);
});
test('zero velocity retains diffusion and gives no finite advection clock', () => {
  const q = pecletAt({ ...p, velocity: 0 }, .5);
  assert.equal(q.peclet.value, 0); near(q.concentration.value, .6); near(q.totalFlux.value, .04);
  assert.equal(q.advectionTime, null); assert.equal(q.boundaryLayerLength, null); assert.equal(q.advectionTimeStatus, 'absent-zero-velocity');
});
test('zero contrast and zero surface concentration separate undefined measured normalization from declared basis', () => {
  const q = pecletAt({ ...p, leftConcentration: 2, rightConcentration: 2 }, .5);
  assert.equal(q.concentration.value, 2); assert.equal(q.gradient.value, 0); near(q.totalFlux.value, 1);
  assert.equal(q.observedNormalizedShape, null); assert.ok(q.basisShape.value! > 0);
  const r = damkohlerAt({ ...d, surfaceConcentration: 0 }, .5);
  for (const key of ['concentration', 'gradient', 'curvature', 'diffusiveFlux', 'totalConsumption', 'storedMass'] as const) assert.equal(r[key].value, 0);
  assert.equal(r.observedNormalizedShape, null); assert.equal(r.observedEffectiveness, null); assert.ok(r.basisShape.value! > 0);
});
test('zero reaction gives uniform storage with exact zero uptake and absent reaction scales', () => {
  const q = damkohlerAt({ ...d, reactionRate: 0 }, .5);
  near(q.concentration.value, 2); near(q.storedMass.value, 3); assert.equal(q.totalConsumption.value, 0);
  assert.equal(q.reactionTime, null); assert.equal(q.penetrationLength, null); assert.equal(q.observedEffectiveness, null);
  assert.equal(q.analyticEffectiveness.value, 1); assert.equal(q.stableDepletion.value, 0);
});
test('tiny nonzero velocity and reaction retain corrections hidden by rounded fields', () => {
  const q = pecletAt({ ...p, velocity: 1e-20 }, .5);
  assert.ok(q.stableShapeCorrection.value! < 0); near(q.stableShapeCorrection.value, -2.5e-20);
  const r = damkohlerAt({ ...d, reactionRate: 1e-20 }, .5);
  assert.equal(r.concentration.value, 2); near(r.stableDepletion.value, 1.5e-20); assert.ok(r.stableEffectivenessDeficit.value! > 0);
});
test('tiny products that underflow remain nonzero signed logarithms', () => {
  const q = pecletAt({ ...p, velocity: Number.MIN_VALUE, diffusivity: 4, length: .25 }, Number.MIN_VALUE);
  assert.equal(q.peclet.value, null); assert.equal(q.peclet.sign, 1);
  assert.equal(q.stableShapeCorrection.value, null); assert.equal(q.stableShapeCorrection.sign, -1);
  const r = damkohlerAt({ ...d, reactionRate: Number.MIN_VALUE, diffusivity: 4, length: .25 }, Number.MIN_VALUE);
  assert.equal(r.damkohler.value, null); assert.equal(r.damkohler.sign, 1);
  assert.equal(r.stableDepletion.value, null); assert.equal(r.stableDepletion.sign, 1);
  assert.notEqual(r.thiele.value, null); assert.notEqual(r.penetrationLength, null);
});
test('large signed Pe locates the opposite boundary layers and preserves counter-flow flux tails', () => {
  const extreme = { ...p, velocity: 2, diffusivity: 1e-4, length: 4, leftConcentration: 0, rightConcentration: 1 };
  const positive = pecletAt(extreme, .5), negative = pecletAt({ ...extreme, velocity: -2, leftConcentration: 1, rightConcentration: 0 }, .5);
  near(positive.peclet.value, 80000); assert.equal(positive.concentration.value, null); assert.equal(positive.concentration.sign, 1);
  assert.equal(positive.totalFlux.value, null); assert.equal(positive.totalFlux.sign, -1);
  assert.equal(negative.totalFlux.value, null); assert.equal(negative.totalFlux.sign, 1);
  near(positive.totalFlux.logMagnitude, Math.log(2) - 80000);
  near(positive.totalFlux.logMagnitude, negative.totalFlux.logMagnitude!);
  assert.equal(positive.roundedTotalFlux, null); // Unavailable operands must not produce a fabricated zero residual.
});
test('large-phi sealed tail retains the reflecting-wall factor two', () => {
  const q = damkohlerAt({ ...d, diffusivity: 1e-4, length: 4, reactionRate: 128 }, 1);
  assert.equal(q.concentration.value, null); assert.equal(q.concentration.sign, 1); assert.equal(q.diffusiveFlux.value, 0);
  near(q.concentration.logMagnitude, Math.log(4) - Math.sqrt(128 * 16 / 1e-4));
});
test('Peclet similarity preserves shape while physical transport and diffusion clocks scale differently', () => {
  const a = 1.5, b = .8, first = pecletAt(p, .37);
  const second = pecletAt({ ...p, length: a*p.length, diffusivity: b*p.diffusivity, velocity: b/a*p.velocity, area: a*a*p.area }, .37);
  near(second.peclet.value, first.peclet.value!); near(second.basisShape.value, first.basisShape.value!);
  near(second.diffusionTime.value, a*a/b*first.diffusionTime.value!); near(second.totalFlux.value, b/a*first.totalFlux.value!);
  near(second.totalTransport.value, a*b*first.totalTransport.value!);
});
test('Damkohler similarity preserves shape but scales mass and actual consumption with geometry', () => {
  const a = 1.5, b = .8, first = damkohlerAt(d, .37);
  const second = damkohlerAt({ ...d, length: a*d.length, diffusivity: b*d.diffusivity, reactionRate: b/(a*a)*d.reactionRate, area: a*a*d.area }, .37);
  near(second.damkohler.value, first.damkohler.value!); near(second.basisShape.value, first.basisShape.value!);
  near(second.storedMass.value, a**3*first.storedMass.value!); near(second.totalConsumption.value, a*b*first.totalConsumption.value!);
});
test('reversing velocity and boundaries reflects the profile and reverses actual flux', () => {
  const a = pecletAt(p, .2), b = pecletAt({ ...p, velocity: -p.velocity, leftConcentration: p.rightConcentration, rightConcentration: p.leftConcentration }, .8);
  near(a.concentration.value, b.concentration.value!); near(a.totalFlux.value, -b.totalFlux.value!); near(a.gradient.value, -b.gradient.value!);
});
test('actual finite differences verify both ODEs and physical flux divergence at ordinary parameters', () => {
  const xi = .37, h = 1e-4;
  for (const [parameters, fn] of [[p, pecletAt], [d, damkohlerAt]] as const) {
    const q = fn(parameters as never, xi), lo = fn(parameters as never, xi-h), hi = fn(parameters as never, xi+h);
    near((hi.concentration.value!-lo.concentration.value!)/(2*h*parameters.length), q.gradient.value!, 2e-6);
    near((hi.gradient.value!-lo.gradient.value!)/(2*h*parameters.length), q.curvature.value!, 2e-6);
  }
  const q = pecletAt(p, xi); near(p.velocity*q.gradient.value!, p.diffusivity*q.curvature.value!);
  const r = damkohlerAt(d, xi); near(d.diffusivity*r.curvature.value!, d.reactionRate*r.concentration.value!);
});
test('invalid physical domains and nonfinite inputs are rejected instead of silently clamped', () => {
  for (const bad of [0, -1, Infinity, NaN]) {
    assert.throws(() => pecletAt({ ...p, diffusivity: bad }, .5), RangeError);
    assert.throws(() => damkohlerAt({ ...d, length: bad }, .5), RangeError);
  }
  assert.throws(() => pecletAt({ ...p, velocity: 3 }, .5), RangeError);
  assert.throws(() => damkohlerAt({ ...d, reactionRate: -1 }, .5), RangeError);
  assert.throws(() => pecletAt(p, -Number.MIN_VALUE), RangeError);
  assert.throws(() => damkohlerAt(d, 1.1), RangeError);
});
