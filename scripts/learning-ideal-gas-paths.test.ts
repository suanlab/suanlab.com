import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/ideal-gas-path-independent.json';
import { idealGasPathState as state, idealGasPathComparison as compare, type IdealGasPathParameters, type IdealGasPath } from '../src/lib/learning-ideal-gas-paths';
import type { BinaryObservation } from '../src/lib/learning-binary-products';
const base: IdealGasPathParameters = { moles: 1, initialTemperature: 300, finalTemperature: 330, initialVolume: .01, finalVolume: .02 };
interface Reference { value: string; sign: number; logMagnitude: string | null }
function near(actual: number, expected: number, tolerance = 4e-13) {
  assert.ok(Number.isFinite(actual));
  assert.ok(Math.abs(actual - expected) <= Math.max(2 * Number.MIN_VALUE, tolerance * Math.abs(expected)), `${actual} != ${expected}`);
}
function observation(actual: BinaryObservation | null, expected: Reference) {
  assert.notEqual(actual, null, 'reference cases must resolve sign and accuracy');
  assert.equal(actual!.sign, expected.sign);
  const target = Number(expected.value);
  if (expected.sign === 0) { assert.equal(actual!.value, 0); assert.equal(actual!.logMagnitude, null); }
  else {
    assert.notEqual(actual!.value, 0);
    if (target === 0 || !Number.isFinite(target)) assert.equal(actual!.value, null);
    else { assert.notEqual(actual!.value, null); near(actual!.value!, target); }
    assert.ok(Math.abs(actual!.logMagnitude! - Number(expected.logMagnitude)) < 3e-11);
  }
}
for (const [family, cases] of Object.entries({ ordinary: refs.ordinary, limits: refs.limits })) {
  for (const [i, row] of cases.entries()) test(`${family} independent ideal-gas path reference ${i + 1}`, () => {
    const s = state(row.parameters, row.path as IdealGasPath, row.progress);
    for (const [key, expected] of Object.entries(row.expected)) {
      const actual = (s as unknown as Record<string, number | BinaryObservation | null>)[key];
      if (typeof actual === 'number') near(actual, Number(expected.value));
      else observation(actual, expected);
    }
    assert.equal(s.heatStatus, 'resolved'); assert.equal(s.entropyProduction, 0);
  });
}
test('reference coverage contains24 actual-coordinate integrations and21 high-precision limits', () => {
  assert.equal(refs.ordinary.length, 24); assert.equal(refs.limits.length, 21);
  assert.ok(refs.ordinary.every(x => x.physicalCoordinateIntegrals));
});
test('worked example has different heat and work with identical energy and entropy changes', () => {
  const { a, b, workDifference } = compare(base);
  near(a.work.value!, 1901.8382861425132); near(b.work.value!, 1728.9438964931938);
  near(a.heat!.value!, 2275.989103959409); near(b.heat!.value!, 2103.09471431009);
  near(a.energyChange.value!, 374.1508178168958); near(a.entropyChange.value!, 6.951825712312764);
  assert.deepEqual(a.energyChange, b.energyChange); assert.deepEqual(a.entropyChange, b.entropyChange);
  near(workDifference.value!, 172.8943896493194);
});
test('path reversal exchanges A/B and negates full work, heat and state changes', () => {
  const reversed = { ...base, initialTemperature: base.finalTemperature, finalTemperature: base.initialTemperature, initialVolume: base.finalVolume, finalVolume: base.initialVolume };
  const forward = compare(base), back = compare(reversed);
  for (const [a, b] of [[forward.a, back.b], [forward.b, back.a]]) {
    for (const key of ['work', 'energyChange', 'entropyChange', 'heat'] as const) near(a[key]!.value!, -b[key]!.value!);
  }
});
test('linear path interpolation has correct endpoints and the expected isochoric/isothermal stages', () => {
  for (const path of ['A', 'B'] as const) {
    const start = state(base, path, 0), middle = state(base, path, .5), end = state(base, path, 1);
    assert.equal(start.temperature, base.initialTemperature); assert.equal(start.volume, base.initialVolume);
    assert.equal(end.temperature, base.finalTemperature); assert.equal(end.volume, base.finalVolume);
    assert.equal(middle.temperature, path === 'A' ? base.finalTemperature : base.initialTemperature);
    assert.equal(middle.volume, path === 'A' ? base.initialVolume : base.finalVolume);
    assert.equal(middle.stage, path === 'A' ? 'isochoric' : 'isothermal');
    assert.notEqual(state(base, path, .75).stage, middle.stage);
  }
});
test('constant temperature and constant volume collapse the two paths to the same final budgets', () => {
  for (const p of [{ ...base, finalTemperature: 300 }, { ...base, finalVolume: .01 }]) {
    const { a, b, workDifference } = compare(p);
    assert.deepEqual(a.work, b.work); assert.deepEqual(a.heat, b.heat); assert.equal(workDifference.value, 0);
  }
  assert.equal(compare({ ...base, finalVolume: .01 }).a.work.value, 0);
  assert.equal(compare({ ...base, finalTemperature: 300 }).a.energyChange.value, 0);
});
test('first law holds over the complete path while pressure follows the same state equation', () => {
  for (const path of ['A', 'B'] as const) for (let i = 0; i <= 64; i++) {
    const s = state(base, path, i / 64);
    if (s.energyChange.sign !== 0) near(s.heat!.value! - s.work.value!, s.energyChange.value!);
    else near(s.heat!.value!, s.work.value!);
    near(s.pressure * s.volume, base.moles * refs.gasConstantBinary * s.temperature);
  }
});
test('tiny progress retains positive heat and work even when displayed states have not changed', () => {
  const a = state(base, 'A', Number.MIN_VALUE), b = state(base, 'B', Number.MIN_VALUE);
  assert.equal(a.temperature, 300); assert.ok(a.energyChange.value! > 0); assert.ok(a.heat!.value! > 0);
  assert.equal(b.volume, .01); assert.equal(b.volumeChange.value, null); assert.equal(b.volumeChange.sign, 1);
  assert.ok(b.work.value! > 0); assert.ok(b.heat!.value! > 0);
});
test('an exact rational adiabatic state relation gives zero endpoint entropy, not a logarithmic residue', () => {
  const p = { ...base, initialTemperature: 252, finalTemperature: 343, initialVolume: 343 / 32768, finalVolume: 216 / 32768 };
  for (const path of ['A', 'B'] as const) {
    const s = state(p, path, 1); assert.equal(s.entropyChange.value, 0); assert.equal(s.entropyChange.sign, 0);
    assert.notEqual(s.heat!.value, 0); // The two-stage path itself is not an adiabat.
  }
  assert.equal(refs.limits[8].expected.entropyChange.sign, 1);
  assert.equal(refs.limits[9].expected.entropyChange.sign, -1);
});
test('opposing heat terms are resolved with outward logarithm bounds around both cancellation points', () => {
  const signs = new Set<number>();
  for (const row of refs.limits.slice(14, 20)) {
    const s = state(row.parameters, row.path as IdealGasPath, 1);
    assert.ok(s.heatPrecisionBits! >= 128); assert.notEqual(s.heat, null);
    observation(s.heat, row.expected.heat); signs.add(s.heat!.sign);
    assert.ok(Math.abs(s.heat!.value!) < 1e-11);
  }
  assert.ok(signs.has(-1)); assert.ok(signs.has(1));
});
test('a reversible cycle has matching net heat and work without identifying either path as a single reservoir process', () => {
  const c = compare(base); near(c.a.heat!.value! - c.b.heat!.value!, c.cycleHeat.value!);
  near(c.a.work.value! - c.b.work.value!, c.workDifference.value!);
});
test('scaling amount scales extensive quantities while temperature and volume path coordinates stay fixed', () => {
  const a = state(base, 'A', .75), b = state({ ...base, moles: 2 }, 'A', .75);
  assert.equal(a.temperature, b.temperature); assert.equal(a.volume, b.volume); near(b.pressure, 2 * a.pressure);
  for (const key of ['work', 'heat', 'energyChange', 'entropyChange'] as const) near(b[key]!.value!, 2 * a[key]!.value!);
});
test('outputs serialize without BigInt, NaN or Infinity', () => {
  const walk = (v: unknown): void => { if (typeof v === 'number') assert.ok(Number.isFinite(v)); else if (v && typeof v === 'object') Object.values(v).forEach(walk); };
  for (const path of ['A', 'B'] as const) for (const s of [0, Number.MIN_VALUE, .25, .5, .75, 1]) {
    const result = state(base, path, s); walk(result); assert.doesNotThrow(() => JSON.stringify(result));
  }
});
test('invalid physical states, paths and progress are rejected', () => {
  for (const key of Object.keys(base) as (keyof IdealGasPathParameters)[]) for (const bad of [NaN, Infinity, -Infinity, 0, -1]) assert.throws(() => state({ ...base, [key]: bad }, 'A', 1), RangeError);
  for (const bad of [-1, 1.01, NaN, Infinity]) assert.throws(() => state(base, 'A', bad), RangeError);
  assert.throws(() => state(base, 'C' as IdealGasPath, 1), RangeError);
  assert.throws(() => state({ ...base, moles: 3 }, 'A', 1), RangeError);
});

test('positive heat is retained when rounded energy and work sum to a false zero', () => {
  const row = refs.limits[20], q = state(row.parameters, 'A', 1);
  assert.equal(q.energyChange.value! + q.work.value!, 0);
  assert.equal(q.heat!.sign, 1); near(q.heat!.value!, 3.468780578357212e-14);
  observation(q.heat, row.expected.heat);
});
