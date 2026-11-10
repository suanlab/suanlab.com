import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/thermal-contact-independent.json';
import { thermalContactState as state, thermalContactWindow as window, type ThermalContactParameters, type ThermalObservation } from '../src/lib/learning-thermal-contact';

const base: ThermalContactParameters = { capacity1: 2, capacity2: 3, initialTemperature1: 330, initialTemperature2: 270, conductance: .4 };
function near(actual: number, expected: number, tolerance = 4e-13) {
  assert.ok(Number.isFinite(actual));
  assert.ok(Math.abs(actual - expected) <= Math.max(Number.MIN_VALUE * 2, tolerance * Math.abs(expected)), `${actual} != ${expected}`);
}
interface Reference { value: string; sign: number; logMagnitude: string | null }
function observation(actual: ThermalObservation, expected: Reference) {
  assert.equal(actual.sign, expected.sign);
  const target = Number(expected.value);
  if (expected.sign === 0) { assert.equal(actual.value, 0); assert.equal(actual.logMagnitude, null); }
  else {
    assert.notEqual(actual.value, 0, 'nonzero values cannot become physical zero');
    if (!Number.isFinite(target) || target === 0) assert.equal(actual.value, null);
    else { assert.notEqual(actual.value, null); near(actual.value!, target); }
    assert.notEqual(actual.logMagnitude, null);
    assert.ok(Math.abs(actual.logMagnitude! - Number(expected.logMagnitude)) < 3e-11);
  }
}
const windowKeys = new Set(['heat', 'energyChange1', 'energyChange2', 'temperatureChange1', 'temperatureChange2', 'entropyChange1', 'entropyChange2', 'entropyChange']);
for (const [family, cases] of Object.entries({ ordinary: refs.ordinary, limits: refs.limits })) {
  for (const [i, row] of cases.entries()) {
    test(`${family} independent matrix/entropy reference ${i + 1}`, () => {
      const s = state(row.parameters, row.end);
      const w = window(row.parameters, row.start, row.end);
      for (const [key, expected] of Object.entries(row.expected)) {
        const result = (windowKeys.has(key) ? w : s) as unknown as Record<string, number | ThermalObservation>;
        if (typeof result[key] === 'number') near(result[key] as number, Number(expected.value));
        else observation(result[key] as ThermalObservation, expected);
      }
      assert.equal(s.connected, row.parameters.conductance > 0);
      if (!s.connected) assert.equal(s.relaxationTime, null);
    });
  }
}
test('reference scope includes24 ordinary,12 limits and three actual-time quadratures', () => {
  assert.equal(refs.ordinary.length, 24); assert.equal(refs.limits.length, 12);
  assert.equal(refs.ordinary.filter(r => 'physicalTimeIntegrals' in r).length, 3);
});
test('worked example has conserved energy, positive production and capacity-weighted equilibrium', () => {
  const s = state(base, 3);
  near(s.equilibriumTemperature, 294); near(s.temperature1, 307.2436598821719);
  near(s.temperature2, 285.1708934118854); near(s.heat.value!, 45.51268023565615);
  near(s.entropyChange.value!, .02109692017293753); near(s.relaxationTime!.value!, 3);
  assert.equal(s.energyChange1.value! + s.energyChange2.value!, 0);
  assert.ok(s.entropyProductionRate.value! > 0);
});
test('exchanging body labels reverses transfer while preserving entropy generation', () => {
  const flipped = { ...base, capacity1: base.capacity2, capacity2: base.capacity1, initialTemperature1: base.initialTemperature2, initialTemperature2: base.initialTemperature1 };
  for (const time of [0, .01, 3, 8]) {
    const a = state(base, time), b = state(flipped, time);
    near(a.temperature1, b.temperature2); near(a.temperature2, b.temperature1);
    near(a.heat.value!, -b.heat.value!); near(a.entropyChange.value!, b.entropyChange.value!);
  }
});
test('equal initial temperatures, disconnected bodies and zero duration are distinct exact-zero cases', () => {
  for (const [p, t] of [[base, 0], [{ ...base, conductance: 0 }, 8], [{ ...base, initialTemperature2: 330 }, 8]] as const) {
    const s = state(p, t);
    for (const key of ['heat', 'entropyChange', 'energyChange1', 'energyChange2'] as const) {
      assert.equal(s[key].value, 0); assert.equal(s[key].sign, 0); assert.equal(s[key].logMagnitude, null);
    }
    assert.equal(s.temperature1, p.initialTemperature1); assert.equal(s.temperature2, p.initialTemperature2);
  }
  assert.equal(state({ ...base, conductance: 0 }, 8).connected, false);
  assert.equal(state({ ...base, initialTemperature2: 330 }, 8).connected, true);
});
test('rescaling capacities and conductance preserves temperatures and rescales energy and entropy', () => {
  const p = { ...base, capacity1: 1, capacity2: 1.5, conductance: .2 };
  const a = state(p, 3), b = state(base, 3);
  near(a.temperature1, b.temperature1); near(a.temperature2, b.temperature2);
  near(a.heat.value! * 2, b.heat.value!); near(a.entropyChange.value! * 2, b.entropyChange.value!);
  near(a.relaxationTime!.value!, b.relaxationTime!.value!);
});
test('stronger conductance speeds relaxation without changing the final energy or equilibrium temperature', () => {
  const a = state(base, 2), b = state({ ...base, conductance: .8 }, 1);
  near(a.temperature1, b.temperature1); near(a.temperature2, b.temperature2); near(a.heat.value!, b.heat.value!);
  near(a.entropyChange.value!, b.entropyChange.value!);
});
test('direct windows add physically and do not subtract nearly equal accumulated entropy values', () => {
  const a = window(base, 0, 2), b = window(base, 2, 8), c = window(base, 0, 8);
  near(a.heat.value! + b.heat.value!, c.heat.value!);
  near(a.entropyChange.value! + b.entropyChange.value!, c.entropyChange.value!);
  const narrow = window(base, 3, 3 + Number.EPSILON * 2);
  assert.ok(narrow.heat.value! > 0); assert.ok(narrow.entropyChange.value! > 0);
});
test('near-equal temperatures retain a positive entropy increase even when individual terms cancel', () => {
  const p = { ...base, initialTemperature1: 300 + 2 ** -44, initialTemperature2: 300 };
  const s = state(p, 8);
  assert.ok(s.entropyChange.value! > 0); assert.ok(s.entropyProductionRate.value! > 0);
  assert.equal(s.temperature1, s.temperature2);
  assert.ok(s.temperatureDifference.value! > 0);
});
test('late equal display temperatures retain the nonzero model difference and heat rate', () => {
  const s = state({ ...base, capacity1: .5, capacity2: .5, conductance: 2, initialTemperature1: 350, initialTemperature2: 250 }, 8);
  assert.equal(s.temperature1, s.temperature2);
  assert.ok(s.temperatureDifference.value! > 0); assert.ok(s.heatRate.value! > 0);
});
test('an unrepresentable relaxation rate does not erase a representable accumulated transfer', () => {
  const s = state({ ...base, capacity1: 4, capacity2: 4, conductance: Number.MIN_VALUE }, 8);
  assert.equal(s.relaxationRate.value, null); assert.equal(s.relaxationRate.sign, 1);
  assert.equal(s.relaxationTime!.value, null); assert.equal(s.relaxationTime!.sign, 1);
  assert.ok(s.heat.value! > 0);
  assert.equal(s.entropyChange.value, null); assert.equal(s.entropyChange.sign, 1);
  assert.ok(s.entropyChange.logMagnitude! < Math.log(Number.MIN_VALUE));
  assert.equal(s.temperature1, base.initialTemperature1);
});
test('underflowed physical transfer retains sign and log magnitude rather than becoming exact zero', () => {
  const s = state({ ...base, conductance: Number.MIN_VALUE }, Number.MIN_VALUE);
  assert.equal(s.heat.value, null); assert.equal(s.heat.sign, 1); assert.ok(s.heat.logMagnitude! < -1400);
  assert.equal(s.entropyChange.value, null); assert.equal(s.entropyChange.sign, 1);
});
test('temperature difference decays monotonically while accumulated entropy increases', () => {
  for (const conductance of [0, Number.MIN_VALUE, 1e-100, .01, .4, 2]) {
    const p = { ...base, conductance };
    let priorDifference = 60, priorEntropy = 0;
    for (let i = 0; i <= 64; i++) {
      const s = state(p, i / 8);
      assert.ok(s.temperature1 >= 250 && s.temperature1 <= 350);
      assert.ok(s.temperature2 >= 250 && s.temperature2 <= 350);
      assert.ok(s.temperatureDifference.value! <= priorDifference + 1e-12);
      if (s.entropyChange.value !== null) {
        assert.ok(s.entropyChange.value >= priorEntropy - 1e-18); priorEntropy = s.entropyChange.value;
      }
      priorDifference = s.temperatureDifference.value!;
    }
  }
});
test('all exposed values serialize without BigInt, NaN or Infinity', () => {
  for (const k of [0, Number.MIN_VALUE, .4, 2]) for (const t of [0, Number.MIN_VALUE, 3, 8]) {
    const s = state({ ...base, conductance: k }, t);
    const walk = (v: unknown): void => {
      if (typeof v === 'number') assert.ok(Number.isFinite(v));
      else if (v && typeof v === 'object') Object.values(v).forEach(walk);
    };
    walk(s); assert.doesNotThrow(() => JSON.stringify(s));
  }
});
test('invalid physical domains and reversed windows are rejected', () => {
  for (const key of Object.keys(base) as (keyof ThermalContactParameters)[]) {
    for (const value of [NaN, Infinity, -Infinity, -1]) assert.throws(() => state({ ...base, [key]: value }, 3), RangeError);
  }
  for (const t of [-1, 9, NaN, Infinity]) assert.throws(() => state(base, t), RangeError);
  assert.throws(() => state({ ...base, capacity1: 0 }, 3), RangeError);
  assert.throws(() => state({ ...base, initialTemperature1: 0 }, 3), RangeError);
  assert.throws(() => state({ ...base, conductance: 3 }, 3), RangeError);
  assert.throws(() => window(base, 4, 3), RangeError);
});
