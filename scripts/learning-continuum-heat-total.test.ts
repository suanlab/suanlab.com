import test from 'node:test';
import assert from 'node:assert/strict';
import { heatModeAt, heatModeWindow } from '../src/lib/learning-continuum-diffusion';
import refs from './fixtures/continuum-heat-total-independent.json';

function check(actual: number | null, expected: {value: number | null; sign: number}) {
  if (expected.value === null || expected.sign === 0) {
    assert.equal(actual, expected.value);
    return;
  }
  assert.notEqual(actual, null);
  assert.equal(Math.sign(actual!), expected.sign);
  assert.ok(Math.abs(actual! - expected.value) <=
    2 * Number.MIN_VALUE + Math.abs(expected.value) * 2e-11,
  `${actual} vs ${expected.value}`);
}

for (const ref of refs.cases) {
  test(`1600-digit heat total and separate perturbation: ${ref.name}`, () => {
    const point = heatModeAt(ref.parameters, ref.x, ref.time);
    const window = heatModeWindow(ref.parameters, ref.center, ref.halfWidth, ref.time);
    check(point.temperature, ref.expected.temperature);
    check(point.perturbation, ref.expected.perturbation);
    check(window.sensibleEnergy, ref.expected.sensibleEnergy);
    check(window.modeHeat, ref.expected.modeHeat);
    assert.equal(point.roundedTemperature, point.temperature);
    assert.ok(point.temperature >= point.temperatureLowerBound);
  });
}

test('finite totals do not turn an unavailable positive perturbation into an exact-zero control', () => {
  const ref = refs.cases.find(r => r.name.startsWith('under-range-contrast-') && r.parameters.meanTemperature === 300)!;
  const p = heatModeAt(ref.parameters, 0, 4);
  const w = heatModeWindow(ref.parameters, 0, 1, 4);
  assert.equal(p.temperature, 300);
  assert.equal(p.perturbation, null);
  assert.equal(w.modeHeat, null);
  assert.ok(w.sensibleEnergy! > 13);
  assert.equal(heatModeAt({...ref.parameters, contrast: 0}, 0, 4).perturbation, 0);
  assert.equal(heatModeWindow({...ref.parameters, contrast: 0}, 0, 1, 4).modeHeat, 0);
});
