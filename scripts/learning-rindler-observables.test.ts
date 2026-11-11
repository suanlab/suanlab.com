import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateRindlerReadings, readRindlerCoordinates } from '../src/lib/learning-rindler-observables';
import { intervalSquared } from '../src/lib/learning-lorentz-boost';

const close = (actual: number, expected: number) => assert(Math.abs(actual - expected) < 1e-10, `${actual} vs ${expected}`);

test('chart conversion recovers the event and proper-time interval', () => {
  for (const rho of [.5, 1, 3]) {
    for (const eta of [0, .8, 2]) {
      const data = calculateRindlerReadings(rho, eta);
      const recovered = readRindlerCoordinates(data.time, data.position);
      close(recovered.rho, rho);
      close(recovered.eta, eta);
      close(data.inertialProperTime, Math.sqrt(intervalSquared({ ct: 0, x: rho }, { ct: data.time, x: data.position })));
      assert(data.inertialProperTime >= data.acceleratedProperTime);
    }
  }
  assert.throws(() => readRindlerCoordinates(1, 1), RangeError);
});

test('proper clock and accelerometer readings differ while local light speed stays invariant', () => {
  const near = calculateRindlerReadings(1, 1), far = calculateRindlerReadings(3, 1);
  assert.equal(far.acceleratedProperTime, 3 * near.acceleratedProperTime);
  close(far.acceleration, near.acceleration / 3);
  assert.equal(far.coordinateLightSpeed, 3 * near.coordinateLightSpeed);
  assert.equal(far.measuredLightSpeed, near.measuredLightSpeed);
});
