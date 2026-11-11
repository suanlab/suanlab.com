import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateAvailabilityLeakage } from '../src/lib/learning-availability-leakage';

test('unavailable future information creates a gain without legitimate predictability', () => {
  assert.deepEqual(calculateAvailabilityLeakage(0), { availableMse: 1, futureMse: .36, targetMse: 0, availableCoefficient: 0 });
  assert.equal(calculateAvailabilityLeakage(1).availableMse, .36);
});

test('the analytic risk matches enumeration of the independent generating variables', () => {
  for (const rho of [0, .5, -.8, 1]) {
    let error = 0;
    for (const x of [-1, 1]) for (const u of [-1, 1]) for (const n of [-1, 1]) {
      const future = rho * x + Math.sqrt(1 - rho ** 2) * u;
      const target = .8 * future + .6 * n;
      error += (target - .8 * rho * x) ** 2 / 8;
    }
    assert(Math.abs(error - calculateAvailabilityLeakage(rho).availableMse) < 1e-12);
  }
});
