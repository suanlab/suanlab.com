import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateWeakFieldClock } from '../src/lib/learning-weak-field-clock';

test('the exact circular rate agrees with the static and local-motion factors', () => {
  for (const r of [6, 8, 12, 48, 100]) {
    const rate = calculateWeakFieldClock(r);
    assert(Math.abs(rate.staticRate * Math.sqrt(1 - rate.localSpeed ** 2) - rate.exact) < 1e-14);
    assert(Math.abs(rate.angularFrequency ** 2 * r ** 3 - 1) < 1e-14);
    assert(Math.abs(rate.error - (rate.linear - rate.exact)) < 1e-14);
  }
  assert.throws(() => calculateWeakFieldClock(5), RangeError);
});

test('the positive first-order error tends to 9u²/8 as the field weakens', () => {
  const strong = calculateWeakFieldClock(6), moderate = calculateWeakFieldClock(12);
  const weak = calculateWeakFieldClock(48), asymptotic = calculateWeakFieldClock(1e6);
  assert(strong.error > moderate.error && moderate.error > weak.error);
  assert(moderate.error < .01 && moderate.error > .001);
  assert(Math.abs(asymptotic.error * 1e12 - 9 / 8) < 1e-5);
});
