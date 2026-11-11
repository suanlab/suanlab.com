import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculatePairedEnergy } from '../src/lib/learning-energy-measurement';

const a = [60, 62, 59, 61, 63, 60];
const b = [55, 58, 56, 56, 59, 55];

test('paired differences cancel a shared block-dependent additive shift', () => {
  const baseline = calculatePairedEnergy(a, b, .01);
  const offset = [4, -3, 2, 1, 9, -7];
  const shift = (values: number[]) => values.map((value, i) => value + offset[i]);
  assert.deepEqual(calculatePairedEnergy(shift(a), shift(b), .01), baseline);
  assert.deepEqual(baseline.differences, [5, 4, 3, 5, 4, 5]);
  assert(Math.abs(baseline.mean - 26 / 6) < 1e-12);
  assert(Math.abs(baseline.standardError - 1 / 3) < 1e-12);
});

test('repeatability and a shared scale uncertainty respond to different inputs', () => {
  const perfectRepeat = calculatePairedEnergy([60, 61, 62], [55, 56, 57], .01);
  assert.equal(perfectRepeat.standardError, 0);
  assert.equal(perfectRepeat.calibrationUncertainty, .05);
  const twiceUncertain = calculatePairedEnergy([60, 61, 62], [55, 56, 57], .02);
  assert.equal(twiceUncertain.calibrationUncertainty, .1);
  assert.throws(() => calculatePairedEnergy([60], [55], .01), RangeError);
});
