import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateModelDiscrepancy } from '../src/lib/learning-model-discrepancy';

test('one exact calibration match can conceal a biased rate and later discrepancy', () => {
  const atCalibration = calculateModelDiscrepancy(.25, 1, 1);
  assert(Math.abs(atCalibration.discrepancy) < 1e-12);
  assert(atCalibration.fittedRate < 1);
  const late = calculateModelDiscrepancy(.25, 1, 3);
  assert(late.discrepancy > .14);
  assert(late.prediction < late.truth);
});

test('without an omitted equilibrium the fitted rate and predictions are exact', () => {
  for (const time of [0, 1, 3]) {
    const result = calculateModelDiscrepancy(0, 1.2, time);
    assert.equal(result.fittedRate, 1.2);
    assert.equal(result.discrepancy, 0);
  }
  assert.throws(() => calculateModelDiscrepancy(1, 1, 3), RangeError);
});
