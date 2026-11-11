import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculatePhysicalValidation } from '../src/lib/learning-physical-validation';

test('basis calibration can conceal clipping that appears in a constructive held-out input', () => {
  for (const input of [[1, 0], [0, 1], [1, -1], [.5, .5]] as const) {
    const model = calculatePhysicalValidation(input, 1, .8);
    assert.equal(model.absoluteError, 0);
    assert.equal(model.calibratedError, 0);
  }
  const constructive = calculatePhysicalValidation([1, 1], 1, .8);
  assert.deepEqual(constructive.target, [1, 1]);
  assert.deepEqual(constructive.measured, [.8, .8]);
  assert(Math.abs(constructive.calibratedError - Math.hypot(.2, .2)) < 1e-12);
});

test('calibration absorbs linear gain error while a higher readout range removes clipping', () => {
  const linear = calculatePhysicalValidation([1, 1], .9, 1);
  assert(linear.absoluteError > .1);
  assert(Math.abs(linear.calibratedError) < 1e-12);
  assert.equal(calculatePhysicalValidation([1, 1], 1, 1).absoluteError, 0);
});
