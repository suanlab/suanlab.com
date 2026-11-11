import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateLandauerErasure } from '../src/lib/learning-landauer-erasure';

test('the unbiased perfect reset yields kT log 2 and a known input has zero entropy', () => {
  const fair = calculateLandauerErasure(.5, 0, 300);
  assert.equal(fair.initialEntropy, Math.LN2);
  assert.equal(fair.heatFloor, 1.380649e-23 * 300 * Math.LN2);
  assert.equal(calculateLandauerErasure(0, 0, 300).heatFloor, 0);
  assert.equal(calculateLandauerErasure(1, 0, 300).heatFloor, 0);
  assert.equal(calculateLandauerErasure(.5, 0, 600).heatFloor, 2 * fair.heatFloor);
});

test('allowing residual entropy reduces the bound and can make it nonpositive', () => {
  const perfect = calculateLandauerErasure(.5, 0, 300);
  const imperfect = calculateLandauerErasure(.5, .1, 300);
  assert(imperfect.heatFloor > 0 && imperfect.heatFloor < perfect.heatFloor);
  assert.equal(calculateLandauerErasure(.1, .1, 300).heatFloor, 0);
  assert(calculateLandauerErasure(0, .1, 300).heatFloor < 0);
  assert.throws(() => calculateLandauerErasure(.5, .6, 300), RangeError);
});
