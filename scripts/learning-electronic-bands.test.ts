import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateElectronicBands } from '../src/lib/learning-electronic-bands';

const close = (a: number, b: number, tolerance = 1e-10) =>
  assert(Math.abs(a - b) < tolerance, `${a} vs ${b}`);

test('gap, folding and site weights agree at the zone boundary', () => {
  const hopping = calculateElectronicBands(1, .6, 0, Math.PI);
  const onsite = calculateElectronicBands(1, 1, .4, Math.PI);
  close(hopping.gap, onsite.gap);
  close(hopping.upper - hopping.lower, hopping.gap);
  assert.equal(hopping.upperWeightA, .5);
  assert.equal(onsite.upperWeightA, 1);
  const crossing = calculateElectronicBands(1, 1, 0, Math.PI);
  assert.equal(crossing.gap, 0);
  assert.equal(crossing.upperSlope, null);
  assert.equal(crossing.upperWeightA, null);
});

test('the closed-form eigenvalues and slopes match the determinant and finite differences', () => {
  for (const [t1, t2, delta] of [[1, .6, 0], [1, 1, .4], [0, 1, -.5]]) {
    for (const q of [-2, -.4, .4, 2]) {
      const result = calculateElectronicBands(t1, t2, delta, q);
      const swapped = calculateElectronicBands(t2, t1, delta, q);
      const determinant = -(delta ** 2 + t1 ** 2 + t2 ** 2 + 2 * t1 * t2 * Math.cos(q));
      close(result.lower * result.upper, determinant);
      close(result.upper, swapped.upper);
      const epsilon = 1e-5;
      const slope = (calculateElectronicBands(t1, t2, delta, q + epsilon).upper -
        calculateElectronicBands(t1, t2, delta, q - epsilon).upper) / (2 * epsilon);
      assert(result.upperSlope !== null);
      close(result.upperSlope, slope);
    }
  }
});
