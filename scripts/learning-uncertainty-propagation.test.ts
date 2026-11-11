import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateCorrelatedVariance } from '../src/lib/learning-uncertainty-propagation';

test('correlation shifts variance between the sum and difference', () => {
  assert.deepEqual(calculateCorrelatedVariance(1, 1, .75), { covariance: .75, sumVariance: 3.5, differenceVariance: .5 });
  const asymmetric = calculateCorrelatedVariance(2, 3, -.5);
  assert.deepEqual(asymmetric, { covariance: -3, sumVariance: 7, differenceVariance: 19 });
});

test('perfectly correlated equal scales have a zero difference variance', () => {
  assert.equal(calculateCorrelatedVariance(1, 1, 1).differenceVariance, 0);
  assert.equal(calculateCorrelatedVariance(1, 1, -1).sumVariance, 0);
  assert.equal(calculateCorrelatedVariance(1, 2, 1).differenceVariance, 1);
  assert.throws(() => calculateCorrelatedVariance(1, 1, 1.1), RangeError);
});
