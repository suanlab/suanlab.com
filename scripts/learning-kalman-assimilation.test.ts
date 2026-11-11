import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateKalmanAssimilation } from '../src/lib/learning-kalman-assimilation';

const close = (a: number, b: number) => assert(Math.abs(a - b) < 1e-12, `${a} vs ${b}`);

test('the first observation matches the scalar Gaussian conditioning result', () => {
  const [first] = calculateKalmanAssimilation(0, 1, .9, .2, .25, [1]);
  close(first.forecastVariance, 1.01);
  close(first.mean, 1.01 / 1.26);
  close(first.variance, 1.01 * .25 / 1.26);
  assert.equal(first.innovation, 1);
});

test('a missing reading carries the forecast forward and noisier readings decrease the gain', () => {
  const [first, missing, third] = calculateKalmanAssimilation(0, 1, .9, .2, .25, [1, null, .2]);
  assert.equal(missing.innovation, null);
  assert.equal(missing.gain, 0);
  close(missing.mean, .9 * first.mean);
  close(missing.variance, .81 * first.variance + .2);
  assert(third.variance < third.forecastVariance);
  const [noisy] = calculateKalmanAssimilation(0, 1, .9, .2, 2, [1]);
  assert(noisy.gain < first.gain);
});
