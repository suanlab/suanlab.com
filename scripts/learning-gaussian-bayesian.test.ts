import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateGaussianPosterior, gaussianDensity } from '../src/lib/learning-gaussian-bayesian';

test('weak sensors retain the prior, and the posterior predicts independent new readings', () => {
  const result = calculateGaussianPosterior(0, 1, .5, .25, 1);
  assert.deepEqual(result, { mean: 1, variance: .5, predictiveMean: .5, predictiveVariance: .375 });
  assert.deepEqual(calculateGaussianPosterior(-2, 3, 0, .25, 1), {
    mean: -2, variance: 3, predictiveMean: -0, predictiveVariance: .25,
  });
});

test('numerically integrating prior times likelihood reproduces posterior moments', () => {
  const result = calculateGaussianPosterior(.2, .9, -.7, .4, -.2);
  let mass = 0, first = 0, second = 0;
  for (let i = 0; i <= 16000; i++) {
    const x = i / 1000 - 8;
    const weight = gaussianDensity(x, .2, .9) * gaussianDensity(-.2, -.7 * x, .4);
    mass += weight;
    first += weight * x;
    second += weight * x ** 2;
  }
  assert(Math.abs(first / mass - result.mean) < 1e-10);
  assert(Math.abs(second / mass - (first / mass) ** 2 - result.variance) < 1e-10);
});
