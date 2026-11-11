import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateSchwarzschildRedshift } from '../src/lib/learning-schwarzschild-redshift';

test('static and moving observers give the expected redshift and cancellation', () => {
  const staticCase = calculateSchwarzschildRedshift(4, 12, 0);
  assert(Math.abs(staticCase.gravitationalFactor - Math.sqrt(.6)) < 1e-14);
  assert(Math.abs(staticCase.cancellationSpeed - .25) < 1e-14);
  assert(staticCase.redshift > 0);
  assert(Math.abs(calculateSchwarzschildRedshift(4, 12, .25).ratio - 1) < 1e-14);
  assert(calculateSchwarzschildRedshift(4, 12, .5).redshift < 0);
  assert(calculateSchwarzschildRedshift(4, 12, -.25).redshift > staticCase.redshift);
});

test('local contraction reproduces the Doppler factor and invalid radii fail', () => {
  const beta = -.6;
  const ratio = calculateSchwarzschildRedshift(8, 8, beta).ratio;
  const measuredEmission = (1 - beta) / Math.sqrt(1 - beta ** 2);
  assert(Math.abs(ratio * measuredEmission - 1) < 1e-14);
  assert.throws(() => calculateSchwarzschildRedshift(2, 8, 0), RangeError);
  assert.throws(() => calculateSchwarzschildRedshift(8, 4, 0), RangeError);
});
