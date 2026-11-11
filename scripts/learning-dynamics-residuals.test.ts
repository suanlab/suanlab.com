import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateDynamicsResidual } from '../src/lib/learning-dynamics-residuals';

test('all candidates match the training observations even when their equation residual is nonzero', () => {
  for (const time of [0, 1, 2]) {
    const result = calculateDynamicsResidual(.2, time);
    assert.equal(result.observationResidual, 0);
    assert(Math.abs(result.equationResidual) > .6);
  }
  assert.equal(calculateDynamicsResidual(.2, .5).observationResidual, .2);
});

test('finite differences and a uniform-grid average recover the equation residual and its RMS', () => {
  const amplitude = -.18, eps = 1e-6, time = .4;
  const model = (t: number) => calculateDynamicsResidual(amplitude, t);
  const derivative = (model(time + eps).candidate - model(time - eps).candidate) / (2 * eps);
  assert(Math.abs(derivative + model(time).candidate - model(time).equationResidual) < 1e-9);
  let squared = 0;
  for (let i = 0; i < 1000; i++) squared += model(i / 500).equationResidual ** 2;
  assert(Math.abs(Math.sqrt(squared / 1000) - model(time).equationRms) < 1e-12);
});
