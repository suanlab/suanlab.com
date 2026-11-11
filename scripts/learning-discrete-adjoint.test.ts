import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateDiscreteAdjoint } from '../src/lib/learning-discrete-adjoint';

test('the two-step example reproduces the forward trajectory and its reverse sensitivities', () => {
  const result = calculateDiscreteAdjoint(1, .5, [0, 0], 0, .1);
  assert.deepEqual(result.states, [1, .5, .25]);
  assert.deepEqual(result.adjoint, [.0625, .125, .25]);
  assert.deepEqual(result.controlGradient, [.125, .25]);
  assert.equal(result.persistenceGradient, .25);
});

test('a directional finite difference checks all controls and persistence together', () => {
  const a = .7, u = [.15, -.1, .2], d = [.2, .4, -.3], da = .1, eps = 1e-6;
  const exact = calculateDiscreteAdjoint(1, a, u, .3, .1);
  const loss = (step: number) => calculateDiscreteAdjoint(1, a + step * da, u.map((value, i) => value + step * d[i]), .3, .1).loss;
  const numeric = (loss(eps) - loss(-eps)) / (2 * eps);
  const derivative = exact.persistenceGradient * da + exact.controlGradient.reduce((total, value, i) => total + value * d[i], 0);
  assert(Math.abs(derivative - numeric) < 1e-9);
});
