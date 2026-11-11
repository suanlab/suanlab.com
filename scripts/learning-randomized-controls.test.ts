import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateRandomizedControls } from '../src/lib/learning-randomized-controls';

test('independent randomized inputs distinguish frozen and refitted interventions', () => {
  const result = calculateRandomizedControls(.3, 0);
  assert.equal(result.full, .01);
  assert(Math.abs(result.removed - .1) < 1e-12);
  assert(Math.abs(result.frozenRandom - .19) < 1e-12);
  assert.equal(result.refittedRandom, result.removed);
  assert.equal(result.refittedCoefficient, 0);
});

test('the analytic formula agrees with exact enumeration of independent sign variables', () => {
  const squared: number[] = [];
  for (const z of [-1, 1]) for (const random of [-1, 1]) for (const noise of [-1, 1]) {
    squared.push((.3 * (z - random) + .1 * noise) ** 2);
  }
  assert(Math.abs(squared.reduce((a, b) => a + b) / squared.length - calculateRandomizedControls(.3, 0).frozenRandom) < 1e-12);
  assert.equal(calculateRandomizedControls(.3, 1).refittedRandom, .01);
  assert.equal(calculateRandomizedControls(.3, 1).frozenRandom, .01);
});
