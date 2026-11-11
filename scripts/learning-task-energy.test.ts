import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateTaskEnergy } from '../src/lib/learning-task-energy';

test('uneven samples integrate the piecewise linear power trace', () => {
  const result = calculateTaskEnergy([0, .05, .2], [10, 20, 50], 100, 20);
  assert(Math.abs(result.gross - 6) < 1e-12);
  assert(Math.abs(result.incremental - 2) < 1e-12);
  assert(Math.abs(result.grossPerTask - .06) < 1e-12);
  assert.throws(() => calculateTaskEnergy([0, .2, .2], [10, 20, 50], 100, 20), RangeError);
  assert.throws(() => calculateTaskEnergy([0, .2], [10], 100, 20), RangeError);
});

test('higher power can use less gross energy with the same incremental energy', () => {
  const slow = calculateTaskEnergy([0, .2], [40, 40], 100, 20);
  const fast = calculateTaskEnergy([0, .1], [60, 60], 100, 20);
  assert.equal(slow.gross, 8);
  assert.equal(fast.gross, 6);
  assert.equal(slow.incremental, 4);
  assert.equal(fast.incremental, 4);
});
