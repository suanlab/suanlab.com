import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateSensorIdentifiability } from '../src/lib/learning-sensor-identifiability';

test('complementary sensors recover both amplitudes and nearly redundant sensors magnify noise', () => {
  const opposite = calculateSensorIdentifiability(-1, .1, 0, 0);
  assert.deepEqual(opposite.estimate, { a: 2, b: 3 });
  assert.equal(opposite.second, -1);
  const weak = calculateSensorIdentifiability(.9, .1, 0, 0);
  assert.deepEqual(weak.estimate, { a: 2, b: 3 });
  assert(opposite.bDeviation !== null && weak.bDeviation !== null);
  assert(Math.abs(weak.bDeviation / opposite.bDeviation - 20) < 1e-12);
  const perturbed = calculateSensorIdentifiability(.9, .1, 0, .1);
  assert(Math.abs((perturbed.estimate?.b ?? 0) - 2) < 1e-12);
});

test('identical sensors with exact readings constrain only a sum', () => {
  assert.equal(calculateSensorIdentifiability(1, .1, 0, 0).state, 'nonunique');
  assert.equal(calculateSensorIdentifiability(1, .1, 0, .1).state, 'inconsistent');
  assert.equal(calculateSensorIdentifiability(1, .1, 0, 0).estimate, null);
});
