import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateSensorAllocations } from '../src/lib/learning-optimal-sensor-allocation';

test('trace and determinant produce different optimal allocations in the asymmetric example', () => {
  const result = calculateSensorAllocations(12, 1, 3);
  assert.deepEqual([result.traceBest.first, result.traceBest.second], [3, 9]);
  assert.deepEqual([result.determinantBest.first, result.determinantBest.second], [6, 6]);
  assert.equal(result.traceBest.trace, 4 / 3);
  assert.equal(result.determinantBest.determinant, .25);
});

test('every plan respects the budget and the two reported optima attain the minima', () => {
  for (const [total, a, b] of [[2, 1, 1], [9, 5, .5], [20, .7, 2.3]]) {
    const result = calculateSensorAllocations(total, a, b);
    for (const plan of result.plans) {
      assert.equal(plan.first + plan.second, total);
      assert(plan.first > 0 && plan.second > 0);
      assert(result.traceBest.trace <= plan.trace);
      assert(result.determinantBest.determinant <= plan.determinant);
    }
  }
});
