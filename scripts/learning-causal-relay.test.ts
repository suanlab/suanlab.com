import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateCausalRelay } from '../src/lib/learning-causal-relay';

test('relay positions satisfy both cone constraints, including null and empty limits', () => {
  for (const [b, interval] of [[2, { lower: -1, upper: 3 }], [6, { lower: 3, upper: 3 }], [7, null]] as const) {
    assert.deepEqual(calculateCausalRelay(b, 3, 3, 0).interval, interval);
  }
  for (let t = 0; t <= 6; t += .5) {
    for (let x = -7; x <= 7; x += .5) {
      const result = calculateCausalRelay(2, t, x, 0);
      const inside = result.interval !== null && x >= result.interval.lower && x <= result.interval.upper;
      assert.equal(inside, result.reachableFromSender && result.reachesDeadline);
    }
  }
});

test('boosts preserve the time order for causal deadlines and can reverse spacelike order', () => {
  for (const beta of [-.9, 0, .9]) {
    assert(calculateCausalRelay(6, 3, 3, beta).deadlineTimeInBoostedFrame > 0);
  }
  const before = calculateCausalRelay(7, 3, 3, 0);
  const after = calculateCausalRelay(7, 3, 3, .9);
  assert.equal(after.deadlineCausalType, 'spacelike');
  assert(before.deadlineTimeInBoostedFrame > 0 && after.deadlineTimeInBoostedFrame < 0);
  assert.equal(after.interval, null);
});
