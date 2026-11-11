import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculatePairedSignTail } from '../src/lib/learning-paired-sign-test';

test('six and five positive signs reproduce the exact example tails', () => {
  assert.equal(calculatePairedSignTail(6, 6).tail, 1 / 64);
  assert.equal(calculatePairedSignTail(6, 5).tail, 7 / 64);
  assert.equal(calculatePairedSignTail(6, 0).tail, 1);
});

test('the probability distribution normalizes and the tail matches enumeration', () => {
  for (let n = 1; n <= 8; n++) {
    for (let k = 0; k <= n; k++) {
      let accepted = 0;
      for (let pattern = 0; pattern < 2 ** n; pattern++) {
        const positives = pattern.toString(2).replaceAll('0', '').length;
        if (positives >= k) accepted++;
      }
      const result = calculatePairedSignTail(n, k);
      assert(Math.abs(result.tail - accepted / 2 ** n) < 1e-12);
      assert(Math.abs(result.probabilities.reduce((a, b) => a + b) - 1) < 1e-12);
    }
  }
});
