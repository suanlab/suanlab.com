import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateBaselineBudget } from '../src/lib/learning-baseline-budgets';

test('the synthetic comparison reverses at the derived crossover budget', () => {
  const small = calculateBaselineBudget(4), crossover = calculateBaselineBudget(16), large = calculateBaselineBudget(64);
  assert(small.first < small.second);
  assert.equal(crossover.first, crossover.second);
  assert(large.second < large.first);
});

test('increasing budget lowers each curve and a nonpositive budget is rejected', () => {
  for (const budget of [1, 2, 4, 16]) {
    const before = calculateBaselineBudget(budget), after = calculateBaselineBudget(2 * budget);
    assert(after.first < before.first);
    assert(after.second < before.second);
  }
  assert.throws(() => calculateBaselineBudget(0), RangeError);
});
