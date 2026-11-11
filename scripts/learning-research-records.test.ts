import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateResearchRecordSummary } from '../src/lib/learning-research-records';

test('reporting the best and the overall mean describes different quantities', () => {
  const losses = [.42, .55, .50, .60, .48, .57, .45, .65, .53, .59];
  const snapshot = [...losses];
  const result = calculateResearchRecordSummary(losses, 1);
  assert.equal(result.selectedMean, .42);
  assert.equal(result.allMean, .534);
  assert.deepEqual(losses, snapshot);
  assert.equal(calculateResearchRecordSummary(losses, 10).selectedMean, result.allMean);
});

test('selection preserves exact record counts and rejects missing final scores', () => {
  const result = calculateResearchRecordSummary([1, 2, 3], 2);
  assert.equal(result.completedCount, 3);
  assert.equal(result.selected.length, 2);
  assert.throws(() => calculateResearchRecordSummary([1, NaN, 3], 2), RangeError);
  assert.throws(() => calculateResearchRecordSummary([1, 2, 3], 4), RangeError);
});
