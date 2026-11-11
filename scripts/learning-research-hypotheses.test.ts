import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateResearchEffect } from '../src/lib/learning-research-hypotheses';

test('the worked example estimates a paired mean improvement of 0.105 K', () => {
  const result = calculateResearchEffect([.65, .52, .44, .60, .55, .50], [.52, .38, .36, .42, .48, .47]);
  assert(Math.abs(result.average - .105) < 1e-12);
  assert(result.standardError > 0);
});

test('the same mean can summarize paired effects of different variability', () => {
  const stable = calculateResearchEffect([4, 4, 4], [3, 3, 3]);
  const variable = calculateResearchEffect([4, 4, 4], [4, 3, 2]);
  assert.equal(stable.average, variable.average);
  assert.equal(stable.standardError, 0);
  assert.equal(variable.standardError, Math.sqrt(1 / 3));
  assert.throws(() => calculateResearchEffect([1, 2], [1]), RangeError);
});
