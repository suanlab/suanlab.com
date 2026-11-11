import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateAggregationEffects } from '../src/lib/learning-aggregation-proof';

test('unequal counts can reverse the sign of an average improvement', () => {
  const result = calculateAggregationEffects([1, 9], [.4, -.1]);
  assert(Math.abs(result.episodeMean - .15) < 1e-12);
  assert.equal(result.observationMean, -.05);
  assert.equal(result.covariance, -1);
});

test('the covariance identity holds and equal counts make both estimands agree', () => {
  for (const counts of [[1, 9, 4], [4, 4, 4], [8, 2, 5]]) {
    const result = calculateAggregationEffects(counts, [.4, -.1, .2]);
    assert(Math.abs(result.observationMean - result.episodeMean - result.covariance / result.meanCount) < 1e-12);
  }
  const same = calculateAggregationEffects([4, 4], [.4, -.1]);
  assert.equal(same.episodeMean, same.observationMean);
});
