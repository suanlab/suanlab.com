import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateSingularModeInverse } from '../src/lib/learning-ill-posedness';

test('the weak direction amplifies error by the reciprocal singular value', () => {
  assert.equal(calculateSingularModeInverse(.5, .01, 'weak').estimateError, .02);
  const weak = calculateSingularModeInverse(.01, .01, 'weak');
  const strong = calculateSingularModeInverse(.01, .01, 'strong');
  assert.equal(weak.condition, 100);
  assert.equal(weak.secondEstimate, 2);
  assert.equal(weak.estimateError, 1);
  assert.equal(strong.estimateError, .01);
  assert.equal(strong.secondEstimate, 1);
});

test('zero singular value causes nonuniqueness for compatible data and inconsistency otherwise', () => {
  const compatible = calculateSingularModeInverse(0, .01, 'strong');
  assert.equal(compatible.state, 'nonunique');
  assert.equal(compatible.firstEstimate, 1.01);
  assert.equal(compatible.secondEstimate, null);
  const inconsistent = calculateSingularModeInverse(0, .01, 'weak');
  assert.equal(inconsistent.state, 'inconsistent');
  assert.equal(inconsistent.firstEstimate, null);
  assert.equal(inconsistent.condition, null);
});
