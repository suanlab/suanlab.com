import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateHoeffdingEvidence } from '../src/lib/learning-evidence-roles';

test('the stated one-model and simultaneous-model radii follow the bound', () => {
  const one = calculateHoeffdingEvidence(100, 1, .05, .18);
  const many = calculateHoeffdingEvidence(100, 100, .05, .18);
  assert(Math.abs(one.radius - .13581015157406195) < 1e-12);
  assert(Math.abs(many.radius - .20364245186235147) < 1e-12);
  assert.equal(many.lower, 0);
  assert(many.upper > one.upper);
});

test('more independent observations shrink the radius and intervals respect the risk range', () => {
  const first = calculateHoeffdingEvidence(100, 10, .05, .9);
  const refined = calculateHoeffdingEvidence(400, 10, .05, .9);
  assert.equal(first.radius, 2 * refined.radius);
  assert.equal(first.upper, 1);
  assert.throws(() => calculateHoeffdingEvidence(100, 10, 0, .9), RangeError);
});
