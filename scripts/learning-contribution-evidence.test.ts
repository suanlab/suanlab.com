import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateContributionEvidence } from '../src/lib/learning-contribution-evidence';

const close = (a: number, b: number) => assert(Math.abs(a - b) < 1e-12, `${a} vs ${b}`);

test('matched comparisons isolate the physical term within each context', () => {
  const result = calculateContributionEvidence(.6, .5, .54, .4);
  close(result.shortGain, .06);
  close(result.longGain, .1);
  close(result.interaction, .04);
  close(result.combinedGain, result.contextGainWithoutPhysics + result.longGain);
});

test('equal improvements across contexts give zero interaction even with different base errors', () => {
  const result = calculateContributionEvidence(.75, .5, .625, .375);
  assert.equal(result.interaction, 0);
  assert.equal(result.shortGain, .125);
  assert.throws(() => calculateContributionEvidence(.75, .5, -.625, .375), RangeError);
});
