import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateEinsteinFluidCurvature } from '../src/lib/learning-einstein-fluid-curvature';

test('trace-free radiation retains nonzero Ricci curvature', () => {
  const dust = calculateEinsteinFluidCurvature(3, 0, 0);
  const radiation = calculateEinsteinFluidCurvature(3, 1, 0);
  assert.equal(dust.scalarCurvature, 3);
  assert.equal(radiation.stressTrace, 0);
  assert.equal(radiation.scalarCurvature, 0);
  assert.equal(radiation.timeRicci, 3);
  assert.equal(radiation.spatialRicci, 1);
});

test('rest-frame components recover both the trace and the field equation', () => {
  for (const [rho, p, lambda] of [[3, 1, 0], [2, 0, 1], [0, 0, .5], [4, -4, 0]]) {
    const result = calculateEinsteinFluidCurvature(rho, p, lambda);
    assert.equal(-result.timeRicci + 3 * result.spatialRicci, result.scalarCurvature);
    assert.equal(result.timeEinstein - lambda, rho);
    assert.equal(result.spatialEinstein + lambda, p);
  }
});
