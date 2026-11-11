import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateRotationalCurvature } from '../src/lib/learning-rotational-curvature';

test('a rotating polar basis does not give flat space curvature', () => {
  const plane = calculateRotationalCurvature('plane', 2, .5);
  assert.equal(plane.radialConnection, -1);
  assert.equal(plane.angularConnection, 1);
  assert.equal(plane.curvature, 0);
  assert.equal(plane.holonomy, 0);
  assert.equal(plane.frameAngleChange, -2 * Math.PI);
});

test('constant curvature agrees with finite differences and integrated loop rotation', () => {
  for (const surface of ['sphere', 'hyperbolic'] as const) {
    const scale = 2, radialRatio = .7, step = 1e-3;
    const sample = (offset: number) => calculateRotationalCurvature(surface, scale, radialRatio + offset / scale);
    const left = sample(-step), center = sample(0), right = sample(step);
    const curvature = -(left.circumference - 2 * center.circumference + right.circumference) / (step ** 2 * center.circumference);
    assert(Math.abs(curvature - center.curvature) < 1e-7);
    assert(Math.abs(center.holonomy - (center.frameAngleChange + 2 * Math.PI)) < 1e-12);
  }
});
