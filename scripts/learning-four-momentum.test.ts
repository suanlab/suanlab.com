import assert from 'node:assert/strict';
import { test } from 'node:test';
import { compareCollisionGeometry } from '../src/lib/learning-four-momentum';
import { boostSpacetimeEvent } from '../src/lib/learning-lorentz-boost';

test('a fixed target and opposing beams have different thresholds at equal lab energy', () => {
  const collision = compareCollisionGeometry(1, 2.5, 3);
  assert.equal(collision.fixedTarget.centerEnergy, 3);
  assert.equal(collision.opposingBeams.centerEnergy, 4.5);
  assert.equal(collision.fixedTarget.threshold, 2.5);
  assert.equal(collision.opposingBeams.threshold, 1);
  assert(collision.fixedTarget.allowed && collision.opposingBeams.allowed);
  const below = compareCollisionGeometry(1, 1.5, 3);
  assert(!below.fixedTarget.allowed && below.opposingBeams.allowed);
});

test('boosting the total four-momentum removes its spatial component in the center frame', () => {
  for (const mass of [.5, 1, 3]) {
    for (const kinetic of [0, .5, 2.5, 10]) {
      const result = compareCollisionGeometry(mass, kinetic, 3);
      const boosted = boostSpacetimeEvent({ ct: result.labEnergy, x: result.fixedTarget.momentum }, result.fixedTarget.centerSpeed);
      assert(Math.abs(boosted.x) < 1e-12);
      assert(Math.abs(boosted.ct - result.fixedTarget.centerEnergy) < 1e-12);
    }
  }
});
