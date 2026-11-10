import assert from 'node:assert/strict';
import { test } from 'node:test';
import { boostSpacetimeEvent, intervalSquared } from '../src/lib/learning-lorentz-boost';
import { compareMinkowskiPaths, departure, reunion } from '../src/lib/learning-minkowski-path';

test('broken paths distinguish timelike, null and spacelike legs', () => {
  assert.equal(compareMinkowskiPaths({ ct: 3, x: 2 }).properTime, 2 * Math.sqrt(5));
  assert.equal(compareMinkowskiPaths({ ct: 3, x: 3 }).properTime, 0);
  const mixed = compareMinkowskiPaths({ ct: 2, x: 3 });
  assert.equal(mixed.outgoing.kind, 'spacelike');
  assert.equal(mixed.returning.kind, 'timelike');
  assert.equal(mixed.properTime, null);
});

test('feasible broken paths preserve proper time under a boost and never exceed the straight clock', () => {
  for (let ct = .5; ct < 6; ct += .5) {
    for (let x = -Math.min(ct, 6 - ct); x <= Math.min(ct, 6 - ct); x += .25) {
      const turn = { ct, x };
      const result = compareMinkowskiPaths(turn);
      assert(result.properTime !== null && result.properTime <= 6);
      if (x === 0) assert.equal(result.properTime, 6);
      const boosted = [departure, turn, reunion].map(event => boostSpacetimeEvent(event, .6));
      const properTime = Math.sqrt(Math.max(0, intervalSquared(boosted[0], boosted[1]))) +
        Math.sqrt(Math.max(0, intervalSquared(boosted[1], boosted[2])));
      assert(Math.abs(properTime - result.properTime) < 1e-7);
    }
  }
});
