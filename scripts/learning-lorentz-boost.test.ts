import assert from 'node:assert/strict';
import { test } from 'node:test';
import { boostSpacetimeEvent, intervalSquared } from '../src/lib/learning-lorentz-boost';

const close = (actual: number, expected: number) =>
  assert(Math.abs(actual - expected) < 1e-10, `${actual} vs ${expected}`);

test('origin, light rays and simultaneous events transform as predicted', () => {
  close(boostSpacetimeEvent({ ct: 5, x: 3 }, .6).x, 0);
  close(boostSpacetimeEvent({ ct: 0, x: 3 }, .6).ct, -2.25);
  close(boostSpacetimeEvent({ ct: 0, x: 3 }, -.6).ct, 2.25);
  for (const sign of [-1, 1]) {
    const ray = boostSpacetimeEvent({ ct: 1, x: sign }, .9);
    close(ray.x, sign * ray.ct);
  }
});

test('boosts preserve intervals, invert and compose by adding rapidities', () => {
  const origin = { ct: 0, x: 0 };
  for (const event of [{ ct: 2, x: 1 }, { ct: 1, x: 2 }, { ct: 2, x: -2 }]) {
    const first = boostSpacetimeEvent(event, .6);
    const next = boostSpacetimeEvent(first, -.3);
    const direct = boostSpacetimeEvent(event, Math.tanh(Math.atanh(.6) + Math.atanh(-.3)));
    close(intervalSquared(origin, next), intervalSquared(origin, event));
    close(next.ct, direct.ct);
    close(next.x, direct.x);
    const original = boostSpacetimeEvent(first, -.6);
    close(original.ct, event.ct);
    close(original.x, event.x);
  }
  assert.throws(() => boostSpacetimeEvent(origin, 1), RangeError);
});
