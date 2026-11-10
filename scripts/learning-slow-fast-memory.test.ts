import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateSlowFastMemory } from '../src/lib/learning-slow-fast-memory';

const close = (actual: number, expected: number, tolerance = 1e-10) =>
  assert(Math.abs(actual - expected) <= tolerance, `${actual} vs ${expected}`);

test('exact modes satisfy the original differential equations and initial values', () => {
  for (const fastRate of [2, 8, 64]) {
    for (const mixing of [0, .6, .95]) {
      const sample = (time: number) => calculateSlowFastMemory({ fastRate, mixing, initialFast: .3, time });
      close(sample(0).slow, 1);
      close(sample(0).fast, .3);
      const t = .25, step = 1e-5;
      const left = sample(t - step), at = sample(t), right = sample(t + step);
      close((right.slow - left.slow) / (2 * step), -at.slow + at.fast, 1e-7);
      close((right.fast - left.fast) / (2 * step), fastRate * (mixing * at.slow - at.fast), 1e-7);
    }
  }
});

test('the memory-lag bound holds for prepared and unprepared initial states', () => {
  for (const fastRate of [2, 4, 16, 64]) {
    for (const initialFast of [0, .6, 1.6]) {
      for (let i = 0; i <= 100; i++) {
        const sample = calculateSlowFastMemory({ fastRate, mixing: .6, initialFast, time: i / 20 });
        assert(sample.closureResidual <= sample.closureBound + 1e-14);
      }
    }
  }
  const error = (fastRate: number) => calculateSlowFastMemory({ fastRate, mixing: .6, initialFast: .6, time: 2 }).reductionError;
  assert(error(32) < error(4));
});
