import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateHeavyModeMatching } from '../src/lib/learning-heavy-mode-matching';

const defaults = { massRatio: 4, mixing: .5, frequencyRatio: .5, order: 2 };
const close = (actual: number, expected: number) => {
  assert(Math.abs(actual - expected) <= 1e-10 * Math.max(1, Math.abs(expected)), `${actual} vs ${expected}`);
};

test('light response equals the light entry of the full inverse kernel', () => {
  for (const massRatio of [2, 4, 16]) {
    for (const mixing of [0, .5, .99]) {
      for (const frequencyRatio of [0, .01, .5, 1, 2]) {
        const result = calculateHeavyModeMatching({ ...defaults, massRatio, mixing, frequencyRatio });
        const omega = massRatio * frequencyRatio;
        const light = 1 + omega ** 2, heavy = massRatio ** 2 + omega ** 2;
        const determinant = light * heavy - (mixing * massRatio) ** 2;
        close(result.exactResponse, heavy / determinant);
      }
    }
  }
});

test('geometric remainder controls kernel and response errors and detects a negative approximation', () => {
  for (const frequencyRatio of [.25, .5, 1, 2]) {
    for (let order = 0; order <= 6; order++) {
      const result = calculateHeavyModeMatching({ ...defaults, order, frequencyRatio });
      close(result.kernelError, Math.abs(result.exactKernel - result.approximateKernel));
      if (result.approximateResponse !== null) {
        assert(result.responseError !== null);
        close(result.responseError, Math.abs(result.exactResponse - result.approximateResponse));
      } else {
        assert(result.approximateKernel <= 0);
        assert.equal(result.responseError, null);
      }
    }
  }
  const highOrder = calculateHeavyModeMatching({ ...defaults, frequencyRatio: 2, order: 6 });
  assert.equal(highOrder.approximateResponse, null);
});
