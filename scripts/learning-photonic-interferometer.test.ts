import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculatePhotonicProduct } from '../src/lib/learning-photonic-interferometer';

const close = (a: number, b: number) => assert(Math.abs(a - b) < 1e-12, `${a} vs ${b}`);

test('the example transforms signed amplitudes while power is invariant under global sign', () => {
  const positive = calculatePhotonicProduct(1, -1, Math.PI / 4, .5);
  const negative = calculatePhotonicProduct(-1, 1, Math.PI / 4, .5);
  close(positive.y1, .5);
  close(positive.y2, -.5);
  close(positive.power1, negative.power1);
  close(positive.power2, negative.power2);
  close(positive.y1, -negative.y1);
  close(positive.y2, -negative.y2);
});

test('attenuation accounts for power loss and unattenuated rotations cancel', () => {
  for (const theta of [0, .35, Math.PI / 4, Math.PI / 2]) {
    for (const d of [0, .4, 1]) {
      const result = calculatePhotonicProduct(.75, -.5, theta, d);
      close(result.inputPower, result.power1 + result.power2 + result.dissipatedPower);
      if (d === 1) {
        close(result.y1, .75);
        close(result.y2, -.5);
      }
    }
  }
});
