import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateTikhonovMode } from '../src/lib/learning-tikhonov-noise-gain';

const close = (a: number, b: number) => assert(Math.abs(a - b) < 1e-12, `${a} vs ${b}`);

test('a penalty equal to squared strength halves signal retention and noise gain', () => {
  const direct = calculateTikhonovMode(.05, 0, 1, .05, .05);
  const filtered = calculateTikhonovMode(.05, .05 ** 2, 1, .05, .05);
  assert(direct && filtered);
  close(direct.estimate, 2);
  close(direct.meanSquaredError, 1);
  close(filtered.signalRetention, .5);
  close(filtered.estimate, 1);
  close(filtered.meanSquaredError, .5);
});

test('filtered estimates solve the scalar normal equation including unobserved modes', () => {
  for (const s of [0, .05, .5, 1]) {
    const result = calculateTikhonovMode(s, .01, 3, .05, -.02);
    assert(result);
    close((s ** 2 + .01) * result.estimate, s * (s * 3 - .02));
  }
  assert.equal(calculateTikhonovMode(0, 0, 1, .05, .05), null);
});
