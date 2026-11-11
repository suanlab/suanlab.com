import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculatePhononModes } from '../src/lib/learning-phonon-dispersion';

const close = (a: number, b: number, tolerance = 1e-10) => assert(Math.abs(a - b) < tolerance, `${a} vs ${b}`);

test('long-wavelength sound, the zone edge and equal-mass touching follow analytic limits', () => {
  const center = calculatePhononModes(1, 4, 0);
  assert.equal(center.acoustic, 0);
  close(center.optical ** 2, 2.5);
  close(calculatePhononModes(1, 4, 1e-5).acoustic / 1e-5, center.soundSpeed);
  const edge = calculatePhononModes(1, 4, Math.PI);
  close(edge.acoustic ** 2, .5);
  close(edge.optical ** 2, 2);
  const folded = calculatePhononModes(1, 1, Math.PI);
  assert.equal(folded.acoustic, folded.optical);
  assert.equal(folded.displacements, null);
});

test('physical displacements solve the coupled equations and are mass-orthogonal', () => {
  for (const [mA, mB] of [[1, 4], [4, 1], [2, 2]]) {
    for (const q of [0, .7, 2.1, Math.PI - .001]) {
      const result = calculatePhononModes(mA, mB, q);
      assert(result.displacements);
      const { acoustic: lower, optical: upper } = result.displacements;
      close(mA * lower[0] * upper[0] + mB * lower[1] * upper[1], 0);
      for (const branch of ['acoustic', 'optical'] as const) {
        const [a, b] = result.displacements[branch];
        const omegaSquared = result[branch] ** 2, c = 2 * Math.cos(q / 2);
        close(2 * a - c * b, omegaSquared * mA * a);
        close(2 * b - c * a, omegaSquared * mB * b);
      }
    }
  }
});
