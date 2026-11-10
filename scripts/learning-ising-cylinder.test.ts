import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateIsingCylinder, calculateIsingSweep } from '../src/lib/learning-ising-cylinder';

function close(actual: number, expected: number) {
  assert(Math.abs(actual - expected) <= 1e-11 * Math.max(1, Math.abs(expected)),
    `Expected ${expected}, got ${actual}`);
}

test('independent spins give the analytic moments and Binder ratio', () => {
  for (const length of [3, 4, 6, 8]) {
    const sites = length ** 2;
    const result = calculateIsingCylinder({ length, horizontalCoupling: 0, verticalCoupling: 0 });
    assert.equal(result.meanMagnetization, 0);
    close(result.secondMoment, sites);
    close(result.fourthMoment, 3 * sites ** 2 - 2 * sites);
    close(result.binderRatio, 2 / (3 * sites));
    close(result.logPartition, sites * Math.log(2));
  }
});

test('row transfer matches exhaustive enumeration on an anisotropic cylinder', () => {
  const length = 4, horizontalCoupling = .37, verticalCoupling = .53;
  let z = 0, second = 0, fourth = 0;
  for (let state = 0; state < 1 << (length ** 2); state++) {
    const spin = (x: number, y: number) => 2 * ((state >> (y * length + x)) & 1) - 1;
    let horizontal = 0, vertical = 0, magnetization = 0;
    for (let y = 0; y < length; y++) {
      for (let x = 0; x < length; x++) {
        const s = spin(x, y);
        magnetization += s;
        horizontal += s * spin((x + 1) % length, y);
        if (y + 1 < length) vertical += s * spin(x, y + 1);
      }
    }
    const weight = Math.exp(horizontalCoupling * horizontal + verticalCoupling * vertical);
    z += weight;
    second += weight * magnetization ** 2;
    fourth += weight * magnetization ** 4;
  }
  const result = calculateIsingCylinder({ length, horizontalCoupling, verticalCoupling });
  close(result.logPartition, Math.log(z));
  close(result.secondMoment, second / z);
  close(result.fourthMoment, fourth / z);
  close(result.binderRatio, 1 - fourth * z / (3 * second ** 2));
});

test('small finite crossings have a measurable size correction relative to the infinite critical point', () => {
  const gap = (k: number, a: number, b: number) => {
    const ratio = (length: number) => calculateIsingCylinder({
      length, horizontalCoupling: k, verticalCoupling: k,
    }).binderRatio;
    return ratio(a) - ratio(b);
  };
  const critical = Math.asinh(1) / 2;
  assert(gap(critical, 3, 4) > 0);
  assert(gap(.469, 3, 4) > 0 && gap(.47, 3, 4) < 0);
  assert(gap(.447, 6, 8) > 0 && gap(.448, 6, 8) < 0);
});

test('the browser sweeps preserve the critical equation and finite crossings', () => {
  const sweeps = [.5, 1, 1.5].map(calculateIsingSweep);
  assert(sweeps[0].critical > sweeps[1].critical && sweeps[1].critical > sweeps[2].critical);
  for (const sweep of sweeps) {
    assert(Math.abs(Math.sinh(2 * sweep.critical) * Math.sinh(2 * sweep.ratio * sweep.critical) - 1) < 1e-6);
    assert(sweep.curves.every(curve => curve.samples.length === 41));
    for (const { lengths, kx } of sweep.crossings) {
      assert(kx !== null);
      const observe = (length: number) => calculateIsingCylinder({
        length, horizontalCoupling: kx, verticalCoupling: kx * sweep.ratio,
      }).binderRatio;
      assert(Math.abs(observe(lengths[0]) - observe(lengths[1])) < 1e-8);
    }
  }
  assert.throws(() => calculateIsingSweep(2), RangeError);
});
