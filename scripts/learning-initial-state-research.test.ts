import assert from 'node:assert/strict';
import test from 'node:test';
import { invertedSpringInitialState as spring, delayedNewtonRelease as release } from '../src/lib/learning-initial-state-role';

test('two quoted native position observations recover the hidden initial velocities and predict a third position', () => {
  for (const gap of [1, 2]) {
    const observed = spring([1, 1, 10, -10, 10, 10, 10 * gap]);
    const future = spring([1, 1, 10, -10, 10, 10, 10 * (gap + 1)]);
    for (const [last, target, actualVelocity] of [
      [observed.queryA.position, future.queryA.position, -1],
      [observed.queryB.position, future.queryB.position, 1],
    ]) {
      const velocity = (last - Math.cosh(gap)) / Math.sinh(gap);
      assert.ok(Math.abs(velocity - actualVelocity) < 2e-13);
      const predicted = (-Math.sinh(1) + Math.sinh(gap + 1) * last) / Math.sinh(gap);
      assert.ok(Math.abs(predicted - target) < 2e-12);
      for (const firstError of [-.001, .001]) for (const lastError of [-.001, .001]) {
        const noisy = ((last + lastError) - (1 + firstError) * Math.cosh(gap)) / Math.sinh(gap);
        const bound = .001 * (1 + Math.cosh(gap)) / Math.sinh(gap);
        assert.ok(Math.abs(noisy - actualVelocity) <= bound + 2e-13);
      }
    }
  }
});

test('the actual zero-stiffness controls use the finite-difference observation inverse', () => {
  const r = spring([2, 0, 10, 5, 10, -5, 20]);
  assert.equal(r.freeLimit, true);
  assert.equal(r.queryA.position, 2);
  assert.equal(r.queryB.position, 0);
  assert.equal((r.queryA.position - 1) / 2, .5);
  assert.equal((r.queryB.position - 1) / 2, -.5);
});

test('native zero-state branches remain original solutions rather than implicit regularization controls', () => {
  for (const [time, immediate, delayed] of [[10, '1', '0'], [20, '16', '1']] as const) {
    const r = release([1, 2, 0, 10, time]);
    assert.equal(r.queryA.position.label, immediate);
    assert.equal(r.queryB.position.label, delayed);
    assert.equal(r.stationary.position.label, '0');
    assert.equal(r.sameCompleteInitialState, true);
    assert.equal(r.nonuniqueConstructed, true);
    assert.equal(r.queryA.newtonResidual.label, '0');
    assert.equal(r.queryB.energyResidual.label, '0');
  }
  const zero = release([1, 0, 0, 10, 20]);
  assert.equal(zero.nonuniqueConstructed, false);
  assert.equal(zero.queryA.position.label, '0');
  assert.equal(zero.queryB.position.label, '0');
});
