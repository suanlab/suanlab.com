import assert from 'node:assert/strict';
import test from 'node:test';
import { frictionlessRodImpact as rod, drivenWheelContact as wheel } from '../src/lib/learning-rigid-contact';

test('quoted mass-metric examples change the contact normal mode while retaining tangential motion', () => {
  for (const [e, impulse, vy, omega, loss] of [
    [0, '2/5', '-3/5', '3/5', '1/5'],
    [5, '3/5', '-2/5', '9/10', '3/20'],
    [10, '4/5', '-1/5', '6/5', '0'],
  ] as const) {
    const r = rod([1, 2, 2, 0, -10, 0, e]);
    assert.equal(r.inertia.label, '2/3');
    assert.equal(r.inverseEffectiveNormalMass.label, '5/2');
    assert.equal(r.normalImpulse.label, impulse);
    assert.equal(r.after.velocity[1].label, vy);
    assert.equal(r.after.omega.label, omega);
    assert.equal(r.energyLoss.label, loss);
    assert.equal(r.after.contactVelocity[1].value, e / 10);
    assert.equal(r.contactAngularResidual.label, '0');
  }
});

test('native grazing bracket distinguishes opening, zero speed and closing without an attractive impulse', () => {
  for (const vy of [-9, -10, -11]) {
    const r = rod([1, 2, 2, 0, vy, 10, 5]);
    assert.equal(r.active, vy < -10);
    if (vy >= -10) {
      assert.equal(r.normalImpulse.label, '0');
      assert.deepEqual(r.before, r.after);
      assert.equal(r.restitutionResidual, null);
    } else {
      assert.equal(r.before.contactVelocity[1].label, '-1/10');
      assert.equal(r.normalImpulse.label, '3/50');
      assert.equal(r.after.contactVelocity[1].label, '1/20');
      assert.equal(r.after.velocity[1].label, '-26/25');
      assert.equal(r.after.omega.label, '109/100');
      assert.equal(r.energyLoss.label, '3/2000');
    }
  }
});

test('four native zero-drive friction coefficients have different waiting times but the quoted terminal losses', () => {
  for (const [mu, event] of [[0, null], [1, '1/3'], [2, '1/6'], [5, '1/15']] as const) {
    const r = wheel([1, 2, mu, 0, 5, 0, 0, 10, 0, 10]);
    assert.equal(r.requiredRollingFriction.label, '0');
    assert.equal(r.zeroSlipEventTime?.label ?? null, event);
    assert.equal(r.query.velocity.label, mu ? '2/3' : '1');
    assert.equal(r.query.clockwiseOmega.label, mu ? '2/3' : '0');
    assert.equal(r.query.heat.label, mu ? '1/6' : '0');
    assert.equal(r.query.kinetic.label, mu ? '1/3' : '1/2');
    assert.equal(r.query.rolling, mu > 0);
    assert.equal(r.query.energyResidual.label, '0');
    assert.equal(r.query.contactHeatResidual.label, '0');
  }
});

test('native times bracket the one-third-second switch and preserve the independently decomposed heat', () => {
  const before = wheel([1, 2, 1, 0, 5, 0, 0, 10, 0, 3]);
  const after = wheel([1, 2, 1, 0, 5, 0, 0, 10, 0, 4]);
  assert.equal(before.query.slipVelocity.label, '1/10');
  assert.equal(before.query.velocity.label, '7/10');
  assert.equal(before.query.clockwiseOmega.label, '3/5');
  assert.equal(before.query.heat.label, '33/200');
  assert.equal(before.query.rolling, false);
  assert.equal(after.query.slipVelocity.label, '0');
  assert.equal(after.query.heat.label, '1/6');
  assert.equal(after.query.position.label, '29/90');
  assert.equal(after.query.clockwiseAngle.label, '7/45');
  assert.equal(after.query.rolling, true);
});

test('an initially spinning wheel gains COM speed while its relative-slip energy is dissipated', () => {
  const sliding = wheel([1, 2, 1, 0, 5, 0, 0, 0, 10, 1]);
  assert.equal(sliding.initialSlip.label, '-1');
  assert.equal(sliding.query.friction.label, '1');
  assert.equal(sliding.query.velocity.label, '1/10');
  assert.equal(sliding.query.clockwiseOmega.label, '4/5');
  assert.equal(sliding.query.heat.label, '17/200');
  const rolling = wheel([1, 2, 1, 0, 5, 0, 0, 0, 10, 10]);
  assert.equal(rolling.query.velocity.label, '1/3');
  assert.equal(rolling.query.clockwiseOmega.label, '1/3');
  assert.equal(rolling.query.heat.label, '1/6');
  assert.equal(rolling.query.kinetic.label, '1/12');
});
