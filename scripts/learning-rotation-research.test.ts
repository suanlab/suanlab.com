import test from 'node:test';
import assert from 'node:assert/strict';
import { pointMassInertia, torqueFreeEuler } from '../src/lib/learning-rotation-inertia';

const shapeA = [4, 1, 1, 1, 2, 2, 0, 10, 10, 0, 0, 0];
const shapeB = [1, 1, 4, 2, 2, 1, 0, 10, 10, 0, 0, 0];
const labels = (matrix: { label: string }[][]) => matrix.map(row => row.map(x => x.label));
const close = (a: number, b: number) => assert.ok(Math.abs(a - b) < 2e-9, `${a} versus ${b}`);

test('quoted equal-mass shapes have identical COM tensors but different fourth spatial moments', () => {
  const a = pointMassInertia(shapeA), b = pointMassInertia(shapeB);
  assert.equal(a.mass.label, '12');
  assert.equal(b.mass.label, '12');
  for (const r of [a, b]) {
    assert.deepEqual(labels(r.tensorCOM), [['4', '0', '0'], ['0', '4', '0'], ['0', '0', '4']]);
    assert.equal(r.kineticCOM.label, '4');
    assert.deepEqual(r.angularMomentumCOM.map(x => x.label), ['4', '4', '0']);
    assert.ok(r.centerOfMass.every(x => x.label === '0'));
  }
  const fourthX = (r: typeof a) => r.points.reduce((s, p) => s + p.mass.value * p.position[0].value ** 4, 0);
  assert.equal(fourthX(a), .5);
  assert.equal(fourthX(b), 2);
});

test('common pivot, spin and orientation probes cannot distinguish the two second-moment-equivalent shapes', () => {
  for (const pivot of [-10, 0, 10]) for (const spin of [[10, 0, 0], [0, 10, 0], [10, 10, 0], [-10, 5, 20]]) {
    for (const orientation of [0, 3, 6]) {
      const results = [shapeA, shapeB].map(shape => {
        const p = shape.slice();
        p.splice(7, 3, ...spin);
        p[10] = pivot;
        p[11] = orientation;
        return pointMassInertia(p);
      });
      const [a, b] = results;
      assert.deepEqual(a.tensorPivot, b.tensorPivot);
      assert.deepEqual(a.kineticPivot, b.kineticPivot);
      assert.deepEqual(a.angularMomentumPivot, b.angularMomentumPivot);
      assert.deepEqual(a.labTensor, b.labTensor);
      assert.deepEqual(a.labAngularMomentum, b.labAngularMomentum);
    }
  }
  const pivotA = shapeA.slice();
  pivotA[10] = 10;
  const r = pointMassInertia(pivotA);
  assert.deepEqual(labels(r.tensorPivot), [['4', '0', '0'], ['0', '16', '0'], ['0', '0', '16']]);
  assert.equal(r.kineticPivot.label, '10');
  assert.equal(r.translationKinetic.label, '6');
  assert.equal(r.pivotChangesVelocityField, true);
});

test('six native energy observations recover the quoted sheared tensor including its negative cross term', () => {
  const base = [1, 1, 1, 2, 4, 6, 1, 10, 10, 0, 0, 0];
  const spins = [[10, 0, 0], [0, 10, 0], [0, 0, 10], [10, 10, 0], [10, 0, 10], [0, 10, 10]];
  const energy = spins.map(spin => {
    const p = base.slice();
    p.splice(7, 3, ...spin);
    return pointMassInertia(p).kineticCOM;
  });
  assert.deepEqual(energy.map(x => x.label), ['13', '41/4', '21/4', '93/4', '61/4', '31/2']);
  const k = energy.map(x => x.value);
  const recovered = [[2 * k[0], k[3] - k[0] - k[1], k[4] - k[0] - k[2]],
    [k[3] - k[0] - k[1], 2 * k[1], k[5] - k[1] - k[2]],
    [k[4] - k[0] - k[2], k[5] - k[1] - k[2], 2 * k[2]]];
  assert.deepEqual(recovered, [[26, 0, -3], [0, 20.5, 0], [-3, 0, 10.5]]);
  assert.deepEqual(recovered, pointMassInertia(base).tensorCOM.map(row => row.map(x => x.value)));
});

test('documented native intermediate spin and finite perturbations remain distinct from the analytic separatrix', () => {
  const pure = torqueFreeEuler([3, 2, 1, 0, 10, 0, 0, 100]);
  assert.deepEqual(pure.principalMoments, [3, 4, 5]);
  assert.equal(pure.initialEnergy.label, '2');
  assert.equal(pure.initialAngularMomentumNormSquared.label, '16');
  assert.ok(pure.states.every(q => q.bodyOmega.every((x, j) => x === [0, 1, 0][j])));
  for (const time of [0, 37, 100]) {
    const a = torqueFreeEuler([3, 2, 1, 1, 10, 0, 0, time]);
    const b = torqueFreeEuler([3, 2, 1, -1, 10, 0, 0, time]);
    for (const r of [a, b]) {
      assert.equal(r.initialEnergy.label, '403/200');
      assert.equal(r.initialAngularMomentumNormSquared.label, '1609/100');
      close(r.query.energyResidual, 0);
      close(r.query.angularNormResidual, 0);
      r.query.labAngularResidual.forEach(x => close(x, 0));
    }
    close(a.query.bodyOmega[0], -b.query.bodyOmega[0]);
    close(a.query.bodyOmega[1], b.query.bodyOmega[1]);
    close(a.query.bodyOmega[2], -b.query.bodyOmega[2]);
  }
});
