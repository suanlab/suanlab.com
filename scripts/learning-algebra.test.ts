import assert from 'node:assert/strict';
import test from 'node:test';
import { applySymmetric2, covarianceRing, leastSquaresState, populationCovariance2, projectionState, symmetricEigen2 } from '../src/lib/learning-linear-algebra';
import { add3, dot3, norm3, type Vec2, type Vec3 } from '../src/lib/learning-simulations';
import { algebraPresets, algebraValues, buildAlgebraScene, defaultAlgebra, validAlgebra, type AlgebraKind } from '../src/components/learning/algebra-model';
import { learningGuides, learningUnits } from '../src/data/interactive-book';
const close = (a: number, b: number) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);
const vector = (a: number[], b: number[]) => { assert.equal(a.length, b.length); a.forEach((v, i) => close(v, b[i])); };
test('projection reference separates dot product, scalar coefficient and vector projection', () => {
  const p = projectionState([3, 2, 0], [2, 0, 0]); close(p.dot, 6); close(p.coefficient!, 1.5);
  vector(p.projected!, [3, 0, 0]); vector(p.residual!, [0, 2, 0]); close(p.residualDot!, 0); close(p.angle!, Math.atan2(2, 3) * 180 / Math.PI);
});
for (const [u, v] of [ [[3, 2, 0], [1, 1, 0]], [[1, 2, 3], [1, 1, 1]], [[-3, 0, 0], [2, 0, 0]], [[3, 0, 0], [0, 2, 0]] ] as [Vec3, Vec3][]) test(`orthogonal decomposition and scale invariance: ${u}/${v}`, () => {
  const p = projectionState(u, v); vector(add3(p.projected!, p.residual!), u); close(dot3(v, p.residual!), 0);
  close(norm3(u) ** 2, norm3(p.projected!) ** 2 + norm3(p.residual!) ** 2);
  vector(projectionState(u, v.map(x => x * 2) as Vec3).projected!, p.projected!);
});
test('zero-vector cases report undefined angle and reference direction honestly', () => {
  const noReference = projectionState([3, 2, 0], [0, 0, 0]); assert.equal(noReference.projected, null); assert.equal(noReference.angle, null);
  const zeroInput = projectionState([0, 0, 0], [2, 0, 0]); vector(zeroInput.projected!, [0, 0, 0]); assert.equal(zeroInput.angle, null);
});
test('regression worked fractions and projection orthogonality agree', () => {
  const initial = leastSquaresState([1, 2, 2], .5); close(initial.loss, .75); close(initial.gradient, -4); close(initial.optimalSlope, 11 / 14); close(initial.optimalLoss, 5 / 28);
  const optimal = leastSquaresState([1, 2, 2], 11 / 14); vector(optimal.residual, [3 / 14, 6 / 14, -5 / 14]); close(optimal.normalResidual, 0); close(optimal.gradient, 0); close(optimal.squaredError, 5 / 14);
  const p = projectionState([1, 2, 2], [1, 2, 3]); vector(p.projected!, optimal.predicted);
});
test('regression exact fit, zero response and large observation have independent reference values', () => {
  close(leastSquaresState([1, 2, 3], 1).loss, 0); close(leastSquaresState([0, 0, 0], 0).loss, 0);
  const outlier = leastSquaresState([1, 2, 5], 10 / 7); close(outlier.optimalSlope, 10 / 7); close(outlier.loss, 5 / 7);
});
for (const [a, b, d, expected] of [[2, 1, 2, [3, 1]], [3, 0, 1, [3, 1]], [2, -1, 2, [3, 1]], [1, 2, 1, [3, -1]], [2, 0, 2, [2, 2]], [0, 0, 0, [0, 0]], [1, 1, 1, [2, 0]]] as [number, number, number, Vec2][]) test(`symmetric eigenproblem reference: ${a},${b},${d}`, () => {
  const e = symmetricEigen2(a, b, d); vector(e.values, expected); close(e.first[0] * e.second[0] + e.first[1] * e.second[1], 0);
  [e.first, e.second].forEach((q, i) => vector(applySymmetric2(a, b, d, q), q.map(v => v * e.values[i])));
});
test('PCA variance, repeated/zero/indefinite cases preserve their meanings', () => {
  close(symmetricEigen2(2, 1, 2).varianceRatio!, .75);
  assert.equal(symmetricEigen2(2, 0, 2).repeated, true); close(symmetricEigen2(2, 0, 2).varianceRatio!, .5);
  assert.equal(symmetricEigen2(0, 0, 0).varianceRatio, null); assert.equal(symmetricEigen2(1, 2, 1).positiveSemidefinite, false); assert.equal(symmetricEigen2(1, 2, 1).varianceRatio, null); assert.equal(covarianceRing(1, 2, 1), null);
});
for (const [a, b, d] of [[2, 1, 2], [2, -1, 2], [1, 1, 1], [0, 0, 0]]) test(`deterministic population covariance and reconstruction error: ${a},${b},${d}`, () => {
  const points = covarianceRing(a, b, d)!; const sample = populationCovariance2(points); vector(sample.mean, [0, 0]); vector(sample.matrix.flat(), [a, b, b, d]);
  const e = symmetricEigen2(a, b, d); close(points.reduce((sum, p) => sum + (p[0] * e.second[0] + p[1] * e.second[1]) ** 2, 0) / points.length, e.values[1]);
});
test('all algebra presets, scene coordinates and guided selections are valid', () => {
  assert.ok(validAlgebra(defaultAlgebra)); assert.ok(!validAlgebra({ ...defaultAlgebra, response: [1, 2] })); assert.ok(!validAlgebra({ ...defaultAlgebra, angle: Infinity }));
  for (const kind of Object.keys(algebraPresets) as AlgebraKind[]) {
    for (const patch of algebraPresets[kind]) {
      const s = { ...defaultAlgebra, ...patch }; assert.ok(validAlgebra(s)); const before = JSON.stringify(s); const scene = buildAlgebraScene(kind, s);
      assert.ok(scene.gridUnit > 0); assert.ok([...scene.lines.flatMap(l => l.points.flat()), ...scene.arrows.flatMap(a => [...a.start, ...a.end]), ...scene.dots.flatMap(d => d.position)].every(Number.isFinite)); assert.equal(JSON.stringify(s), before);
    }
    learningGuides[kind].experiments.forEach(g => assert.ok(algebraPresets[kind][g.preset]));
  }
  assert.deepEqual(buildAlgebraScene('least-squares', defaultAlgebra).axisLabels, ['obs₁', 'obs₂', 'obs₃']);
  close(algebraValues({ ...defaultAlgebra, angle: 45 }).cross, 0);
});
test('nonfinite inputs and invalid covariance sample counts are rejected', () => {
  assert.throws(() => projectionState([NaN, 0, 0], [1, 0, 0])); assert.throws(() => leastSquaresState([1, 2, 3], Infinity));
  assert.throws(() => symmetricEigen2(1, NaN, 1)); assert.throws(() => covarianceRing(2, 1, 2, 1)); assert.throws(() => populationCovariance2([]));
});

test('unit reading order respects its explicit prerequisite graph and stable IDs', () => {
  const index = new Map(learningUnits.map((unit, i) => [unit.id, i])); assert.equal(index.size, learningUnits.length);
  learningUnits.forEach((unit, i) => unit.prerequisiteIds.forEach(id => assert.ok(index.has(id) && index.get(id)! < i, `${unit.id}: prerequisite ${id}`)));
});
test('projection avoids denominator underflow when the reference direction is rescaled', () => {
  vector(projectionState([3, 2, 0], [2e-200, 0, 0]).projected!, [3, 0, 0]);
});
