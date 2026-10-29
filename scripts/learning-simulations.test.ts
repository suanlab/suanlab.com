import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { add3, scale3, determinant3, gradientPath, gradientStep, particleState, quadratic, transform3, type Mat3, type Vec2, type Vec3 } from '../src/lib/learning-simulations';
import { buildLabScene, defaultLab, validLab } from '../src/components/learning/lab-model';
import { getLearningContent } from '../src/lib/learning-content';
import { learningUnits } from '../src/data/interactive-book';

function close(actual: number | number[], expected: number | number[]) {
  const a = typeof actual === 'number' ? [actual] : actual; const b = typeof expected === 'number' ? [expected] : expected;
  assert.equal(a.length, b.length);
  a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) <= 1e-9, `${v} != ${b[i]}`));
}
function fixture<T>(file: string): { referenceCases: T[] } {
  return JSON.parse(fs.readFileSync(path.join('docs/online-learning/physics-math-for-ai', file), 'utf8'));
}
interface ParticleCase { id: string; force1N: Vec3; force2N: Vec3; massKg: number; timeSeconds: number; initialVelocityMps: Vec3; initialPositionM: Vec3; expected: { netForceN: Vec3; netForceMagnitudeN: number; accelerationMps2: Vec3; velocityMps: Vec3; positionM: Vec3; kineticEnergyJ: number; workJ: number }; }
for (const c of fixture<ParticleCase>('01-vectors-and-forces.scene.json').referenceCases) test(`constant-force reference: ${c.id}`, () => {
  const actual = particleState(c.force1N, c.force2N, c.massKg, c.timeSeconds, c.initialVelocityMps, c.initialPositionM); const e = c.expected;
  close(actual.force, e.netForceN); close(actual.magnitude, e.netForceMagnitudeN); close(actual.acceleration, e.accelerationMps2);
  close(actual.velocity, e.velocityMps); close(actual.position, e.positionM); close(actual.kineticEnergy, e.kineticEnergyJ); close(actual.work, e.workJ);
  close(actual.kineticEnergy - .5 * c.massKg * c.initialVelocityMps.reduce((sum, v) => sum + v * v, 0), actual.work);
});
interface MatrixCase { id: string; matrix: Mat3; point: Vec3; expected: { point: Vec3; determinant: number; unitCubeVolumeScale: number; invertible: boolean }; }
for (const c of fixture<MatrixCase>('02-linear-transforms.scene.json').referenceCases) test(`column-vector reference: ${c.id}`, () => {
  close(transform3(c.matrix, c.point), c.expected.point); close(determinant3(c.matrix), c.expected.determinant);
  close(Math.abs(determinant3(c.matrix)), c.expected.unitCubeVolumeScale); assert.equal(determinant3(c.matrix) !== 0, c.expected.invertible);
});
interface GradientCase { id: string; a: number; b: number; point: Vec2; learningRate: number; expected: { gradient: Vec2; nextPoint: Vec2; loss: number; nextLoss: number }; }
for (const c of fixture<GradientCase>('04-gradients.scene.json').referenceCases) test(`gradient reference: ${c.id}`, () => {
  const current = quadratic(c.a, c.b, c.point); const next = gradientStep(c.a, c.b, c.point, c.learningRate);
  close(current.gradient, c.expected.gradient); close(current.loss, c.expected.loss); close(next, c.expected.nextPoint); close(quadratic(c.a, c.b, next).loss, c.expected.nextLoss);
});
test('convergence, oscillation and display-limit stopping do not clamp true values', () => {
  const stable = gradientPath(2, 1, [2, 2], .25, 50); assert.ok(quadratic(2, 1, stable.at(-1)!).loss < 1e-10);
  const boundary = gradientPath(2, 1, [2, 2], 1, 8); close(boundary.at(-1)!, [2, 0]);
  const growing = gradientPath(2, 1, [2, 2], 1.25, 50); assert.ok(growing.length < 51); assert.ok(Math.abs(growing.at(-1)![0]) > 1000);
  assert.deepEqual(gradientPath(2, 1, [0, 0], 2, 4), [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]);
});
test('invalid physical inputs and malformed saved settings are rejected', () => {
  assert.throws(() => particleState([0, 0, 0], [0, 0, 0], 0, 0)); assert.throws(() => particleState([Infinity, 0, 0], [0, 0, 0], 1, 1));
  assert.throws(() => particleState([0, 0, 0], [0, 0, 0], 1, -1)); assert.throws(() => quadratic(0, 1, [1, 1]));
  assert.throws(() => gradientStep(1, 1, [1, 1], 0)); assert.throws(() => gradientPath(1, 1, [1, 1], .1, 51));
  assert.throws(() => gradientPath(1, 1, [NaN, 0], .1, 0)); assert.throws(() => gradientPath(1, 1, [1, 1], 0, 0));
  assert.ok(validLab(defaultLab)); assert.ok(!validLab({ ...defaultLab, matrix: [[1, 0]] })); assert.ok(!validLab({ ...defaultLab, time: Infinity }));
});
test('render adapters preserve column-vector directions and never change the model', () => {
  const settings = { ...defaultLab, matrix: [[0, -1, 0], [1, 0, 0], [0, 0, 1]] as Mat3, point: [1, 0, 0] as Vec3 };
  const scene = buildLabScene('linear-transforms', settings);
  close(scene.arrows[0].end.map(v => v * scene.gridUnit), [0, 1, 0]); close(scene.dots[1].position.map(v => v * scene.gridUnit), [0, 1, 0]);
  const original = JSON.stringify(defaultLab);
  for (const kind of ['vectors-and-forces', 'linear-transforms', 'gradients'] as const) {
    const view = buildLabScene(kind, defaultLab); assert.ok(view.gridUnit > 0);
    assert.ok(view.lines.flatMap(l => l.points.flat()).every(Number.isFinite)); assert.ok(view.dots.flatMap(d => d.position).every(Number.isFinite));
  }
  assert.equal(JSON.stringify(defaultLab), original);
});
for (const unit of learningUnits) test(`complete bilingual theory and valid equations: ${unit.id}`, async () => {
  const content = await getLearningContent(unit.id);
  for (const locale of ['ko', 'en'] as const) {
    assert.ok(content[locale].includes('katex-display')); assert.ok(content[locale].includes('<math')); assert.ok(!content[locale].includes('katex-error'));
    assert.ok(content[locale].length > 4000); assert.ok(!/<h1/.test(content[locale]));
    const headings = content.outline[locale]; assert.equal(headings.length, (content[locale].match(/<h2 /g) || []).length);
    assert.equal(new Set(headings.map(h => h.id)).size, headings.length);
    headings.forEach(h => { assert.ok(h.title); assert.ok(content[locale].includes(`id="${h.id}"`)); });
  }
  assert.ok(!/[가-힣]/.test(content.en));
});

test('graduate worked examples agree across arithmetic, coordinate maps and loss changes', () => {
  close(add3(scale3([2, 0, 0], .25), scale3([0, 4, 0], .75)), [.5, 3, 0]);
  const scaling: Mat3 = [[2, 0, 0], [0, .5, 0], [0, 0, 1]]; const rotation: Mat3 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];
  close(transform3(rotation, transform3(scaling, [1, 0, 0])), [0, 2, 0]);
  close(transform3(scaling, transform3(rotation, [1, 0, 0])), [0, .5, 0]);
  close(transform3([[1, 1, 0], [1, -1, 0], [0, 0, 1]], [2, 1, 0]), [3, 1, 0]);
  const points = gradientPath(2, 1, [2, 2], .25, 3);
  close(points.map(p => quadratic(2, 1, p).loss), [6, 2.125, .8828125, .41845703125]);
  close((quadratic(2, 1, [2.1, 2]).loss - 6) / .1, 4.1);
  close((quadratic(2, 1, [2.01, 2]).loss - 6) / .01, 4.01);
  for (const rate of [.25, 1.25]) close(quadratic(2, 1, gradientStep(2, 1, [2, 2], rate)).loss - 6, -20 * rate + 18 * rate ** 2);
  close(quadratic(2, 1, gradientPath(2, 1, [2, 2], .75, 2).at(-1)!).loss, .2578125);
});
