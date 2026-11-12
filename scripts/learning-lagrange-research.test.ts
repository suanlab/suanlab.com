import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';
import { polarLagrangeNewtonEquivalence, doublePendulumLagrangeDynamics } from '../src/lib/learning-lagrange-equations';
const close = (a: number, b: number, tolerance = 1e-9) => assert.ok(Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(a), Math.abs(b)), `${a} vs ${b}`);

for (const [id, equations, image] of [
  ['polar-lagrange-newton-equivalence', 7, 'analytical-lagrange-kinetic'],
  ['double-pendulum-lagrange-dynamics', 6, 'analytical-lagrange-scale'],
] as const) {
  test(`Lagrange research: paired proofs, table and executable inference example: ${id}`, async () => {
    const pairs = ['en', 'ko'].map(lang => {
      const source = readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
      const start = source.search(/^## (Research extension:|연구 확장:)/m);
      const end = source.indexOf(lang === 'en' ? '## Exercises with explanations' : '## 해설이 있는 연습 문제', start);
      assert.ok(start >= 0 && end > start);
      const tree = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(source.slice(start, end));
      const math = tree.children.flatMap(node => node.type === 'math' ? [node.value] : []);
      const python = tree.children.flatMap(node => node.type === 'code' && node.lang === 'python' ? [node.value] : []);
      const tables = tree.children.filter(node => node.type === 'table');
      assert.equal(math.length, equations);
      assert.equal(python.length, 1);
      assert.equal(tables.length, 1);
      for (const row of tables[0].children) assert.equal(row.children.length, 4);
      if (lang === 'en') assert.doesNotMatch(source, /[가-힣]/);
      const stdout = execFileSync('python3', ['-c', python[0]], { encoding: 'utf8', timeout: 20_000 });
      return { math, python, stdout };
    });
    assert.deepEqual(pairs[0], pairs[1]);
    const content = await getLearningContent(id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!content[lang].includes('katex-error'));
      assert.ok(content[lang].includes(`${image}.${lang}.svg`));
      assert.equal(content.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent kinetic/Cartesian projection checks and 402 native RK4 sample states agree', () => {
  const reference = JSON.parse(execFileSync('python3', ['scripts/learning/verify-lagrange-research.py', '--json'], { encoding: 'utf8', timeout: 20_000 })) as {
    kineticStates: number; boostChecks: number; massScaleStates: number; forcedScaleStates: number;
    trajectoryStatesPerScale: number; maximumScaleGap: number;
    trajectories: { scale: number; states: number[][] }[];
  };
  assert.equal(reference.kineticStates, 72);
  assert.equal(reference.boostChecks, 12);
  assert.equal(reference.massScaleStates, 216);
  assert.equal(reference.forcedScaleStates, 432);
  assert.equal(reference.trajectoryStatesPerScale, 201);
  assert.ok(reference.maximumScaleGap < 5e-12);
  let compared = 0;
  for (const row of reference.trajectories.filter(row => row.scale === 1 || row.scale === 2)) {
    const r = doublePendulumLagrangeDynamics([row.scale, row.scale, 2, 2, 6, 0, 10, -10, 20, 2]);
    assert.equal(row.states.length, 201);
    row.states.forEach((state, j) => {
      for (let k = 0; k < 4; k++) close(r.curves[k][j][1], state[k], 1e-8);
      compared++;
    });
  }
  assert.equal(compared, 402);
});

test('native free branch supplies the timed Cartesian path used by the alternative kinetic candidates', () => {
  const r = polarLagrangeNewtonEquivalence([1, 0, 2, 0, 2, 2, 20]);
  assert.equal(r.freeLimit, true);
  assert.deepEqual(r.query.position, [2, 1]);
  assert.deepEqual(r.query.velocity, [1, 1]);
  assert.ok(r.query.acceleration.every(x => x === 0));
  close(r.query.kinetic, 1);
});

test('three native mass-scale pairs preserve 603 timed states while energies and rod forces scale', () => {
  let compared = 0;
  for (const controls of [
    [1, 1, 2, 2, 6, 0, 10, -10, 20, 2],
    [1, 2, 2, 3, 3, -4, 5, 3, 30, 2],
    [2, 1, 3, 2, -4, 5, -3, 7, 40, 2],
  ]) {
    const a = doublePendulumLagrangeDynamics(controls);
    const b = doublePendulumLagrangeDynamics(controls.map((x, i) => i < 2 ? 2 * x : x));
    for (let j = 0; j < 201; j++) {
      for (let k = 0; k < 4; k++) close(a.curves[k][j][1], b.curves[k][j][1]);
      compared++;
    }
    for (const key of ['initial', 'query', 'final'] as const) {
      close(b[key].energy, 2 * a[key].energy);
      for (const field of ['rod1Force', 'rod2Force'] as const) {
        for (let j = 0; j < 2; j++) close(b[key][field][j], 2 * a[key][field][j]);
      }
    }
  }
  assert.equal(compared, 603);
});
