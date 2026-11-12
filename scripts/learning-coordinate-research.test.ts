import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';
import { polarGeneralizedCoordinates, doublePendulumGeneralizedMetric } from '../src/lib/learning-generalized-coordinates';

for (const [id, equations, image] of [
  ['polar-generalized-velocity-force-metric', 6, 'analytical-polar-hessian'],
  ['double-pendulum-generalized-mass-metric', 7, 'analytical-pendulum-modes'],
] as const) {
  test(`coordinate research: paired proofs, table and executable cell render correctly: ${id}`, async () => {
    const pairs = ['en', 'ko'].map(lang => {
      const source = readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
      const start = source.search(/^## (Research extension:|연구 확장:)/m);
      const end = source.indexOf(lang === 'en' ? '## Exercises with explained solutions' : '## 해설이 있는 연습 문제', start);
      assert.ok(start >= 0 && end > start);
      const tree = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(source.slice(start, end));
      const math = tree.children.flatMap(node => node.type === 'math' ? [node.value] : []);
      const python = tree.children.flatMap(node => node.type === 'code' && node.lang === 'python' ? [node.value] : []);
      const tables = tree.children.filter(node => node.type === 'table');
      assert.equal(math.length, equations);
      assert.equal(python.length, 1);
      assert.equal(tables.length, 1);
      for (const row of tables[0].children) assert.equal(row.children.length, 3);
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

test('independent coordinate research checks composed Hessians, generalized spectra and finite-time nonlinear errors', () => {
  const stdout = execFileSync('python3', ['scripts/learning/verify-coordinate-research.py'], { encoding: 'utf8', timeout: 20_000 });
  assert.match(stdout, /216 polar Hessian states passed/);
  assert.match(stdout, /54 exact mass\/stiffness coordinate pairs passed/);
  assert.match(stdout, /6 nonlinear trajectories passed with 6 step-halving comparisons/);
});

test('native polar geometry evaluates the force at the non-equilibrium negative-curvature point', () => {
  const r = polarGeneralizedCoordinates([1, 2, 12, 0, 0, 20, 0, 0, 0, 0]);
  assert.equal(r.localChartRegular, true);
  assert.deepEqual(r.cartesianAcceleration, [2, 0]);
  assert.ok(Math.abs(r.generalizedAcceleration![0] + 2) < 1e-14);
  assert.ok(Math.abs(r.generalizedAcceleration![1]) < 1e-14);
});

test('native double-pendulum metric matches both coordinate bases of the research equilibrium', () => {
  const r = doublePendulumGeneralizedMetric([2, 1, 2, 2, 0, 0, 10, -10, 0, 0, 10, -10, 50]);
  assert.deepEqual(r.metricAbsolute, [[3, 1], [1, 1]]);
  assert.deepEqual(r.metricRelative, [[6, 2], [2, 1]]);
  assert.equal(r.kineticCartesian, 1);
  assert.equal(r.kineticRelative, 1);
  assert.deepEqual(r.relativeRates, [1, -2]);
});
