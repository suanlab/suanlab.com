import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';
import { uniformForceHamiltonJacobiFamily, oscillatorHamiltonJacobiCausticAtlas } from '../src/lib/learning-hamilton-jacobi';
for (const [id, equations, image] of [
  ['uniform-force-hamilton-jacobi-family', 8, 'analytical-hj-residual-gradients'],
  ['oscillator-hamilton-jacobi-caustic-atlas', 9, 'analytical-hj-value-selection'],
] as const) {
  test(`Hamilton-Jacobi research paired proof, table and executable cell: ${id}`, async () => {
    const pairs = ['en', 'ko'].map(lang => {
      const source = readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
      assert.doesNotMatch(source, /[\u0000-\u0008\u000b-\u001f]/);
      const start = source.search(/^## (Research extension:|연구 확장:)/m);
      const tail = source.slice(start);
      const relativeEnd = tail.search(/^## (Work through six checks with explanations|해설과 함께 여섯 가지를 확인한다)$/m);
      const end = relativeEnd < 0 ? -1 : start+relativeEnd;
      assert.ok(start >= 0 && end > start);
      const tree = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(source.slice(start, end));
      const rejectAccidentalEmphasis = (node: { type: string; children?: { type: string; children?: any[] }[] }): void => {
        assert.notEqual(node.type, 'emphasis', 'Starred reference symbols must use inline math rather than Markdown emphasis.');
        node.children?.forEach(rejectAccidentalEmphasis);
      };
      rejectAccidentalEmphasis(tree);
      const math = tree.children.flatMap(node => node.type === 'math' ? [node.value] : []);
      const python = tree.children.flatMap(node => node.type === 'code' && node.lang === 'python' ? [node.value] : []);
      const tables = tree.children.filter(node => node.type === 'table');
      assert.equal(math.length, equations);
      assert.equal(python.length, 1);
      assert.equal(tables.length, 1);
      for (const row of tables[0].children) assert.equal(row.children.length, 4);
      if (lang === 'en') assert.doesNotMatch(source, /[가-힣]/);
      return { math, python, stdout: execFileSync('python3', ['-c', python[0]], { encoding: 'utf8', timeout: 20_000 }) };
    });
    assert.deepEqual(pairs[0], pairs[1]);
    const rows = pairs[0].stdout.trim().split('\n').map(line => line.split(/\s+/).map(value => value === 'None' ? null : Number(value)));
    assert.deepEqual(rows, id === 'uniform-force-hamilton-jacobi-family'
      ? [[0, .63662, .405285, -1], [1, .127324, .016211, -1], [4, .037448, .001402, -1], [16, .009794, .000096, -1]]
      : [[.5, 1, 1, 0], [1, 1, 1, 0], [1.5, 3, 2, -.027778], [2, 3, 2, -.0625]]);
    const content = await getLearningContent(id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!content[lang].includes('katex-error'));
      assert.ok(content[lang].includes(`${image}.${lang}.svg`));
      assert.equal(content.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent derivative residuals and global-minimum checks cover the stated families', () => {
  const result = JSON.parse(execFileSync('python3', ['scripts/learning/verify-hj-research.py', '--json'], { encoding: 'utf8', timeout: 20_000 }));
  assert.deepEqual(result, { characteristicIdentityStates: 2496, differentiablePdeChecks: 648, hopfLaxStates: 810, oscillatoryStates: 6561, shockTests: 108, singularMixedHessianCases: 96, smallResidualCounterexamples: 567 });
});

test('native uniform-force integral retains both phase equations and the action increment', () => {
  const r = uniformForceHamiltonJacobiFamily([1, 2, 2, 0, 0, 0, 20, 0]);
  assert.equal(r.globalHamiltonJacobiSolution, true);
  assert.equal(r.wholeFrozenPathPhysical, true);
  assert.equal(r.query.position.label, '3/2');
  assert.equal(r.query.gradientMomentum.label, '2');
  assert.equal(r.query.mechanicalAction.label, '11/6');
});

test('native oscillator chart loss leaves the smooth physical phase state available', () => {
  const q = oscillatorHamiltonJacobiCausticAtlas([1, 1, 2, 2, 1, 0, 2, 4]).query;
  assert.equal(q.position.label, '1');
  assert.equal(q.momentum.label, '-1');
  assert.equal(q.energy.label, '1');
  assert.equal(q.mixedChart, null);
  assert.equal(q.endpointChartRegular, true);
});
