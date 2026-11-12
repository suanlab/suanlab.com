import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';
import { routhCyclicMomentumReduction, circleDiracConstraintGeometry } from '../src/lib/learning-constraints-reduction';
for (const [id, equations, image] of [
  ['routh-cyclic-momentum-reduction', 8, 'analytical-constraint-nonholonomic'],
  ['circle-dirac-constraint-geometry', 8, 'analytical-constraint-moving-circle'],
] as const) {
  test(`constraint research paired proof, table and executable cell: ${id}`, async () => {
    const pairs = ['en', 'ko'].map(lang => {
      const source = readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
      const start = source.search(/^## (Research extension:|연구 확장:)/m);
      const tail = source.slice(start);
      const relativeEnd = tail.search(/^## (Exercises with explained solutions|.*연습.*)$/m);
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
    assert.deepEqual(rows, id === 'routh-cyclic-momentum-reduction'
      ? [[0, 1, 0, 1], [.5, .894427, .447214, 1], [1, .707107, .707107, 1], [2, .447214, .894427, 1]]
      : [[0, 1, .625, -.5], [.5, 1.25, .445, -.256], [1, 1.5, .347222, -.148148], [2, 2, .25, -.0625]]);
    const content = await getLearningContent(id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!content[lang].includes('katex-error'));
      assert.ok(content[lang].includes(`${image}.${lang}.svg`));
      assert.equal(content.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent nonholonomic and moving constraint checks cover their declared domains', () => {
  const result = JSON.parse(execFileSync('python3', ['scripts/learning/verify-constraint-research.py', '--json'], { encoding: 'utf8', timeout: 20_000 }));
  assert.deepEqual(result, { admissibleLoopChecks: 48, collapsedCircleRejections: 12, movingCircleExactStates: 972, movingCirclePathStates: 420, nonholonomicPathStates: 405, nonholonomicStates: 405, vakonomicSnapshotChecks: 1215 });
});

test('native Routh reduction still requires its specified fixed canonical momentum', () => {
  const q = routhCyclicMomentumReduction([1, 1, 0, 2, 0, 10, 2, 2]).query!;
  assert.equal(q.canonicalAngularMomentum.label, '1');
  assert.equal(q.canonicalAngularMomentumRate.label, '0');
  assert.equal(q.trueRadialForce.label, '0');
  assert.equal(q.naiveRadialForce.label, '-2');
});

test('native static circle does not silently reinterpret radial momentum as boundary motion', () => {
  const r = circleDiracConstraintGeometry([1, 2, 2, 0, 1, 2, 0, 0]);
  assert.equal(r.regularDiracBracket, true);
  assert.equal(r.onPhysicalConstraintSurface, false);
  assert.equal(r.constraints[1].label, '1/2');
  assert.equal(r.geometry!.physicalReactionPower, null);
  assert.equal(r.geometry!.hamiltonianRate.label, '0');
});
