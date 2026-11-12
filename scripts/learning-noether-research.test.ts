import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';
import { noetherPointSymmetryDefects, noetherBoostTimeBoundaries } from '../src/lib/learning-noether-theorem';
for (const [id, equations, image] of [
  ['noether-point-symmetry-defects', 6, 'analytical-noether-rank'],
  ['noether-boost-time-boundaries', 7, 'analytical-noether-boost'],
] as const) {
  test(`Noether research paired proof, table and executable cell: ${id}`, async () => {
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
    assert.deepEqual(rows, id === 'noether-point-symmetry-defects'
      ? [[1, 2, .02, .01], [.1, 2, .2, .01], [.01, 2, 2, .01], [0, 1, null, .01]]
      : [[0, -1, 1, 1], [.5, -.5, 1, 1], [1, 0, 1, 1], [2, 1, 1, 1]]);
    const content = await getLearningContent(id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!content[lang].includes('katex-error'));
      assert.ok(content[lang].includes(`${image}.${lang}.svg`));
      assert.equal(content.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent characteristic rank, error amplification and phase-action identities', () => {
  const result = JSON.parse(execFileSync('python3', ['scripts/learning/verify-noether-research.py', '--json'], { encoding: 'utf8', timeout: 20_000 }));
  assert.deepEqual(result, { characteristicStates: 324, errorBoundChecks: 1296, phaseGeneratorStates: 432, quarticBoostStates: 324, rankDeficientPathStates: 32 });
});

test('native point model keeps independent translation residual projections', () => {
  for (const generator of [0, 1]) {
    const r = noetherPointSymmetryDefects([1, 0, 0, 0, 0, 1, generator, 0, 20]);
    assert.equal(r.globalQuasiSymmetry, true);
    assert.equal(r.query.defect.label, '0');
    assert.equal(r.query.chargeRate.label, r.query.newtonResidual[generator].label);
    assert.equal(r.wholePathSatisfiesDynamics, false);
  }
});

test('native quadratic boost remains quasi-symmetric for a free physical path', () => {
  const r = noetherBoostTimeBoundaries([1, 0, 0, 2, 2, 0, 2, 0, 20]);
  assert.equal(r.globalBoostQuasiSymmetry, true);
  assert.equal(r.wholePathSatisfiesDynamics, true);
  assert.equal(r.wholePathBoostChargeConstant, true);
  assert.equal(r.query.boostCharge.label, '-1');
  assert.equal(r.query.boostIdentity.label, '0');
});
