import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';
import { freeParticleActionBoundaryGauge, harmonicActionStationarySaddle } from '../src/lib/learning-stationary-action';

for (const [id, equations, image] of [
  ['free-particle-action-boundary-gauge', 6, 'analytical-terminal-cost'],
  ['harmonic-action-stationary-saddle', 7, 'analytical-action-index'],
] as const) {
  test(`action research: paired proofs, four-column table and executable cell: ${id}`, async () => {
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

test('independent action proof checks include soft endpoints, complete signatures and conjugate data', () => {
  const stdout = execFileSync('python3', ['scripts/learning/verify-action-research.py'], { encoding: 'utf8', timeout: 20_000 });
  assert.match(stdout, /1152 exact free-endpoint paths passed/);
  assert.match(stdout, /117 exact Morse signatures and 162 independent quadratic-form integrals passed/);
  assert.match(stdout, /36 critical endpoint cases and 40 near-critical endpoint perturbations passed/);
});

test('native free-particle first variation distinguishes a changed terminal position from fixed endpoints', () => {
  const controls = [1, 10, 0, 10, 0, 0, 0, 0, 10, 0, 0, 50, 50];
  const base = freeParticleActionBoundaryGauge(controls);
  const gauged = freeParticleActionBoundaryGauge(controls.map((x, i) => i === 6 ? -20 : x));
  assert.equal(base.firstVariation.label, '1');
  assert.equal(gauged.firstVariation.label, '-1');
  assert.equal(gauged.stationaryForFixedEndpoints, true);
  assert.equal(gauged.directionFixesEndpoints, false);
  assert.equal(gauged.residualMeanSquare.label, '0');
});

test('native first/third-mode view remains a saddle when the full space has an omitted null or negative second mode', () => {
  for (const durationCode of [20, 25]) {
    const r = harmonicActionStationarySaddle([1, 4, durationCode, 0, 0, 10, 0, 50, 50]);
    assert.equal(r.zeroReferenceNature, 'saddle');
    assert.equal(r.selectedPathStationary, true);
    assert.ok(r.hessianAA.value < 0);
    assert.ok(r.hessianBB.value > 0);
    assert.equal(r.d1, 1600 - (4 * durationCode) ** 2);
  }
});
