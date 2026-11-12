import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';
import { polarHamiltonianLegendreMap } from '../src/lib/learning-hamiltonian-canonical';
import { poissonObservableCanonicalFlow, angularMomentumPoissonAlgebra } from '../src/lib/learning-poisson-brackets';

for (const [id, equations, image] of [
  ['polar-hamiltonian-legendre-map', 6, 'analytical-canonical-dual'],
  ['poisson-observable-canonical-flow', 6, 'analytical-poisson-clock'],
  ['angular-momentum-poisson-algebra', 7, 'analytical-angular-closure'],
] as const) {
  test(`canonical research paired proof, table and executable cell: ${id}`, async () => {
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
      return { math, python, stdout: execFileSync('python3', ['-c', python[0]], { encoding: 'utf8', timeout: 20_000 }) };
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

test('independent nonlinear inversion, clock and reduction checks cover all stated examples', () => {
  const result = JSON.parse(execFileSync('python3', ['scripts/learning/verify-canonical-research.py', '--json'], { encoding: 'utf8', timeout: 20_000 }));
  assert.deepEqual(result, { legendreStates: 270, nonlinearChartStates: 108, canonicalAngularStates: 729, clockExamples: 5, sameAngularAndEnergyCounterexamples: 2, zeroLeafCounterexample: 1 });
});

test('native quadratic branch retains its velocity-eliminated Hamiltonian', () => {
  const r = polarHamiltonianLegendreMap([1, 0, 2, 0, 2, 2, 20]);
  assert.ok(r.query.canonical);
  assert.equal(r.query.physicalEnergy, 1);
  assert.equal(r.query.canonical.hamiltonian, 1);
  assert.equal(r.query.canonical.angularMomentum, 1);
  assert.ok(Math.abs(r.query.canonical.legendreDiscrepancy) < 1e-12);
});

test('native linear bracket example still needs its induced tensor for physical rates', () => {
  const q = poissonObservableCanonicalFlow([1, 0, 2, 2, 2, 0, -2, 20]).query;
  assert.equal(q.badFundamentalBracket, 4);
  assert.equal(q.physicalQRate, 2);
  assert.equal(q.badNaiveQRate, .5);
  assert.equal(q.badTensorCorrectedBracket, q.poisson);
});

test('native full oscillator can leave the rank-zero angular point', () => {
  const r = angularMomentumPoissonAlgebra([1, 0, 1, 2, 1, 3, 0, 0]);
  assert.ok(r.query.angularMomentum.every(x => x === 0));
  assert.ok(r.query.bracketMatrix.flat().every(x => x === 0));
  assert.ok(Math.hypot(...r.query.torque) > 0);
  assert.ok(r.curves[0].some(([, value]) => Math.abs(value) > .1));
});
