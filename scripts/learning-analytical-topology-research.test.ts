import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';
import { lagrangianMomentumGraph } from '../src/lib/learning-symplectic-geometry';
import { timeGaugeHamiltonianEnergy } from '../src/lib/learning-hamiltonian-canonical';

for (const [id, columns, image] of [
  ['lagrangian-momentum-graphs', 3, 'analytical-topology-period'],
  ['time-gauge-hamiltonian-energy', 4, 'analytical-circle-gauge'],
] as const) {
  test(`global-action companion: paired math, table, executable example and real renderer: ${id}`, async () => {
    const cells = ['en', 'ko'].map(lang => {
      const source = readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
      const start = source.search(/^## (Research extension:|연구 확장:)/m);
      const end = source.indexOf(lang === 'en' ? '## Exercises with explanations' : '## 해설이 있는 연습 문제', start);
      assert.ok(start >= 0 && end > start);
      const tree = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(source.slice(start, end));
      const math = tree.children.flatMap(node => node.type === 'math' ? [node.value] : []);
      const python = tree.children.flatMap(node => node.type === 'code' && node.lang === 'python' ? [node.value] : []);
      const tables = tree.children.filter(node => node.type === 'table');
      assert.equal(math.length, 5);
      assert.equal(python.length, 1);
      assert.equal(tables.length, 1);
      for (const row of tables[0].children) assert.equal(row.children.length, columns);
      if (lang === 'en') assert.doesNotMatch(source, /[가-힣]/);
      const stdout = execFileSync('python3', ['-c', python[0]], { encoding: 'utf8', timeout: 20_000 });
      return { math, python, stdout };
    });
    assert.deepEqual(cells[0], cells[1]);
    const content = await getLearningContent(id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!content[lang].includes('katex-error'));
      assert.ok(content[lang].includes(`${image}.${lang}.svg`));
      assert.equal(content.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent periods and Hamiltonian tangency include both signs, repeated winding and the zero sector', () => {
  const stdout = execFileSync('python3', ['scripts/learning/verify-analytical-topology.py'], { encoding: 'utf8', timeout: 20_000 });
  assert.match(stdout, /76 loop periods passed/);
  assert.match(stdout, /48 Hamiltonian tangency states/);
  assert.match(stdout, /36 exact circle states passed/);
});

test('native constant-matrix experiment distinguishes exact indefinite graphs from nonclosed graphs', () => {
  const exact = lagrangianMomentumGraph([2, -2, 2, 0, 1, -1, 3, 2, 0, 0]);
  assert.equal(exact.isLagrangianGraph, true);
  assert.equal(exact.symmetricActionSignature, 'indefinite');
  assert.equal(exact.closedLoopAction.label, '0');
  const nonclosed = lagrangianMomentumGraph([0, 0, 0, 2, 0, 0, 2, 2, 0, 80]);
  assert.equal(nonclosed.isLagrangianGraph, false);
  assert.equal(nonclosed.closedLoopAction.label, '-2');
});

test('native time-dependent gauge keeps physical motion while changing canonical energy', () => {
  const gauged = timeGaugeHamiltonianEnergy([1, 0, 0, 2, 4, 20]).query;
  const base = timeGaugeHamiltonianEnergy([1, 0, 0, 2, 0, 20]).query;
  assert.equal(gauged.position, base.position);
  assert.equal(gauged.velocity, base.velocity);
  assert.equal(gauged.canonicalMomentum, 3);
  assert.equal(base.canonicalMomentum, 1);
  assert.equal(gauged.physicalEnergy, .5);
  assert.equal(gauged.hamiltonian, -.5);
  assert.equal(base.hamiltonian, .5);
});
