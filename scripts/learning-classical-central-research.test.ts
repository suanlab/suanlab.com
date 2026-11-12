import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { inverseSquareReduction as reduce, ellipticKeplerMotion as orbit, solveEllipticKepler as solve } from '../src/lib/learning-central-force';
import { getLearningContent } from '../src/lib/learning-content';

for (const [id, columns, equations] of [
  ['inverse-square-effective-potential', [4], 6],
  ['elliptic-kepler-time-area', [3], 6],
] as const) {
  test(`paired research equations, code, table structure and real renderer: ${id}`, async () => {
    const pairs = ['en', 'ko'].map(lang => {
      const raw = readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
      const start = raw.search(/^## (Research extension:|연구 확장:)/m);
      const end = raw.indexOf(lang === 'en' ? '## Worked examples and exercises with solutions' : '## 예제와 해설이 있는 연습문제', start);
      assert.ok(start >= 0 && end > start);
      const tree = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(raw.slice(start, end));
      const tables = tree.children.filter(node => node.type === 'table');
      assert.equal(tables.length, columns.length);
      for (const [i, table] of tables.entries()) for (const row of table.children) assert.equal(row.children.length, columns[i]);
      const math = tree.children.flatMap(node => node.type === 'math' ? [node.value] : []);
      const code = tree.children.flatMap(node => node.type === 'code' && node.lang === 'python' ? [node.value] : []);
      assert.equal(math.length, equations);
      assert.equal(code.length, 1);
      return { math, code };
    });
    assert.deepEqual(pairs[0], pairs[1]);
    const content = await getLearningContent(id);
    for (const lang of ['ko', 'en'] as const) {
      assert.ok(!content[lang].includes('katex-error'));
      assert.equal(content.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
      assert.ok(content[lang].includes(`${id.startsWith('inverse') ? 'classical-apsis-uncertainty' : 'classical-kepler-passage-limit'}.${lang}.svg`));
    }
  });
}

test('quoted native speed anchors retain exact energy signs and distinct finite/unbound geometry', () => {
  for (const [speed, energy, conic] of [[19, '-39/200', 'ellipse'], [20, '0', 'parabola'], [21, '41/200', 'hyperbola']] as const) {
    const r = reduce([2, 2, 0, speed, 20, 1]);
    assert.equal(r.specificEnergy.label, energy);
    assert.equal(r.conic, conic);
    assert.equal(r.radial, false);
    assert.equal(r.eccentricityIdentityResidual.label, '0');
    assert.ok(Math.abs(r.periapsis!-1) < 1e-14);
    if (speed === 19) assert.ok(Math.abs(r.apoapsis!-361/39) < 1e-12);
    else assert.equal(r.apoapsis, null);
    if (speed === 20) {
      assert.deepEqual(r.query.radialVelocityBranches, [-1, 1]);
      assert.equal(r.query.radialSpeed, 1);
      assert.equal(r.query.tangentialVelocity.label, '1');
    }
  }
});

test('native elliptic clock brackets the independent two-metre passage and reproduces its special root', () => {
  for (const retrograde of [0, 1]) {
    const before = orbit([2, 4, 10, 0, retrograde, 21, 1]);
    const after = orbit([2, 4, 10, 0, retrograde, 22, 1]);
    assert.ok(before.query.radius < 2 && after.query.radius > 2);
    assert.ok(before.time < Math.PI-1 && Math.PI-1 < after.time);
    assert.ok(Math.abs(before.period-4*Math.PI) < 1e-14);
    assert.equal(before.periapsis, 1);
    assert.equal(before.apoapsis, 3);
    assert.ok(Math.abs(before.query.totalEnergy+.5) < 1e-14);
  }
  const root = solve((Math.PI-1)/2, .5);
  assert.ok(Math.abs(root.eccentricAnomaly-Math.PI/2) < 2e-14);
  assert.ok(Math.abs(root.residual) < 1e-14);
});
