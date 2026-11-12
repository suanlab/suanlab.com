import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { isolatedCentralPair as pair, gyroscopicConservation as gyro } from '../src/lib/learning-conservation-laws';
import { getLearningContent } from '../src/lib/learning-content';

for (const [id, columns, equations] of [
  ['isolated-central-pair-conservation', 4, 6],
  ['gyroscopic-force-energy-momentum', 3, 6],
] as const) {
  test(`paired research equations, code, table structure and real renderer: ${id}`, async () => {
    const pairs = ['en', 'ko'].map(lang => {
      const raw = readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
      const start = raw.search(/^## (Research extension:|연구 확장:)/m);
      const end = raw.indexOf(lang === 'en' ? '## Six exercises with worked solutions' : '## 풀이가 있는 여섯 연습', start);
      assert.ok(start >= 0 && end > start);
      const tree = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(raw.slice(start, end));
      const tables = tree.children.filter(node => node.type === 'table');
      assert.equal(tables.length, 1);
      for (const row of tables[0].children) assert.equal(row.children.length, columns);
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
      assert.ok(content[lang].includes(`${id.startsWith('isolated') ? 'classical-parametric-pair-work' : 'classical-midpoint-gyro-phase'}.${lang}.svg`));
    }
  });
}

test('quoted pair states use constant stiffness in the native evaluator and retain distinct energy budgets', () => {
  for (const code of [0, 1, 10, 24, 50, 100]) {
    for (const vy of [0, 5]) {
      const r = pair([2, 2, 4, 5, 0, -5, 0, 0, vy, 0, -vy, 0, 0, code]);
      assert.equal(r.reducedMass, 1);
      assert.equal(r.omega, 2);
      assert.equal(r.initialEnergy.label, vy === 0 ? '2' : '5/2');
      assert.equal(r.initialAngularMomentum.label, vy === 0 ? '0' : '1');
      assert.ok(Math.abs(r.query.energy - (vy === 0 ? 2 : 2.5)) < 2e-12);
      assert.ok(Math.abs(r.query.angularMomentum - (vy === 0 ? 0 : 1)) < 2e-12);
    }
  }
  const high = pair([2, 2, 4, 5, 0, -5, 0, 0, 0, 0, 0, 0, 0, 0]);
  const low = pair([2, 2, 1, 5, 0, -5, 0, 0, 0, 0, 0, 0, 0, 0]);
  assert.deepEqual(high.query.position1, low.query.position1);
  assert.deepEqual(high.query.velocity1, low.query.velocity1);
  assert.equal(low.initialEnergy.value-high.initialEnergy.value, -1.5);
});

test('native analytic eight-second states expose the midpoint return error despite matching invariants', () => {
  for (const sign of [-1, 1]) {
    const r = gyro([1, sign*10, 0, 0, 10, 0, 80]);
    assert.ok(Math.abs(r.query.position[0]-Math.sin(8)) < 1e-14);
    assert.ok(Math.abs(r.query.position[1]-sign*2*Math.sin(4)**2) < 1e-14);
    assert.ok(Math.abs(Math.hypot(...r.query.position)-2*Math.abs(Math.sin(4))) < 1e-14);
    assert.equal(r.query.kinetic, .5);
    assert.ok(Math.abs(r.query.modifiedAngularMomentum) < 1e-14);
    assert.deepEqual(r.exactGuidingCenter!.map(x=>x.label), ['0', String(sign)]);
    assert.ok(Math.abs(r.query.velocity[0]-Math.cos(8)) < 1e-14);
    assert.ok(Math.abs(r.query.velocity[1]-sign*Math.sin(8)) < 1e-14);
  }
});
