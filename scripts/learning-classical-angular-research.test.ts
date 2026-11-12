import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { fixedOriginAngularMomentum as fixed, centralHarmonicAngularMomentum as central } from '../src/lib/learning-angular-momentum';
import { getLearningContent } from '../src/lib/learning-content';

for (const [id, columns, equations] of [
  ['fixed-origin-angular-momentum-torque', 4, 6],
  ['central-harmonic-angular-momentum-area', 5, 6],
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
      assert.ok(content[lang].includes(`${id.startsWith('fixed') ? 'classical-moving-reference' : 'classical-harmonic-invariant-geometry'}.${lang}.svg`));
    }
  });
}

test('native fixed references retain their lab-momentum convention for the quoted free particle', () => {
  for (const time of [0, 5, 10, 20]) {
    const r = fixed([1, 10, 0, 0, 10, 0, 0, 0, 0, 10, 0, time]);
    assert.equal(r.angularMomentum.label, '1');
    assert.equal(r.shiftedAngularMomentum.label, '0');
    assert.equal(r.torque.label, '0');
    assert.equal(r.angularResidual.label, '0');
  }
});

test('quoted central states distinguish rotated ellipses, direction, circle, radial and stationary motion', () => {
  const cases = [
    { q: [20, 0], v: [0, 10], E: 2.5, L: '2', radial: false, circle: false, rest: false },
    { q: [0, 20], v: [-10, 0], E: 2.5, L: '2', radial: false, circle: false, rest: false },
    { q: [20, 0], v: [0, -10], E: 2.5, L: '-2', radial: false, circle: false, rest: false },
    { q: [10, 10], v: [-10, 10], E: 2, L: '2', radial: false, circle: true, rest: false },
    { q: [10, 0], v: [10, 0], E: 1, L: '0', radial: true, circle: false, rest: false },
    { q: [0, 0], v: [0, 0], E: 0, L: '0', radial: true, circle: false, rest: true },
  ];
  for (const c of cases) {
    const r = central([1, 1, ...c.q, ...c.v, 0, 0, 0]);
    assert.equal(r.initialEnergy, c.E);
    assert.equal(r.initialAngularMomentum.label, c.L);
    assert.equal(r.radialMotion, c.radial);
    assert.equal(r.circularMotion, c.circle);
    assert.equal(r.stationary, c.rest);
  }
  const aboutCenter = central([1, 1, 20, 0, 0, 10, 0, 0, 0]);
  const aboutStart = central([1, 1, 20, 0, 0, 10, 20, 0, 0]);
  assert.equal(aboutStart.query.angularMomentum, 2);
  assert.equal(aboutStart.query.shiftedAngularMomentum, 0);
  assert.deepEqual(aboutCenter.curves[0], aboutStart.curves[0]);
});
