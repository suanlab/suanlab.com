import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { finiteForcePulse, twoParticleCenterOfMass } from '../src/lib/learning-impulse-center-of-mass';
import { getLearningContent } from '../src/lib/learning-content';

for (const [id, columns, equations] of [
  ['finite-force-pulse-impulse-histories', 4, 8],
  ['two-particle-momentum-center-of-mass', 5, 8],
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
      assert.ok(content[lang].includes(`${id.startsWith('finite') ? 'classical-pulse-moments' : 'classical-center-force-identification'}.${lang}.svg`));
    }
  });
}


test('quoted native flat profile reaches the first minimum-effort target', () => {
  const result = finiteForcePulse([1, 5, 0, 20, 20]);
  assert.equal(result.flat.position.label, '1');
  assert.equal(result.flat.velocity.label, '1');
  assert.equal(result.flat.work.label, '1/2');
  assert.equal(result.flat.force.label, '0');
  // The article's quadratic profile is standalone, not the native ramp.
  assert.equal(result.ramp.position.label, '2/3');
});

test('all quoted native force assignments have identical trajectories and the stated work split', () => {
  const base = [1, 2, -10, 10, 0, 0, 10, 0, 0, 10, 0];
  const reference = twoParticleCenterOfMass(base);
  const cases = [
    { forces: [-10, 20, -20], internal: '-3/4', external: '3/2' },
    { forces: [10, 0, 0], internal: '3/4', external: '0' },
    { forces: [30, -20, 20], internal: '9/4', external: '-3/2' },
    { forces: [7, 3, -3], internal: '21/40', external: '9/40' },
  ];
  for (const entry of cases) {
    for (const selected of [0, 1, 2]) {
      const values = [...base.slice(0, 6), ...entry.forces, 10, selected];
      const result = twoParticleCenterOfMass(values);
      assert.deepEqual(result.curves, reference.curves);
      assert.equal(result.centerOfMass.label, '1/3');
      assert.equal(result.kinetic.label, '3/4');
      assert.equal(result.internalWork.label, entry.internal);
      assert.equal(result.externalWork.label, entry.external);
      assert.equal(result.selectedExternalForce.label, ['0', '1', '-1'][selected]);
      assert.equal(result.workResidual.label, '0');
      assert.equal(result.mechanicalResidual.label, '0');
    }
  }
});
