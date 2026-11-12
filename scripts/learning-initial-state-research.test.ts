import assert from 'node:assert/strict';
import test from 'node:test';
import { invertedSpringInitialState as spring, delayedNewtonRelease as release } from '../src/lib/learning-initial-state-role';
import { readFileSync } from 'node:fs';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';

for (const [id, equations, columns, image] of [
  ['inverted-spring-initial-state-sensitivity', 5, 3, 'classical-initial-observation'],
  ['nonlipschitz-newton-delayed-release', 7, 4, 'classical-initial-selection'],
] as const) {
  test(`paired initial-state research equations, executable cell, table and real renderer: ${id}`, async () => {
    const pairs = ['en', 'ko'].map(lang => {
      const source = readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
      const start = source.search(/^## (Research extension:|연구 확장:)/m);
      const end = source.indexOf(lang === 'en' ? '## Exercises with explained solutions' : '## 해설이 있는 연습 문제', start);
      assert.ok(start >= 0 && end > start);
      const tree = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(source.slice(start, end));
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
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!content[lang].includes('katex-error'));
      assert.equal(content.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
      assert.ok(content[lang].includes(`${image}.${lang}.svg`));
    }
  });
}

test('two quoted native position observations recover the hidden initial velocities and predict a third position', () => {
  for (const gap of [1, 2]) {
    const observed = spring([1, 1, 10, -10, 10, 10, 10 * gap]);
    const future = spring([1, 1, 10, -10, 10, 10, 10 * (gap + 1)]);
    for (const [last, target, actualVelocity] of [
      [observed.queryA.position, future.queryA.position, -1],
      [observed.queryB.position, future.queryB.position, 1],
    ]) {
      const velocity = (last - Math.cosh(gap)) / Math.sinh(gap);
      assert.ok(Math.abs(velocity - actualVelocity) < 2e-13);
      const predicted = (-Math.sinh(1) + Math.sinh(gap + 1) * last) / Math.sinh(gap);
      assert.ok(Math.abs(predicted - target) < 2e-12);
      for (const firstError of [-.001, .001]) for (const lastError of [-.001, .001]) {
        const noisy = ((last + lastError) - (1 + firstError) * Math.cosh(gap)) / Math.sinh(gap);
        const bound = .001 * (1 + Math.cosh(gap)) / Math.sinh(gap);
        assert.ok(Math.abs(noisy - actualVelocity) <= bound + 2e-13);
      }
    }
  }
});

test('the actual zero-stiffness controls use the finite-difference observation inverse', () => {
  const r = spring([2, 0, 10, 5, 10, -5, 20]);
  assert.equal(r.freeLimit, true);
  assert.equal(r.queryA.position, 2);
  assert.equal(r.queryB.position, 0);
  assert.equal((r.queryA.position - 1) / 2, .5);
  assert.equal((r.queryB.position - 1) / 2, -.5);
});

test('native zero-state branches remain original solutions rather than implicit regularization controls', () => {
  for (const [time, immediate, delayed] of [[10, '1', '0'], [20, '16', '1']] as const) {
    const r = release([1, 2, 0, 10, time]);
    assert.equal(r.queryA.position.label, immediate);
    assert.equal(r.queryB.position.label, delayed);
    assert.equal(r.stationary.position.label, '0');
    assert.equal(r.sameCompleteInitialState, true);
    assert.equal(r.nonuniqueConstructed, true);
    assert.equal(r.queryA.newtonResidual.label, '0');
    assert.equal(r.queryB.energyResidual.label, '0');
  }
  const zero = release([1, 0, 0, 10, 20]);
  assert.equal(zero.nonuniqueConstructed, false);
  assert.equal(zero.queryA.position.label, '0');
  assert.equal(zero.queryB.position.label, '0');
});
