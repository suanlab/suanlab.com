import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';
for (const [id, equations, image] of [
  ['four-dimensional-symplectic-maps', 7, 'analytical-symplectic-dimension'],
] as const) {
  test(`symplectic dimension research paired proof, table and executable cell: ${id}`, async () => {
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

test('independent full pushforward, Jacobi triples and invariant density checks', () => {
  const result = JSON.parse(execFileSync('python3', ['scripts/learning/verify-symplectic-dimension-research.py', '--json'], { encoding: 'utf8', timeout: 20_000 }));
  assert.deepEqual(result, { energyChecks: 810, hiddenPairProbes: 3, pushforwardStates: 405, tensorStates: 405, weightedDivergenceChecks: 810 });
});
