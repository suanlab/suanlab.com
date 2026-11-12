import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { evaluateFoundation } from '../src/lib/learning-foundations';
import { getLearningContent } from '../src/lib/learning-content';

const close = (actual: number, expected: number, tolerance = 2e-10) => {
  assert.ok(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);
};

// Closed matrix-power solution; does not rerun the solver's step loop.
for (const [h, k, q0] of [[.1, 1, 1], [.1, 4, 1], [.5, 4, 1], [.9, 4, 1], [.2, 3, -2]]) {
  test(`oscillator matrix powers, physical energy and coercive bounds: ${h},${k},${q0}`, () => {
    const result = evaluateFoundation('mechanics-energy', [h, k, q0]);
    const z = h * Math.sqrt(k), theta = 2 * Math.asin(z / 2);
    const initial = k * q0 * q0 / 2;
    const energies: number[] = [];
    for (let n = 0; n <= 100; n++) {
      const q = q0 * (Math.cos(n * theta) - z * z / (2 * Math.sin(theta)) * Math.sin(n * theta));
      const v = -q0 * h * k * Math.sin(n * theta) / Math.sin(theta);
      const energy = (v * v + k * q * q) / 2;
      close(result.curves[1][n][1], q);
      close((v * v + k * q * q - h * k * q * v) / 2, initial);
      assert.ok(energy >= initial / (1 + z / 2) - 1e-10);
      assert.ok(energy <= initial / (1 - z / 2) + 1e-10);
      energies.push(energy);
      if (n === 100) {
        const state = result.stats['q(T), v(T)'] as number[];
        close(state[0], q); close(state[1], v);
        close(result.stats['E(T) [J]'] as number, energy);
      }
    }
    close(result.stats['max |E−E₀| [J]'] as number, Math.max(...energies.map(e => Math.abs(e - initial))));
    assert.equal(result.warning, undefined);
  });
}

test('defective unit-modulus boundary has quadratic physical-energy growth and a constant degenerate form', () => {
  const result = evaluateFoundation('mechanics-energy', [1, 4, 1]);
  assert.equal(result.warning, 'stability');
  for (let n = 0; n <= 100; n++) {
    const q = (-1) ** n * (1 + 2 * n), v = (-1) ** n * 4 * n;
    assert.equal(result.curves[1][n][1], q);
    assert.equal((v - 2 * q) ** 2 / 2, 2);
    assert.equal((v * v + 4 * q * q) / 2, 16 * n * n + 8 * n + 2);
  }
  assert.deepEqual(result.stats['q(T), v(T)'], [201, 400]);
  assert.equal(result.stats['E(T) [J]'], 160802);
  assert.equal(result.stats['max |E−E₀| [J]'], 160800);
});

test('zero initial state stays zero without defining an energy ratio', () => {
  const result = evaluateFoundation('mechanics-energy', [.5, 4, 0]);
  assert.ok(result.curves.flat().every(point => point[1] === 0));
  assert.equal(result.stats['E₀ [J]'], 0);
  assert.equal(result.stats['E(T) [J]'], 0);
});

test('paired extension has identical equations, executable example and a valid four-column table', async () => {
  const sections = ['ko', 'en'].map(lang => {
    const raw = readFileSync(`content/learning/physics-math-for-ai/mechanics-energy.${lang}.md`, 'utf8');
    const start = raw.search(/^## (Research extension:|연구 확장:)/m);
    const end = raw.indexOf(lang === 'ko' ? '## 연습과 근거' : '## Exercises and reference', start);
    assert.ok(start >= 0 && end > start);
    const section = raw.slice(start, end);
    const tree = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(section);
    const tables = tree.children.filter(node => node.type === 'table');
    assert.equal(tables.length, 1);
    for (const row of tables[0].children) assert.equal(row.children.length, 4);
    return {
      equations: tree.children.filter(node => node.type === 'math').map(node => node.value),
      code: tree.children.flatMap(node => node.type === 'code' && node.lang === 'python' ? [node.value] : []),
    };
  });
  assert.equal(sections[0].equations.length, 6);
  assert.equal(sections[0].code.length, 1);
  assert.deepEqual(sections[0], sections[1]);
  const rendered = await getLearningContent('mechanics-energy');
  for (const lang of ['ko', 'en'] as const) {
    assert.ok(!rendered[lang].includes('katex-error'));
    assert.ok(rendered[lang].includes(`classical-energy-invariant.${lang}.svg`));
    assert.equal(rendered.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
  }
});
