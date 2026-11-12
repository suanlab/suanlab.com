import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { quarticWorkEnergy, dampedWorkPower } from '../src/lib/learning-work-energy';
import { getLearningContent } from '../src/lib/learning-content';

for (const [id, columns, equations] of [
  ['quartic-potential-work-allowed-states', 3, 8],
  ['damped-oscillator-work-power-heat', 4, 9],
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
      assert.ok(content[lang].includes(`${id.startsWith('quartic') ? 'classical-quartic-separatrix' : 'classical-damped-certificate'}.${lang}.svg`));
    }
  });
}


test('quoted zero-energy states distinguish an outer turn from the stationary origin', () => {
  const turn = quarticWorkEnergy([1, 1, -2, 0, 0, 20, 0]);
  const origin = quarticWorkEnergy([1, 1, -2, 0, 0, 0, 0]);
  assert.deepEqual(turn.intervals, [[-2, 2]]);
  assert.deepEqual(origin.intervals, turn.intervals);
  assert.equal(turn.queryForce.label, '-4');
  assert.equal(turn.isTurning, true);
  assert.equal(turn.stationaryQuery, false);
  assert.equal(origin.stationaryQuery, true);
  assert.equal(origin.queryForce.label, '0');
  assert.equal(quarticWorkEnergy([1, 1, -2, -1, 0, 0, 0]).queryAllowed, false);
  assert.ok(quarticWorkEnergy([1, 1, -2, 1, 0, 0, 0]).speed! > 0);
});

test('native exact flows obey the derived decay certificate and reproduce the critical force-error example', () => {
  for (const [m, k, c] of [[1, 1, 1], [1, 1, 2], [1, 1, 10], [4, 3, 2]]) {
    const eps = c / (2 * m), s11 = 1 + eps * c / k, off = eps * Math.sqrt(m / k);
    const hi = (s11 + 1 + Math.hypot(s11 - 1, 2 * off)) / 2;
    const lo = (1 + eps * (c - eps * m) / k) / hi;
    const gamma = 2 * Math.min(eps, c / m - eps) / hi;
    for (const code of [0, 1, 5, 10, 20, 50, 100]) {
      const result = dampedWorkPower([m, k, c, 0, 10, 0, code]);
      const q = result.query;
      const L = q.energy + eps * m * q.position * q.velocity + eps * c * q.position ** 2 / 2;
      assert.ok(lo * q.energy <= L + 1e-10);
      assert.ok(L <= hi * q.energy + 1e-10);
      assert.ok(q.energy <= hi / lo * Math.exp(-gamma * code / 10) * k / 2 + 1e-10);
    }
  }
  const rest = dampedWorkPower([1, 1, 2, 0, 10, 0, 0]).query;
  assert.equal(rest.energy, .5); assert.equal(rest.energyRate, 0); assert.equal(rest.acceleration, -1);
  for (const code of [0, 1, 5, 10, 20, 50, 100]) {
    const t = code / 10, result = dampedWorkPower([1, 1, 2, 1, 0, 0, code]).query;
    const expected = .1 * (1 - (1 + t) * Math.exp(-t));
    assert.ok(Math.abs(result.position - expected) < 1e-12);
    const lo = 2 - Math.SQRT2, eta = 1 / (2 + Math.SQRT2);
    const bound = Math.sqrt(2 * .01 / (eta * lo) * (-Math.expm1(-eta * t)));
    assert.ok(Math.abs(result.position) <= bound + 1e-12);
  }
});
