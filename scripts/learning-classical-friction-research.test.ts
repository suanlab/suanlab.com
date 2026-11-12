import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { dryFrictionMotion as dry, circularGuideMotion as guide } from '../src/lib/learning-friction-constraints';
import { getLearningContent } from '../src/lib/learning-content';

for (const [id, columns, equations] of [
  ['dry-friction-stopping-release', [6, 4], 6],
  ['circular-guide-reaction-work', [4], 6],
] as const) {
  test(`paired research equations, code, table structure and real renderer: ${id}`, async () => {
    const pairs = ['en', 'ko'].map(lang => {
      const raw = readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
      const start = raw.search(/^## (Research extension:|연구 확장:)/m);
      const end = raw.indexOf(lang === 'en' ? '## Six exercises with worked solutions' : '## 풀이가 있는 여섯 연습', start);
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
      assert.ok(content[lang].includes(`${id.startsWith('dry') ? 'classical-friction-identification' : 'classical-moving-guide-work'}.${lang}.svg`));
    }
  });
}

test('the three quoted friction candidates share forward position, velocity and kinetic curves but differ under the reverse probe', () => {
  const forward = [0, 1, 2].map(j => dry([1, j+1, 2-j, 10*j, 10, 10]));
  for (const j of [0, 1, 2]) {
    const r = forward[j];
    assert.equal(r.staticLimit.label, '3');
    assert.equal(r.kineticLimit.value, j+1);
    assert.equal(r.stopTime!.label, '1');
    assert.equal(r.query.position.label, '1/2');
    assert.equal(r.query.velocity.label, '0');
    assert.equal(r.query.heat.label, ['1/2', '1', '3/2'][j]);
    assert.equal(r.query.driveWork.label, ['0', '1/2', '1'][j]);
    assert.equal(r.query.sticking, true);
    for (const curve of [0, 1, 4]) assert.deepEqual(r.curves[curve], forward[0].curves[curve]);
    const pos = dry([1, j+1, 2-j, 10*j, 10, 0]);
    const neg = dry([1, j+1, 2-j, 10*j, -10, 0]);
    assert.equal(pos.query.acceleration.label, '-1');
    assert.equal(neg.query.acceleration.label, ['1', '3', '5'][j]);
    assert.equal((pos.query.acceleration.value+neg.query.acceleration.value)/2, j);
    assert.equal((neg.query.acceleration.value-pos.query.acceleration.value)/2, j+1);
    const stop = dry([1, j+1, 2-j, 10*j, -10, 10]);
    assert.equal(stop.stopTime!.label, ['1', '1/3', '1/5'][j]);
    assert.equal(stop.query.position.label, ['-1/2', '-1/6', '-1/10'][j]);
    assert.equal(stop.query.heat.label, ['1/2', '1/3', '3/10'][j]);
    assert.equal(stop.query.driveWork.label, ['0', '-1/6', '-1/5'][j]);
    assert.equal(stop.query.energyResidual.label, '0');
  }
});

test('the quoted zero-rate limit remains the native fixed-guide analytic trajectory with zero guide work', () => {
  const r = guide([1, 4, 0, 0, 5, 10]);
  assert.equal(r.radius.label, '2');
  assert.equal(r.query.omega.label, '1/2');
  assert.equal(r.query.angularDisplacement.label, '1/2');
  assert.equal(r.query.normalReaction.label, '-1/2');
  assert.equal(r.query.kinetic.label, '1/2');
  assert.equal(r.query.angularMomentum.label, '2');
  assert.equal(r.query.reactionWork.label, '0');
  assert.equal(r.query.energyResidual.label, '0');
  assert.ok(Math.abs(r.query.position[0]-2*Math.cos(.5)) < 1e-14);
  assert.ok(Math.abs(r.query.position[1]-2*Math.sin(.5)) < 1e-14);
  assert.ok(Math.abs(r.query.tangentResidual) < 1e-14);
  assert.ok(Math.abs(r.query.reactionPower) < 1e-14);
});
