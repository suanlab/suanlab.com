import assert from 'node:assert/strict';
import test from 'node:test';
import { frictionlessRodImpact as rod, drivenWheelContact as wheel } from '../src/lib/learning-rigid-contact';
import { readFileSync } from 'node:fs';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';

for (const [id, equations, image] of [
  ['frictionless-rod-contact-impulse', 7, 'classical-contact-grazing'],
  ['driven-wheel-rolling-sliding', 6, 'classical-contact-friction-limit'],
] as const) {
  test(`paired contact research equations, executable cell, tables and real renderer: ${id}`, async () => {
    const pairs = ['en', 'ko'].map(lang => {
      const source = readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
      const start = source.search(/^## (Research extension:|연구 확장:)/m);
      const heading = lang === 'en' ? '## Worked examples and exercises with solutions' : '## 예제와 해설이 있는 연습문제';
      const end = source.indexOf(heading, start);
      assert.ok(start >= 0 && end > start);
      const tree = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(source.slice(start, end));
      const tables = tree.children.filter(node => node.type === 'table');
      assert.equal(tables.length, 1);
      for (const row of tables[0].children) assert.equal(row.children.length, 4);
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

test('quoted mass-metric examples change the contact normal mode while retaining tangential motion', () => {
  for (const [e, impulse, vy, omega, loss] of [
    [0, '2/5', '-3/5', '3/5', '1/5'],
    [5, '3/5', '-2/5', '9/10', '3/20'],
    [10, '4/5', '-1/5', '6/5', '0'],
  ] as const) {
    const r = rod([1, 2, 2, 0, -10, 0, e]);
    assert.equal(r.inertia.label, '2/3');
    assert.equal(r.inverseEffectiveNormalMass.label, '5/2');
    assert.equal(r.normalImpulse.label, impulse);
    assert.equal(r.after.velocity[1].label, vy);
    assert.equal(r.after.omega.label, omega);
    assert.equal(r.energyLoss.label, loss);
    assert.equal(r.after.contactVelocity[1].value, e / 10);
    assert.equal(r.contactAngularResidual.label, '0');
  }
});

test('native grazing bracket distinguishes opening, zero speed and closing without an attractive impulse', () => {
  for (const vy of [-9, -10, -11]) {
    const r = rod([1, 2, 2, 0, vy, 10, 5]);
    assert.equal(r.active, vy < -10);
    if (vy >= -10) {
      assert.equal(r.normalImpulse.label, '0');
      assert.deepEqual(r.before, r.after);
      assert.equal(r.restitutionResidual, null);
    } else {
      assert.equal(r.before.contactVelocity[1].label, '-1/10');
      assert.equal(r.normalImpulse.label, '3/50');
      assert.equal(r.after.contactVelocity[1].label, '1/20');
      assert.equal(r.after.velocity[1].label, '-26/25');
      assert.equal(r.after.omega.label, '109/100');
      assert.equal(r.energyLoss.label, '3/2000');
    }
  }
});

test('four native zero-drive friction coefficients have different waiting times but the quoted terminal losses', () => {
  for (const [mu, event] of [[0, null], [1, '1/3'], [2, '1/6'], [5, '1/15']] as const) {
    const r = wheel([1, 2, mu, 0, 5, 0, 0, 10, 0, 10]);
    assert.equal(r.requiredRollingFriction.label, '0');
    assert.equal(r.zeroSlipEventTime?.label ?? null, event);
    assert.equal(r.query.velocity.label, mu ? '2/3' : '1');
    assert.equal(r.query.clockwiseOmega.label, mu ? '2/3' : '0');
    assert.equal(r.query.heat.label, mu ? '1/6' : '0');
    assert.equal(r.query.kinetic.label, mu ? '1/3' : '1/2');
    assert.equal(r.query.rolling, mu > 0);
    assert.equal(r.query.energyResidual.label, '0');
    assert.equal(r.query.contactHeatResidual.label, '0');
  }
});

test('native times bracket the one-third-second switch and preserve the independently decomposed heat', () => {
  const before = wheel([1, 2, 1, 0, 5, 0, 0, 10, 0, 3]);
  const after = wheel([1, 2, 1, 0, 5, 0, 0, 10, 0, 4]);
  assert.equal(before.query.slipVelocity.label, '1/10');
  assert.equal(before.query.velocity.label, '7/10');
  assert.equal(before.query.clockwiseOmega.label, '3/5');
  assert.equal(before.query.heat.label, '33/200');
  assert.equal(before.query.rolling, false);
  assert.equal(after.query.slipVelocity.label, '0');
  assert.equal(after.query.heat.label, '1/6');
  assert.equal(after.query.position.label, '29/90');
  assert.equal(after.query.clockwiseAngle.label, '7/45');
  assert.equal(after.query.rolling, true);
});

test('an initially spinning wheel gains COM speed while its relative-slip energy is dissipated', () => {
  const sliding = wheel([1, 2, 1, 0, 5, 0, 0, 0, 10, 1]);
  assert.equal(sliding.initialSlip.label, '-1');
  assert.equal(sliding.query.friction.label, '1');
  assert.equal(sliding.query.velocity.label, '1/10');
  assert.equal(sliding.query.clockwiseOmega.label, '4/5');
  assert.equal(sliding.query.heat.label, '17/200');
  const rolling = wheel([1, 2, 1, 0, 5, 0, 0, 0, 10, 10]);
  assert.equal(rolling.query.velocity.label, '1/3');
  assert.equal(rolling.query.clockwiseOmega.label, '1/3');
  assert.equal(rolling.query.heat.label, '1/6');
  assert.equal(rolling.query.kinetic.label, '1/12');
});
