import test from 'node:test';
import assert from 'node:assert/strict';
import {continuumKinds, evaluateContinuum} from '../src/lib/learning-continuum';
import {continuumLabs, learningUnits, learningGuides} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';

function translated(value: unknown): void {
  if (!value || typeof value !== 'object') return;
  const object = value as Record<string, unknown>;
  if ('ko' in object && 'en' in object) {
    assert.ok(typeof object.ko === 'string' && object.ko.length > 0);
    assert.ok(typeof object.en === 'string' && object.en.length > 0);
    assert.doesNotMatch(object.en as string, /[가-힣]/);
  }
  Object.values(object).forEach(translated);
}
function finiteTree(value: unknown): void {
  if (typeof value === 'number') assert.ok(Number.isFinite(value));
  else if (value && typeof value === 'object') Object.values(value).forEach(finiteTree);
}
for (const [index, id] of continuumKinds.entries()) {
  const spec = continuumLabs[id];
  test(`${id}: registered bilingual lesson, prerequisites and curriculum`, () => {
    const unit = learningUnits.find(u => u.id === id)!;
    assert.ok(unit);
    unit.prerequisiteIds.forEach(p => assert.ok(learningUnits.some(u => u.id === p)));
    assert.ok(curriculumDomains.some(d => d.unitIds.includes(id)));
    assert.equal(unit.questions.length, 2);
    assert.equal(learningGuides[id].steps.length, 4);
    translated({unit, guide: learningGuides[id], spec});
    assert.equal(spec.controls.length, index === 0 ? 10 : 12);
    assert.equal(spec.presets.length, index === 0 ? 14 : 16);
  });
  test(`${id}: every preset produces finite or explicitly missing observations without mutating inputs`, () => {
    for (const preset of spec.presets) {
      const values = [...preset.values], original = [...values];
      const result = evaluateContinuum(id, values);
      assert.deepEqual(values, original);
      finiteTree(result);
      const lab = result.massTransport ?? result.gasConservation!;
      assert.equal(lab.temporal.length, 81);
      assert.equal(lab.temporal[0].time, -1);
      assert.equal(lab.temporal.at(-1)!.time, 1);
      assert.equal(lab.spatial[0].x, -5);
      assert.equal(lab.spatial.at(-1)!.x, 5);
      assert.equal(lab.particles.length, 9);
    }
  });
  test(`${id}: malformed, sparse and out-of-range controls are rejected`, () => {
    const defaults = spec.controls.map(c => c.initial);
    assert.throws(() => evaluateContinuum(id, defaults.slice(1)), RangeError);
    for (let i = 0; i < defaults.length; i++) {
      for (const invalid of [NaN, Infinity, spec.controls[i].max + 1]) {
        const values = [...defaults]; values[i] = invalid;
        assert.throws(() => evaluateContinuum(id, values), RangeError);
      }
      const sparse = [...defaults]; delete sparse[i];
      assert.throws(() => evaluateContinuum(id, sparse), RangeError);
    }
    if (index === 1) {const values = [...defaults]; values[9] = .5; assert.throws(() => evaluateContinuum(id, values), RangeError);}
  });
}
test('worked mass and energy examples preserve their physical units and signed changes', () => {
  const mass = evaluateContinuum(continuumKinds[0], continuumLabs[continuumKinds[0]].presets[0].values).massTransport!;
  assert.ok(Math.abs(mass.balance.massChange! - Math.SQRT2 / Math.PI) < 1e-14);
  const gas = evaluateContinuum(continuumKinds[1], continuumLabs[continuumKinds[1]].presets[0].values).gasConservation!;
  assert.equal(gas.query.storage[0], 2);
  assert.ok(Math.abs(gas.query.storage[2]! - 17 / 12) < 1e-14);
  assert.ok(Math.abs(gas.balance.changes[2]! + 2 / 3) < 1e-14);
});
test('mass spatial sampling resolves the highest supported wavenumber', () => {
  const values = continuumLabs[continuumKinds[0]].controls.map(c => c.initial); values[2] = 8;
  const lab = evaluateContinuum(continuumKinds[0], values).massTransport!;
  assert.ok((lab.spatial.length - 1) / (10 * 8 / (2 * Math.PI)) >= 16);
});
test('material transport and exact zero width retain distinct observations', () => {
  const values = continuumLabs[continuumKinds[0]].controls.map(c => c.initial); values[7] = values[3];
  const material = evaluateContinuum(continuumKinds[0], values).massTransport!;
  assert.equal(material.query.massRate, 0); assert.equal(material.balance.massChange, 0);
  values[6] = 0;
  const zero = evaluateContinuum(continuumKinds[0], values).massTransport!;
  assert.equal(zero.query.mass, 0); assert.equal(zero.query.observedMeanDensity, null);
  assert.equal(zero.query.leftOutward, null);
});
