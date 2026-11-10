import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { learningUnits, learningGuides } from '../src/data/interactive-book';
import { curriculumDomains } from '../src/data/learning-curriculum';
import { partitionKinds, partitionGuides } from '../src/lib/learning-partition';
const baseline = JSON.parse(readFileSync('scripts/fixtures/pre-partition-registry-hashes.json', 'utf8'));
const hash = (v: unknown) => createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('all 515 prior units and guides retain their complete values and iteration order', () => {
  assert.equal(baseline.ids.length, 515); assert.ok(learningUnits.length >= 518);
  assert.deepEqual(learningUnits.slice(0, 515).map(u => u.id), baseline.ids);
  assert.deepEqual(Object.keys(learningGuides).slice(0, 515), baseline.guideOrder);
  for (const u of learningUnits.slice(0, 515)) { assert.equal(hash(u), baseline.unitHashes[u.id], u.id); assert.equal(hash(learningGuides[u.id]), baseline.guideHashes[u.id], u.id); }
  assert.deepEqual(learningUnits.slice(515, 518).map(u => u.id), partitionKinds);
  assert.equal(learningUnits.slice(0, 518).reduce((n, u) => n + learningGuides[u.id].experiments.length, 0), 1554);
});
test('registered families contain detailed bilingual concepts, matching guides and explained quizzes', () => {
  for (const kind of partitionKinds) {
    const unit = learningUnits.find(u => u.id === kind); assert.ok(unit);
    assert.ok(unit.prerequisiteIds.every(id => { const index = learningUnits.findIndex(u => u.id === id); return index >= 0 && index < learningUnits.indexOf(unit); }));
    const guide = learningGuides[kind]; assert.equal(guide.steps.length, 4); assert.deepEqual(guide.experiments, partitionGuides[kind]);
    for (const s of guide.steps) {
      assert.ok(s.explanation.ko.length > 100); assert.ok(s.explanation.en.length > 150);
      assert.ok(s.formula.length > 15); assert.ok(s.diagram.ko.length > 8); assert.ok(s.diagram.en.length > 15);
      assert.doesNotMatch(s.explanation.en, /[가-힣]/);
    }
    assert.equal(unit.questions.length, 2);
    for (const q of unit.questions) {
      assert.equal(q.options.length, 3); assert.ok(Number.isInteger(q.answer) && q.answer >= 0 && q.answer < 3);
      assert.ok(q.explanation.ko.length > 80); assert.ok(q.explanation.en.length > 140);
      assert.doesNotMatch(q.explanation.en, /[가-힣]/);
    }
  }
});
test('all three units are mapped into statistical thermodynamics without removing its previous units', () => {
  const domain = curriculumDomains.find(d => d.id === 'thermo-statmech'); assert.ok(domain);
  for (const id of partitionKinds) assert.ok(domain.unitIds.includes(id));
  for (const id of ['thermodynamics', 'canonical-binary-energy-fluctuations', 'grand-canonical-finite-occupancy-fluctuations']) assert.ok(domain.unitIds.some(u => u === id));
});
