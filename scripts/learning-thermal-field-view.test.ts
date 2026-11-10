import assert from 'node:assert/strict';
import { test } from 'node:test';
import { thermalDraft, parseThermalDraft, thermalProjection, thermalMatrixView } from '../src/lib/learning-thermal-field-view';
import { thermalFieldDefault, thermalFieldExamples } from '../src/lib/learning-thermal-field-input';
import { thermalFieldInputKey } from '../src/lib/learning-thermal-field-protocol';
import { calculateThermalField } from '../src/lib/learning-thermal-field';
import { thermalFieldSteps, thermalFieldQuestions, thermalFieldExperiments, thermalFieldLabels } from '../src/data/thermal-field-teaching';
test('all14 example drafts round-trip geometry and every derivative; invalid edits cannot silently resize vectors', () => {
  for (const { input } of thermalFieldExamples) assert.equal(thermalFieldInputKey(parseThermalDraft(thermalDraft(input))!), thermalFieldInputKey(input));
  for (const change of [{ shape: '2.0' }, { shape: '2e0' }, { shape: '8,8' }, { shape: '2,2,2' }, { temperature: '1e-50' },
    { source: { value: '1,0', first: '0', second: '0,0' } }]) assert.equal(parseThermalDraft({ ...thermalDraft(thermalFieldDefault), ...change }), null);
  const d = thermalDraft(thermalFieldDefault); d.mass.first = '1';
  assert.equal(thermalFieldDefault.mass.first, '0');
});
test('signed logarithmic projection handles huge ratios and never presents tiny nonzero values as exact zero', () => {
  assert.equal(thermalProjection('0', 'signed-log'), 0);
  assert.equal(thermalProjection('1' + '0'.repeat(1000), 'linear'), null);
  assert(Math.abs(thermalProjection('1' + '0'.repeat(1000), 'signed-log')! - 1000) < 1e-10);
  assert(Math.abs(thermalProjection('-' + '1' + '0'.repeat(1000), 'signed-log')! + 1000) < 1e-10);
  assert.equal(thermalProjection('1/' + '1' + '0'.repeat(1000), 'linear'), null);
  assert.equal(thermalProjection('1/' + '1' + '0'.repeat(1000), 'signed-log'), null);
  assert(Math.abs(thermalProjection('1' + '0'.repeat(1000) + '/1' + '0'.repeat(999), 'linear')! - 10) < 1e-10);
});
test('matrix coordinates distinguish pair indices from physical axes and preserve selected exact entries', () => {
  const exact = calculateThermalField(thermalFieldExamples.find(x => x.id === 'two-dimensional')!.input); assert(exact.kind === 'normalizable');
  for (const kind of ['covariance', 'precision', 'inverseOnSpace'] as const) {
    const view = thermalMatrixView(exact, kind, 'linear', 8); assert.equal(view.row, 1); assert.equal(view.column, 2);
    assert.equal(view.points.filter(x => x.selected).length, 1); assert.equal(view.points.length, 36);
    assert.equal(view.points[8].value, thermalProjection(exact[kind][1][2], 'linear'));
  }
});
test('teaching comparisons reproduce actual controlled calculations and point to real examples', () => {
  const example = (id: string) => { const result = calculateThermalField(thermalFieldExamples.find(x => x.id === id)!.input); assert(result.kind === 'normalizable'); return result; };
  const base = example('coupled-source'), background = example('same-law-background'), scale = example('coordinate-rescaling');
  assert.deepEqual(background.mean, base.mean); assert.deepEqual(background.covariance, base.covariance);
  assert.equal(base.thermalEnergy, '5/3'); assert.equal(background.thermalEnergy, '-145/3');
  assert.equal(base.heatCapacity, '1'); assert.equal(background.heatCapacity, '-55');
  assert.equal(base.logPartition.constant, scale.logPartition.constant);
  for (const guide of thermalFieldExperiments) for (const id of guide.examples) assert(thermalFieldExamples.some(x => x.id === id) && thermalFieldLabels[id]);
  assert.equal(thermalFieldSteps.length, 6); assert.equal(thermalFieldQuestions.length, 8);
  for (const q of thermalFieldQuestions) { assert(q.correct >= 0 && q.correct < q.options.length); assert(q.explanation.en.length > 60); assert(!/[가-힣]/.test(q.prompt.en + q.explanation.en + q.options.map(x => x.en).join(' '))); }
});
