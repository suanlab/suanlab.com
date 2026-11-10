import test from 'node:test';
import assert from 'node:assert/strict';
import {
  quantumVariationalKinds as kinds, quantumVariationalPresets as presets,
  quantumVariationalPresetInput as preset, quantumVariationalNewSpace as space,
  editQuantumVariationalField as edit,
} from '../src/lib/learning-quantum-variational';
import { isQuantumVariationalInput as valid, isQuantumVariationalSuccessForInput as certified } from '../src/lib/learning-quantum-variational-protocol';
import { evaluateQuantumVariationalRequest as evaluate } from '../src/lib/learning-quantum-variational-evaluator';
import { quantumRitzGeometry as geometry } from '../src/lib/learning-quantum-variational-geometry';
import { quantumVariationalTeaching as teaching } from '../src/data/quantum-variational-teaching';
import { learningUnits } from '../src/data/interactive-book';

for (const kind of kinds) for (let i = 0; i < presets[kind].length; i++) {
  test(`teaching preparation is valid and certified: ${kind}/${i}`, () => {
    const input = preset(kind, i);
    assert(valid(input));
    const response = evaluate({ version: 1, requestId: 1, input });
    assert(response.ok);
    assert(certified(response, input, 1));
    if (response.kind === kinds[1]) for (const component of ['full', 'inside', 'outside'] as const) {
      const g = geometry(response.result, component, 10);
      if (!g) continue;
      assert(g.axes.flat().every(Number.isFinite));
      for (const axis of g.axes) assert(Math.abs(Math.hypot(...axis) - 1) < 1e-12);
      if (g.axes.length === 2) assert(Math.abs(g.axes[0].reduce((sum, x, j) => sum + x * g.axes[1][j], 0)) < 1e-12);
      for (const axis of g.axes) for (let j = 0; j < 3; j++) {
        const projected = response.result.spaceProjector[j].reduce((sum, q, k) => sum + q.real.value! * axis[k], 0);
        assert(Math.abs(projected - axis[j]) < 1e-10);
      }
    }
  });
}

test('new spaces cover all supported dimensions without truncating a previous input', () => {
  for (let n = 1; n <= 4; n++) {
    assert(valid(space(kinds[0], n)));
    for (let k = 1; k <= n; k++) assert(valid(space(kinds[1], n, k)));
  }
  for (const [n, k] of [[0, 1], [5, 1], [2, 3], [3, 0]]) assert.throws(() => space(kinds[1], n, k));
});

test('invalid field drafts cannot replace a valid physical input', () => {
  const input = preset(kinds[0], 0);
  for (const raw of ['', '1.5', '1.00000000000000001', 'Infinity', '1000001', 'NaN']) assert.equal(edit(input, 'stateReal0', raw), null, raw);
  assert.equal(edit(input, 'stateReal0', '1e2')?.parameters.stateReal0, 100);
  assert.equal(edit(input, 'hDenominator', '0'), null);
  assert.equal(edit(input, 'stateReal0', '0'), null);
  assert.equal(edit(input, 'stateReal3', '1'), null);
  assert.equal(edit(input, 'not-a-field', '1'), null);
  assert(valid(edit(input, 'stateReal0', '2')));
  assert.equal(input.parameters.stateReal0, 1);
});

test('real-slice geometry refuses imaginary coordinates, higher dimensions and invalid scales', () => {
  const input = preset(kinds[1], 0), response = evaluate({ version: 1, requestId: 1, input });
  assert(response.ok && response.kind === kinds[1]);
  assert(geometry(response.result, 'full', 1));
  for (const value of [0, -1, Infinity, NaN]) assert.equal(geometry(response.result, 'full', value), null);
  const changed = structuredClone(response.result);
  changed.normalizedState[0].imaginary.sign = 1;
  assert.equal(geometry(changed, 'full', 1), null);
  const four = evaluate({ version: 1, requestId: 2, input: space(kinds[1], 4, 2) });
  assert(four.ok && four.kind === kinds[1]);
  assert.equal(geometry(four.result, 'full', 1), null);
});

test('registered teaching has resolvable prerequisites, guided examples and explained bilingual quizzes', () => {
  const ids = new Set<string>([...learningUnits.map(u => u.id), ...kinds]);
  assert.equal(teaching.length, 2);
  for (const { unit, guide } of teaching) {
    assert.equal(learningUnits.some(u => (u.id as string) === unit.id), true);
    assert(unit.prerequisiteIds.every(id => ids.has(id)));
    assert.equal(unit.questions.length, 3);
    assert.equal(guide.experiments.length, 3);
    for (const experiment of guide.experiments) assert(presets[unit.id][experiment.preset!]);
    for (const q of unit.questions) {
      assert(q.options[q.answer]);
      assert(q.explanation.ko.length > 40);
      assert(q.explanation.en.length > 80);
      assert(!/[가-힣]/.test(q.question.en + q.explanation.en));
    }
  }
});
