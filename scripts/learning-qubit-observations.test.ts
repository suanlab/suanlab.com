import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evaluateQubitRequest as calculate } from '../src/lib/learning-qubit-evaluator';
import { evaluateQubitValidationRequest as validate } from '../src/lib/learning-qubit-validation-evaluator';
import { qubitSuccessHeader } from '../src/lib/learning-qubit-protocol';
import { qubitWorkerRepresentatives as reps } from './learning/qubit-worker-cases';
import { qubitTeachingPresets } from '../src/lib/learning-qubit-teaching-input';
import type { QubitInput } from '../src/lib/learning-qubit-input';
import type { QubitObservations, QubitObservationState } from '../src/lib/learning-qubit-observations-schema';

const run = (input: QubitInput) => {
  const response = calculate({ version: 1, requestId: 1, input });
  assert(response.ok); return response;
};
function alterCoordinate(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false;
  const r = value as Record<string, unknown>;
  if (typeof r.coordinate === 'number') { r.coordinate += 1 / 4; return true; }
  return Object.values(r).some(alterCoordinate);
}
for (const c of reps) {
  test(`owned ${c.input.kind} display survives trusted validation`, () => {
    const response = run(c.input);
    assert(Object.isFrozen(response.observations) && Object.isFrozen(response.observations.coordinates));
    const receipt = validate({ version: 1, requestId: 1, input: c.input, response });
    assert.equal(receipt.phase, 'validated');
    if (receipt.phase === 'validated') assert.deepEqual(receipt.response.observations, response.observations);
  });
  test(`${c.input.kind} rejects plausible coordinate corruption without changing exact mathematics`, () => {
    const response = structuredClone(run(c.input));
    assert(alterCoordinate(response.observations.coordinates));
    assert.equal(validate({ version: 1, requestId: 1, input: c.input, response }).phase, 'rejected');
  });
  test(`${c.input.kind} requires the full observation schema and exact report structure`, () => {
    const good = run(c.input);
    for (const observations of [undefined, {}, { ...good.observations, schemaVersion: 2 },
      { ...good.observations, states: { ...good.observations.states, inventedRole: null } },
      { ...good.observations, coordinates: null },
      { ...good.observations, extra: true }]) {
      const response = { ...good, observations };
      assert.equal(validate({ version: 1, requestId: 1, input: c.input, response }).phase, 'rejected');
    }
    const { observations, ...old } = good;
    assert(observations);
    assert.equal(qubitSuccessHeader(old, c.input, 1), false);
  });
}
const presets = Object.values(qubitTeachingPresets).flat();
const preset = (id: string) => { const p = presets.find(p => p.id === id); assert(p); return p.input; };
const state = (observations: QubitObservations, role: string): QubitObservationState => {
  const entry = observations.states[role]; assert(entry); return entry;
};
const realCoordinate = (entry: QubitObservationState, axis: number) => {
  assert(entry.observation.isDensity && entry.observation.bloch);
  return entry.observation.bloch[axis].display.real.coordinate;
};
test('conditional-X feedback retains both before/after branches and record-system states', () => {
  const { observations } = run(preset('corrected-plus'));
  assert.equal(realCoordinate(state(observations, 'conditionalBeforeFeedback1'), 2), -1);
  assert.equal(realCoordinate(state(observations, 'conditionalAfterFeedback1'), 2), 1);
  assert.equal(realCoordinate(state(observations, 'systemAfterFeedback'), 2), 1);
  assert.equal(realCoordinate(state(observations, 'forgottenBeforeFeedback'), 2), 0);
  const before = state(observations, 'classicalQuantumBeforeFeedback');
  const after = state(observations, 'classicalQuantumAfterFeedback');
  assert.notDeepEqual(before.matrix, after.matrix);
  assert.equal(after.matrix[2][2].real[0], '1/2');
  assert.equal(after.matrix[3][3].real[0], '0');
});
test('zero-probability branches remain absent on both sides of feedback', () => {
  const { observations } = run(preset('zero-branch'));
  assert.equal(observations.states.conditionalBeforeFeedback1, null);
  assert.equal(observations.states.conditionalAfterFeedback1, null);
});
test('coarse measurement outcomes retain mixed conditional purity and Y', () => {
  const { observations } = run(preset('coarse-record'));
  const conditional = state(observations, 'conditionalAfterFeedback0');
  assert(conditional.observation.isDensity && conditional.observation.purity);
  assert.equal(conditional.observation.purity.exact.real[0], '5/9');
  assert.equal(conditional.observation.bloch![1].exact.real[0], '1/3');
});
test('transpose counterexample and failed selection never become normalized state panels', () => {
  const transpose = run(preset('transpose'));
  assert.equal(state(transpose.observations, 'output').observation.isDensity, true);
  const bell = state(transpose.observations, 'bellOutput').observation;
  assert.equal(bell.isDensity, false); assert.equal(bell.purity, null); assert.equal(bell.bloch, null);
  for (const id of ['impossible-selection', 'trace-growing']) {
    const response = run(preset(id)), output = state(response.observations, 'output').observation;
    assert.equal(output.isDensity, false); assert.equal(output.purity, null); assert.equal(output.bloch, null);
    assert.equal(validate({ version: 1, requestId: 1, input: response.input, response }).phase, 'validated');
  }
});
test('coherent/dephased records distinguish Y while preserving the final system', () => {
  const { observations } = run(preset('deferred'));
  assert.equal(realCoordinate(state(observations, 'coherentRecord'), 1), 1);
  assert.equal(realCoordinate(state(observations, 'dephasedRecord'), 1), 0);
  assert.deepEqual(state(observations, 'coherentSystem').matrix, state(observations, 'dephasedSystem').matrix);
  assert.equal(state(observations, 'coherentJoint').observation.purity!.exact.real[0], '1');
  assert.equal(state(observations, 'dephasedJoint').observation.purity!.exact.real[0], '1/2');
});
test('all staged state fields reject matrix, purity, Bloch, role and normalization forgeries', () => {
  for (const id of ['corrected-plus', 'deferred', 'coarse-record']) {
    const input = preset(id), good = run(input);
    for (const role of Object.keys(good.observations.states)) {
      const initial = good.observations.states[role];
      if (initial === null) continue;
      for (const mutate of [
        (entry: QubitObservationState) => { (entry.matrix[0][0].real as string[])[0] = '999'; },
        (entry: QubitObservationState) => { entry.observation.purity!.display.real.coordinate = -5; },
        (entry: QubitObservationState) => { entry.matrixCoordinates[0][0].real.coordinate = -5; },
        (entry: QubitObservationState) => { entry.observation.unitTrace = false; },
        (entry: QubitObservationState) => { if (entry.observation.bloch) entry.observation.bloch[0].display.real.coordinate = -5; else entry.observation.dimension = 2; },
      ]) {
        const response = structuredClone(good); mutate(response.observations.states[role]!);
        assert.equal(validate({ version: 1, requestId: 1, input, response }).phase, 'rejected', `${id}/${role}`);
      }
    }
  }
});
