import test from 'node:test';
import assert from 'node:assert/strict';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { useEnsembleCalculation } from '../src/components/learning/use-ensemble-calculation';
import { evaluateEnsembleRequest as evaluate } from '../src/lib/learning-ensemble-evaluator';
import { isEnsembleInput, snapshotEnsembleInput, ensembleInputKey, ensembleResultMatchesInput, type EnsembleInput } from '../src/lib/learning-ensemble-protocol';
import { EnsembleCalculationController as Controller, type EnsembleWorkerPort, type EnsembleCalculationState } from '../src/lib/learning-ensemble-controller';
const reservoir: EnsembleInput = { kind: 'finite-binary-reservoir-canonical-comparison', parameters: { subsystemSites: 2, reservoirSites: 2, totalExcitations: 2 } };
const canonical: EnsembleInput = { kind: 'canonical-binary-energy-fluctuations', parameters: { sites: 4, energyGap: 1.380649e-23, temperature: 1 } };
const grand: EnsembleInput = { kind: 'grand-canonical-finite-occupancy-fluctuations', parameters: { sites: 2, energyGap: 1, chemicalPotential: 1, temperature: 0 } };
class Port implements EnsembleWorkerPort {
  onmessage: Worker['onmessage'] = null; onerror: Worker['onerror'] = null; onmessageerror: Worker['onmessageerror'] = null;
  messages: unknown[] = []; terminated = 0; postFailure = false;
  postMessage(value: unknown) { if (this.postFailure) throw new Error('clone failure'); this.messages.push(structuredClone(value)); }
  terminate() { this.terminated++; }
  emit(data: unknown) { this.onmessage?.call(this as unknown as Worker, new MessageEvent('message', { data })); }
}
function setup() {
  const ports: Port[] = [], states: EnsembleCalculationState[] = [];
  const controller = new Controller(() => { const port = new Port(); ports.push(port); return port; }, state => states.push(state));
  return { controller, ports, states };
}
for (const input of [reservoir, canonical, grand]) test(`serializable worker dispatch matches the submitted preparation: ${input.kind}`, () => {
  assert.equal(isEnsembleInput(input), true);
  const result = evaluate({ version: 1, requestId: 7, input }); assert.ok(result.ok);
  assert.equal(result.requestId, 7); assert.equal(ensembleResultMatchesInput(input, result.result), true);
  assert.doesNotThrow(() => structuredClone(result)); assert.doesNotThrow(() => JSON.stringify(result));
  if (result.kind === reservoir.kind) assert.equal(result.result.varianceExcitations.value, 1 / 3);
  if (result.kind === canonical.kind) assert.ok(result.result.entropyInKB.value! > 2.3);
  if (result.kind === grand.kind) { assert.equal(result.result.numberSusceptibility, null); assert.equal(result.result.numberVariance.value, .5); }
});
test('malformed headers and unknown inputs fail before calculation with generic errors', () => {
  for (const value of [null, {}, { version: 2, requestId: 1 }, { version: 1, requestId: 0 }, { version: 1, requestId: 1.5 }, { version: 1, requestId: Number.MAX_SAFE_INTEGER + 1 }])
    assert.deepEqual(evaluate(value).ok, false);
  const inputs = [null, [], {}, { kind: 'other', parameters: {} },
    { ...reservoir, parameters: { subsystemSites: 0, reservoirSites: 0, totalExcitations: 0 } },
    { ...reservoir, parameters: { subsystemSites: 4096, reservoirSites: 1, totalExcitations: 0 } },
    { ...reservoir, parameters: { subsystemSites: 1, reservoirSites: 1, totalExcitations: 3 } },
    { ...canonical, parameters: { sites: 1.5, energyGap: 1, temperature: 1 } },
    { ...canonical, parameters: { sites: 1, energyGap: -1, temperature: 1 } },
    { ...grand, parameters: { sites: 1, energyGap: 1, temperature: 1 } }];
  for (const input of inputs) {
    assert.equal(isEnsembleInput(input), false);
    assert.deepEqual(evaluate({ version: 1, requestId: 1, input }), { version: 1, requestId: 1, ok: false, error: 'invalid-input' });
  }
  for (const input of [canonical, grand]) for (const key of Object.keys(input.parameters)) for (const bad of [NaN, Infinity, -Infinity, '1', undefined])
    assert.equal(isEnsembleInput({ ...input, parameters: { ...input.parameters, [key]: bad } }), false);
});
test('zero, negative temperature and extreme finite scalar inputs retain model semantics', () => {
  const cases: EnsembleInput[] = [
    { kind: reservoir.kind, parameters: { subsystemSites: 0, reservoirSites: 4, totalExcitations: 2 } },
    { kind: reservoir.kind, parameters: { subsystemSites: 2, reservoirSites: 0, totalExcitations: 1 } },
    { kind: canonical.kind, parameters: { sites: 0, energyGap: 0, temperature: 0 } },
    { kind: canonical.kind, parameters: { sites: 2, energyGap: Number.MIN_VALUE, temperature: Number.MAX_VALUE } },
    { kind: canonical.kind, parameters: { sites: 2, energyGap: 1, temperature: -Number.MIN_VALUE } },
    { kind: grand.kind, parameters: { sites: 2, energyGap: 1, chemicalPotential: 2, temperature: 0 } },
    { kind: grand.kind, parameters: { sites: 0, energyGap: 1, chemicalPotential: -1, temperature: 0 } },
    { kind: grand.kind, parameters: { sites: 2, energyGap: Number.MAX_VALUE, chemicalPotential: Number.MAX_VALUE, temperature: Number.MIN_VALUE } },
  ];
  for (const input of cases) {
    const response = evaluate({ version: 1, requestId: 1, input }); assert.ok(response.ok);
    assert.ok(ensembleResultMatchesInput(input, response.result));
  }
});
test('frozen scalar snapshots and stable keys ignore property order and unrelated fields', () => {
  const input = structuredClone(canonical), snapshot = snapshotEnsembleInput(input), key = ensembleInputKey(snapshot);
  assert.ok(Object.isFrozen(snapshot)); assert.ok(Object.isFrozen(snapshot.parameters));
  (input.parameters as { temperature: number }).temperature = 2;
  assert.equal(ensembleInputKey(snapshot), key); assert.notEqual(ensembleInputKey(input), key);
  const reordered = { kind: canonical.kind, parameters: { temperature: 1, energyGap: 1.380649e-23, sites: 4, unrelated: true } };
  assert.ok(isEnsembleInput(reordered)); assert.equal(ensembleInputKey(reordered), key);
  assert.throws(() => snapshotEnsembleInput(null as unknown as EnsembleInput), RangeError);
});
test('worker evaluation strips unrelated nonserializable input properties', () => {
  const input = { ...canonical, parameters: { ...canonical.parameters, unrelated: () => 'not a calculator field' } };
  const response = evaluate({ version: 1, requestId: 1, input }); assert.ok(response.ok);
  assert.equal('unrelated' in response.result.parameters, false); assert.doesNotThrow(() => structuredClone(response));
});
test('successful completion terminates the worker and keeps the submitted snapshot', () => {
  const { controller, ports } = setup(); controller.run(reservoir); ports[0].emit(evaluate(ports[0].messages[0]));
  const state = controller.getSnapshot(); assert.equal(state.status, 'ready');
  if (state.status === 'ready') assert.equal(state.inputKey, ensembleInputKey(reservoir));
  assert.equal(ports[0].terminated, 1); assert.equal(ports[0].onmessage, null);
});
test('replacement rejects captured stale result, error and started callbacks', () => {
  const { controller, ports, states } = setup(); controller.run(reservoir);
  const late = ports[0].onmessage!, lateError = ports[0].onerror!, old = ports[0].messages[0]; controller.run(grand);
  assert.equal(ports[0].terminated, 1); const count = states.length;
  for (const data of [evaluate(old), { version: 1, requestId: 1, phase: 'started', kind: reservoir.kind }])
    late.call(ports[0] as unknown as Worker, new MessageEvent('message', { data }));
  lateError.call(ports[0] as unknown as Worker, new Event('error') as ErrorEvent);
  assert.equal(states.length, count); ports[1].emit(evaluate(ports[1].messages[0]));
  assert.equal(controller.getSnapshot().status, 'ready'); assert.equal(controller.getSnapshot().requestId, 2);
});
test('wrong ids are ignored; wrong kind/version/parameters and malformed numeric shape fail', () => {
  for (const mode of ['kind', 'version', 'parameters', 'rows', 'sparse', 'nonfinite', 'missing']) {
    const { controller, ports } = setup(); controller.run(canonical);
    ports[0].emit({ version: 1, requestId: 2, ok: true }); assert.equal(controller.getSnapshot().status, 'pending');
    const response = structuredClone(evaluate(ports[0].messages[0])); assert.ok(response.ok && response.kind === canonical.kind);
    if (mode === 'kind') ports[0].emit({ ...response, kind: grand.kind });
    else if (mode === 'version') ports[0].emit({ ...response, version: 2 });
    else {
      if (mode === 'parameters') response.result.parameters.temperature = 2;
      if (mode === 'rows') response.result.distribution.pop();
      if (mode === 'sparse') delete (response.result.distribution as unknown[])[0];
      if (mode === 'nonfinite') response.result.distribution[0].probability.logNegativeLogMagnitude = Infinity;
      if (mode === 'missing') delete (response.result as unknown as Record<string, unknown>).entropy;
      ports[0].emit(response);
    }
    const state = controller.getSnapshot(); assert.equal(state.status, 'error');
    if (state.status === 'error') assert.equal(state.error, 'invalid-response');
    assert.equal(ports[0].terminated, 1);
  }
});
test('invalid input and unavailable worker do not start synchronous calculator fallbacks', () => {
  let created = 0; const controller = new Controller(() => { created++; throw new Error('unavailable'); }, () => {});
  controller.run(null); assert.equal(created, 0); controller.run(canonical); assert.equal(created, 1);
  const state = controller.getSnapshot(); assert.equal(state.status, 'error');
  if (state.status === 'error') assert.equal(state.error, 'unavailable');
});
test('post, worker, deserialization and calculator errors clean up the active port', () => {
  for (const mode of ['post', 'error', 'messageerror', 'model']) {
    const port = new Port(); port.postFailure = mode === 'post'; const controller = new Controller(() => port, () => {});
    controller.run(grand);
    if (mode === 'error') port.onerror!.call(port as unknown as Worker, new Event('error') as ErrorEvent);
    if (mode === 'messageerror') port.onmessageerror!.call(port as unknown as Worker, new MessageEvent('messageerror'));
    if (mode === 'model') port.emit({ version: 1, requestId: 1, ok: false, error: 'calculation-failed' });
    assert.equal(controller.getSnapshot().status, 'error'); assert.equal(port.terminated, 1);
  }
});
test('cancel/dispose are idempotent and remount starts an independent controller', () => {
  const { controller, ports, states } = setup(); controller.run(reservoir); controller.cancel(); assert.equal(controller.getSnapshot().status, 'idle');
  controller.run(canonical); const count = states.length; controller.dispose(); controller.dispose(); controller.cancel();
  assert.equal(controller.run(grand), null); assert.equal(states.length, count); assert.equal(ports[1].terminated, 1);
  const next = setup(); next.controller.run(grand); next.ports[0].emit(evaluate(next.ports[0].messages[0])); assert.equal(next.controller.getSnapshot().status, 'ready');
});
test('reentrant notifications cancel before worker creation and factory failures preserve cancellation', () => {
  let created = 0;
  const controller = new Controller(() => { created++; return new Port(); }, state => { if (state.status === 'pending') controller.cancel(); });
  controller.run(reservoir); assert.equal(created, 0); assert.equal(controller.getSnapshot().status, 'idle');
  const failed = new Controller(() => { failed.cancel(); throw new Error('failed'); }, () => {});
  failed.run(canonical); assert.equal(failed.getSnapshot().status, 'idle');
});
test('reentrant factory replacement cannot attach the superseded port', () => {
  const oldPort = new Port(), nextPort = new Port(); let called = 0;
  const controller = new Controller(() => { called++; if (called === 1) { controller.run(grand); return oldPort; } return nextPort; }, () => {});
  controller.run(reservoir); assert.equal(oldPort.terminated, 1); assert.equal(oldPort.messages.length, 0);
  nextPort.emit(evaluate(nextPort.messages[0])); assert.equal(controller.getSnapshot().status, 'ready'); assert.equal(controller.getSnapshot().requestId, 2);
});
test('actual-start acknowledgment updates phase without resolving the result', () => {
  const { controller, ports } = setup(); controller.run(reservoir);
  ports[0].emit({ version: 1, requestId: 1, phase: 'started', kind: reservoir.kind });
  const state = controller.getSnapshot(); assert.equal(state.status, 'pending');
  if (state.status === 'pending') assert.equal(state.phase, 'calculating');
  ports[0].emit(evaluate(ports[0].messages[0])); assert.equal(controller.getSnapshot().status, 'ready');
});
test('calculator exceptions return no error detail or input echo', () => {
  const original = Math.exp;
  try {
    Math.exp = () => { throw new Error('private diagnostic'); };
    assert.deepEqual(evaluate({ version: 1, requestId: 1, input: canonical }), { version: 1, requestId: 1, ok: false, error: 'calculation-failed' });
  } finally { Math.exp = original; }
});
test('server rendering never accesses Worker or evaluates a synchronous fallback', () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let accesses = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { accesses++; throw new Error('Worker unavailable during SSR'); } });
  function Probe() {
    const state = useEnsembleCalculation(canonical);
    assert.equal(state.response, null); assert.equal(state.requestId, null);
    return createElement('output', null, state.status);
  }
  try { assert.equal(renderToStaticMarkup(createElement(Probe)), '<output>idle</output>'); assert.equal(accesses, 0); }
  finally { if (descriptor) Object.defineProperty(globalThis, 'Worker', descriptor); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
