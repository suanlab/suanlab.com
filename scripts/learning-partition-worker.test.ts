import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { usePartitionCalculation } from '../src/components/learning/use-partition-calculation';
import { evaluatePartitionRequest as evaluate } from '../src/lib/learning-partition-evaluator';
import { partitionPresets, partitionKinds } from '../src/lib/learning-partition';
import { isPartitionInput, snapshotPartitionInput, partitionInputKey, partitionResultMatchesInput, isPartitionObservation, type PartitionInput } from '../src/lib/learning-partition-protocol';
import { PartitionCalculationController as Controller, type PartitionWorkerPort, type PartitionCalculationState } from '../src/lib/learning-partition-controller';
const inputs = partitionKinds.map(kind => partitionPresets[kind][0].input), spectrum = inputs[0], pair = inputs[1];
class Port implements PartitionWorkerPort {
  onmessage: Worker['onmessage'] = null; onerror: Worker['onerror'] = null; onmessageerror: Worker['onmessageerror'] = null;
  messages: unknown[] = []; terminated = 0; postFailure = false;
  postMessage(value: unknown) { if (this.postFailure) throw new Error('clone failure'); this.messages.push(structuredClone(value)); }
  terminate() { this.terminated++; }
  emit(data: unknown) { this.onmessage?.call(this as unknown as Worker, new MessageEvent('message', { data })); }
}
function setup() {
  const ports: Port[] = [], states: PartitionCalculationState[] = [];
  const controller = new Controller(() => { const port = new Port(); ports.push(port); return port; }, state => states.push(state));
  return { controller, ports, states };
}
for (const input of inputs) test(`typed dispatch round-trips a serializable model: ${input.kind}`, () => {
  const response = evaluate({ version: 1, requestId: 1, input }); assert.ok(response.ok);
  assert.ok(partitionResultMatchesInput(input, response.result));
  assert.doesNotThrow(() => structuredClone(response)); assert.doesNotThrow(() => JSON.stringify(response));
});
test('all teaching presets retain exact support, range and unresolved transport semantics', () => {
  for (const kind of partitionKinds) for (const preset of partitionPresets[kind]) {
    const response = evaluate({ version: 1, requestId: 2, input: preset.input }); assert.ok(response.ok, preset.label.en);
    assert.ok(partitionResultMatchesInput(preset.input, response.result), preset.label.en);
    assert.doesNotThrow(() => JSON.stringify(response));
  }
});
test('invalid headers, sparse levels and nonfinite or missing fields fail before dispatch', () => {
  for (const request of [null, {}, { version: 2, requestId: 1 }, { version: 1, requestId: 0 }, { version: 1, requestId: .5 }]) assert.equal(evaluate(request).ok, false);
  const invalid = [null, {}, { kind: 'unknown', parameters: {} },
    { ...spectrum, parameters: { energyLevels: [], energyOffset: 0, temperature: 1 } },
    { ...spectrum, parameters: { energyLevels: Array(3), energyOffset: 0, temperature: 1 } },
    { ...spectrum, parameters: { energyLevels: [0, NaN], energyOffset: 0, temperature: 1 } },
    { ...pair, parameters: { couplingEnergy: 0, fieldEnergy: 0, temperature: 1, support: 'diagonal' } },
    { kind: inputs[2].kind, parameters: { gapEnergy: -1, temperature: 1, cutoff: 2 } },
    { kind: inputs[2].kind, parameters: { gapEnergy: 1, temperature: 1, cutoff: 2.5 } }];
  for (const input of invalid) { assert.equal(isPartitionInput(input), false); assert.equal(evaluate({ version: 1, requestId: 1, input }).ok, false); }
  for (const input of inputs) for (const bad of [Infinity, NaN, '1', undefined]) assert.equal(isPartitionInput({ ...input, parameters: { ...input.parameters, temperature: bad } }), false);
});
test('immutable snapshots isolate both the parameter object and the caller level array', () => {
  assert.equal(spectrum.kind, 'finite-spectrum-partition-cumulant-generators');
  if (spectrum.kind !== 'finite-spectrum-partition-cumulant-generators') return;
  const levels = [0, 1, 2], raw: PartitionInput = { kind: spectrum.kind, parameters: { energyLevels: levels, energyOffset: 0, temperature: 1 } }, snapshot = snapshotPartitionInput(raw);
  const key = partitionInputKey(snapshot); levels[0] = 99;
  assert.notEqual(partitionInputKey(raw), key); assert.equal(partitionInputKey(snapshot), key);
  assert.ok(Object.isFrozen(snapshot.parameters)); assert.equal(snapshot.kind, spectrum.kind);
  if (snapshot.kind === spectrum.kind) assert.ok(Object.isFrozen(snapshot.parameters.energyLevels));
  const reordered = { kind: spectrum.kind, parameters: { temperature: 1, energyOffset: 0, energyLevels: [0, 1, 2], unrelated: () => 1 } };
  assert.ok(isPartitionInput(reordered)); assert.equal(partitionInputKey(reordered), key);
  assert.doesNotThrow(() => structuredClone(evaluate({ version: 1, requestId: 1, input: reordered })));
});
test('started acknowledgment, completion and cleanup retain the submitted snapshot', () => {
  const { controller, ports } = setup(); controller.run(spectrum);
  ports[0].emit({ version: 1, requestId: 1, phase: 'started', kind: spectrum.kind });
  const pending = controller.getSnapshot(); assert.equal(pending.status, 'pending'); if (pending.status === 'pending') assert.equal(pending.phase, 'calculating');
  ports[0].emit(evaluate(ports[0].messages[0])); const state = controller.getSnapshot(); assert.equal(state.status, 'ready');
  if (state.status === 'ready') assert.equal(state.inputKey, partitionInputKey(spectrum));
  assert.equal(ports[0].terminated, 1); assert.equal(ports[0].onmessage, null);
});
test('replacement terminates synchronous work and rejects captured late callbacks', () => {
  const { controller, ports, states } = setup(); controller.run(spectrum);
  const callback = ports[0].onmessage!, error = ports[0].onerror!, request = ports[0].messages[0];
  controller.run(pair); const count = states.length;
  callback.call(ports[0] as unknown as Worker, new MessageEvent('message', { data: evaluate(request) }));
  error.call(ports[0] as unknown as Worker, new Event('error') as ErrorEvent);
  assert.equal(states.length, count); assert.equal(ports[0].terminated, 1);
  ports[1].emit(evaluate(ports[1].messages[0])); assert.equal(controller.getSnapshot().status, 'ready'); assert.equal(controller.getSnapshot().requestId, 2);
});
test('malformed results reject mismatched parameters, missing states, invalid observations and rational bounds', () => {
  for (const mode of ['parameters', 'kind', 'version', 'sparse', 'nonfinite', 'missing', 'bounds', 'zero']) {
    const { controller, ports } = setup(); controller.run(spectrum);
    ports[0].emit({ version: 1, requestId: 900, ok: true }); assert.equal(controller.getSnapshot().status, 'pending');
    const response = structuredClone(evaluate(ports[0].messages[0])); assert.ok(response.ok && response.kind === spectrum.kind);
    if (!response.ok || response.kind !== 'finite-spectrum-partition-cumulant-generators') return;
    if (mode === 'kind') ports[0].emit({ ...response, kind: pair.kind });
    else if (mode === 'version') ports[0].emit({ ...response, version: 2 });
    else {
      if (mode === 'parameters') response.result.parameters.temperature = 9;
      if (mode === 'sparse') delete (response.result.states as unknown[])[0];
      if (mode === 'nonfinite') response.result.states[0].probability.logMagnitude = Infinity;
      if (mode === 'missing') delete (response.result as unknown as Record<string, unknown>).freeEnergy;
      if (mode === 'bounds') response.result.states[0].probability.bounds!.lower.denominator.coefficient = '0';
      if (mode === 'zero') response.result.states[0].probability.value = 0;
      ports[0].emit(response);
    }
    const state = controller.getSnapshot(); assert.equal(state.status, 'error'); if (state.status === 'error') assert.equal(state.error, 'invalid-response');
    assert.equal(ports[0].terminated, 1);
  }
});
test('numeric status contract rejects invented zeros and accepts honest uncertainty', () => {
  assert.equal(isPartitionObservation({ value: 0, sign: 1, logMagnitude: -800, resolution: 'derived-exponential', logResolution: 'finite', bounds: null }), false);
  assert.equal(isPartitionObservation({ value: null, sign: 1, logMagnitude: null, resolution: 'unresolved-at-precision-ceiling', logResolution: 'unresolved', bounds: null }), true);
  assert.equal(isPartitionObservation({ value: null, sign: null, logMagnitude: null, resolution: 'mathematical-infinity', logResolution: 'not-applicable', bounds: null }), false);
});
test('unavailable workers and invalid input never trigger a synchronous UI fallback', () => {
  let created = 0; const controller = new Controller(() => { created++; throw new Error('blocked'); }, () => {});
  controller.run(null); assert.equal(created, 0); controller.run(spectrum); assert.equal(created, 1);
  const state = controller.getSnapshot(); assert.equal(state.status, 'error'); if (state.status === 'error') assert.equal(state.error, 'unavailable');
});
test('post, execution and deserialization failures terminate the active worker', () => {
  for (const mode of ['post', 'error', 'messageerror', 'model']) {
    const port = new Port(); port.postFailure = mode === 'post'; const controller = new Controller(() => port, () => {}); controller.run(spectrum);
    if (mode === 'error') port.onerror!.call(port as unknown as Worker, new Event('error') as ErrorEvent);
    if (mode === 'messageerror') port.onmessageerror!.call(port as unknown as Worker, new MessageEvent('messageerror'));
    if (mode === 'model') port.emit({ version: 1, requestId: 1, ok: false, error: 'calculation-failed' });
    assert.equal(controller.getSnapshot().status, 'error'); assert.equal(port.terminated, 1);
  }
});
test('cancellation and disposal are idempotent; remount has a fresh sequence', () => {
  const { controller, ports, states } = setup(); controller.run(pair); controller.cancel(); assert.equal(controller.getSnapshot().status, 'idle');
  controller.run(spectrum); const count = states.length; controller.dispose(); controller.dispose(); controller.cancel();
  assert.equal(controller.run(pair), null); assert.equal(states.length, count); assert.equal(ports[0].terminated, 1); assert.equal(ports[1].terminated, 1);
  const next = setup(); next.controller.run(pair); assert.equal(next.controller.getSnapshot().requestId, 1); next.controller.dispose();
});
test('reentrant notification and factory replacement cannot attach stale ports', () => {
  let created = 0; const controller = new Controller(() => { created++; return new Port(); }, state => { if (state.status === 'pending') controller.cancel(); });
  controller.run(spectrum); assert.equal(created, 0);
  const old = new Port(), current = new Port(); let calls = 0;
  const replacement = new Controller(() => { if (++calls === 1) { replacement.run(pair); return old; } return current; }, () => {});
  replacement.run(spectrum); assert.equal(old.terminated, 1); assert.equal(old.messages.length, 0);
  current.emit(evaluate(current.messages[0])); assert.equal(replacement.getSnapshot().status, 'ready');
});
test('SSR never reads Worker or executes the calculator through the hook', () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let reads = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { reads++; throw new Error('SSR Worker read'); } });
  function Probe() { const state = usePartitionCalculation(spectrum); assert.equal(state.response, null); return React.createElement('output', null, state.status); }
  try { assert.equal(renderToStaticMarkup(React.createElement(Probe)), '<output>idle</output>'); assert.equal(reads, 0); }
  finally { if (descriptor) Object.defineProperty(globalThis, 'Worker', descriptor); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
