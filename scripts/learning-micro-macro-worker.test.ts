import test from 'node:test';
import assert from 'node:assert/strict';
import { evaluateMicroMacroRequest as evaluate } from '../src/lib/learning-micro-macro-evaluator';
import { isMicroMacroInput, snapshotMicroMacroInput, microMacroInputKey, type MicroMacroInput } from '../src/lib/learning-micro-macro-protocol';
import { MicroMacroCalculationController as Controller, type MicroMacroWorkerPort, type MicroMacroCalculationState } from '../src/lib/learning-micro-macro-controller';
const binary: MicroMacroInput = { kind: 'binary-shell-multiplicity-block-counts', parameters: { sites: 4, excitations: 2, blockASites: 2, energyGap: 1 } };
const partition: MicroMacroInput = { kind: 'finite-macro-partition-conditional-entropy', parameters: { trialWeights: [6, 2, 1, 1], targetWeights: [4, 4, 1, 1], mapping: [0, 0, 1, 1], groupCount: 3 } };
const noisy: MicroMacroInput = { kind: 'finite-noisy-macro-observation-channel', parameters: { preparationWeights: [1, 1], channelWeights: [[9, 1], [1, 9]] } };
class Port implements MicroMacroWorkerPort {
  onmessage: Worker['onmessage'] = null;
  onerror: Worker['onerror'] = null;
  onmessageerror: Worker['onmessageerror'] = null;
  messages: unknown[] = []; terminated = 0; postFailure = false;
  postMessage(value: unknown) { if (this.postFailure) throw new Error('clone failure'); this.messages.push(structuredClone(value)); }
  terminate() { this.terminated++; }
  emit(data: unknown) { this.onmessage?.call(this as unknown as Worker, new MessageEvent('message', { data })); }
}
function setup() {
  const ports: Port[] = [], states: MicroMacroCalculationState[] = [];
  const controller = new Controller(() => { const port = new Port(); ports.push(port); return port; }, state => states.push(state));
  return { controller, ports, states };
}
for (const input of [binary, partition, noisy]) test(`dispatches serializable calculator result: ${input.kind}`, () => {
  assert.equal(isMicroMacroInput(input), true);
  const result = evaluate({ version: 1, requestId: 7, input });
  assert.equal(result.ok, true); assert.equal(result.requestId, 7); assert.doesNotThrow(() => structuredClone(result));
  if (result.ok && result.kind === binary.kind) assert.equal(result.result.selectedShell.multiplicity, '6');
  if (result.ok && result.kind === partition.kind) assert.equal(result.result.macroDivergence.value, 0);
  if (result.ok && result.kind === noisy.kind) assert.ok(result.result.mutualInformation.value! > .36);
});
test('boundary rejects malformed requests, sparse input and invalid probabilities before model work', () => {
  for (const v of [null, {}, { version: 2, requestId: 1 }, { version: 1, requestId: 0 }, { version: 1, requestId: 1.5 }, { version: 1, requestId: Infinity }])
    assert.deepEqual(evaluate(v).ok, false);
  for (const input of [{ ...binary, parameters: { ...binary.parameters, sites: 4097 } },
    { ...binary, parameters: { ...binary.parameters, energyGap: 0 } },
    { ...partition, parameters: { ...partition.parameters, mapping: Array(4) } },
    { ...noisy, parameters: { preparationWeights: [1, 0], channelWeights: [[1, 0], [0, 0]] } },
    { ...noisy, parameters: { preparationWeights: [1, 1], channelWeights: [[1], [1, 1]] } }, { kind: 'other', parameters: {} }]) {
    assert.equal(isMicroMacroInput(input), false);
    assert.deepEqual(evaluate({ version: 1, requestId: 1, input }), { version: 1, requestId: 1, ok: false, error: 'invalid-input' });
  }
});
test('frozen deep snapshots preserve parameters and stable keys after caller mutation', () => {
  const value = structuredClone(noisy), snapshot = snapshotMicroMacroInput(value), key = microMacroInputKey(snapshot);
  assert.equal(snapshot.kind, noisy.kind); assert.ok(Object.isFrozen(snapshot.parameters));
  if (value.kind === noisy.kind) (value.parameters.channelWeights[0] as number[])[0] = 0;
  assert.equal(microMacroInputKey(snapshot), key); assert.notEqual(microMacroInputKey(value), key);
  if (snapshot.kind === noisy.kind) assert.ok(Object.isFrozen(snapshot.parameters.channelWeights[0]));
});
test('successful result terminates its worker and retains the exact submitted input', () => {
  const { controller, ports, states } = setup(); controller.run(binary);
  assert.equal(states[0].status, 'pending'); ports[0].emit(evaluate(ports[0].messages[0]));
  const ready = controller.getSnapshot(); assert.equal(ready.status, 'ready');
  if (ready.status === 'ready') { assert.equal(ready.response.kind, binary.kind); assert.equal(ready.inputKey, microMacroInputKey(binary)); }
  assert.equal(ports[0].terminated, 1); assert.equal(ports[0].onmessage, null);
});
test('new request immediately terminates old work and ignores even captured stale callbacks', () => {
  const { controller, ports, states } = setup(); controller.run(binary);
  const late = ports[0].onmessage!, old = ports[0].messages[0]; controller.run(noisy);
  assert.equal(ports[0].terminated, 1); assert.equal(ports[1].messages.length, 1);
  late.call(ports[0] as unknown as Worker, new MessageEvent('message', { data: evaluate(old) }));
  assert.equal(states.length, 2); ports[1].emit(evaluate(ports[1].messages[0]));
  assert.equal(controller.getSnapshot().status, 'ready'); assert.equal(controller.getSnapshot().requestId, 2);
});
test('wrong request ids cannot resolve current work and wrong kind/version/scope fails', () => {
  for (const mutation of ['kind', 'version', 'scope']) {
    const { controller, ports } = setup(); controller.run(binary);
    ports[0].emit({ version: 1, requestId: 2, ok: true }); assert.equal(controller.getSnapshot().status, 'pending');
    const response = evaluate(ports[0].messages[0]); assert.ok(response.ok);
    const invalid = mutation === 'kind' ? { ...response, kind: noisy.kind } : mutation === 'version' ? { ...response, version: 2 }
      : { ...response, result: { scope: 'wrong' } };
    ports[0].emit(invalid); const state = controller.getSnapshot(); assert.equal(state.status, 'error');
    if (state.status === 'error') assert.equal(state.error, 'invalid-response');
    assert.equal(ports[0].terminated, 1);
  }
});
test('invalid input and unavailable worker never execute a synchronous calculator fallback', () => {
  let created = 0; const states: MicroMacroCalculationState[] = [];
  const controller = new Controller(() => { created++; throw new Error('unsupported'); }, state => states.push(state));
  controller.run(null); assert.equal(created, 0); controller.run(binary); assert.equal(created, 1);
  assert.equal(states.at(-1)?.status, 'error');
  const last = states.at(-1); if (last?.status === 'error') assert.equal(last.error, 'unavailable');
});
test('post failure, worker errors, invalid messages and model errors clear the active worker', () => {
  for (const mode of ['post', 'error', 'messageerror', 'model']) {
    const states: MicroMacroCalculationState[] = [], port = new Port(); port.postFailure = mode === 'post';
    const controller = new Controller(() => port, state => states.push(state)); controller.run(binary);
    if (mode === 'error') port.onerror!.call(port as unknown as Worker, new Event('error') as ErrorEvent);
    if (mode === 'messageerror') port.onmessageerror!.call(port as unknown as Worker, new MessageEvent('messageerror'));
    if (mode === 'model') port.emit({ version: 1, requestId: 1, ok: false, error: 'calculation-failed' });
    assert.equal(states.at(-1)?.status, 'error'); assert.equal(port.terminated, 1);
  }
});
test('cancel, disposal and a remount-like new controller reject all late events', () => {
  const { controller, ports, states } = setup(); controller.run(binary); const late = ports[0].onmessage!;
  controller.cancel(); assert.equal(controller.getSnapshot().status, 'idle'); controller.run(noisy); const count = states.length;
  controller.dispose(); controller.dispose(); assert.equal(controller.run(binary), null); controller.cancel();
  late.call(ports[0] as unknown as Worker, new MessageEvent('message', { data: evaluate(ports[0].messages[0]) }));
  assert.equal(states.length, count); assert.equal(ports[1].terminated, 1);
  const next = setup(); next.controller.run(binary); next.ports[0].emit(evaluate(next.ports[0].messages[0])); assert.equal(next.controller.getSnapshot().status, 'ready');
});
test('reentrant pending notification cancels before a worker is started', () => {
  let created = 0;
  const controller = new Controller(() => { created++; return new Port(); }, state => { if (state.status === 'pending') controller.cancel(); });
  controller.run(binary); assert.equal(created, 0); assert.equal(controller.getSnapshot().status, 'idle');
});

test('a matching started acknowledgment identifies actual calculation without resolving a result', () => {
  const { controller, ports } = setup(); controller.run(binary);
  ports[0].emit({ version: 1, requestId: 1, phase: 'started', kind: binary.kind });
  const running = controller.getSnapshot(); assert.equal(running.status, 'pending');
  if (running.status === 'pending') assert.equal(running.phase, 'calculating');
  ports[0].emit(evaluate(ports[0].messages[0])); assert.equal(controller.getSnapshot().status, 'ready');
});
test('a reentrant factory failure cannot overwrite a newer cancellation state', () => {
  const controller = new Controller(() => { controller.cancel(); throw new Error('factory failed'); }, () => {});
  controller.run(binary); assert.equal(controller.getSnapshot().status, 'idle');
});

test('canonical keys ignore caller property order and unrelated input fields', () => {
  const reordered = { kind: binary.kind, parameters: { energyGap: 1, blockASites: 2, excitations: 2, sites: 4, unrelated: true } };
  assert.ok(isMicroMacroInput(reordered));
  assert.equal(microMacroInputKey(reordered), microMacroInputKey(binary));
});
