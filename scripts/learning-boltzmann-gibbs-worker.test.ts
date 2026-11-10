import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { readFileSync } from 'node:fs';
import { renderToStaticMarkup } from 'react-dom/server';
import { useBoltzmannGibbsCalculation } from '../src/components/learning/use-boltzmann-gibbs-calculation';
import { useBoltzmannGibbsSeries } from '../src/components/learning/use-boltzmann-gibbs-series';
import { evaluateBoltzmannGibbsRequest as evaluate } from '../src/lib/learning-boltzmann-gibbs-evaluator';
import { boltzmannGibbsKinds as kinds, boltzmannGibbsPresets as presets, boltzmannGibbsPresetInput as preset } from '../src/lib/learning-boltzmann-gibbs';
import { isBoltzmannGibbsInput, snapshotBoltzmannGibbsInput as snapshot, boltzmannGibbsInputKey as key, isBoltzmannGibbsSuccessForInput as valid,
  isBoltzmannGibbsObservation as observation, parseBoltzmannNumber as parse, parseBoltzmannNumberList as parseList, boltzmannTransportLimits, type BoltzmannGibbsInput } from '../src/lib/learning-boltzmann-gibbs-protocol';
import { BoltzmannGibbsCalculationController as Controller, type BoltzmannGibbsWorkerPort, type BoltzmannGibbsCalculationState } from '../src/lib/learning-boltzmann-gibbs-controller';
class Port implements BoltzmannGibbsWorkerPort {
  onmessage: Worker['onmessage'] = null; onerror: Worker['onerror'] = null; onmessageerror: Worker['onmessageerror'] = null;
  messages: unknown[] = []; terminated = 0; postFailure = false;
  postMessage(v: unknown) { if (this.postFailure) throw new Error('clone'); this.messages.push(structuredClone(v)); }
  terminate() { this.terminated++; }
  emit(data: unknown) { this.onmessage?.call(this as unknown as Worker, new MessageEvent('message', { data })); }
}
function setup() {
  const ports: Port[] = [], states: BoltzmannGibbsCalculationState[] = [];
  const controller = new Controller(() => { const port = new Port(); ports.push(port); return port; }, state => states.push(state));
  return { controller, ports, states };
}
const weighted = preset(kinds[0], 0), gibbs = preset(kinds[1], 0), gaussian = preset(kinds[2], 0);
for (const kind of kinds) for (const p of presets[kind]) test(`${p.id}: serializable dispatch validates scientific status and exact input identity`, () => {
  const response = evaluate({ version: 1, requestId: 1, input: p.input }); assert.ok(response.ok); assert.ok(valid(response, p.input, 1));
  assert.doesNotThrow(() => structuredClone(response)); assert.doesNotThrow(() => JSON.stringify(response));
});
test('all independent-reference parameter sets pass the actual transport contract, including tails and normalization failures', () => {
  for (const [file, kind] of [['weighted-boltzmann', kinds[0]], ['finite-gibbs', kinds[1]], ['quadratic-gibbs', kinds[2]]] as const) {
    const cases = JSON.parse(readFileSync(`scripts/fixtures/${file}-oracle.json`, 'utf8')).cases;
    for (const c of cases) { const input = { kind, parameters: c.parameters } as BoltzmannGibbsInput, r = evaluate({ version: 1, requestId: 3, input }); assert.ok(r.ok, c.id); assert.ok(valid(r, input, 3), c.id); }
  }
});
test('input validators reject sparse arrays, missing support, null defaults, invalid signs and path options', () => {
  const invalid = [null, {}, { kind: 'unknown', parameters: {} }, { ...weighted, parameters: { ...weighted.parameters, energyLevels: Array(3) } },
    { ...weighted, parameters: { ...weighted.parameters, referenceWeights: Array(3) } }, { ...weighted, parameters: { ...weighted.parameters, referenceWeights: [0, 0, 0] } },
    { ...weighted, parameters: { ...weighted.parameters, referenceWeights: [1, -1, 1] } }, { ...weighted, path: { seed: 1, bits: 8, steps: 1, mode: 'random' } },
    { ...gibbs, parameters: { ...gibbs.parameters, temperature: 0 } }, { ...gibbs, parameters: { ...gibbs.parameters, support: 'unknown' } },
    { ...gibbs, parameters: { ...gibbs.parameters, observationSteps: Array(2) } }, { ...gibbs, parameters: { ...gibbs.parameters, observationSteps: [2, 1] } },
    { ...gibbs, parameters: { ...gibbs.parameters, initialStateIndex: 4 } }, { ...gibbs, path: null }, { ...gibbs, path: { seed: -1, bits: 0, steps: 9000, mode: 'random' } },
    { ...gaussian, parameters: { ...gaussian.parameters, coordinateScale: [0, 1] } }, { ...gaussian, parameters: { ...gaussian.parameters, coordinateShift: null } },
    { ...gaussian, parameters: { ...gaussian.parameters, temperature: -1 } }];
  for (const input of invalid) { assert.equal(isBoltzmannGibbsInput(input), false); assert.equal(evaluate({ version: 1, requestId: 1, input }).ok, false); }
  for (const input of [weighted, gibbs, gaussian]) for (const temperature of [NaN, Infinity, '1', undefined]) assert.equal(isBoltzmannGibbsInput({ ...input, parameters: { ...input.parameters, temperature } }), false);
});
test('canonical defaults, arrays and sampling options are owned immutable snapshots', () => {
  const base = preset(kinds[1], 0); assert.ok(base.kind === kinds[1]);
  const raw = { ...base, path: { mode: 'random' as const, seed: 1, bits: 8, steps: 16 } }, s = snapshot(raw), before = key(s);
  raw.path.seed = 9; assert.equal(key(s), before); assert.notEqual(key(raw), before);
  assert.ok(s.kind === kinds[1]); assert.ok(Object.isFrozen(s.path)); assert.ok(Object.isFrozen(s.parameters.observationSteps));
  assert.equal(key(base), key({ ...base, parameters: { ...base.parameters, initialStateIndex: 0, observationSteps: [1, 2, 4] } }));
  const w = preset(kinds[0], 0); assert.ok(w.kind === kinds[0]); const rawWeights = [1, 2, 1], copy = snapshot({ ...w, parameters: { ...w.parameters, referenceWeights: rawWeights } }); rawWeights[0] = 99;
  assert.ok(copy.kind === kinds[0]); assert.equal(copy.parameters.referenceWeights[0], 1); assert.ok(Object.isFrozen(copy.parameters.referenceWeights));
  const g = preset(kinds[2], 0); assert.ok(g.kind === kinds[2]); assert.equal(key(g), key({ ...g, parameters: { ...g.parameters, coordinateScale: [1, 1], coordinateShift: [0, 0] } }));
});
test('decimal parsing does not silently turn a nonzero out-of-range literal into a zero boundary', () => {
  for (const s of ['1e-400', '-1e-9999', '1e309']) assert.deepEqual(parse(s), { ok: false, error: 'outside-input-range' });
  for (const s of ['0', '-0', '.0e9999']) assert.deepEqual(parse(s), { ok: true, value: 0 });
  assert.deepEqual(parse('5e-324'), { ok: true, value: Number.MIN_VALUE }); assert.deepEqual(parse('-.125'), { ok: true, value: -.125 });
  for (const s of ['', 'Infinity', 'NaN', '0x10', '1,2', '1e']) assert.equal(parse(s).ok, false);
  assert.deepEqual(parseList('1, 2, 3'), { ok: true, values: [1, 2, 3] }); assert.deepEqual(parseList('1 2\n3'), { ok: true, values: [1, 2, 3] });
  for (const s of ['1,,2', '1,', ',1', '', '1 2,3', '1e-400 2']) assert.equal(parseList(s).ok, false);
});
test('request headers are checked before numerical dispatch', () => {
  for (const v of [null, {}, { version: 2, requestId: 1 }, { version: 1, requestId: 0 }, { version: 1, requestId: NaN }, { version: 1, requestId: .5 }]) assert.equal(evaluate(v).ok, false);
});
test('started acknowledgment, completed response and cleanup keep the submitted snapshot', () => {
  const { controller, ports } = setup(); controller.run(weighted);
  ports[0].emit({ version: 1, requestId: 1, phase: 'started', kind: weighted.kind }); const state = controller.getSnapshot(); assert.equal(state.status, 'pending'); if (state.status === 'pending') assert.equal(state.phase, 'calculating');
  ports[0].emit(evaluate(ports[0].messages[0])); assert.equal(controller.getSnapshot().status, 'ready'); assert.equal(ports[0].terminated, 1); assert.equal(ports[0].onmessage, null);
});
test('replacement terminates prior work and ignores captured callbacks and wrong valid request IDs', () => {
  const { controller, ports, states } = setup(); controller.run(weighted); const late = ports[0].onmessage!, error = ports[0].onerror!, old = ports[0].messages[0];
  controller.run(gibbs); const count = states.length;
  late.call(ports[0] as unknown as Worker, new MessageEvent('message', { data: evaluate(old) })); error.call(ports[0] as unknown as Worker, new Event('error') as ErrorEvent);
  ports[1].emit({ version: 1, requestId: 999, ok: true }); assert.equal(states.length, count);
  ports[1].emit(evaluate(ports[1].messages[0])); assert.equal(controller.getSnapshot().status, 'ready'); assert.equal(controller.getSnapshot().requestId, 2); assert.equal(ports[0].terminated, 1);
});
test('malformed identifiers and result envelopes fail and terminate the worker', () => {
  for (const mode of ['nan-id', 'version', 'kind', 'key', 'input', 'missing', 'bounds', 'invented-zero', 'resolved-flag', 'cycle']) {
    const { controller, ports } = setup(); controller.run(weighted);
    const r = structuredClone(evaluate(ports[0].messages[0])); assert.ok(r.ok && r.kind === kinds[0]);
    const v = r as unknown as Record<string, unknown>, result = r.result;
    if (mode === 'nan-id') v.requestId = NaN;
    if (mode === 'version') v.version = 2;
    if (mode === 'kind') v.kind = kinds[1];
    if (mode === 'key') v.inputKey = 'wrong';
    if (mode === 'input') r.input.parameters.temperature = 9;
    if (mode === 'missing') delete (result as unknown as Record<string, unknown>).freeEnergy;
    if (mode === 'bounds') result.states[0].probability.bounds!.lower.denominator.coefficient = '0';
    if (mode === 'invented-zero') result.states[0].probability.value = 0;
    if (mode === 'resolved-flag') result.fullyResolved = false;
    if (mode === 'cycle') v.input = { ...r.input, parameters: { ...r.input.parameters, energyLevels: Array(3) } };
    ports[0].emit(r); const s = controller.getSnapshot(); assert.equal(s.status, 'error', mode); if (s.status === 'error') assert.equal(s.error, 'invalid-response'); assert.equal(ports[0].terminated, 1);
  }
});
test('Gibbs support graphs and Gaussian invalid-density branches reject stale display shapes', () => {
  for (const [input, mutate] of [
    [gibbs, (r: Record<string, unknown>) => { const m = r.kernels as Record<string, unknown[][]>; m.first[0][1] = m.first[0][0]; }],
    [gibbs, (r: Record<string, unknown>) => { (r.diagnostics as Record<string, Record<string, unknown>>).random.irreducible = false; }],
    [preset(kinds[2], 4), (r: Record<string, unknown>) => { r.mean = [0, 0]; }],
    [gaussian, (r: Record<string, unknown>) => { (r.covariance as Record<string, unknown>[][])[0][0].sign = 0; }],
  ] as const) {
    const response = evaluate({ version: 1, requestId: 1, input }); assert.ok(response.ok); mutate(response.result as unknown as Record<string, unknown>); assert.equal(valid(response, input, 1), false);
  }
});
test('worker creation, post, execution and deserialization errors have no synchronous fallback', () => {
  let calls = 0; const unavailable = new Controller(() => { calls++; throw new Error('blocked'); }, () => {}); unavailable.run(null); assert.equal(calls, 0); unavailable.run(weighted); assert.equal(calls, 1); assert.equal(unavailable.getSnapshot().status, 'error');
  for (const mode of ['post', 'error', 'messageerror', 'model']) {
    const port = new Port(); port.postFailure = mode === 'post'; const controller = new Controller(() => port, () => {}); controller.run(weighted);
    if (mode === 'error') port.onerror!.call(port as unknown as Worker, new Event('error') as ErrorEvent);
    if (mode === 'messageerror') port.onmessageerror!.call(port as unknown as Worker, new MessageEvent('messageerror'));
    if (mode === 'model') port.emit({ version: 1, requestId: 1, ok: false, error: 'calculation-failed' });
    assert.equal(controller.getSnapshot().status, 'error'); assert.equal(port.terminated, 1);
  }
});
test('cancellation/disposal and reentrant notification/factory replacement detach every port', () => {
  const { controller, ports, states } = setup(); controller.run(weighted); controller.cancel(); controller.run(gibbs); const count = states.length;
  controller.dispose(); controller.dispose(); controller.cancel(); assert.equal(controller.run(gaussian), null); assert.equal(states.length, count); assert.equal(ports[0].terminated, 1); assert.equal(ports[1].terminated, 1);
  let created = 0; const reentrant = new Controller(() => { created++; return new Port(); }, state => { if (state.status === 'pending') reentrant.cancel(); }); reentrant.run(weighted); assert.equal(created, 0);
  const old = new Port(), current = new Port(); let calls = 0;
  const replacement = new Controller(() => { if (++calls === 1) { replacement.run(gibbs); return old; } return current; }, () => {});
  replacement.run(weighted); assert.equal(old.terminated, 1); assert.equal(old.messages.length, 0); current.emit(evaluate(current.messages[0])); assert.equal(replacement.getSnapshot().status, 'ready');
});
test('SSR calculation and series hooks never read Worker or execute numerical work', () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let reads = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { reads++; throw new Error('SSR Worker'); } });
  function Probe() { const state = useBoltzmannGibbsCalculation(weighted), series = useBoltzmannGibbsSeries([null] as unknown as BoltzmannGibbsInput[]); assert.equal(state.response, null); assert.deepEqual(series.responses, []); return React.createElement('output', null, `${state.status}/${series.status}`); }
  try { assert.equal(renderToStaticMarkup(React.createElement(Probe)), '<output>idle/pending</output>'); assert.equal(reads, 0); }
  finally { if (descriptor) Object.defineProperty(globalThis, 'Worker', descriptor); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
test('observation transport distinguishes zero, unknown, finite sign and bounded rational payloads', () => {
  const r = evaluate({ version: 1, requestId: 1, input: weighted }); assert.ok(r.ok && r.kind === kinds[0]); const q = structuredClone(r.result.states[0].probability);
  assert.ok(observation(q)); q.bounds!.lower.denominator.coefficient = '0'; assert.equal(observation(q), false);
  assert.equal(observation({ value: null, sign: 1, logMagnitude: null, resolution: 'unresolved-at-precision-ceiling', logResolution: 'unresolved', bounds: null }), true);
  assert.equal(observation({ value: 0, sign: 1, logMagnitude: -800, resolution: 'interval-resolved', logResolution: 'finite', bounds: null }), false);
  assert.equal(observation({ value: null, sign: null, logMagnitude: null, resolution: 'mathematical-infinity', logResolution: 'not-applicable', bounds: null }), false);
  const huge = structuredClone(r.result.states[0].probability); huge.bounds!.lower.numerator.coefficient = '1'.repeat(boltzmannTransportLimits.coefficientCharacters + 1); assert.equal(observation(huge), false);
});

test('cyclic unexpected observation properties fail without stranding a worker', () => {
  const {controller,ports}=setup();controller.run(gibbs);const r=evaluate(ports[0].messages[0]);assert.ok(r.ok&&r.kind===kinds[1]);
  const flow=r.result.systematicDetailedBalanceWitness.forwardFlow as unknown as Record<string,unknown>;flow.extra=flow;
  ports[0].emit(r);assert.equal(controller.getSnapshot().status,'error');assert.equal(ports[0].terminated,1);
});
test('long dyadic contraction bounds larger than the old transport limit retain their honest unresolved scope', () => {
  const kB=1.380649e-23,input:BoltzmannGibbsInput={kind:kinds[1],parameters:{couplingEnergy:Number.MIN_VALUE,firstFieldEnergy:800*kB*Number.MAX_VALUE,secondFieldEnergy:800*kB*Number.MAX_VALUE,
    energyOffset:0,temperature:Number.MAX_VALUE,support:'full-product',observationSteps:[64]}};
  const r=evaluate({version:1,requestId:1,input});assert.ok(r.ok&&r.kind===kinds[1]);
  const q=r.result.totalVariationToTarget.systematic[0];assert.ok(q.bounds!.lower.numerator.coefficient.length>100000);assert.ok(valid(r,input,1));
  assert.equal(r.result.fullyResolved,false);assert.equal(q.sign,1);assert.equal(q.resolution,'interval-resolved');
});
