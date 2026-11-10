import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { build } from 'esbuild';
import { Worker as ThreadWorker } from 'node:worker_threads';
import { evaluateFiniteSizeRequest as evaluate } from '../src/lib/learning-finite-size-evaluator';
import { finiteSizeKinds as kinds, isFiniteSizeInput, snapshotFiniteSizeInput as snapshot, finiteSizeInputKey as key, isFiniteSizeSuccessForInput as valid, isFiniteSizeObservation as observation, type FiniteSizeInput } from '../src/lib/learning-finite-size-protocol';
import { FiniteSizeCalculationController as Controller, type FiniteSizeWorkerPort, type FiniteSizeCalculationState } from '../src/lib/learning-finite-size-controller';
import { useFiniteSizeCalculation } from '../src/components/learning/use-finite-size-calculation';
const boundary: FiniteSizeInput = { kind: kinds[0], parameters: { dimension: 2, lengthOne: 2, lengthTwo: 2, width: 3, growthMode: 'uniform-dilation', scale: 4, spinOne: 1, spinTwo: -1, couplingNumerator: 1, couplingDenominator: 1 } };
const spatial: FiniteSizeInput = { kind: kinds[1], parameters: { siteCount: 4, occupiedWeight: 1, emptyWeight: 1, commonNumerator: 1, commonDenominator: 2, commonScaling: 'fixed', epsilonNumerator: 1, epsilonDenominator: 2 } };
const twoPhase: FiniteSizeInput = { kind: kinds[2], parameters: { siteCount: 4, activityNumerator: 2, activityDenominator: 1, sizePath: 'fixed', approachSide: 1, offsetNumerator: 1, offsetDenominator: 2 } };
const normal = [boundary, spatial, twoPhase];
const heavy: FiniteSizeInput[] = [
 { ...boundary, parameters: { ...boundary.parameters, dimension: 3, lengthOne: 1000000, lengthTwo: 1000000, width: 1000000, scale: 1000000, couplingDenominator: 1000000 } },
 { ...spatial, parameters: { ...spatial.parameters, siteCount: 512, occupiedWeight: 1, emptyWeight: 999999, commonNumerator: 1, commonDenominator: 1000000, commonScaling: 'inverse-size-squared', epsilonNumerator: 999999, epsilonDenominator: 1000000 } },
 { ...twoPhase, parameters: { ...twoPhase.parameters, siteCount: 512, activityNumerator: 1, activityDenominator: 1000000 } },
];
class Port implements FiniteSizeWorkerPort {
 onmessage: Worker['onmessage'] = null; onerror: Worker['onerror'] = null; onmessageerror: Worker['onmessageerror'] = null;
 messages: unknown[] = []; terminated = 0; postFailure = false;
 postMessage(value: unknown) { if (this.postFailure) throw Error('clone'); this.messages.push(structuredClone(value)); }
 terminate() { this.terminated++; }
 emit(data: unknown) { this.onmessage?.call(this as unknown as Worker, new MessageEvent('message', { data })); }
}
function setup() { const ports: Port[] = [], states: FiniteSizeCalculationState[] = []; const controller = new Controller(() => { const p = new Port(); ports.push(p); return p; }, s => states.push(s)); return { ports, states, controller }; }
for (const [i, file] of ['boundary-extensivity', 'spatial-self-averaging', 'two-phase-rounding'].entries()) test(`${kinds[i]}: all independent preparations pass real dispatch and complete cloned transport validation`, () => {
 let cases = 0;
 for (const c of JSON.parse(readFileSync(`scripts/fixtures/${file}-oracle.json`, 'utf8')).cases) { const input = { kind: kinds[i], parameters: c.parameters } as FiniteSizeInput; const response = evaluate({ version: 1, requestId: 1, input }); assert.ok(response.ok, c.id); assert.ok(valid(response, input, 1), c.id); assert.ok(valid(structuredClone(response), input, 1), c.id); assert.doesNotThrow(() => JSON.stringify(response)); cases++; }
 assert.equal(cases, [27, 32, 34][i]);
});
test('canonical snapshots own all scalar fields, normalize signed zero and ignore property order', () => {
 for (const input of normal) { const reverse = { parameters: Object.fromEntries(Object.entries(input.parameters).reverse()), kind: input.kind }; assert.ok(isFiniteSizeInput(reverse)); assert.equal(key(reverse), key(input)); const snap = snapshot(input); assert.ok(Object.isFrozen(snap)); assert.ok(Object.isFrozen(snap.parameters)); assert.notEqual(snap.parameters, input.parameters); }
 assert.equal(key({ ...spatial, parameters: { ...spatial.parameters, commonNumerator: -0 } }), key({ ...spatial, parameters: { ...spatial.parameters, commonNumerator: 0 } }));
});
test('malformed inputs and inherited, extra or invalid request envelopes are rejected', () => {
 const bad = [null, {}, [], { ...boundary, parameters: { ...boundary.parameters, dimension: 1 } }, { ...boundary, parameters: { ...boundary.parameters, couplingDenominator: 0 } }, { ...spatial, parameters: { ...spatial.parameters, siteCount: 513 } }, { ...spatial, parameters: { ...spatial.parameters, occupiedWeight: 0, emptyWeight: 0 } }, { ...spatial, parameters: { ...spatial.parameters, commonNumerator: 3 } }, { ...twoPhase, parameters: { ...twoPhase.parameters, offsetNumerator: 3 } }, { ...twoPhase, parameters: { ...twoPhase.parameters, activityNumerator: 0 } }, { ...spatial, parameters: { ...spatial.parameters, siteCount: NaN } }, { ...twoPhase, precision: 8192 }];
 for (const input of bad) { assert.equal(isFiniteSizeInput(input), false); assert.deepEqual(evaluate({ version: 1, requestId: 1, input }), { version: 1, requestId: 1, ok: false, error: 'invalid-input' }); }
 for (const input of [null, {}, { version: 2, requestId: 1, input: boundary }, { version: 1, requestId: 0, input: boundary }, { version: 1, requestId: 1, input: boundary, extra: true }, Object.assign(Object.create({ version: 1 }), { requestId: 1, input: boundary, extra: true })]) { const r = evaluate(input); assert.ok(!r.ok); assert.equal(r.error, 'invalid-request'); }
});
test('current ACK, success and termination belong to the active input', () => {
 const { controller, ports, states } = setup(); controller.run(spatial); ports[0].emit({ version: 1, requestId: 1, phase: 'started', kind: kinds[1] }); const pending = controller.getSnapshot(); assert.ok(pending.status === 'pending'); assert.equal(pending.phase, 'calculating'); ports[0].emit(evaluate(ports[0].messages[0])); assert.equal(controller.getSnapshot().status, 'ready'); assert.equal(ports[0].terminated, 1); assert.equal(ports[0].onmessage, null); assert.equal(states.filter(s => s.status === 'ready').length, 1);
});
test('replacement, old request ids and saved late callbacks cannot publish stale results', () => {
 const { controller, ports } = setup(); controller.run(boundary); const late = ports[0].onmessage!; controller.run(twoPhase); assert.equal(ports[0].terminated, 1); late.call(ports[0] as unknown as Worker, new MessageEvent('message', { data: evaluate(ports[0].messages[0]) })); ports[1].emit({ version: 1, requestId: 1, phase: 'started', kind: kinds[0] }); assert.equal(controller.getSnapshot().status, 'pending'); ports[1].emit(evaluate(ports[1].messages[0])); const ready = controller.getSnapshot(); assert.ok(ready.status === 'ready'); assert.equal(ready.response.kind, kinds[2]); assert.equal(ports[1].terminated, 1);
});
test('every required top-level result field and current field is checked', () => {
 for (const input of normal) {
  const good = evaluate({ version: 1, requestId: 1, input }); assert.ok(good.ok);
  for (const field of Object.keys(good.result)) { const r: any = structuredClone(good); delete r.result[field]; assert.equal(valid(r, input, 1), false, `${input.kind}.${field}`); }
  for (const field of Object.keys(good.result.current)) { const r: any = structuredClone(good); delete r.result.current[field]; assert.equal(valid(r, input, 1), false, `${input.kind}.current.${field}`); }
  for (const field of ['inputKey', 'kind', 'requestId', 'version', 'input']) { const r: any = structuredClone(good); r[field] = null; assert.equal(valid(r, input, 1), false, field); }
  const cycle: any = structuredClone(good); cycle.result.extra = cycle; assert.equal(valid(cycle, input, 1), false);
 }
});
test('history endpoints, distribution counts, event identities, null meanings and root identity reject corruption', () => {
 const change = (input: FiniteSizeInput, mutate: (r: any) => void) => { const r = structuredClone(evaluate({ version: 1, requestId: 1, input })); mutate(r); assert.equal(valid(r, input, 1), false); };
 for (const input of normal) change(input, r => { r.result.history[0] = r.result.history.at(-1); });
 change(spatial, r => { r.result.current.distribution[0].count = 1; }); change(spatial, r => { r.result.current.eventCounts = [1]; });
 const deterministic: FiniteSizeInput = { ...spatial, parameters: { ...spatial.parameters, occupiedWeight: 0 } }; change(deterministic, r => { r.result.current.distinctPairCorrelation = r.result.current.siteVariance; });
 const singleton: FiniteSizeInput = { ...spatial, parameters: { ...spatial.parameters, siteCount: 1 } }; change(singleton, r => { r.result.current.distinctPairCovariance = r.result.current.siteVariance; });
 change(twoPhase, r => { r.result.roots[0].conjugateIndex = 0; }); change(twoPhase, r => { r.result.roots[0].turns = r.result.roots[1].turns; }); change(twoPhase, r => { r.result.rootGeometry.coordinatesAreApproximate = false; }); change(twoPhase, r => { r.result.current.distribution[1].count = 1; });
});
test('point and interval observations reject invalid signs, denominators, exponents, infinity and unsupported precision', () => {
 const r = evaluate({ version: 1, requestId: 1, input: twoPhase }); assert.ok(r.ok && r.kind === kinds[2]); const q = r.result.current.pressureCorrection; assert.ok(observation(q)); assert.equal(observation(q, false), false);
 for (const mode of ['zero-denominator', 'digits', 'exponent', 'sign', 'infinity', 'resolution', 'negative-zero', 'missing']) { const x: any = structuredClone(q); if (mode === 'zero-denominator') x.bounds.lower.denominator.coefficient = '0'; if (mode === 'digits') x.bounds.lower.numerator.coefficient = '1'.repeat(16385); if (mode === 'exponent') x.bounds.lower.numerator.exponent = 1; if (mode === 'sign') x.sign = -1; if (mode === 'infinity') x.value = Infinity; if (mode === 'resolution') x.resolution = 'unresolved-at-precision-ceiling'; if (mode === 'negative-zero') x.bounds.lower.numerator.coefficient = '-0'; if (mode === 'missing') delete x.bounds.upper; assert.equal(observation(x), false, mode); }
 const point: any = structuredClone(r.result.current.occupiedProbability); point.bounds.upper = { ...point.bounds.upper, numerator: { ...point.bounds.upper.numerator, coefficient: '3' } }; assert.equal(observation(point), false);
});
test('creation, posting, malformed ACK/error, execution and message decoding fail without synchronous fallback', () => {
 const unavailable = new Controller(() => { throw Error('blocked'); }, () => {}); unavailable.run(boundary); assert.equal(unavailable.getSnapshot().status, 'error');
 for (const mode of ['post', 'error', 'decode', 'ack', 'failure', 'unknown-error', 'extra-error', 'inherited-ack']) { const port = new Port(); port.postFailure = mode === 'post'; const c = new Controller(() => port, () => {}); c.run(twoPhase);
  if (mode === 'error') port.onerror!.call(port as unknown as Worker, new Event('error') as ErrorEvent); if (mode === 'decode') port.onmessageerror!.call(port as unknown as Worker, new MessageEvent('messageerror')); if (mode === 'ack') port.emit({ version: 1, requestId: 1, phase: 'started', kind: kinds[2], extra: true }); if (mode === 'failure') port.emit({ version: 1, requestId: 1, ok: false, error: 'calculation-failed' }); if (mode === 'unknown-error') port.emit({ version: 1, requestId: 1, ok: false, error: 'unknown' }); if (mode === 'extra-error') port.emit({ version: 1, requestId: 1, ok: false, error: 'invalid-input', extra: true }); if (mode === 'inherited-ack') port.emit(Object.assign(Object.create({ version: 1 }), { requestId: 1, phase: 'started', kind: kinds[2], extra: true }));
  assert.equal(c.getSnapshot().status, 'error', mode); assert.equal(port.terminated, 1, mode);
 }
});
test('cancel, dispose, invalid replacement and reentrant notification terminate each owned port once', () => {
 const { controller, ports, states } = setup(); controller.run(spatial); controller.cancel(); assert.equal(ports[0].terminated, 1); assert.equal(controller.getSnapshot().status, 'idle'); controller.run(boundary); controller.run(null); assert.equal(ports[1].terminated, 1); assert.equal(controller.getSnapshot().status, 'error'); controller.run(twoPhase); controller.dispose(); controller.dispose(); assert.equal(ports[2].terminated, 1); const count = states.length; controller.run(boundary); controller.cancel(); assert.equal(states.length, count);
 let factories = 0; let c: Controller; c = new Controller(() => { factories++; return new Port(); }, s => { if (s.status === 'pending') c.cancel(); }); c.run(boundary); assert.equal(factories, 0);
});
test('SSR uses no Worker and the hook main bundle excludes all numerical implementations', async () => {
 const original = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let reads = 0; Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { reads++; throw Error('SSR Worker'); } });
 function Probe({ input, enabled = true }: { input: FiniteSizeInput | null; enabled?: boolean }) { const state = useFiniteSizeCalculation(input, enabled); assert.equal(state.status, 'idle'); assert.equal(state.response, null); return React.createElement('output', null, state.status); }
 try { for (const input of [...normal, null]) renderToStaticMarkup(React.createElement(Probe, { input })); renderToStaticMarkup(React.createElement(Probe, { input: boundary, enabled: false })); assert.equal(reads, 0); } finally { if (original) Object.defineProperty(globalThis, 'Worker', original); else Reflect.deleteProperty(globalThis, 'Worker'); }
 const r = await build({ entryPoints: ['src/components/learning/use-finite-size-calculation.ts'], bundle: true, write: false, format: 'esm', platform: 'browser', metafile: true, external: ['react'], logLevel: 'silent' }); const paths = Object.keys(r.metafile!.inputs);
 for (const name of ['learning-boundary-extensivity.ts', 'learning-spatial-self-averaging.ts', 'learning-two-phase-rounding.ts', 'learning-finite-size-evaluator.ts', 'learning-finite-markov-exact.ts', 'learning-binary-products.ts', 'learning-interval-transcendentals.ts', 'learning-rational-gcd.ts']) assert.ok(!paths.some(p => p.endsWith(name)), name);
 assert.ok(paths.some(p => p.endsWith('learning-finite-size-worker-factory.ts'))); mkdirSync('.runtime/learning/finite-size-worker', { recursive: true }); writeFileSync('.runtime/learning/finite-size-worker/hook-metafile.json', JSON.stringify(r.metafile, null, 2) + '\n');
});
test('real Node workers cancel on started ACKs and recover with fresh successful requests', async () => {
 const built = await build({ entryPoints: ['src/lib/learning-finite-size.worker.ts'], bundle: true, write: false, format: 'cjs', platform: 'node', target: 'es2022', logLevel: 'silent' });
 const code = `const {parentPort}=require('node:worker_threads');globalThis.self={postMessage(v){parentPort.postMessage(v)},set onmessage(fn){parentPort.on('message',data=>fn({data}))}};\n${built.outputFiles[0].text}`;
 const records: unknown[] = [];
 for (let i = 0; i < 3; i++) {
  const ports: { thread: ThreadWorker; ended: Promise<number>; terminated: number }[] = []; let cancelAtStart = true; let cancelResolve = () => {}, readyResolve = () => {}; const cancelled = new Promise<void>(r => { cancelResolve = r; });
  const c = new Controller(() => { const thread = new ThreadWorker(code, { eval: true }), item = { thread, ended: Promise.resolve(0), terminated: 0 }; item.ended = new Promise(resolve => thread.once('exit', resolve)); const port: FiniteSizeWorkerPort = { onmessage: null, onerror: null, onmessageerror: null, postMessage: v => thread.postMessage(v), terminate() { item.terminated++; void thread.terminate(); } }; thread.on('message', data => port.onmessage?.call(port as unknown as Worker, new MessageEvent('message', { data }))); thread.on('error', () => port.onerror?.call(port as unknown as Worker, new Event('error') as ErrorEvent)); ports.push(item); return port; }, state => { if (cancelAtStart && state.status === 'pending' && state.phase === 'calculating') { c.cancel(); cancelResolve(); } if (state.status === 'ready') readyResolve(); });
  const timeout = () => new Promise<never>((_, reject) => { const timer = setTimeout(() => reject(Error('native worker timeout')), 30000); timer.unref(); });
  try { c.run(heavy[i]); await Promise.race([cancelled, timeout()]); await ports[0].ended; assert.equal(c.getSnapshot().status, 'idle'); cancelAtStart = false; const ready = new Promise<void>(r => { readyResolve = r; }); c.run(normal[i]); await Promise.race([ready, timeout()]); await ports[1].ended; const state = c.getSnapshot(); assert.ok(state.status === 'ready'); assert.ok(valid(state.response, normal[i], 2)); assert.ok(ports.every(p => p.terminated === 1)); records.push({ kind: kinds[i], startedAckCancellation: true, freshThreadRecovery: true, ownedPorts: ports.length }); }
  finally { c.dispose(); await Promise.all(ports.map(async p => { await p.thread.terminate(); await p.ended; })); }
 }
 writeFileSync('.runtime/learning/finite-size-worker/node-verification.json', JSON.stringify({ scope: 'Native Node worker entry; cancellation on observed started ACK; no CPU interruption timing claim for short boundary calculations', records }, null, 2) + '\n');
});
