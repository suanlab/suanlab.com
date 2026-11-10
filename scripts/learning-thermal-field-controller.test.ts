import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { renderToStaticMarkup } from 'react-dom/server';
import { createElement } from 'react';
import { build } from 'esbuild';
import { ThermalFieldCalculationController as Controller, type ThermalFieldCalculationState as State, type ThermalFieldWorkerPort as PortType } from '../src/lib/learning-thermal-field-controller';
import { evaluateThermalFieldRequest as calculate } from '../src/lib/learning-thermal-field-evaluator';
import { evaluateThermalFieldValidationRequest as validate } from '../src/lib/learning-thermal-field-validation-evaluator';
import { thermalFieldExamples, thermalFieldDefault } from '../src/lib/learning-thermal-field-input';
import { useThermalFieldCalculation } from '../src/components/learning/use-thermal-field-calculation';

class Port implements PortType {
  onmessage: PortType['onmessage'] = null; onerror: PortType['onerror'] = null; onmessageerror: PortType['onmessageerror'] = null;
  sent: unknown[] = []; terminated = 0; throwPost = false;
  postMessage(value: unknown) { if (this.throwPost) throw Error('post'); this.sent.push(structuredClone(value)); }
  terminate() { this.terminated++; }
  emit(data: unknown) { this.onmessage?.call(this as unknown as Worker, { data } as MessageEvent); }
}
const controllers: Controller[] = [];
after(() => controllers.forEach(c => c.dispose()));
function harness(options: { pause?: () => Promise<void>; nodesPerTask?: number } = {}, timeout = 30000) {
  const calculations: Port[] = [], validations: Port[] = [], states: State[] = [];
  const factories = { calculate: () => { const p = new Port(); calculations.push(p); return p; }, validate: () => { const p = new Port(); validations.push(p); return p; } };
  const controller = new Controller(factories, state => states.push(state), { pause: async () => {}, ...options }, timeout);
  controllers.push(controller);
  return { controller, calculations, validations, states, factories };
}
async function drain(check: () => boolean) {
  for (let i = 0; !check() && i < 10000; i++) await Promise.resolve();
  assert(check(), 'asynchronous copy did not finish');
}
function deliver(port: Port) { const result = calculate(port.sent[0], m => port.emit(m)); port.emit(result); return result; }
async function finish(h: ReturnType<typeof harness>) {
  await drain(() => h.validations.length > 0);
  const port = h.validations.at(-1)!; port.emit(validate(port.sent[0]));
  await drain(() => h.controller.getSnapshot().status !== 'pending');
}
for (const id of ['coupled-source', 'squared-temperature', 'massless-constraint', 'massless-divergence', 'two-dimensional']) {
  test(`two owned ports admit independently verified ${id} and release handlers`, async () => {
    const input = thermalFieldExamples.find(x => x.id === id)!.input, h = harness();
    h.controller.run(input); deliver(h.calculations[0]); await finish(h);
    const state = h.controller.getSnapshot(); assert.equal(state.status, 'ready');
    if (state.status !== 'ready') assert.fail('not ready');
    assert.deepEqual(state.response.input, input); assert(Object.isFrozen(state.response.result.exact));
    assert.deepEqual(h.states.map(s => s.status === 'pending' ? s.phase : s.status), ['starting', 'calculating', 'validating', 'ready']);
    for (const p of [...h.calculations, ...h.validations]) { assert.equal(p.terminated, 1); assert.equal(p.onmessage, null); assert.equal(p.onerror, null); assert.equal(p.onmessageerror, null); }
    if (id === 'massless-divergence') { assert.equal(state.response.result.exact.kind, 'not-normalizable'); assert.equal(state.response.result.observations, null); }
  });
}
test('a well-attributed response with wrong mathematics never reaches ready', async () => {
  const h = harness(); h.controller.run(thermalFieldDefault);
  const result = structuredClone(calculate(h.calculations[0].sent[0])); assert(result.ok);
  if (result.result.exact.kind !== 'normalizable') assert.fail('wrong branch');
  result.result.exact.thermalEnergy = '999'; h.calculations[0].emit(result); await finish(h);
  const state = h.controller.getSnapshot(); assert(state.status === 'error' && state.error === 'invalid-response');
  assert(!h.states.some(s => s.status === 'ready'));
});
test('replacement and cancellation release old ports and ignore captured late callbacks', async () => {
  const h = harness(); h.controller.run(thermalFieldDefault);
  const old = h.calculations[0], late = old.onmessage!;
  h.controller.run(thermalFieldExamples[2].input);
  late.call(old as unknown as Worker, { data: { bad: true } } as MessageEvent);
  assert.equal(h.controller.getSnapshot().requestId, 2); assert.equal(old.terminated, 1);
  deliver(h.calculations[1]); await drain(() => h.validations.length === 1);
  const gate = h.validations[0], lateGate = gate.onmessage!, response = validate(gate.sent[0]);
  h.controller.cancel(); lateGate.call(gate as unknown as Worker, { data: response } as MessageEvent);
  assert.equal(h.controller.getSnapshot().status, 'idle'); assert.equal(gate.terminated, 1);
  h.controller.run(thermalFieldDefault); deliver(h.calculations[2]);
  await drain(() => h.validations.length === 2); h.validations[1].emit(validate(h.validations[1].sent[0]));
  await drain(() => h.controller.getSnapshot().status === 'ready');
});
test('cancel during each ownership copy prevents validator creation or ready publication', async () => {
  let resume = () => {};
  const first = harness({ nodesPerTask: 1, pause: () => new Promise<void>(r => { resume = r; }) });
  first.controller.run(thermalFieldDefault); deliver(first.calculations[0]); first.controller.cancel(); resume(); await Promise.resolve();
  assert.equal(first.validations.length, 0); assert.equal(first.controller.getSnapshot().status, 'idle');
  let admission = false;
  const second = harness({ nodesPerTask: 1, pause: () => admission ? new Promise<void>(r => { resume = r; }) : Promise.resolve() });
  second.controller.run(thermalFieldDefault); deliver(second.calculations[0]); await drain(() => second.validations.length === 1);
  admission = true; const p = second.validations[0]; p.emit(validate(p.sent[0])); second.controller.cancel(); resume(); await Promise.resolve();
  assert(!second.states.some(s => s.status === 'ready')); assert.equal(p.terminated, 1);
});
test('timeout stops an unresponsive calculation or validator and an immediate retry can succeed', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  for (const phase of ['calculate', 'validate']) {
    const h = harness({}, 50); h.controller.run(thermalFieldDefault);
    if (phase === 'validate') { deliver(h.calculations[0]); await drain(() => h.validations.length === 1); }
    t.mock.timers.tick(50);
    const state = h.controller.getSnapshot(); assert(state.status === 'error' && state.error === 'timeout');
    assert([...h.calculations, ...h.validations].every(p => p.terminated === 1));
    h.controller.run(thermalFieldDefault); deliver(h.calculations[1]);
    await drain(() => h.validations.length === (phase === 'validate' ? 2 : 1));
    const p = h.validations.at(-1)!; p.emit(validate(p.sent[0])); await drain(() => h.controller.getSnapshot().status === 'ready');
  }
});
test('factory, postMessage, worker and deserialization failures cannot strand either phase', async () => {
  const input = thermalFieldDefault;
  for (const phase of ['calculate', 'validate'] as const) for (const reason of ['factory', 'post', 'error', 'messageerror']) {
    const h = harness();
    if (reason === 'factory') h.factories[phase] = () => { throw Error('unavailable'); };
    if (reason === 'post') {
      h.factories[phase] = () => { const p = new Port(); p.throwPost = true; (phase === 'calculate' ? h.calculations : h.validations).push(p); return p; };
    }
    h.controller.run(input);
    if (phase === 'validate') { deliver(h.calculations[0]); await drain(() => h.validations.length > 0 || h.controller.getSnapshot().status === 'error'); }
    const port = (phase === 'calculate' ? h.calculations : h.validations).at(-1);
    if (reason === 'error') port!.onerror?.call(port as unknown as Worker, { preventDefault() {} } as ErrorEvent);
    if (reason === 'messageerror') port!.onmessageerror?.call(port as unknown as Worker, {} as MessageEvent);
    assert.equal(h.controller.getSnapshot().status, 'error', `${phase}:${reason}`);
    assert([...h.calculations, ...h.validations].every(p => p.terminated === 1));
  }
});
test('invalid frames, getters, resource limits and disposal are terminal without executing caller accessors', () => {
  const h = harness(); h.controller.run({}); assert.equal(h.calculations.length, 0);
  h.controller.run(thermalFieldDefault); const p = h.calculations[0];
  p.emit({ version: 1, requestId: 99, ok: true }); assert.equal(h.controller.getSnapshot().status, 'pending');
  let calls = 0; p.emit({ get version() { calls++; return 1; } }); assert.equal(calls, 0); assert.equal(h.controller.getSnapshot().status, 'error');
  h.controller.run(thermalFieldDefault); h.calculations[1].emit({ version: 1, requestId: 3, ok: false, error: 'resource-limit' });
  const state = h.controller.getSnapshot(); assert(state.status === 'error' && state.error === 'resource-limit');
  h.controller.dispose(); const count = h.calculations.length; assert.equal(h.controller.run(thermalFieldDefault), null); assert.equal(h.calculations.length, count);
});
test('SSR stays idle without constructing Worker, and the hook excludes mathematical engines', async () => {
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let calls = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { calls++; throw Error('SSR Worker'); } });
  try {
    function Probe() { return createElement('span', null, useThermalFieldCalculation(thermalFieldDefault, true).status); }
    assert.equal(renderToStaticMarkup(createElement(Probe)), '<span>idle</span>'); assert.equal(calls, 0);
  } finally { if (saved) Object.defineProperty(globalThis, 'Worker', saved); else Reflect.deleteProperty(globalThis, 'Worker'); }
  const bundle = await build({ entryPoints: ['src/components/learning/use-thermal-field-calculation.ts'], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true });
  for (const file of ['learning-thermal-field.ts', 'learning-thermal-field-validation.ts', 'learning-thermal-field-observations.ts', 'learning-quantum-hilbert-core.ts']) assert(!Object.keys(bundle.metafile!.inputs).includes(`src/lib/${file}`));
});
