import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { build } from 'esbuild';
import { evaluateBrownianBridgeRequest as calculate } from '../src/lib/learning-brownian-bridge-evaluator';
import { evaluateBrownianBridgeValidationRequest as validate } from '../src/lib/learning-brownian-bridge-validation-evaluator';
import { brownianBridgeDefault, type BrownianBridgeInput } from '../src/lib/learning-brownian-bridge-input';
import { brownianBridgeInputKey } from '../src/lib/learning-brownian-bridge-protocol';
const root = '.runtime/learning/brownian-bridge-worker';
mkdirSync(root, { recursive: true });
const fixture = JSON.parse(readFileSync('scripts/fixtures/brownian-bridge-references.json', 'utf8')) as { cases: { id: string; input: BrownianBridgeInput; expected: unknown }[] };
for (const c of fixture.cases) test('complete calculation and verification envelopes: ' + c.id, () => {
  const events: unknown[] = [], response = calculate({ version: 1, requestId: 1, input: c.input }, message => events.push(message));
  assert(response.ok); assert.deepEqual(events, [{ version: 1, requestId: 1, phase: 'started', kind: 'brownian-bridge' }]);
  assert.deepEqual(response.result, c.expected); assert(Object.isFrozen(response) && Object.isFrozen(response.result));
  const receipt = validate({ version: 1, requestId: 1, input: c.input, response }); assert.equal(receipt.phase, 'validated');
  if (receipt.phase === 'validated') { assert.deepEqual(receipt.response, response); assert.notEqual(receipt.response, response); assert(Object.isFrozen(receipt.response.result.covariance[0])); }
});
test('request IDs, kinds, original fraction spellings and nested input attribution must all agree', () => {
  const input = { ...brownianBridgeDefault, time: '2/2' }, response = calculate({ version: 1, requestId: 3, input }); assert(response.ok);
  for (const request of [{ version: 1, requestId: 4, input, response }, { version: 1, requestId: 3, input: brownianBridgeDefault, response }, { version: 1, requestId: 3, input: { ...input, probe: 1 }, response }]) assert.equal(validate(request).phase, 'rejected');
  for (const change of [
    { ...response, kind: 'other' }, { ...response, inputKey: 'unrelated' }, { ...response, requestId: 4 },
    { ...response, input: { ...input, end: '3' } }, { ...response, result: { ...response.result, input: { ...input, source: '2' } } },
    { ...response, result: { ...response.result, area: { ...response.result.area, variance: '0' } } },
  ]) assert.equal(validate({ version: 1, requestId: 3, input, response: change }).phase, 'rejected');
});
test('field insertion order cannot split controller and hook attribution, while fraction spellings remain distinct', () => {
  const reordered = Object.fromEntries(Object.entries(brownianBridgeDefault).reverse()) as unknown as BrownianBridgeInput;
  assert.equal(brownianBridgeInputKey(reordered), brownianBridgeInputKey(brownianBridgeDefault));
  assert.notEqual(brownianBridgeInputKey({ ...reordered, time: '2/2' }), brownianBridgeInputKey(brownianBridgeDefault));
  const response = calculate({ version: 1, requestId: 3, input: reordered }); assert(response.ok);
  assert.equal(validate({ version: 1, requestId: 3, input: reordered, response }).phase, 'validated');
});
test('invalid inputs and malformed requests do not emit started or create a successful result', () => {
  for (const request of [null, {}, { version: 2, requestId: 1, input: brownianBridgeDefault }, ...[0, -1, 1.5, Infinity, Number.MAX_SAFE_INTEGER + 1].map(requestId => ({ version: 1, requestId, input: brownianBridgeDefault })), { version: 1, requestId: 1, input: { ...brownianBridgeDefault, diffusion: '0' } }]) {
    const events: unknown[] = []; assert.equal(calculate(request, event => events.push(event)).ok, false); assert.deepEqual(events, []);
    assert.equal(validate(request).phase, 'rejected');
  }
});
test('hostile envelopes reject accessors and report bounded-copy limits without evaluating candidate mathematics', () => {
  let reads = 0; const getter = { get version() { reads++; return 1; }, requestId: 1, input: brownianBridgeDefault };
  assert.equal(calculate(getter).ok, false); assert.equal(validate(getter).phase, 'rejected'); assert.equal(reads, 0);
  const response = calculate({ version: 1, requestId: 1, input: brownianBridgeDefault }); assert(response.ok);
  const receipt = validate({ version: 1, requestId: 1, input: brownianBridgeDefault, response: { ...response, result: { ...response.result, kernel: { ...response.result.kernel, formula: 'x'.repeat(65537) } } } });
  assert.deepEqual(receipt, { version: 1, requestId: 1, phase: 'rejected', error: 'resource-limit' });
  const proxy = new Proxy({}, { getPrototypeOf() { throw Error('invalid object'); } });
  assert.equal(calculate(proxy).ok, false); assert.equal(validate(proxy).phase, 'rejected');
});
test('reader hook excludes mathematical engines and the validator Worker never imports its producer', async () => {
  for (const [label, entry] of [['hook', 'src/components/learning/use-brownian-bridge-calculation.ts'], ['calculate', 'src/lib/learning-brownian-bridge.worker.ts'], ['validate', 'src/lib/learning-brownian-bridge-validation.worker.ts']] as const) {
    const bundle = await build({ entryPoints: [entry], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true }), inputs = Object.keys(bundle.metafile!.inputs);
    if (label !== 'calculate') { assert(!inputs.includes('src/lib/learning-brownian-bridge.ts')); assert(!inputs.includes('src/lib/learning-quantum-hilbert-core.ts')); }
    if (label === 'hook') { assert(!inputs.includes('src/lib/learning-brownian-bridge-validation.ts')); assert(!bundle.outputFiles[0].text.includes('BigInt(')); }
    writeFileSync(`${root}/${label}-dependency-graph.json`, JSON.stringify(bundle.metafile, null, 2) + '\n');
  }
});
test('hook server rendering stays idle without reading the Worker global', async () => {
  const React = await import('react'), { renderToStaticMarkup } = await import('react-dom/server'), { useBrownianBridgeCalculation } = await import('../src/components/learning/use-brownian-bridge-calculation');
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let reads = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { reads++; throw Error('SSR Worker'); } });
  try { function Probe() { return React.createElement('span', null, useBrownianBridgeCalculation(brownianBridgeDefault, true).status); } assert.equal(renderToStaticMarkup(React.createElement(Probe)), '<span>idle</span>'); assert.equal(reads, 0); }
  finally { if (saved) Object.defineProperty(globalThis, 'Worker', saved); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
