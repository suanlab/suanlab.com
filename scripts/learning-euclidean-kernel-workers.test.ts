import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { build } from 'esbuild';
import { evaluateEuclideanKernelRequest as calculate } from '../src/lib/learning-euclidean-kernel-evaluator';
import { evaluateEuclideanKernelValidationRequest as validate } from '../src/lib/learning-euclidean-kernel-validation-evaluator';
import { euclideanKernelDefault, type EuclideanKernelInput } from '../src/lib/learning-euclidean-kernel-input';
import { euclideanKernelInputKey } from '../src/lib/learning-euclidean-kernel-protocol';
const root = '.runtime/learning/euclidean-kernel-worker';
mkdirSync(root, { recursive: true });
const fixture = JSON.parse(readFileSync('scripts/fixtures/euclidean-kernel-references.json', 'utf8')) as { cases: { id: string; input: EuclideanKernelInput; expected: unknown }[] };
for (const c of fixture.cases) test('complete calculation and verification envelopes: ' + c.id, () => {
  const events: unknown[] = [], response = calculate({ version: 1, requestId: 1, input: c.input }, message => events.push(message));
  assert(response.ok); assert.deepEqual(events, [{ version: 1, requestId: 1, phase: 'started', kind: 'euclidean-kernel' }]);
  assert.deepEqual(response.result, c.expected); assert(Object.isFrozen(response) && Object.isFrozen(response.result));
  const receipt = validate({ version: 1, requestId: 1, input: c.input, response }); assert.equal(receipt.phase, 'validated');
  if (receipt.phase === 'validated') { assert.deepEqual(receipt.response, response); assert.notEqual(receipt.response, response); assert(Object.isFrozen(receipt.response.result.covariance.values[0])); }
});
test('request IDs, kinds, original fraction spellings and nested input attribution must all agree', () => {
  const input = { ...euclideanKernelDefault, time: '2/2' }, response = calculate({ version: 1, requestId: 3, input }); assert(response.ok);
  for (const request of [{ version: 1, requestId: 4, input, response }, { version: 1, requestId: 3, input: euclideanKernelDefault, response }, { version: 1, requestId: 3, input: { ...input, probe: 0 }, response }]) assert.equal(validate(request).phase, 'rejected');
  for (const change of [
    { ...response, version: 2 }, { ...response, ok: false }, { ...response, kind: 'other' }, { ...response, inputKey: 'unrelated' }, { ...response, requestId: 4 },
    { ...response, input: { ...input, end: '3' } }, { ...response, result: { ...response.result, input: { ...input, hbar: '2' } } },
    { ...response, result: { ...response.result, kernel: { ...response.result.kernel, minimumAction: '0' } } },
  ]) assert.equal(validate({ version: 1, requestId: 3, input, response: change }).phase, 'rejected');
});
test('field insertion order cannot split controller and hook attribution, while fraction spellings remain distinct', () => {
  const reordered = Object.fromEntries(Object.entries(euclideanKernelDefault).reverse()) as unknown as EuclideanKernelInput;
  assert.equal(euclideanKernelInputKey(reordered), euclideanKernelInputKey(euclideanKernelDefault));
  assert.notEqual(euclideanKernelInputKey({ ...reordered, time: '2/2' }), euclideanKernelInputKey(euclideanKernelDefault));
  const response = calculate({ version: 1, requestId: 3, input: reordered }); assert(response.ok);
  assert.equal(validate({ version: 1, requestId: 3, input: reordered, response }).phase, 'validated');
});
test('invalid inputs and malformed requests do not emit started or create a successful result', () => {
  for (const request of [null, {}, { version: 2, requestId: 1, input: euclideanKernelDefault }, ...[0, -1, 1.5, Infinity, Number.MAX_SAFE_INTEGER + 1].map(requestId => ({ version: 1, requestId, input: euclideanKernelDefault })), { version: 1, requestId: 1, input: { ...euclideanKernelDefault, hbar: '0' } }]) {
    const events: unknown[] = []; assert.equal(calculate(request, event => events.push(event)).ok, false); assert.deepEqual(events, []);
    assert.equal(validate(request).phase, 'rejected');
  }
});
test('hostile envelopes reject accessors and report bounded-copy limits without evaluating candidate mathematics', () => {
  let reads = 0; const getter = { get version() { reads++; return 1; }, requestId: 1, input: euclideanKernelDefault };
  assert.equal(calculate(getter).ok, false); assert.equal(validate(getter).phase, 'rejected'); assert.equal(reads, 0);
  const response = calculate({ version: 1, requestId: 1, input: euclideanKernelDefault }); assert(response.ok);
  const receipt = validate({ version: 1, requestId: 1, input: euclideanKernelDefault, response: { ...response, result: { ...response.result, kernel: { ...response.result.kernel, minimumAction: 'x'.repeat(65537) } } } });
  assert.deepEqual(receipt, { version: 1, requestId: 1, phase: 'rejected', error: 'resource-limit' });
  const proxy = new Proxy({}, { getPrototypeOf() { throw Error('invalid object'); } });
  assert.equal(calculate(proxy).ok, false); assert.equal(validate(proxy).phase, 'rejected');
});
test('reader hook excludes mathematical engines and the validator Worker never imports its producer', async () => {
  for (const [label, entry] of [['hook', 'src/components/learning/use-euclidean-kernel-calculation.ts'], ['calculate', 'src/lib/learning-euclidean-kernel.worker.ts'], ['validate', 'src/lib/learning-euclidean-kernel-validation.worker.ts']] as const) {
    const bundle = await build({ entryPoints: [entry], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true }), inputs = Object.keys(bundle.metafile!.inputs);
    if (label !== 'calculate') { assert(!inputs.includes('src/lib/learning-euclidean-kernel.ts')); assert(!inputs.includes('src/lib/learning-quantum-hilbert-core.ts')); }
    if (label === 'hook') { assert(!inputs.includes('src/lib/learning-euclidean-kernel-validation.ts')); assert(!bundle.outputFiles[0].text.includes('BigInt(')); }
    writeFileSync(`${root}/${label}-dependency-graph.json`, JSON.stringify(bundle.metafile, null, 2) + '\n');
  }
});
test('hook server rendering stays idle without reading the Worker global', async () => {
  const React = await import('react'), { renderToStaticMarkup } = await import('react-dom/server'), { useEuclideanKernelCalculation } = await import('../src/components/learning/use-euclidean-kernel-calculation');
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let reads = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { reads++; throw Error('SSR Worker'); } });
  try { function Probe() { return React.createElement('span', null, useEuclideanKernelCalculation(euclideanKernelDefault, true).status); } assert.equal(renderToStaticMarkup(React.createElement(Probe)), '<span>idle</span>'); assert.equal(reads, 0); }
  finally { if (saved) Object.defineProperty(globalThis, 'Worker', saved); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
