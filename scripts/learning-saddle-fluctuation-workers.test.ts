import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { build } from 'esbuild';
import { evaluateSaddleFluctuationRequest as calculate } from '../src/lib/learning-saddle-fluctuation-evaluator';
import { evaluateSaddleFluctuationValidationRequest as validate } from '../src/lib/learning-saddle-fluctuation-validation-evaluator';
import { saddleFluctuationDefault, type SaddleFluctuationInput } from '../src/lib/learning-saddle-fluctuation-input';
import { saddleFluctuationInputKey } from '../src/lib/learning-saddle-fluctuation-protocol';
const root = '.runtime/learning/saddle-fluctuation-worker';
mkdirSync(root, { recursive: true });
const fixture = JSON.parse(readFileSync('scripts/fixtures/saddle-fluctuation-references.json', 'utf8')) as { cases: { id: string; input: SaddleFluctuationInput; expected: unknown }[] };
for (const c of fixture.cases) test('complete calculation and verification envelopes: ' + c.id, () => {
  const events: unknown[] = [], response = calculate({ version: 1, requestId: 1, input: c.input }, message => events.push(message));
  assert(response.ok); assert.deepEqual(events, [{ version: 1, requestId: 1, phase: 'started', kind: 'saddle-fluctuation' }]);
  assert(Object.isFrozen(response) && Object.isFrozen(response.result));
  const receipt = validate({ version: 1, requestId: 1, input: c.input, response }); assert.equal(receipt.phase, 'validated');
  if (receipt.phase === 'validated') { assert.deepEqual(receipt.response, response); assert.notEqual(receipt.response, response); assert(Object.isFrozen(receipt.response.result.input)); }
});
test('request IDs, kinds, original fraction spellings and nested input attribution must all agree', () => {
  const input = { ...saddleFluctuationDefault, time: '2/2' }, response = calculate({ version: 1, requestId: 3, input }); assert(response.ok);
  for (const request of [{ version: 1, requestId: 4, input, response }, { version: 1, requestId: 3, input: saddleFluctuationDefault, response }, { version: 1, requestId: 3, input: { ...input, probe: 0 }, response }]) assert.equal(validate(request).phase, 'rejected');
  for (const change of [
    { ...response, version: 2 }, { ...response, ok: false }, { ...response, kind: 'other' }, { ...response, inputKey: 'unrelated' }, { ...response, requestId: 4 },
    { ...response, input: { ...input, end: '3' } }, { ...response, result: { ...response.result, input: { ...input, hbar: '2' } } },
    { ...response, result: { ...response.result, scope: { ...response.result.scope, unknownField: '0' } } },
  ]) assert.equal(validate({ version: 1, requestId: 3, input, response: change }).phase, 'rejected');
});
test('field insertion order cannot split controller and hook attribution, while fraction spellings remain distinct', () => {
  const reordered = Object.fromEntries(Object.entries(saddleFluctuationDefault).reverse()) as unknown as typeof saddleFluctuationDefault;
  assert.equal(saddleFluctuationInputKey(reordered), saddleFluctuationInputKey(saddleFluctuationDefault));
  assert.notEqual(saddleFluctuationInputKey({ ...reordered, time: '2/2' }), saddleFluctuationInputKey(saddleFluctuationDefault));
  const response = calculate({ version: 1, requestId: 3, input: reordered }); assert(response.ok);
  assert.equal(validate({ version: 1, requestId: 3, input: reordered, response }).phase, 'validated');
});
test('invalid inputs and malformed requests do not emit started or create a successful result', () => {
  for (const request of [null, {}, { version: 2, requestId: 1, input: saddleFluctuationDefault }, ...[0, -1, 1.5, Infinity, Number.MAX_SAFE_INTEGER + 1].map(requestId => ({ version: 1, requestId, input: saddleFluctuationDefault })), { version: 1, requestId: 1, input: { ...saddleFluctuationDefault, hbar: '0' } }]) {
    const events: unknown[] = []; assert.equal(calculate(request, event => events.push(event)).ok, false); assert.deepEqual(events, []);
    assert.equal(validate(request).phase, 'rejected');
  }
});
test('hostile envelopes reject accessors and report bounded-copy limits without evaluating candidate mathematics', () => {
  let reads = 0; const getter = { get version() { reads++; return 1; }, requestId: 1, input: saddleFluctuationDefault };
  assert.equal(calculate(getter).ok, false); assert.equal(validate(getter).phase, 'rejected'); assert.equal(reads, 0);
  const response = calculate({ version: 1, requestId: 1, input: saddleFluctuationDefault }); assert(response.ok);
  const receipt = validate({ version: 1, requestId: 1, input: saddleFluctuationDefault, response: { ...response, result: { ...response.result, scope: { ...response.result.scope, unknownField: 'x'.repeat(65537) } } } });
  assert.deepEqual(receipt, { version: 1, requestId: 1, phase: 'rejected', error: 'resource-limit' });
  const proxy = new Proxy({}, { getPrototypeOf() { throw Error('invalid object'); } });
  assert.equal(calculate(proxy).ok, false); assert.equal(validate(proxy).phase, 'rejected');
});
test('reader hook excludes mathematical engines and the validator Worker never imports its producer', async () => {
  for (const [label, entry] of [['hook', 'src/components/learning/use-saddle-fluctuation-calculation.ts'], ['calculate', 'src/lib/learning-saddle-fluctuation.worker.ts'], ['validate', 'src/lib/learning-saddle-fluctuation-validation.worker.ts']] as const) {
    const bundle = await build({ entryPoints: [entry], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true }), inputs = Object.keys(bundle.metafile!.inputs);
    if (label !== 'calculate') { assert(!inputs.includes('src/lib/learning-saddle-fluctuation.ts')); assert(!inputs.includes('src/lib/learning-quantum-hilbert-core.ts')); }
    if (label === 'hook') { assert(!inputs.includes('src/lib/learning-saddle-fluctuation-validation.ts')); assert(!bundle.outputFiles[0].text.includes('BigInt(')); }
    writeFileSync(`${root}/${label}-dependency-graph.json`, JSON.stringify(bundle.metafile, null, 2) + '\n');
  }
});
test('hook server rendering stays idle without reading the Worker global', async () => {
  const React = await import('react'), { renderToStaticMarkup } = await import('react-dom/server'), { useSaddleFluctuationCalculation } = await import('../src/components/learning/use-saddle-fluctuation-calculation');
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let reads = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { reads++; throw Error('SSR Worker'); } });
  try { function Probe() { return React.createElement('span', null, useSaddleFluctuationCalculation(saddleFluctuationDefault, true).status); } assert.equal(renderToStaticMarkup(React.createElement(Probe)), '<span>idle</span>'); assert.equal(reads, 0); }
  finally { if (saved) Object.defineProperty(globalThis, 'Worker', saved); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
