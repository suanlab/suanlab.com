import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { build } from 'esbuild';
import { evaluateQuarticAsymptoticRequest as calculate } from '../src/lib/learning-quartic-asymptotic-evaluator';
import { evaluateQuarticAsymptoticValidationRequest as validate } from '../src/lib/learning-quartic-asymptotic-validation-evaluator';
import { validateQuarticAsymptoticResult } from '../src/lib/learning-quartic-asymptotic-validation';
import { quarticAsymptoticDefault, quarticAsymptoticExamples, type QuarticAsymptoticInput } from '../src/lib/learning-quartic-asymptotic-input';
import type { QuarticAsymptoticSuccess } from '../src/lib/learning-quartic-asymptotic-protocol';
const root = '.runtime/learning/quartic-asymptotic-worker';
const cases = JSON.parse(readFileSync('scripts/fixtures/quartic-asymptotic-reference.json', 'utf8')).cases as { id: string; input: QuarticAsymptoticInput; coefficients: string[]; globallyBestBoundOrder: number }[];
for (const c of cases) test(`independent full quartic response: ${c.id}`, () => {
  const started: unknown[] = [], response = calculate({ version: 1, requestId: 1, input: c.input }, message => started.push(message));
  assert(response.ok); assert.deepEqual(response.result.coefficients, c.coefficients); assert.equal(response.result.globallyBestBoundOrder, c.globallyBestBoundOrder);
  assert.deepEqual(started, [{ version: 1, requestId: 1, phase: 'started', kind: 'quartic-asymptotic' }]);
  const receipt = validate({ version: 1, requestId: 1, input: c.input, response });
  assert.equal(receipt.phase, 'validated', c.id);
  if (receipt.phase === 'validated') { assert.deepEqual(receipt.response, response); assert(Object.isFrozen(receipt.response.result.gaussianNormalization)); }
});

test('each response leaf, including quadrature bounds and global recommendations, rejects mutation', () => {
  const input = quarticAsymptoticDefault, response = calculate({ version: 1, requestId: 2, input }); assert(response.ok);
  const paths: (string | number)[][] = [];
  function visit(value: unknown, path: (string | number)[]) {
    if (value && typeof value === 'object') { for (const [key, child] of Object.entries(value)) if (key !== 'input') visit(child, [...path, Array.isArray(value) ? Number(key) : key]); }
    else paths.push(path);
  }
  visit(response.result, []);
  for (const path of paths) {
    const changed: QuarticAsymptoticSuccess = structuredClone(response); let parent = changed.result as unknown as Record<string | number, unknown>;
    for (const key of path.slice(0, -1)) parent = parent[key] as Record<string | number, unknown>;
    const key = path.at(-1)!, old = parent[key]; parent[key] = typeof old === 'boolean' ? !old : typeof old === 'number' ? old + 1 : old === '0' ? '1' : '0';
    assert.equal(validate({ version: 1, requestId: 2, input, response: changed }).phase, 'rejected', path.join('.'));
  }
  mkdirSync(root, { recursive: true }); writeFileSync(`${root}/leaf-mutations.json`, JSON.stringify({ rejected: paths.length, oneLeafPerClone: true }, null, 2) + '\n');
  assert(paths.length > 170);
});

test('raw spelling, retained order, request IDs and message kind retain attribution', () => {
  const input = quarticAsymptoticDefault, response = calculate({ version: 1, requestId: 3, input }); assert(response.ok);
  for (const other of [{ ...input, coupling: '2/20' }, { ...input, order: 13 }]) assert.equal(validate({ version: 1, requestId: 3, input: other, response }).phase, 'rejected');
  assert.equal(validate({ version: 1, requestId: 4, input, response }).phase, 'rejected');
  const changed = structuredClone(response) as unknown as Record<string, unknown>; changed.kind = 'regular-perturbation';
  assert.equal(validate({ version: 1, requestId: 3, input, response: changed }).phase, 'rejected');
});

test('nonfinite orders, negative couplings, extra fields and hostile wire data are rejected', () => {
  const input = quarticAsymptoticDefault;
  for (const request of [{ version: 2, requestId: 1, input }, { version: 1, requestId: 0, input }, { version: 1, requestId: 1.5, input }, { version: 1, requestId: 1, input, extra: 1 }, { version: 1, requestId: 1, input: { coupling: '-1', order: 12 } }]) assert.equal(calculate(request).ok, false);
  let calls = 0; assert.equal(calculate({ get version() { calls++; return 1; }, requestId: 1, input }).ok, false);
  const response = calculate({ version: 1, requestId: 5, input }); assert(response.ok);
  const getter = structuredClone(response.result); Object.defineProperty(getter.quadrature, 'halfLineLower', { get() { calls++; return '0'; } });
  assert.equal(validateQuarticAsymptoticResult(input, getter), null); assert.equal(calls, 0);
  const huge = structuredClone(response); huge.result.quadratureInterval.lower = '1'.repeat(65537);
  const receipt = validate({ version: 1, requestId: 5, input, response: huge }); assert.equal(receipt.phase, 'rejected'); if (receipt.phase === 'rejected') assert.equal(receipt.error, 'resource-limit');
  const extra = structuredClone(response.result) as unknown as Record<string, unknown>; extra.unclaimed = true; assert.equal(validateQuarticAsymptoticResult(input, extra), null);
});

test('untrusted output fractions are compared as text, never parsed for arithmetic', () => {
  const input = quarticAsymptoticDefault, response = calculate({ version: 1, requestId: 6, input }); assert(response.ok);
  const changed = structuredClone(response); changed.result.quadratureInterval.lower = '9'.repeat(64000);
  let called = false; const original = globalThis.BigInt;
  globalThis.BigInt = ((value: Parameters<typeof BigInt>[0]) => { if (typeof value === 'string' && value.length > 25) { called = true; throw Error('Untrusted operand'); } return original(value); }) as typeof BigInt;
  try { assert.equal(validateQuarticAsymptoticResult(input, changed.result), null); assert.equal(called, false); } finally { globalThis.BigInt = original; }
});

test('Gaussian, maximum and tiny positive cases cannot be forged into different physical conclusions', () => {
  for (const id of ['gaussian-exact', 'tiny-positive', 'maximum-coprime']) {
    const input = quarticAsymptoticExamples.find(c => c.id === id)!.input, response = calculate({ version: 1, requestId: 7, input }); assert(response.ok);
    const changed: QuarticAsymptoticSuccess = structuredClone(response);
    changed.result.series = response.result.series === 'exact-gaussian' ? 'divergent-asymptotic' : 'exact-gaussian';
    changed.result.budgetReachesBestBound = !changed.result.budgetReachesBestBound;
    assert.equal(validate({ version: 1, requestId: 7, input, response: changed }).phase, 'rejected');
  }
});

test('reader hook excludes arithmetic; independent validator excludes producer and transcendental engines', async () => {
  for (const entry of ['src/components/learning/use-quartic-asymptotic-calculation.ts', 'src/lib/learning-quartic-asymptotic-validation.worker.ts']) {
    const bundle = await build({ entryPoints: [entry], bundle: true, write: false, platform: 'browser', metafile: true, logLevel: 'silent' });
    const inputs = Object.keys(bundle.metafile!.inputs);
    for (const name of ['learning-quartic-asymptotic.ts', 'learning-quartic-asymptotic-integral.ts', 'learning-quartic-asymptotic-evaluator.ts', 'learning-interval-transcendentals.ts', 'learning-quantum-hilbert-core.ts']) assert(!inputs.includes('src/lib/' + name));
    if (!entry.includes('validation.worker')) { assert(!inputs.includes('src/lib/learning-quartic-asymptotic-validation.ts')); assert(!bundle.outputFiles[0].text.includes('BigInt(')); }
  }
});

test('React hook renders idle on the server without creating a Worker', async () => {
  const { createElement } = await import('react'), { renderToString } = await import('react-dom/server'), { useQuarticAsymptoticCalculation } = await import('../src/components/learning/use-quartic-asymptotic-calculation');
  const before = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let calls = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { calls++; throw Error('SSR Worker'); } });
  try { function Probe() { return createElement('span', null, useQuarticAsymptoticCalculation(quarticAsymptoticDefault, true).status); } assert.equal(renderToString(createElement(Probe)), '<span>idle</span>'); assert.equal(calls, 0); }
  finally { if (before) Object.defineProperty(globalThis, 'Worker', before); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
