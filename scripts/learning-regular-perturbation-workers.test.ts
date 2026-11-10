import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { createHash } from 'node:crypto';
import { validateRegularPerturbationResult } from '../src/lib/learning-regular-perturbation-validation';
import { evaluateRegularPerturbationRequest as calculate } from '../src/lib/learning-regular-perturbation-evaluator';
import { evaluateRegularPerturbationValidationRequest as validate } from '../src/lib/learning-regular-perturbation-validation-evaluator';
import { regularPerturbationExamples, regularPerturbationDefault, type RegularPerturbationInput } from '../src/lib/learning-regular-perturbation-input';
import type { RegularPerturbationSuccess } from '../src/lib/learning-regular-perturbation-protocol';

const root = '.runtime/learning/regular-perturbation-worker';
const cases = JSON.parse(readFileSync('scripts/fixtures/regular-perturbation-reference.json', 'utf8')).cases as { id: string; input: RegularPerturbationInput; coefficients: string[] }[];
const budget = JSON.parse(readFileSync('scripts/fixtures/regular-perturbation-budget-reference.json', 'utf8')) as { id: string; input: RegularPerturbationInput; payloadSha256: string }[];
const stable = (value: unknown): string => Array.isArray(value) ? `[${value.map(stable).join(',')}]` : value && typeof value === 'object' ? `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stable((value as Record<string, unknown>)[key])}`).join(',')}}` : JSON.stringify(value);
for (const c of cases) test(`independent full regular perturbation response: ${c.id}`, () => {
  const started: unknown[] = [], response = calculate({ version: 1, requestId: 1, input: c.input }, message => started.push(message));
  assert(response.ok);
  assert.deepEqual(response.result.coefficients, c.coefficients);
  assert.deepEqual(started, [{ version: 1, requestId: 1, phase: 'started', kind: 'regular-perturbation' }]);
  const receipt = validate({ version: 1, requestId: 1, input: c.input, response });
  assert.equal(receipt.phase, 'validated');
  if (receipt.phase === 'validated') {
    assert.deepEqual(receipt.response, response);
    assert(Object.isFrozen(receipt.response.result.approximations[0].rootDistance));
  }
});

for (const c of budget) test(`coprime maximum input, independent Python payload and owned wire: ${c.id}`, () => {
  const response = calculate({ version: 1, requestId: 8, input: c.input }); assert(response.ok);
  const result = response.result;
  const payload = { coefficients: result.coefficients, residualPolynomial: result.residualPolynomial, dimensionlessCoupling: result.dimensionlessCoupling,
    approximations: result.approximations.map(row => Object.fromEntries(Object.entries(row).filter(([key]) => key !== 'rootDistance'))) };
  assert.equal(createHash('sha256').update(stable(payload)).digest('hex'), c.payloadSha256);
  assert.equal(validate({ version: 1, requestId: 8, input: c.input, response }).phase, 'validated');
});

test('each individual response leaf rejects a one-leaf mutation', () => {
  const input = regularPerturbationDefault, response = calculate({ version: 1, requestId: 2, input });
  assert(response.ok);
  const paths: (string | number)[][] = [];
  function visit(value: unknown, path: (string | number)[]) {
    if (value && typeof value === 'object') for (const [key, child] of Object.entries(value)) { if (key !== 'input') visit(child, [...path, Array.isArray(value) ? Number(key) : key]); }
    else paths.push(path);
  }
  visit(response.result, []);
  for (const path of paths) {
    const altered: RegularPerturbationSuccess = structuredClone(response);
    let parent = altered.result as unknown as Record<string | number, unknown>;
    for (const key of path.slice(0, -1)) parent = parent[key] as Record<string | number, unknown>;
    const key = path.at(-1)!, old = parent[key];
    parent[key] = typeof old === 'number' ? old + 1 : old === '0' ? '1' : '0';
    assert.equal(validate({ version: 1, requestId: 2, input, response: altered }).phase, 'rejected', path.join('.'));
  }
  assert.equal(paths.length, 77);
  mkdirSync(root, { recursive: true });
  writeFileSync(`${root}/leaf-mutations.json`, JSON.stringify({ rejected: paths.length, oneLeafPerClone: true }, null, 2) + '\n');
});

test('radius class, root signs, cancellation, residual and error bounds cannot be forged', () => {
  for (const id of ['outside-radius', 'critical-radius', 'zero-source', 'maximum-soft']) {
    const input = regularPerturbationExamples.find(c => c.id === id)!.input;
    const response = calculate({ version: 1, requestId: 3, input }); assert(response.ok);
    for (const mutate of [
      (x: RegularPerturbationSuccess) => { x.result.convergenceRegion = 'inside-radius'; },
      (x: RegularPerturbationSuccess) => { x.result.root.lowerResidual = '123'; },
      (x: RegularPerturbationSuccess) => { x.result.residualPolynomial[0] = '1'; },
      (x: RegularPerturbationSuccess) => { x.result.approximations.at(-1)!.errorBound = '-1'; },
      (x: RegularPerturbationSuccess) => { x.result.susceptibility.upper = '-1'; },
    ]) {
      const altered: RegularPerturbationSuccess = structuredClone(response); mutate(altered);
      assert.equal(validate({ version: 1, requestId: 3, input, response: altered }).phase, 'rejected', id);
    }
  }
});

test('input spelling, order, source sign, message kind and request IDs retain attribution', () => {
  const input = regularPerturbationDefault, response = calculate({ version: 1, requestId: 4, input }); assert(response.ok);
  for (const other of [{ ...input, stiffness: '2/2' }, { ...input, source: '-1' }, { ...input, order: 5 }]) {
    assert.equal(validate({ version: 1, requestId: 4, input: other, response }).phase, 'rejected');
  }
  assert.equal(validate({ version: 1, requestId: 5, input, response }).phase, 'rejected');
  const changed = structuredClone(response) as unknown as Record<string, unknown>; changed.kind = 'graph-gauss';
  assert.equal(validate({ version: 1, requestId: 4, input, response: changed }).phase, 'rejected');
});

test('bounded owned transport rejects getters, cycles, extra fields and oversized strings', () => {
  const input = regularPerturbationDefault;
  for (const request of [{ version: 2, requestId: 1, input }, { version: 1, requestId: 0, input }, { version: 1, requestId: 1.5, input }, { version: 1, requestId: 1, input, extra: 0 }, { version: 1, requestId: 1, input: { ...input, stiffness: '0' } }]) assert.equal(calculate(request).ok, false);
  let calls = 0;
  assert.equal(calculate({ get version() { calls++; return 1; }, requestId: 1, input }).ok, false);
  const response = calculate({ version: 1, requestId: 6, input }); assert(response.ok);
  const accessor = structuredClone(response.result);
  Object.defineProperty(accessor.root, 'lower', { get() { calls++; return '0'; } });
  assert.equal(validateRegularPerturbationResult(input, accessor), null); assert.equal(calls, 0);
  const oversized = structuredClone(response); oversized.result.linearScale = '1'.repeat(65537);
  const rejected = validate({ version: 1, requestId: 6, input, response: oversized });
  assert.equal(rejected.phase, 'rejected'); if (rejected.phase === 'rejected') assert.equal(rejected.error, 'resource-limit');
  const extra = structuredClone(response.result) as unknown as Record<string, unknown>; extra.unclaimed = true;
  assert.equal(validateRegularPerturbationResult(input, extra), null);
  const cycle: Record<string, unknown> = {}; cycle.value = cycle;
  assert.equal(validateRegularPerturbationResult(input, cycle), null);
});

test('untrusted output text is never parsed as a BigInt', () => {
  const input = regularPerturbationDefault, response = calculate({ version: 1, requestId: 7, input }); assert(response.ok);
  const altered = structuredClone(response); altered.result.approximations[0].errorBound = '9'.repeat(64000);
  let untrustedParsed = false; const original = globalThis.BigInt;
  globalThis.BigInt = ((value: Parameters<typeof BigInt>[0]) => { if (typeof value === 'string' && value.length > 25) { untrustedParsed = true; throw Error('Untrusted arithmetic'); } return original(value); }) as typeof BigInt;
  try { assert.equal(validateRegularPerturbationResult(input, altered.result), null); assert.equal(untrustedParsed, false); }
  finally { globalThis.BigInt = original; }
});

test('reader hook excludes arithmetic and validation bundles exclude the producer', async () => {
  for (const entry of ['src/components/learning/use-regular-perturbation-calculation.ts', 'src/lib/learning-regular-perturbation-validation.worker.ts']) {
    const bundle = await build({ entryPoints: [entry], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true, logLevel: 'silent' });
    const inputs = Object.keys(bundle.metafile!.inputs);
    for (const file of ['learning-regular-perturbation.ts', 'learning-regular-perturbation-evaluator.ts', 'learning-quantum-hilbert-core.ts']) assert(!inputs.includes('src/lib/' + file));
    if (!entry.includes('validation.worker')) { assert(!inputs.includes('src/lib/learning-regular-perturbation-validation.ts')); assert(!bundle.outputFiles[0].text.includes('BigInt(')); }
  }
});

test('the hook server-renders idle without constructing a Worker', async () => {
  const { createElement } = await import('react'), { renderToString } = await import('react-dom/server');
  const { useRegularPerturbationCalculation } = await import('../src/components/learning/use-regular-perturbation-calculation');
  const before = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let calls = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { calls++; throw Error('SSR Worker'); } });
  try {
    function Probe() { return createElement('span', null, useRegularPerturbationCalculation(regularPerturbationDefault, true).status); }
    assert.equal(renderToString(createElement(Probe)), '<span>idle</span>'); assert.equal(calls, 0);
  } finally { if (before) Object.defineProperty(globalThis, 'Worker', before); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
