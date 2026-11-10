import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { build } from 'esbuild';
import { evaluateSpatialSpectrumRequest as calculate } from '../src/lib/learning-spatial-spectrum-evaluator';
import { evaluateSpatialSpectrumValidationRequest as validate } from '../src/lib/learning-spatial-spectrum-validation-evaluator';
import { validateSpatialSpectrumResult } from '../src/lib/learning-spatial-spectrum-validation';
import { spatialSpectrumDefault, spatialSpectrumExamples, type SpatialSpectrumInput } from '../src/lib/learning-spatial-spectrum-input';
import type { SpatialSpectrumSuccess } from '../src/lib/learning-spatial-spectrum-protocol';

const root = '.runtime/learning/spatial-spectrum-worker';
mkdirSync(root, { recursive: true });
const fixture = JSON.parse(readFileSync('scripts/fixtures/spatial-spectrum-reference.json', 'utf8')) as { cases: { id: string; input: SpatialSpectrumInput }[] };
for (const c of fixture.cases) test('independent entire spatial response: ' + c.id, () => {
  const events: unknown[] = [], response = calculate({ version: 1, requestId: 1, input: c.input }, m => events.push(m));
  assert(response.ok);
  assert.deepEqual(events, [{ version: 1, requestId: 1, phase: 'started', kind: 'spatial-spectrum' }]);
  const receipt = validate({ version: 1, requestId: 1, input: c.input, response });
  assert.equal(receipt.phase, 'validated', c.id);
  if (receipt.phase === 'validated') {
    assert.deepEqual(receipt.response, response);
    assert.notEqual(receipt.response, response);
    assert(Object.isFrozen(receipt.response.result.covariance[0]));
  }
});

function success(input: SpatialSpectrumInput, requestId = 2): SpatialSpectrumSuccess {
  const response = calculate({ version: 1, requestId, input }); assert(response.ok); return response;
}
test('each leaf of small complete periodic,pinned,massless and uncoupled responses rejects independent mutation', () => {
  const inputs = [
    { ...spatialSpectrumDefault, sites: 3 },
    { ...spatialSpectrumDefault, sites: 4, boundary: 'dirichlet' as const, constraint: 'mean-zero' as const },
    { ...spatialSpectrumDefault, sites: 3, massSquared: '0', constraint: 'mean-zero' as const },
    { ...spatialSpectrumDefault, sites: 3, speedSquared: '0' }
  ], rows = [];
  for (const input of inputs) {
    const response = success(input), leaves: { path: (string | number)[]; value: unknown }[] = [];
    const walk = (v: unknown, path: (string | number)[]) => {
      if (v === null || typeof v !== 'object') { leaves.push({ path, value: v }); return; }
      for (const [k, x] of Object.entries(v)) walk(x, [...path, Array.isArray(v) ? Number(k) : k]);
    };
    walk(response.result, []);
    for (const leaf of leaves) {
      const changed = structuredClone(response.result);
      let node: unknown = changed;
      for (const key of leaf.path.slice(0, -1)) node = (node as Record<string | number, unknown>)[key];
      (node as Record<string | number, unknown>)[leaf.path.at(-1)!] = typeof leaf.value === 'string' ? leaf.value + '!' : typeof leaf.value === 'number' ? leaf.value + 1 : typeof leaf.value === 'boolean' ? !leaf.value : 'forged';
      assert.equal(validateSpatialSpectrumResult(input, changed), null, leaf.path.join('.'));
    }
    assert.deepEqual(new Set(leaves.map(l => l.path[0])), new Set(Object.keys(response.result)));
    rows.push({ boundary: input.boundary, constraint: input.constraint, massSquared: input.massSquared, speedSquared: input.speedSquared, leaves: leaves.length, rejected: leaves.length });
  }
  writeFileSync(root + '/leaf-mutations.json', JSON.stringify({ rows, rejected: rows.reduce((n, row) => n + row.rejected, 0) }, null, 2) + '\n');
});

test('narrow enclosures must contain independent mathematics,not merely overlap it', () => {
  const input = spatialSpectrumDefault, response = success(input);
  const forged = structuredClone(response.result);
  forged.modes[1].power = { lower: '0', upper: '1' };
  assert.equal(validateSpatialSpectrumResult(input, forged), null);
  forged.modes[1].power = { lower: '0', upper: '0' };
  assert.equal(validateSpatialSpectrumResult(input, forged), null);
  forged.modes[1].power = { lower: '1', upper: '0' };
  assert.equal(validateSpatialSpectrumResult(input, forged), null);
  forged.modes[1].power = { lower: '1/3', upper: '1/3' };
  assert.equal(validateSpatialSpectrumResult(input, forged), null);
  const falseZero = structuredClone(response.result);
  falseZero.fourierCovariance[0][1].real = { lower: '0', upper: '1/' + String(BigInt(1) << BigInt(512)) };
  assert.equal(validateSpatialSpectrumResult(input, falseZero), null);
});

test('every top-level field is checked for the maximum accepted input and the result remains immutable', () => {
  const input = spatialSpectrumExamples.find(c => c.id === 'maximum-coprime')!.input, response = success(input);
  for (const key of Object.keys(response.result)) {
    const changed = { ...response.result, [key]: 'forged' };
    assert.equal(validateSpatialSpectrumResult(input, changed), null, key);
  }
  const checked = validateSpatialSpectrumResult(input, response.result)!;
  assert(checked); assert(Object.isFrozen(checked.fourierCovariance[0][0].real));
  assert.throws(() => { checked.covariance[0][0] = 'bad'; });
  assert(validateSpatialSpectrumResult(input, response.result));
});

test('physical meaning cannot be relabelled through spectrum,length,origin or refinement', () => {
  const response = success(spatialSpectrumDefault), mutations: ((r: SpatialSpectrumSuccess['result']) => void)[] = [
    r => { r.modes[1].signedWave = -1; },
    r => { r.modes[1].continuumNormalizedPower = { lower: '0', upper: '0' }; },
    r => { r.infinite.correlationLength = { lower: '1', upper: '1' }; },
    r => { r.curve[0].site = 1; },
    r => { r.refinements![1].fixedBoxLength = '16'; },
    r => { r.refinements![1].arrayIndex = 7; },
    r => { r.localRatios[0].length = null; },
    r => { r.certificates.trace = '0'; },
    r => { r.fourierPrecision[0][1].real = { lower: '1', upper: '1' }; },
    r => { r.selectedReconstruction[0].real = { lower: '0', upper: '0' }; }
  ];
  for (const mutation of mutations) { const changed = structuredClone(response.result); mutation(changed); assert.equal(validateSpatialSpectrumResult(spatialSpectrumDefault, changed), null); }
});

test('request attribution includes raw fraction spellings and all nested inputs', () => {
  const input = { ...spatialSpectrumDefault, temperature: '2/2' }, response = success(input, 3);
  for (const data of [{ version: 1, requestId: 4, input, response }, { version: 1, requestId: 3, input: spatialSpectrumDefault, response }, { version: 1, requestId: 3, input: { ...input, origin: 1 }, response }]) assert.equal(validate(data).phase, 'rejected');
  const changed = structuredClone(response); changed.result.input = { ...changed.result.input, selectedMode: 0 };
  assert.equal(validate({ version: 1, requestId: 3, input, response: changed }).phase, 'rejected');
  for (const requestId of [0, -1, 1.5, Infinity, Number.MAX_SAFE_INTEGER + 1]) assert.equal(calculate({ version: 1, requestId, input }).ok, false);
});

test('untrusted exact fraction strings never become operands of the independent engine', () => {
  const input = { ...spatialSpectrumDefault, temperature: '13/7' }, response = success(input), changed = structuredClone(response.result);
  changed.covariance[0][0] = 'untrusted-exact-operand';
  const split = String.prototype.split; let parsed = false;
  String.prototype.split = function(this: string, separator: unknown, limit?: number) { if (String(this) === 'untrusted-exact-operand') parsed = true; return Reflect.apply(split, this, [separator, limit]); } as typeof String.prototype.split;
  try { assert.equal(validateSpatialSpectrumResult(input, changed), null); assert.equal(parsed, false); }
  finally { String.prototype.split = split; }
});

test('getters,extra keys,cycles,symbols,prototypes and oversized responses fail without executing caller accessors', () => {
  const response = success(spatialSpectrumDefault); let reads = 0;
  const getter = { ...response.result }; Object.defineProperty(getter, 'sites', { get() { reads++; return 8; } });
  assert.equal(validateSpatialSpectrumResult(spatialSpectrumDefault, getter), null); assert.equal(reads, 0);
  const cycle: Record<string, unknown> = { ...response.result }; cycle.self = cycle;
  const array = structuredClone(response.result); Object.defineProperty(array.covariance, 'extra', { value: '0' });
  for (const value of [cycle, array, { ...response.result, extra: true }, { ...response.result, [Symbol('extra')]: 1 }, { ...response.result, contract: 'x'.repeat(65537) }, Object.create(response.result)]) assert.equal(validateSpatialSpectrumResult(spatialSpectrumDefault, value), null);
  const hostile = new Proxy({}, { getPrototypeOf() { throw Error('hostile'); } });
  assert.equal(calculate(hostile).ok, false); assert.equal(validate(hostile).phase, 'rejected');
});

test('browser hook excludes arithmetic and the validator bundle has no production calculator dependency', async () => {
  for (const entry of ['src/components/learning/use-spatial-spectrum-calculation.ts', 'src/lib/learning-spatial-spectrum-validation.worker.ts']) {
    const bundle = await build({ entryPoints: [entry], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true }), inputs = Object.keys(bundle.metafile!.inputs), hook = entry.includes('use-spatial');
    for (const file of ['learning-spatial-spectrum.ts', 'learning-spatial-spectrum-evaluator.ts', 'learning-quantum-hilbert-core.ts', 'learning-quantum-evolution-interval.ts', 'learning-spin-angle.ts', 'learning-interval-transcendentals.ts']) assert(!inputs.includes('src/lib/' + file), file);
    if (hook) { assert(!inputs.includes('src/lib/learning-spatial-spectrum-validation.ts')); assert(!bundle.outputFiles[0].text.includes('BigInt(')); }
    writeFileSync(root + '/' + (hook ? 'hook' : 'independent') + '-dependency-graph.json', JSON.stringify(bundle.metafile, null, 2) + '\n');
  }
});

test('hook server-renders idle without constructing a browser Worker', async () => {
  const React = await import('react'), { renderToStaticMarkup } = await import('react-dom/server'), { useSpatialSpectrumCalculation } = await import('../src/components/learning/use-spatial-spectrum-calculation');
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let calls = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { calls++; throw Error('SSR Worker'); } });
  try { function Probe() { return React.createElement('span', null, useSpatialSpectrumCalculation(spatialSpectrumDefault, true).status); } assert.equal(renderToStaticMarkup(React.createElement(Probe)), '<span>idle</span>'); assert.equal(calls, 0); }
  finally { if (saved) Object.defineProperty(globalThis, 'Worker', saved); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
