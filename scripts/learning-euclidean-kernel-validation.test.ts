import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { validateEuclideanKernelResult } from '../src/lib/learning-euclidean-kernel-validation';
import { euclideanKernelDefault } from '../src/lib/learning-euclidean-kernel-input';
import type { EuclideanKernelResult } from '../src/lib/learning-euclidean-kernel';
const root = '.runtime/learning/euclidean-kernel-numerics';
mkdirSync(root, { recursive: true });
const fixture = JSON.parse(readFileSync('scripts/fixtures/euclidean-kernel-references.json', 'utf8')) as { cases: { id: string; input: unknown; expected: EuclideanKernelResult }[] };
const maxima = ['minimum-positive-factors', 'maximum-coprime', 'mixed-coprime-denominators'];
const ordered = [...fixture.cases.filter(c => !maxima.includes(c.id)), ...fixture.cases.filter(c => maxima.includes(c.id))];
const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value));
function changed(value: unknown, path: string[], replacement: unknown) {
  const out = clone(value);
  let cursor = out as Record<string, unknown>;
  for (const key of path.slice(0, -1)) cursor = cursor[key] as Record<string, unknown>;
  cursor[path.at(-1)!] = replacement;
  return out;
}
function leaves(value: unknown, path: string[] = []): { path: string[]; value: unknown }[] {
  return value !== null && typeof value === 'object' ? Object.entries(value).flatMap(([key, child]) => leaves(child, [...path, key])) : [{ path, value }];
}
const timing: { id: string; milliseconds: number }[] = [];
for (const c of ordered) test('independent LDL/convolution validates every Python reference field: ' + c.id, () => {
  const start = performance.now(), accepted = validateEuclideanKernelResult(c.input, c.expected);
  assert.deepEqual(accepted, c.expected);
  assert(Object.isFrozen(accepted) && Object.isFrozen(accepted.covariance.values) && Object.isFrozen(accepted.covariance.values[0]) && Object.isFrozen(accepted.covariance.indices[0]));
  if (maxima.includes(c.id)) timing.push({ id: c.id, milliseconds: performance.now() - start });
});

test('selected maximum-input mutations reject wrong matrix,normalization,trace,refinement and certificates', () => {
  const paths = ['covariance.values.0.1', 'covariance.indices.1.1', 'precision.0.0', 'mean.17', 'factorization.determinant', 'factorization.diagonal.0', 'factorization.subdiagonal.0', 'factorization.recurrence.31', 'endpointForcing.0', 'timeStep', 'inverseTemperature', 'kernel.prefactorSquaredTimesPi', 'kernel.diagonalExponentCoefficient', 'kernel.crossExponentCoefficient', 'kernel.minimumAction', 'kernel.rightLogCorrection', 'trace.squared', 'trace.diagonalCurvature', 'endpointIntegral.prefactorSquared', 'endpointIntegral.negativeLogQuadraticCoefficient', 'expectedAction.kinetic', 'expectedAction.potential', 'expectedAction.total', 'refinements.3.meanAtProbe', 'refinements.3.varianceAtProbe', 'refinements.3.traceSquared', 'refinements.3.minimumAction', 'certificates.inverseResidual.0.0', 'scope.fullMatrixIntervals'];
  let mutations = 0;
  for (const id of maxima) {
    const c = fixture.cases.find(c => c.id === id)!;
    for (const path of paths) {
      const keys = path.split('.'); let current: unknown = c.expected;
      for (const key of keys) current = (current as Record<string, unknown>)[key];
      assert(current !== undefined, path);
      const replacement = typeof current === 'number' ? current + 1 : current === '1' ? '2' : '1';
      assert.equal(validateEuclideanKernelResult(c.input, changed(c.expected, keys, replacement)), null, `${id}:${path}`); mutations++;
    }
  }
  writeFileSync(root + '/maximum-validation.json', JSON.stringify({ selectedMutationsRejected: mutations, allMaximumLeavesMutated: false, timing }, null, 2) + '\n');
});
test('every scalar leaf in three small outputs is checked,including indexed fractions and scope flags', () => {
  let mutations = 0;
  for (const id of ['one-variable-hand-check', 'four-intervals', 'free-particle']) {
    const c = fixture.cases.find(c => c.id === id)!;
    assert(validateEuclideanKernelResult(c.input, c.expected));
    for (const { path, value } of leaves(c.expected)) {
      const replacement = typeof value === 'boolean' ? !value : typeof value === 'number' ? value + 1 : value === null ? 'finite' : `${value}x`;
      assert.equal(validateEuclideanKernelResult(c.input, changed(c.expected, path, replacement)), null, `${id}:${path.join('.')}`); mutations++;
    }
  }
  writeFileSync(root + '/leaf-mutations.json', JSON.stringify({ individualSmallLeafMutationsRejected: mutations, smallModels: 3 }, null, 2) + '\n');
});
test('raw input attribution,field ownership and canonical fractions cannot be bypassed by plausible numbers', () => {
  const c = fixture.cases[0], r = c.expected;
  assert.equal(validateEuclideanKernelResult({ ...euclideanKernelDefault, hbar: '2' }, r), null);
  assert.equal(validateEuclideanKernelResult({ ...euclideanKernelDefault, mass: '1/1' }, r), null);
  assert.equal(validateEuclideanKernelResult(c.input, { ...r, extra: 0 }), null);
  const without = clone(r) as Partial<EuclideanKernelResult>; delete without.trace; assert.equal(validateEuclideanKernelResult(c.input, without), null);
  assert.equal(validateEuclideanKernelResult(c.input, changed(r, ['kernel', 'prefactorSquaredTimesPi'], '8/18')), null);
  assert.equal(validateEuclideanKernelResult(c.input, changed(r, ['kernel', 'minimumAction'], '9'.repeat(4000) + '/7')), null);
  assert.equal(validateEuclideanKernelResult(c.input, changed(r, ['covariance', 'values', '0', '1'], '1/0')), null);
  assert.equal(validateEuclideanKernelResult({ ...euclideanKernelDefault, hbar: '0' }, r), null);
});
test('descriptor snapshots reject hostile getters,cycles,prototypes,sparse arrays and resource overflow', () => {
  const c = fixture.cases[0], r = c.expected;
  let calls = 0; const hostile = clone(r);
  Object.defineProperty(hostile.kernel, 'minimumAction', { get() { calls++; return '49/72'; } });
  assert.equal(validateEuclideanKernelResult(c.input, hostile), null); assert.equal(calls, 0);
  const cycle = clone(r) as unknown as Record<string, unknown>; cycle.kernel = cycle;
  const symbol = { ...r, [Symbol('unexpected')]: true };
  const sparse = clone(r); delete sparse.mean[1];
  for (const value of [cycle, symbol, sparse, Object.assign(Object.create(null), r), changed(r, ['kernel', 'minimumAction'], '9'.repeat(65537)), changed(r, ['mean'], Array(257).fill('0'))]) assert.equal(validateEuclideanKernelResult(c.input, value), null);
});
test('accepted response is a deep owned snapshot and caller mutation cannot alter the cache', () => {
  const c = fixture.cases[0], mutable = clone(c.expected), owned = validateEuclideanKernelResult(c.input, mutable)!;
  mutable.kernel.minimumAction = '0'; mutable.covariance.values[0][1] = '0';
  assert.equal(owned.kernel.minimumAction, '49/72'); assert.equal(owned.covariance.values[0][1], '2/9');
  assert.deepEqual(validateEuclideanKernelResult(c.input, c.expected), c.expected);
});
test('independent validator bundle excludes producer arithmetic and producer runtime modules', async () => {
  const result = await build({ entryPoints: ['src/lib/learning-euclidean-kernel-validation.ts'], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true });
  const inputs = Object.keys(result.metafile!.inputs);
  assert(inputs.includes('src/lib/learning-graph-gauge-validation-math.ts'));
  assert(!inputs.some(path => /learning-(?:euclidean-kernel|quantum-hilbert-core|finite-markov-exact)\.ts$/.test(path)));
  writeFileSync(root + '/validator-dependency-graph.json', JSON.stringify(result.metafile, null, 2) + '\n');
});
