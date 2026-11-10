import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { build } from 'esbuild';
import { calculateBrownianBridge } from '../src/lib/learning-brownian-bridge';
import { validateBrownianBridgeResult } from '../src/lib/learning-brownian-bridge-validation';
import { brownianBridgeDefault } from '../src/lib/learning-brownian-bridge-input';
import { Rational as Q } from '../src/lib/learning-graph-gauge-validation-math';
const root = '.runtime/learning/brownian-bridge-numerics';
mkdirSync(root, { recursive: true });
const fixture = JSON.parse(readFileSync('scripts/fixtures/brownian-bridge-references.json', 'utf8')) as { cases: { id: string; input: unknown; expected: unknown }[] };
for (const c of fixture.cases) test('independent full numerical result reconstruction: ' + c.id, () => {
  const result = calculateBrownianBridge(c.input), checked = validateBrownianBridgeResult(c.input, result);
  assert(checked); assert.deepEqual(checked, c.expected); assert(Object.isFrozen(checked) && Object.isFrozen(checked.covariance[0]));
});
type Path = (string | number)[];
function leaves(value: unknown, path: Path = []): Path[] {
  return value !== null && typeof value === 'object' ? Object.entries(value).flatMap(([key, child]) => leaves(child, [...path, Array.isArray(value) ? Number(key) : key])) : [path];
}
function mutate(result: unknown, path: Path) {
  const copy = JSON.parse(JSON.stringify(result)); let parent = copy;
  for (const key of path.slice(0, -1)) parent = parent[key];
  const key = path.at(-1)!, value = parent[key];
  parent[key] = typeof value === 'number' ? value + 1 : typeof value === 'boolean' ? !value : typeof value === 'string' && /^-?\d+(?:\/\d+)?$/.test(value) ? Q.read(value).add(new Q(BigInt(1))).text() : 'tampered';
  return copy;
}
test('every individual result leaf is checked for four structurally different small bridges', () => {
  const rows = [];
  for (const id of ['four-interval-hand-check', 'one-interior-coordinate', 'fixed-start-source', 'fractional-grid']) {
    const c = fixture.cases.find(c => c.id === id)!, result = calculateBrownianBridge(c.input), paths = leaves(result);
    assert(validateBrownianBridgeResult(c.input, result));
    for (const path of paths) assert.equal(validateBrownianBridgeResult(c.input, mutate(result, path)), null, id + ':' + path.join('.'));
    rows.push({ id, individuallyRejectedMutations: paths.length });
  }
  writeFileSync(`${root}/leaf-mutations.json`, JSON.stringify(rows, null, 2) + '\n');
});
test('maximum inputs reject mutations across matrices, normalization, source, area and analytic refinement fields', () => {
  const paths: Path[] = [
    ['precision', 30, 30], ['covariance', 32, 32], ['covariance', 17, 23], ['mean', 17], ['baselineMean', 31],
    ['factorization', 'diagonal', 30], ['factorization', 'subdiagonal', 29], ['factorization', 'determinant'],
    ['kernel', 'prefactorSquaredTimesPi'], ['kernel', 'tiltedLogExponential'], ['kernel', 'baselineNegativeLogExponential'],
    ['source', 'logGenerating'], ['source', 'meanResponse', 17], ['area', 'variance'], ['area', 'varianceGap'], ['area', 'mean'],
    ['increments', 'covariance', 31, 31], ['increments', 'mean', 31], ['increments', 'expectedAction'],
    ['certificates', 'inverseResidual', 30, 30], ['certificates', 'meanResidual', 30],
    ['refinements', 3, 'matchedProbeIndex'], ['refinements', 3, 'areaVariance'], ['refinements', 3, 'expectedQuadraticVariation'],
    ['refinements', 3, 'expectedAction'], ['refinements', 3, 'fullMatrixCalculated'], ['scope', 'continuousMeasureExistenceProved'],
  ];
  for (const id of ['maximum-coprime', 'minimum-positive-factors']) {
    const c = fixture.cases.find(c => c.id === id)!, result = calculateBrownianBridge(c.input);
    assert(validateBrownianBridgeResult(c.input, result));
    for (const path of paths) assert.equal(validateBrownianBridgeResult(c.input, mutate(result, path)), null, id + ':' + path.join('.'));
  }
  writeFileSync(`${root}/maximum-mutations.json`, JSON.stringify({ inputs: 2, selectedLeavesPerInput: paths.length, individuallyRejectedMutations: paths.length * 2, allMaximumLeavesClaimed: false }, null, 2) + '\n');
});
test('untrusted response values never become rational operands and hostile structures cannot invoke getters', () => {
  const result = calculateBrownianBridge(brownianBridgeDefault); let reads = 0;
  const accessor = { ...result }; Object.defineProperty(accessor, 'mean', { get() { reads++; return result.mean; } });
  assert.equal(validateBrownianBridgeResult(brownianBridgeDefault, accessor), null); assert.equal(reads, 0);
  const enormous = JSON.parse(JSON.stringify(result)); enormous.area.variance = '9'.repeat(65537);
  assert.equal(validateBrownianBridgeResult(brownianBridgeDefault, enormous), null);
  assert.equal(validateBrownianBridgeResult(brownianBridgeDefault, { ...result, unexpected: true }), null);
  assert.equal(validateBrownianBridgeResult(brownianBridgeDefault, { ...result, [Symbol('hidden')]: true }), null);
  assert.equal(validateBrownianBridgeResult(brownianBridgeDefault, Object.assign(Object.create({ extra: 1 }), result)), null);
  assert.equal(validateBrownianBridgeResult({ ...brownianBridgeDefault, time: '0' }, result), null);
  const cycle = { ...result, cycle: null as unknown }; cycle.cycle = cycle;
  assert.equal(validateBrownianBridgeResult(brownianBridgeDefault, cycle), null);
});
test('validation rejects results attributed to another input even when their covariance agrees', () => {
  const r = calculateBrownianBridge(brownianBridgeDefault);
  for (const patch of [{ probe: 1 }, { source: '2' }, { start: '3', end: '4' }, { time: '2/2' }]) assert.equal(validateBrownianBridgeResult({ ...brownianBridgeDefault, ...patch }, r), null);
});
test('the independent validation bundle has no runtime producer or producer arithmetic dependencies', async () => {
  const bundle = await build({ entryPoints: ['src/lib/learning-brownian-bridge-validation.ts'], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true });
  const inputs = Object.keys(bundle.metafile!.inputs);
  assert(inputs.includes('src/lib/learning-graph-gauge-validation-math.ts'));
  assert(!inputs.includes('src/lib/learning-brownian-bridge.ts')); assert(!inputs.includes('src/lib/learning-quantum-hilbert-core.ts')); assert(!inputs.includes('src/lib/learning-finite-markov-exact.ts'));
  writeFileSync(`${root}/independent-dependency-graph.json`, JSON.stringify(bundle.metafile, null, 2) + '\n');
});
