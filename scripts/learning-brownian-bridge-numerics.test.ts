import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { brownianBridgeDefault, brownianBridgeExamples, readBrownianBridgeInput } from '../src/lib/learning-brownian-bridge-input';
import { calculateBrownianBridge } from '../src/lib/learning-brownian-bridge';
import { calculateGaussianField } from '../src/lib/learning-gaussian-field';
import { gaussianFieldDefault } from '../src/lib/learning-gaussian-field-input';
import { Rational as Q } from '../src/lib/learning-graph-gauge-validation-math';
import { snapshotQubitWire } from '../src/lib/learning-qubit-wire';
const root = '.runtime/learning/brownian-bridge-numerics';
mkdirSync(root, { recursive: true });
const fixture = JSON.parse(readFileSync('scripts/fixtures/brownian-bridge-references.json', 'utf8')) as { cases: { id: string; input: unknown; expected: unknown }[] };
const result = (id: string) => calculateBrownianBridge(brownianBridgeExamples.find(example => example.id === id)!.input);
const read = Q.read, integer = (n: number) => new Q(BigInt(n));
for (const c of fixture.cases) test('independent incidence / Gauss–Jordan / increment reference: ' + c.id, () => {
  const actual = calculateBrownianBridge(c.input); assert.deepEqual(actual, c.expected); assert(Object.isFrozen(actual.input));
});
test('input grammar enforces exact rational bounds, positive factors and interval/probe counts', () => {
  for (const patch of [
    { intervals: 1 }, { intervals: 33 }, { intervals: 4.1 }, { intervals: '4' }, { probe: -1 }, { probe: 5 }, { probe: 1.5 }, { probe: -0 },
    { time: '0' }, { time: '-1' }, { diffusion: '0' }, { diffusion: '-1' }, { time: '1e-4' }, { diffusion: '0.5' },
    { source: '-0' }, { start: '+1' }, { end: '01' }, { end: '1/0' }, { end: '1/-2' }, { source: '1000000000000/999999999999' },
    { time: '999999999985/62499999999' }, { start: '-17' }, { end: '17' }, { source: '17' },
  ]) assert.equal(readBrownianBridgeInput({ ...brownianBridgeDefault, ...patch }), null, JSON.stringify(patch));
  assert(readBrownianBridgeInput({ ...brownianBridgeDefault, time: '999999999984/62499999999', diffusion: '1/999999999989', start: '-16', end: '16', source: '-16' }));
  assert.throws(() => calculateBrownianBridge({ ...brownianBridgeDefault, intervals: 1 }), RangeError);
});
test('owned input validation rejects getters, missing/extra/symbol fields and does not retain caller objects', () => {
  let reads = 0; const getter = { ...brownianBridgeDefault }; Object.defineProperty(getter, 'time', { get() { reads++; return '1'; } });
  assert.equal(readBrownianBridgeInput(getter), null); assert.equal(reads, 0);
  for (const value of [null, [], '1', { ...brownianBridgeDefault, extra: 1 }, { ...brownianBridgeDefault, [Symbol('extra')]: 1 }, { intervals: 4 }]) assert.equal(readBrownianBridgeInput(value), null);
  const plain = { ...brownianBridgeDefault }, owned = readBrownianBridgeInput(plain)!; plain.end = '16'; assert.equal(owned.end, '1'); assert(Object.isFrozen(owned));
});
test('four intervals reproduce the worked precision, covariance, determinant and action', () => {
  const r = result('four-interval-hand-check');
  assert.deepEqual(r.precision, [['4', '-2', '0'], ['-2', '4', '-2'], ['0', '-2', '4']]);
  assert.deepEqual(r.covariance[2], ['0', '1/4', '1/2', '1/4', '0']);
  assert.deepEqual(r.mean, ['0', '1/4', '1/2', '3/4', '1']);
  assert.equal(r.factorization.determinant, '32'); assert.equal(r.kernel.prefactorSquaredTimesPi, '1/4'); assert.equal(r.kernel.baselineNegativeLogExponential, '1/4');
  assert.equal(r.increments.expectedQuadraticVariation, '7/4'); assert.equal(r.increments.expectedAction, '7/4');
});
test('one interior coordinate receives both endpoint forcings and the point-source tilt', () => {
  const r = calculateBrownianBridge({ ...brownianBridgeDefault, intervals: 2, probe: 1, start: '-2', end: '4', source: '2' });
  assert.deepEqual(r.baselineMean, ['-2', '1', '4']); assert.deepEqual(r.mean, ['-2', '2', '4']);
  assert.equal(r.covariance[1][1], '1/2'); assert.equal(r.source.logGenerating, '3'); assert.equal(r.kernel.tiltedLogExponential, '-6');
});
test('a midpoint source changes the conditioned mean and normalization but leaves covariance unchanged', () => {
  const base = result('four-interval-hand-check'), tilted = result('middle-source');
  assert.deepEqual(tilted.covariance, base.covariance); assert.deepEqual(tilted.mean, ['0', '3/4', '3/2', '5/4', '1']);
  assert.equal(tilted.source.logGenerating, '2'); assert.equal(tilted.area.mean, '1'); assert.equal(tilted.area.variance, '5/32');
  assert.equal(tilted.increments.expectedAction, '11/4');
});
test('a source at a fixed endpoint changes only the unnormalized kernel weight', () => {
  for (const probe of [0, 4]) {
    const base = calculateBrownianBridge({ ...brownianBridgeDefault, start: '-2', probe }), tilted = calculateBrownianBridge({ ...base.input, source: '3' });
    assert.deepEqual(tilted.mean, base.mean); assert.deepEqual(tilted.covariance, base.covariance);
    assert.equal(tilted.source.varianceAtProbe, '0'); assert.equal(tilted.source.logGenerating, probe === 0 ? '-6' : '3');
    assert(tilted.source.meanResponse.every(value => value === '0'));
  }
});
test('diffusion and total-time changes preserve the source-free mean with their distinct area scalings', () => {
  const base = result('four-interval-hand-check'), d = result('double-diffusion'), t = result('double-time');
  for (const r of [d, t]) { assert.deepEqual(r.mean, base.mean); for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) assert.equal(r.covariance[i][j], read(base.covariance[i][j]).mul(integer(2)).text()); }
  assert.equal(d.area.variance, read(base.area.variance).mul(integer(2)).text()); assert.equal(t.area.variance, read(base.area.variance).mul(integer(8)).text());
});
test('joint endpoint translation shifts the tilted law and adds the source constant to its generating function', () => {
  const base = result('middle-source'), translated = calculateBrownianBridge({ ...base.input, start: '3', end: '4' });
  assert.deepEqual(translated.covariance, base.covariance); assert.deepEqual(translated.mean, base.mean.map(m => read(m).add(integer(3)).text()));
  assert.equal(read(translated.source.logGenerating).sub(read(base.source.logGenerating)).text(), '6');
  assert.deepEqual(translated.increments, base.increments);
});
test('explicit fine matrices preserve every coarse-time marginal while their area variances change', () => {
  for (let intervals = 2; intervals <= 16; intervals++) {
    const coarse = calculateBrownianBridge({ ...brownianBridgeDefault, intervals, probe: Math.floor(intervals / 2), source: '-5/7' });
    const fine = calculateBrownianBridge({ ...coarse.input, intervals: intervals * 2, probe: coarse.input.probe * 2 });
    assert.deepEqual(fine.mean.filter((_, i) => i % 2 === 0), coarse.mean);
    assert.deepEqual(fine.covariance.filter((_, i) => i % 2 === 0).map(row => row.filter((_, i) => i % 2 === 0)), coarse.covariance);
    assert.deepEqual(fine.refinements[0], { ...coarse.refinements[1], fullMatrixCalculated: true });
    assert.notEqual(coarse.area.variance, fine.area.variance); assert.equal(coarse.area.mean, fine.area.mean);
  }
});
test('the interior law agrees with the earlier pinned Gaussian model after converting the time-step coefficient', () => {
  for (let n = 4; n <= 9; n++) {
    const probe = Math.floor(n / 2), r = calculateBrownianBridge({ ...brownianBridgeDefault, intervals: n, time: `${n}/2`, start: '1', end: '2', probe, source: '1/2' });
    const j = Array.from({ length: n - 1 }, (_, i) => integer(i === 0 ? 1 : i === n - 2 ? 2 : 0).add(i + 1 === probe ? read('1/2') : integer(0)).text());
    const g = calculateGaussianField({ ...gaussianFieldDefault, sites: n - 1, boundary: 'dirichlet', massSquared: '0', stiffness: '1', temperature: '1', source: j, probes: [0, 0, 0, 0] });
    assert.deepEqual(g.precision, r.precision); assert.deepEqual(g.mean, r.mean.slice(1, n)); assert.deepEqual(g.connectedTwo, r.covariance.slice(1, n).map(row => row.slice(1, n)));
  }
});
test('prepared examples preserve raw fraction spellings, stay frozen and match the independent case list', () => {
  assert.deepEqual(brownianBridgeExamples.map(e => e.id), fixture.cases.slice(0, brownianBridgeExamples.length).map(c => c.id));
  for (const e of brownianBridgeExamples) { assert(Object.isFrozen(e) && Object.isFrozen(e.input)); assert.deepEqual(e.input, fixture.cases.find(c => c.id === e.id)!.input); }
  const a = result('four-interval-hand-check'), b = result('equivalent-spelling'); assert.deepEqual(a.covariance, b.covariance); assert.equal(b.input.diffusion, '3/3');
});
test('maximum inputs retain matrix size, exact fractions and existing owned wire budgets', () => {
  const rows = [];
  for (const id of ['maximum-coprime', 'minimum-positive-factors']) {
    const r = result(id); assert.deepEqual(snapshotQubitWire(r), r); assert.equal(r.covariance.length, 33); assert.equal(r.precision.length, 31);
    assert.equal(r.scope.largestAnalyticRefinementIntervals, 256); assert(r.refinements.slice(1).every(row => !row.fullMatrixCalculated));
    rows.push({ id, bytes: Buffer.byteLength(JSON.stringify(r)), intervals: r.input.intervals });
  }
  writeFileSync(`${root}/resources.json`, JSON.stringify(rows, null, 2) + '\n');
});
test('the reader input and example module excludes exact arithmetic and all numerical engines', async () => {
  const bundle = await build({ entryPoints: ['src/lib/learning-brownian-bridge-input.ts'], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true });
  assert.deepEqual(Object.keys(bundle.metafile!.inputs), ['src/lib/learning-brownian-bridge-input.ts']);
});
