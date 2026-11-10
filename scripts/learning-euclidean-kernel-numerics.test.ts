import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { calculateEuclideanKernel, type EuclideanKernelResult } from '../src/lib/learning-euclidean-kernel';
import { euclideanKernelDefault, euclideanKernelExamples, readEuclideanKernelInput } from '../src/lib/learning-euclidean-kernel-input';
import { calculateBrownianBridge } from '../src/lib/learning-brownian-bridge';
import { Rational as Q } from '../src/lib/learning-graph-gauge-validation-math';
import { snapshotQubitWire } from '../src/lib/learning-qubit-wire';
const root = '.runtime/learning/euclidean-kernel-numerics';
mkdirSync(root, { recursive: true });
const fixture = JSON.parse(readFileSync('scripts/fixtures/euclidean-kernel-references.json', 'utf8')) as { cases: { id: string; input: unknown; expected: EuclideanKernelResult }[] };
const unpack = (r: EuclideanKernelResult) => r.covariance.indices.map(row => row.map(id => r.covariance.values[Math.floor(id / 128)][id % 128]));
const exact = Q.read, integer = (n: number) => new Q(BigInt(n));
const maxIds = ['minimum-positive-factors', 'maximum-coprime', 'mixed-coprime-denominators'];
const resources: { id: string; bytes: number; milliseconds: number; covarianceValues: number; largestFractionCharacters: number }[] = [];

for (const c of fixture.cases) test('independent Python incidence/Gauss-Jordan/periodic/convolution reference: ' + c.id, () => {
  const started = performance.now(), actual = calculateEuclideanKernel(c.input);
  assert.deepEqual(actual, c.expected);
  assert.deepEqual(snapshotQubitWire(actual), actual);
  assert(Object.isFrozen(actual.input));
  if (maxIds.includes(c.id)) resources.push({ id: c.id, bytes: Buffer.byteLength(JSON.stringify(actual)), milliseconds: performance.now() - started, covarianceValues: actual.covariance.values.flat().length, largestFractionCharacters: Math.max(...JSON.stringify(actual).match(/-?\d+(?:\/\d+)?/g)!.map(x => x.length)) });
});

test('owned rational input enforces signs,exact bounds,counts and raw spellings', () => {
  for (const patch of [
    { intervals: 1 }, { intervals: 33 }, { intervals: 2.1 }, { intervals: '2' }, { probe: -0 }, { probe: -1 }, { probe: 3 }, { probe: 0.1 },
    { time: '0' }, { mass: '-1' }, { hbar: '0' }, { omegaSquared: '-1/2' }, { omegaSquared: '-0/3' },
    { time: '1e-4' }, { mass: '.5' }, { hbar: '1.0' }, { end: '+1' }, { start: '01' }, { end: '1/0' }, { end: '1/-2' },
    { start: '-0' }, { end: ' 1' }, { omegaSquared: '1\n' }, { time: '999999999985/62499999999' }, { mass: '1000000000000/999999999999' }, { end: '17' },
  ]) assert.equal(readEuclideanKernelInput({ ...euclideanKernelDefault, ...patch }), null, JSON.stringify(patch));
  const accepted = readEuclideanKernelInput({ ...euclideanKernelDefault, time: '999999999984/62499999999', mass: '1/999999999989', hbar: '16', omegaSquared: '0/7', start: '-16', end: '16' });
  assert(accepted); assert.equal(accepted.omegaSquared, '0/7');
  assert.throws(() => calculateEuclideanKernel({ ...euclideanKernelDefault, intervals: 33 }), RangeError);
});
test('input snapshots reject accessors,symbols,missing fields and exceptions without retaining caller data', () => {
  let calls = 0; const getter = { ...euclideanKernelDefault };
  Object.defineProperty(getter, 'mass', { get() { calls++; return '1'; } });
  assert.equal(readEuclideanKernelInput(getter), null); assert.equal(calls, 0);
  for (const value of [null, [], '1', { ...euclideanKernelDefault, extra: 1 }, { ...euclideanKernelDefault, [Symbol('extra')]: 1 }, { intervals: 2 }, new Proxy({}, { ownKeys() { throw Error('hostile'); } })]) assert.equal(readEuclideanKernelInput(value), null);
  const mutable = { ...euclideanKernelDefault }, owned = readEuclideanKernelInput(mutable)!; mutable.mass = '16'; assert.equal(owned.mass, '1'); assert(Object.isFrozen(owned));
});
test('the one-variable worked example keeps both endpoints,normalization and trace', () => {
  const r = calculateEuclideanKernel(euclideanKernelDefault);
  assert.deepEqual(r.precision, [['9/2']]); assert.deepEqual(r.mean, ['0', '4/9', '1']); assert.equal(unpack(r)[1][1], '2/9');
  assert.equal(r.kernel.actionConstant, '9/8'); assert.equal(r.kernel.minimumAction, '49/72');
  assert.equal(r.kernel.prefactorSquaredTimesPi, '4/9'); assert.equal(r.kernel.diagonalExponentCoefficient, '49/72'); assert.equal(r.kernel.crossExponentCoefficient, '8/9');
  assert.equal(r.trace.squared, '16/17'); assert.equal(r.kernel.rightLogCorrection, '-1/8'); assert.equal(r.endpointIntegral.prefactorSquared, '32/49');
  const both = calculateEuclideanKernel({ ...euclideanKernelDefault, start: '-2', end: '4' });
  assert.deepEqual(both.endpointForcing, ['4']); assert.equal(both.mean[1], '8/9');
});
test('the zero-potential branch matches the earlier free bridge and declares the unbounded trace divergent', () => {
  for (const intervals of [2, 4, 7, 16, 32]) {
    const input = { ...euclideanKernelDefault, intervals, omegaSquared: '0/7', mass: '2', hbar: '3', time: '3/2', start: '-1/2', end: '5/4', probe: Math.floor(intervals / 2) };
    const r = calculateEuclideanKernel(input), b = calculateBrownianBridge({ intervals, time: input.time, diffusion: '3/4', start: input.start, end: input.end, probe: input.probe, source: '0' });
    assert.deepEqual(r.mean, b.mean); assert.deepEqual(unpack(r), b.covariance); assert.deepEqual(r.precision, b.precision);
    assert.equal(r.kernel.prefactorSquaredTimesPi, b.kernel.prefactorSquaredTimesPi); assert.equal(r.kernel.minimumAction, b.kernel.baselineNegativeLogExponential);
    assert.deepEqual(r.trace, { status: 'divergent-unbounded-line', diagonalCurvature: '0', squared: null });
    assert.deepEqual(r.endpointIntegral, { prefactorSquared: '1', negativeLogQuadraticCoefficient: '0', conservesMass: true });
  }
});
test('potential increases reduce variances while mass and hbar have distinct exact scalings', () => {
  const input = { ...euclideanKernelDefault, intervals: 4, probe: 2 }, base = calculateEuclideanKernel(input), heavier = calculateEuclideanKernel({ ...input, mass: '2' }), quantum = calculateEuclideanKernel({ ...input, hbar: '2' });
  assert.deepEqual(heavier.mean, base.mean); assert.deepEqual(quantum.mean, base.mean);
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
    assert.equal(unpack(heavier)[i][j], exact(unpack(base)[i][j]).div(integer(2)).text()); assert.equal(unpack(quantum)[i][j], exact(unpack(base)[i][j]).mul(integer(2)).text());
  }
  assert.equal(heavier.trace.squared, base.trace.squared); assert.equal(quantum.trace.squared, base.trace.squared);
  const free = unpack(calculateEuclideanKernel({ ...input, omegaSquared: '0' })), strong = unpack(calculateEuclideanKernel({ ...input, omegaSquared: '4' }));
  for (let i = 1; i < 4; i++) { assert(exact(free[i][i]).sub(exact(unpack(base)[i][i])).n > BigInt(0)); assert(exact(unpack(base)[i][i]).sub(exact(strong[i][i])).n > BigInt(0)); }
});
test('endpoint reversal preserves the symmetric kernel and flips the right-rule log correction', () => {
  const input = { ...euclideanKernelDefault, intervals: 7, probe: 3, start: '-1/2', end: '5/4' }, a = calculateEuclideanKernel(input), b = calculateEuclideanKernel({ ...input, start: input.end, end: input.start });
  assert.equal(a.kernel.minimumAction, b.kernel.minimumAction); assert.equal(a.kernel.prefactorSquaredTimesPi, b.kernel.prefactorSquaredTimesPi); assert.deepEqual(b.mean, a.mean.slice().reverse());
  assert.equal(b.kernel.rightLogCorrection, exact(a.kernel.rightLogCorrection).neg().text());
  for (const start of ['1', '-1']) assert.equal(calculateEuclideanKernel({ ...euclideanKernelDefault, start }).kernel.rightLogCorrection, '0');
});
test('analytic refinements match explicit full matrices at the same physical probe,without claiming marginal invariance', () => {
  const base = calculateEuclideanKernel({ ...euclideanKernelDefault, intervals: 4, probe: 2 });
  for (const row of base.refinements) {
    const r = row.intervals === 4 ? base : calculateEuclideanKernel({ ...base.input, intervals: row.intervals, probe: row.matchedProbeIndex });
    assert.equal(row.meanAtProbe, r.mean[row.matchedProbeIndex]); assert.equal(row.varianceAtProbe, unpack(r)[row.matchedProbeIndex][row.matchedProbeIndex]);
    assert.equal(row.prefactorSquaredTimesPi, r.kernel.prefactorSquaredTimesPi); assert.equal(row.traceSquared, r.trace.squared); assert.equal(row.minimumAction, r.kernel.minimumAction);
  }
  assert.notEqual(base.refinements[0].varianceAtProbe, base.refinements[1].varianceAtProbe);
});
test('examples retain raw fraction spelling and every planned example has an independent reference', () => {
  assert.deepEqual(euclideanKernelExamples.map(x => x.id), fixture.cases.slice(0, euclideanKernelExamples.length).map(x => x.id));
  for (const e of euclideanKernelExamples) { assert(Object.isFrozen(e) && Object.isFrozen(e.input)); assert.deepEqual(e.input, fixture.cases.find(c => c.id === e.id)!.input); }
  const equivalent = calculateEuclideanKernel(euclideanKernelExamples.find(x => x.id === 'equivalent-spelling')!.input);
  assert.equal(equivalent.input.hbar, '4/4'); assert.deepEqual(equivalent.covariance, calculateEuclideanKernel(euclideanKernelDefault).covariance);
});
test('maximum inputs retain all matrix entries,all analytic rows and unchanged wire limits', () => {
  assert.equal(resources.length, 3); assert(resources.every(row => row.bytes > 500000 && row.largestFractionCharacters < 65536));
  for (const id of maxIds) { const r = fixture.cases.find(c => c.id === id)!.expected; assert.equal(r.covariance.indices.length, 33); assert(r.covariance.indices.every(row => row.length === 33)); assert.equal(r.precision.length, 31); assert.equal(r.refinements[3].intervals, 256); assert(r.refinements.slice(1).every(row => !row.fullMatrixCalculated)); }
  writeFileSync(root + '/resources.json', JSON.stringify(resources, null, 2) + '\n');
});
test('reader input metadata excludes all exact arithmetic and numerical engines', async () => {
  const r = await build({ entryPoints: ['src/lib/learning-euclidean-kernel-input.ts'], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true });
  assert.deepEqual(Object.keys(r.metafile!.inputs), ['src/lib/learning-euclidean-kernel-input.ts']);
});
