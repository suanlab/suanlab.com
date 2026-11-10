import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { Worker } from 'node:worker_threads';
import { manyBodyKinds as kinds, snapshotManyBodyInput as snapshot, readManyBodyInput as read, manyBodyInputKey as key, parseManyBodyInputDraft as draft, parseManyBodyStoredInput as stored, readManyBodyRequest, type ManyBodyInput } from '../src/lib/learning-many-body-protocol';
import { evaluateManyBodyRequest as evaluate } from '../src/lib/learning-many-body-evaluator';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
ensureTestArtifactDirectory('.runtime/learning/many-body-worker', { recursive: true });

const runtime = '.runtime/learning/many-body-worker';
const preparations = kinds.flatMap(kind => {
  const refs = JSON.parse(readFileSync(`scripts/fixtures/${kind}-reference.json`, 'utf8')) as { cases: { label: string; input: Record<string, number>; expected: unknown }[] };
  return refs.cases.map(c => ({ label: `${kind}: ${c.label}`, input: snapshot({ kind, parameters: c.input }), expected: c.expected }));
});
type F = { n: bigint; d: bigint };
const frac = (n: bigint, d = BigInt(1)): F => ({ n, d });
const le = (a: F, b: F) => a.n * b.d <= b.n * a.d;
const equal = (a: F, b: F) => a.n * b.d === b.n * a.d;
const add = (a: F, b: F) => frac(a.n * b.d + b.n * a.d, a.d * b.d);
const multiply = (a: F, b: F) => frac(a.n * b.n, a.d * b.d);
function parse(raw: string): F {
  if (raw.includes('/')) { const [n, d] = raw.split('/'); return frac(BigInt(n), BigInt(d)); }
  const m = /^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(raw); assert(m, raw);
  const n = BigInt(m[1] + m[2] + (m[3] ?? '')), exponent = Number(m[4] ?? 0) - (m[3] ?? '').length;
  return exponent >= 0 ? frac(n * BigInt(10) ** BigInt(exponent)) : frac(n, BigInt(10) ** BigInt(-exponent));
}
function endpoint(b: NonNullable<Q['bounds']>['lower']): F {
  const n = BigInt(b.numerator.coefficient), d = BigInt(b.denominator.coefficient), e = b.numerator.exponent - b.denominator.exponent;
  return e >= 0 ? frac(n * BigInt(10) ** BigInt(e), d) : frac(n, d * BigInt(10) ** BigInt(-e));
}
const exact = (q: Q) => { assert(q.bounds); const lo = endpoint(q.bounds.lower), hi = endpoint(q.bounds.upper); assert(equal(lo, hi)); return lo; };
let exactChecks = 0, directedChecks = 0;
function observation(actual: unknown, expected: string, path: string) {
  assert(isMasterObservation(actual), path); const q = actual as Q; assert(q.bounds, path);
  const lo = endpoint(q.bounds.lower), hi = endpoint(q.bounds.upper); assert(le(lo, hi), path);
  const directed = expected.startsWith('decimal:'), value = parse(directed ? expected.slice(8) : expected);
  const absolute = value.n < BigInt(0) ? -value.n : value.n, scale = absolute > value.d ? absolute : value.d;
  if (directed) {
    const eps = BigInt(10) ** BigInt(205);
    assert(le(lo, frac(value.n * eps + scale, value.d * eps)) && le(frac(value.n * eps - scale, value.d * eps), hi), path + ' reference overlap');
    directedChecks++;
  } else { assert(equal(lo, value) && equal(hi, value), path + ' exact equality'); exactChecks++; }
  const width = frac(hi.n * lo.d - lo.n * hi.d, hi.d * lo.d);
  assert(le(width, frac(scale, value.d * BigInt(10) ** BigInt(125))), path + ' directed width');
}
function compare(actual: unknown, expected: unknown, path = 'result') {
  if (typeof expected === 'string' && (/^-?\d/.test(expected) || expected.startsWith('decimal:'))) return observation(actual, expected, path);
  if (Array.isArray(expected)) { assert(Array.isArray(actual), path); assert.equal(actual.length, expected.length, path); expected.forEach((v, i) => compare(actual[i], v, `${path}[${i}]`)); return; }
  if (expected && typeof expected === 'object') { assert(actual && typeof actual === 'object', path); for (const [key, value] of Object.entries(expected)) compare((actual as Record<string, unknown>)[key], value, `${path}.${key}`); return; }
  assert.equal(actual, expected, path);
}
let directedCertificateChecks = 0;
function directedZero(value: unknown) {
  if (value === null) return;
  if (isMasterObservation(value)) {
    const v = value as Q; assert(v.bounds);
    const lo = endpoint(v.bounds.lower), hi = endpoint(v.bounds.upper);
    assert(le(lo, frac(BigInt(0))) && le(frac(BigInt(0)), hi), 'directed certificate contains0');
    assert(le(frac(hi.n * lo.d - lo.n * hi.d, hi.d * lo.d), frac(BigInt(1), BigInt(10) ** BigInt(125))), 'directed certificate width');
    directedCertificateChecks++; return;
  }
  assert(value && typeof value === 'object'); Object.values(value).forEach(directedZero);
}
for (const p of preparations) test(`many-body owned input and lexical storage: ${p.label}`, () => {
  assert.equal(key(p.input), key({ kind: p.input.kind, parameters: Object.fromEntries(Object.entries(p.input.parameters).reverse()) }));
  assert(Object.isFrozen(p.input)); assert(Object.isFrozen(p.input.parameters));
  assert.deepEqual(draft(JSON.stringify(p.input)), p.input);
  assert.deepEqual(stored(JSON.stringify({ version: 1, value: p.input })), p.input);
  assert.equal(read({ ...p.input, extra: true }), null);
  assert.equal(read({ ...p.input, parameters: { ...p.input.parameters, extra: 1 } }), null);
});

test('unknown kinds, foreign family parameters, nonfinite/rounded integers and storage versions are rejected', () => {
  for (const kind of kinds) {
    const p = preparations.find(p => p.input.kind === kind)!.input;
    assert.equal(read({ kind: '__proto__', parameters: p.parameters }), null);
    assert.equal(read({ kind: 'constructor', parameters: p.parameters }), null);
    const field = Object.keys(p.parameters)[0];
    for (const token of ['1e-9999', '1.00000000000000001', '9007199254740993', '1e9999']) {
      const raw = JSON.stringify(p).replace(new RegExp(`"${field}":-?\\d+`), `"${field}":${token}`);
      assert.equal(draft(raw), null); assert.equal(stored(`{"version":1,"value":${raw}}`), null);
    }
    assert.equal(stored(JSON.stringify({ version: 2, value: p })), null);
    assert.equal(stored(JSON.stringify({ version: 1, value: p, extra: 0 })), null);
    for (const other of kinds.filter(k => k !== kind)) assert.equal(read({ kind: other, parameters: p.parameters }), null);
  }
  assert.equal(draft(' '.repeat(32769)), null); assert.equal(draft('not-json'), null); assert.equal(stored('{}'), null);
});

test('descriptors, symbols, prototypes and hostile reflection never invoke caller getters/get traps', () => {
  for (const kind of kinds) {
    const p = preparations.find(p => p.input.kind === kind)!.input; let reads = 0;
    const getter = { ...p }; Object.defineProperty(getter, 'kind', { get() { reads++; return kind; } });
    assert.equal(read(getter), null);
    const parameters = { ...p.parameters }, name = Object.keys(parameters)[0];
    Object.defineProperty(parameters, name, { get() { reads++; return 1; } });
    assert.equal(read({ ...p, parameters }), null);
    const proxy = new Proxy(p, { get() { reads++; throw Error('Caller get trap'); } });
    assert.deepEqual(read(proxy), p); assert.equal(reads, 0);
    for (const value of [{ ...p, [Symbol('extra')]: 0 }, Object.assign(Object.create(null), p), Object.create(p), new Proxy({}, { getPrototypeOf() { throw Error('Reflection'); } })]) assert.equal(read(value), null);
  }
  let reads = 0;
  const envelope = { version: 1, input: preparations[0].input };
  Object.defineProperty(envelope, 'requestId', { get() { reads++; return 1; } });
  assert.deepEqual(evaluate(envelope), { version: 1, requestId: null, ok: false, error: 'invalid-request' }); assert.equal(reads, 0);
});

test('owned request snapshots isolate caller mutation and progress uses canonical input identity', () => {
  for (const kind of kinds) {
    const original = preparations.find(p => p.input.kind === kind)!.input;
    const input = { kind: original.kind, parameters: { ...original.parameters } }, oldKey = key(input);
    const request = readManyBodyRequest({ version: 1, requestId: 7, input }); assert('request' in request);
    input.parameters[Object.keys(input.parameters)[0]] = -99;
    assert.equal(key(request.request.input), oldKey);
    assert(Object.isFrozen(request.request)); assert(Object.isFrozen(request.request.input.parameters));
  }
});

test('invalid envelopes and input give one failure without a started frame', () => {
  const progress: unknown[] = [], input = preparations[0].input;
  for (const value of [null, {}, { version: 2, requestId: 1, input }, { version: 1, requestId: 0, input }, { version: 1, requestId: 1.5, input }, { version: 1, requestId: Infinity, input }, { version: 1, requestId: 1, input, extra: 0 }, new Proxy({}, { ownKeys() { throw Error('Reflection'); } })]) {
    const r = evaluate(value, p => progress.push(p)); assert(!r.ok); assert.equal(r.error, 'invalid-request');
  }
  const r = evaluate({ version: 1, requestId: 7, input: { kind: 'unknown', parameters: {} } }, p => progress.push(p));
  assert.deepEqual(r, { version: 1, requestId: 7, ok: false, error: 'invalid-input' }); assert.deepEqual(progress, []);
});

test('each dispatch family returns an attributed cloneable result matching an independent preparation', () => {
  for (const kind of kinds) {
    const p = preparations.find(p => p.input.kind === kind)!, progress: unknown[] = [];
    const r = evaluate({ version: 1, requestId: 9, input: p.input }, frame => progress.push(frame));
    assert(r.ok, p.label); assert.equal(r.kind, kind); assert.equal(r.inputKey, key(p.input)); assert.deepEqual(r.input, p.input);
    assert.deepEqual(progress, [{ version: 1, requestId: 9, phase: 'started', kind }]);
    compare(structuredClone(r).result, p.expected, p.label); assert.doesNotThrow(() => JSON.stringify(r));
  }
});

test('reader request and factory bundles contain no heavy numerical engine and SSR constructs no worker', async () => {
  const result = await build({ entryPoints: ['src/lib/learning-many-body-protocol.ts', 'src/lib/learning-many-body-worker-factory.ts'], bundle: true, write: false, outdir: 'unused', metafile: true, platform: 'browser', format: 'esm', target: 'es2020' });
  const inputs = Object.keys(result.metafile!.inputs);
  assert(!inputs.some(p => /learning-(spin-chain|fermion-fock|boson-fock|boson-coherent|mps|schmidt|boson-algebraic|tensor-roots|quantum-hilbert-core|observable-matrix|many-body-evaluator)\.ts$|interval-transcendentals/.test(p)), inputs.join('\n'));
  writeFileSync(`${runtime}/request-dependency-graph.json`, JSON.stringify({ inputs, numericalEnginesExcluded: true }, null, 2) + '\n');
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let reads = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { reads++; throw Error('SSR Worker'); } });
  try { await import('../src/lib/learning-many-body-worker-factory'); assert.equal(reads, 0); }
  finally { if (previous) Object.defineProperty(globalThis, 'Worker', previous); else Reflect.deleteProperty(globalThis, 'Worker'); }
});

test('bundled worker entry executes all 207 independent preparations off-thread and preserves frames', { timeout: 180000 }, async () => {
  assert.equal(preparations.length, 207);
  const code = await build({ entryPoints: ['src/lib/learning-many-body.worker.ts'], bundle: true, write: false, platform: 'browser', format: 'iife', target: 'es2020' });
  writeFileSync(`${runtime}/worker-bundle.js`, code.outputFiles[0].text);
  const worker = new Worker(`const {parentPort}=require('node:worker_threads');globalThis.self={postMessage:value=>parentPort.postMessage(value)};${code.outputFiles[0].text}\nparentPort.on('message',data=>self.onmessage({data}));`, { eval: true });
  try {
    for (const [i, p] of preparations.entries()) {
      const id = i + 1;
      const frames = await new Promise<unknown[]>((resolve, reject) => {
        const frames: unknown[] = [], timer = setTimeout(() => { cleanup(); reject(Error(`Worker timeout ${p.label}`)); }, 30000);
        function cleanup() { clearTimeout(timer); worker.off('message', message); worker.off('error', error); }
        function error(e: Error) { cleanup(); reject(e); }
        function message(value: unknown) { frames.push(value); if (frames.length === 2) { cleanup(); resolve(frames); } }
        worker.on('message', message); worker.once('error', error); worker.postMessage({ version: 1, requestId: id, input: p.input });
      });
      assert.deepEqual(frames[0], { version: 1, requestId: id, phase: 'started', kind: p.input.kind });
      const r = frames[1] as ReturnType<typeof evaluate>; assert(r.ok, p.label);
      assert.equal(r.requestId, id); assert.equal(r.inputKey, key(p.input)); assert.deepEqual(r.input, p.input);
      compare(r.result, p.expected, p.label);
      if (r.kind === 'schmidt') for (const cut of r.result.cuts) { directedZero(cut.directedResiduals); if (cut.truncation.chosen) directedZero(cut.truncation.chosen.directedResiduals); }
    }
    writeFileSync(`${runtime}/node-worker-verification.json`, JSON.stringify({ preparations: preparations.length, startedFrames: preparations.length, independentReferenceResultsCompared: preparations.length, workerCreated: 1, requestIdsAttributed: true, responseValidatorImplemented: false }, null, 2) + '\n');
  } finally { await worker.terminate(); }
});
after(() => writeFileSync(`${runtime}/observation-metrics.json`, JSON.stringify({ exactChecks, directedChecks, directedCertificateChecks }, null, 2) + '\n'));
