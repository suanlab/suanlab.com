import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { manyBodyKinds as kinds, snapshotManyBodyInput as snapshot, manyBodyInputKey as key, type ManyBodyInput } from '../src/lib/learning-many-body-input';
import type { ManyBodyResponse } from '../src/lib/learning-many-body-protocol';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
const runtime = '.runtime/learning/many-body-worker';
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
async function main() {
  const preparations = kinds.flatMap(kind => {
    const refs = JSON.parse(readFileSync(`scripts/fixtures/${kind}-reference.json`, 'utf8')) as { cases: { label: string; input: Record<string, number>; expected: unknown }[] };
    return refs.cases.map(c => ({ label: `${kind}: ${c.label}`, input: snapshot({ kind, parameters: c.input }), expected: c.expected }));
  });
  assert.equal(preparations.length, 207);
  const bundle = await build({ entryPoints: ['src/lib/learning-many-body.worker.ts'], bundle: true, write: false, platform: 'browser', format: 'iife', target: 'es2020' });
  writeFileSync(`${runtime}/native-worker-bundle.js`, bundle.outputFiles[0].text);
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage(), errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    // Isolated harness on the existing preview origin; no app/export mutation.
    await page.route('**/__many_body_worker_harness__/', route => route.fulfill({ contentType: 'text/html', body: '<!doctype html><html lang="en"><title>Worker verification</title><body>Worker verification</body></html>' }));
    await page.goto('http://localhost:4175/__many_body_worker_harness__/');
    const url = await page.evaluate(code => URL.createObjectURL(new Blob([code], { type: 'text/javascript' })), bundle.outputFiles[0].text);
    let created = 0, terminated = 0; const cases: unknown[] = [];
    try {
      for (const [i, p] of preparations.entries()) {
        const requestId = i + 1;
        const run = await page.evaluate(async ({ url, input, requestId }) => {
          const worker = new Worker(url, { type: 'module', name: 'many-body-native-verification' }), start = performance.now();
          try {
            const frames = await new Promise<unknown[]>((resolve, reject) => {
              const frames: unknown[] = [], timeout = setTimeout(() => reject(Error('Native many-body worker timed out')), 30000);
              worker.onmessage = event => { frames.push(event.data); if (frames.length === 2) { clearTimeout(timeout); resolve(frames); } };
              worker.onerror = event => { event.preventDefault(); clearTimeout(timeout); reject(Error(event.message)); };
              worker.onmessageerror = () => { clearTimeout(timeout); reject(Error('Native clone failed')); };
              worker.postMessage({ version: 1, requestId, input });
            });
            return { frames, milliseconds: performance.now() - start, terminated: true };
          } finally { worker.onmessage = null; worker.onerror = null; worker.onmessageerror = null; worker.terminate(); }
        }, { url, input: p.input, requestId });
        created++; terminated += Number(run.terminated);
        assert.deepEqual(run.frames[0], { version: 1, requestId, phase: 'started', kind: p.input.kind });
        const result = run.frames[1] as ManyBodyResponse;
        assert(result.ok, p.label); assert.equal(result.requestId, requestId); assert.equal(result.kind, p.input.kind);
        assert.equal(result.inputKey, key(p.input)); assert.deepEqual(result.input, p.input);
        compare(result.result, p.expected, p.label);
        if (result.kind === 'schmidt') for (const cut of result.result.cuts) { directedZero(cut.directedResiduals); if (cut.truncation.chosen) directedZero(cut.truncation.chosen.directedResiduals); }
        cases.push({ kind: p.input.kind, label: p.label, milliseconds: run.milliseconds });
        if ((i + 1) % 25 === 0) console.log(JSON.stringify({ completed: i + 1, total: preparations.length }));
      }
    } finally { await page.evaluate(url => URL.revokeObjectURL(url), url); }
    assert.equal(created, 207); assert.equal(terminated, created); assert.deepEqual(errors, []);
    const report = { workerType: 'native Chromium module Worker from bundled entry via Blob on isolated preview-origin harness',
      preparations: preparations.length, workersCreated: created, workersTerminated: terminated,
      independentReferenceResultsCompared: cases.length, exactChecks, directedChecks, directedCertificateChecks,
      responseValidatorImplemented: false, cancellationControllerVerified: false, actualReaderVerified: false,
      productionNextWorkerAssetVerified: false, publicDeploymentPerformed: false, errors, cases };
    writeFileSync(`${runtime}/native-verification.json`, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ ...report, cases: undefined }));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
