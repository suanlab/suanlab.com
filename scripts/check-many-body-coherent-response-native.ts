import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { snapshotManyBodyInput as snapshot } from '../src/lib/learning-many-body-input';
import { readFermionFockSuccessForInput } from '../src/lib/learning-many-body-fermion-validation';
import { readSpinChainSuccessForInput } from '../src/lib/learning-many-body-spin-validation';
import { readBosonFockSuccessForInput } from '../src/lib/learning-many-body-boson-validation';
import { readBosonCoherentSuccessForInput } from '../src/lib/learning-many-body-coherent-validation';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
const runtime = '.runtime/learning/many-body-coherent-response';
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
  const families = ['spin-chain', 'fermion-fock', 'boson-fock', 'boson-coherent'] as const;
  const refs = families.flatMap(kind => (JSON.parse(readFileSync(`scripts/fixtures/${kind}-reference.json`, 'utf8')).cases as { label: string; input: Record<string, number>; expected: unknown }[]).map(ref => ({ ...ref, kind })));
  const readPacket = { 'spin-chain': readSpinChainSuccessForInput, 'fermion-fock': readFermionFockSuccessForInput, 'boson-fock': readBosonFockSuccessForInput, 'boson-coherent': readBosonCoherentSuccessForInput };
  const [workerBundle, validatorBundle] = await Promise.all([
    build({ entryPoints: ['src/lib/learning-many-body.worker.ts'], bundle: true, write: false, platform: 'browser', format: 'iife', target: 'es2020' }),
    build({ stdin: { contents: "export {readSpinChainSuccessForInput} from './src/lib/learning-many-body-spin-validation'; export {readFermionFockSuccessForInput} from './src/lib/learning-many-body-fermion-validation'; export {readBosonFockSuccessForInput} from './src/lib/learning-many-body-boson-validation'; export {readBosonCoherentSuccessForInput} from './src/lib/learning-many-body-coherent-validation';", resolveDir: process.cwd(), sourcefile: 'many-body-four-validators.ts' }, bundle: true, write: false, platform: 'browser', format: 'iife', globalName: 'manyBodyCoherentWire', target: 'es2020' }),
  ]);
  writeFileSync(`${runtime}/native-validator-bundle.js`, validatorBundle.outputFiles[0].text);
  writeFileSync(`${runtime}/native-worker-bundle.js`, workerBundle.outputFiles[0].text);
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage(), errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/__many_body_coherent_response__/', route => route.fulfill({ contentType: 'text/html', body: '<!doctype html><html lang="en"><title>Coherent response verification</title><body>Coherent response verification</body></html>' }));
    await page.goto('http://localhost:4175/__many_body_coherent_response__/');
    await page.addScriptTag({ content: validatorBundle.outputFiles[0].text });
    const url = await page.evaluate(code => URL.createObjectURL(new Blob([code], { type: 'text/javascript' })), workerBundle.outputFiles[0].text);
    let created = 0, terminated = 0; const cases: { kind: string; label: string; validationMilliseconds: number }[] = [];
    let coherentExactChecks = 0, coherentDirectedChecks = 0;
    try {
      for (const [index, ref] of refs.entries()) {
        const input = snapshot({ kind: ref.kind, parameters: ref.input }), requestId = index + 1;
        const run = await page.evaluate(async ({ url, input, requestId }) => {
          const readers = (globalThis as unknown as { manyBodyCoherentWire: Record<string, (value: unknown, input: unknown, id: number) => unknown> }).manyBodyCoherentWire;
          const name = input.kind === 'spin-chain' ? 'readSpinChainSuccessForInput' : input.kind === 'fermion-fock' ? 'readFermionFockSuccessForInput' : input.kind === 'boson-fock' ? 'readBosonFockSuccessForInput' : 'readBosonCoherentSuccessForInput';
          const validate = readers[name];
          const worker = new Worker(url, { type: 'module', name: 'many-body-coherent-response-verification' });
          try {
            const frames = await new Promise<any[]>((resolve, reject) => {
              const frames: unknown[] = [], timer = setTimeout(() => reject(Error('Native worker timeout')), 30000);
              worker.onmessage = event => { frames.push(event.data); if (frames.length === 2) { clearTimeout(timer); resolve(frames); } };
              worker.onerror = event => { event.preventDefault(); clearTimeout(timer); reject(Error(event.message)); };
              worker.onmessageerror = () => { clearTimeout(timer); reject(Error('Native clone failure')); };
              worker.postMessage({ version: 1, requestId, input });
            });
            const start = performance.now(), owned = validate(frames[1], input, requestId), milliseconds = performance.now() - start;
            if (!owned || !Object.isFrozen(owned)) throw Error('Browser response validation failed');
            if (validate(frames[1], input, requestId + 1) !== null) throw Error('Wrong request ID accepted');
            const corrupt = structuredClone(frames[1]); const energy = input.kind === 'boson-coherent' ? corrupt.result.normalizedProjection.energy.observation : input.kind === 'boson-fock' ? corrupt.result.energy.observation : corrupt.result.energy; energy.value = 1234567;
            if (validate(corrupt, input, requestId) !== null) throw Error('Corrupt energy metadata accepted');
            return { started: frames[0], owned, validationMilliseconds: milliseconds, wrongIdRejected: true, corruptMetadataRejected: true };
          } finally { worker.onmessage = null; worker.onerror = null; worker.onmessageerror = null; worker.terminate(); }
        }, { url, input, requestId });
        created++; terminated++;
        assert.deepEqual(run.started, { version: 1, requestId, phase: 'started', kind: ref.kind });
        const packet = readPacket[ref.kind](run.owned, input, requestId); assert(packet, ref.label);
        const beforeExact = exactChecks, beforeDirected = directedChecks; compare(packet.result, ref.expected, ref.label);
        if (ref.kind === 'boson-coherent') { coherentExactChecks += exactChecks - beforeExact; coherentDirectedChecks += directedChecks - beforeDirected; }
        cases.push({ kind: ref.kind, label: ref.label, validationMilliseconds: run.validationMilliseconds });
        if ((index + 1) % 20 === 0) console.log(JSON.stringify({ completed: index + 1, total: refs.length }));
      }
    } finally { await page.evaluate(url => URL.revokeObjectURL(url), url); }
    assert.equal(created, 152); assert.equal(terminated, created); assert.deepEqual(errors, []);
    const report = { nativeModuleWorkerPreparations: refs.length, browserValidatorAccepted: refs.length,
      workersCreated: created, workersTerminated: terminated, browserWrongIdsRejected: 152, browserCorruptMetadataRejected: 152,
      exactChecks, directedChecks, maximumBrowserValidationMilliseconds: Math.max(...cases.map(c => c.validationMilliseconds)),
      currentCoherentPreparations: 32, compatibilityPreparations: 120, coherentExactChecks, coherentDirectedChecks, maximumCoherentValidationMilliseconds: Math.max(...cases.filter(c => c.kind === 'boson-coherent').map(c => c.validationMilliseconds)), responseFamiliesVerified: [...families], pendingResponseFamilies: ['mps', 'schmidt'],
      actualReaderVerified: false, cancellationControllerVerified: false, productionNextWorkerAssetVerified: false, errors, cases };
    writeFileSync(`${runtime}/native-verification.json`, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ ...report, cases: undefined }));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
