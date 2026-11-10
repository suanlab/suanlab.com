import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';
import { qubitWorkerCases } from './learning/qubit-worker-cases';
import { evaluateQubitRequest } from '../src/lib/learning-qubit-evaluator';
const root = '.runtime/learning/quantum-qubit-reader-integration';
const base = process.env.QUBIT_PREVIEW_BASE ?? 'http://localhost:4176';
const stable = (v: unknown): string => Array.isArray(v) ? `[${v.map(stable).join(',')}]` : v && typeof v === 'object' ? `{${Object.keys(v).sort().map(k => `${JSON.stringify(k)}:${stable((v as Record<string, unknown>)[k])}`).join(',')}}` : JSON.stringify(v);
const digest = (v: unknown) => createHash('sha256').update(stable(v)).digest('hex');
type Asset = { url: string; name?: string; type: 'classic' | 'module' };
async function main() {
  const expected = qubitWorkerCases.map((c, index) => {
    const response = evaluateQubitRequest({ version: 1, requestId: index + 1, input: c.input }); assert(response.ok);
    return { result: digest(c.expected), observations: digest(response.observations) };
  });
  const browser = await chromium.launch({ headless: true }), errors: string[] = [];
  try {
    const context = await browser.newContext();
    await context.addInitScript({ content: `globalThis.__name=(fn)=>fn;localStorage.setItem('suanlab-language','en');window.__qubitAssetPorts=[];const Native=Worker;window.Worker=class extends Native{constructor(url,opts){super(url,opts);window.__qubitAssetPorts.push({url:String(url),name:opts?.name,type:opts?.type??'classic'});}};` });
    const page = await context.newPage(); page.on('pageerror', error => errors.push(error.message));
    await page.goto(base + '/book/online/physics-math-for-ai/finite-qubit-circuit-order-interference/', { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => document.querySelector('[data-qubit-lab]')?.getAttribute('data-qubit-phase') === 'ready');
    const ports = await page.evaluate(() => (window as unknown as { __qubitAssetPorts: Asset[] }).__qubitAssetPorts);
    const calculate = ports.find(port => port.name === 'qubit-calculation'), validate = ports.find(port => port.name === 'qubit-validation');
    assert(calculate && validate);
    const assets = [];
    for (const port of [calculate, validate]) {
      const url = new URL(port.url, base); assert(url.pathname.startsWith('/_next/static/'));
      const response = await fetch(url); assert.equal(response.status, 200);
      const served = Buffer.from(await response.arrayBuffer()), disk = readFileSync('out' + url.pathname); assert.deepEqual(served, disk);
      assets.push({ ...port, bytes: disk.length, sha256: createHash('sha256').update(disk).digest('hex') });
    }
    const report = await page.evaluate(async ({ assets, inputs }) => {
      const stable = (v: unknown): string => Array.isArray(v) ? `[${v.map(stable).join(',')}]` : v && typeof v === 'object' ? `{${Object.keys(v).sort().map(k => `${JSON.stringify(k)}:${stable((v as Record<string, unknown>)[k])}`).join(',')}}` : JSON.stringify(v);
      const hash = async (v: unknown) => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(stable(v))))).map(x => x.toString(16).padStart(2, '0')).join('');
      const workers = assets.map(asset => new Worker(asset.url, { type: asset.type }));
      const rpc = (worker: Worker, frame: unknown) => new Promise<Record<string, unknown>>((resolve, reject) => {
        const timer = setTimeout(() => reject(Error('Production response timeout')), 30000);
        worker.onerror = error => { clearTimeout(timer); error.preventDefault(); reject(Error(error.message)); };
        worker.onmessage = event => { if (event.data.phase === 'started') return; clearTimeout(timer); resolve(event.data); };
        worker.postMessage(frame);
      });
      const results = [], tamperedKinds = new Set<string>();
      try {
        for (const [index, c] of inputs.entries()) {
          const requestId = index + 1, response = await rpc(workers[0], { version: 1, requestId, input: c.input });
          if (!response.ok) throw Error(`${c.label}: calculation rejected`);
          const receipt = await rpc(workers[1], { version: 1, requestId, input: c.input, response });
          if (receipt.phase !== 'validated') throw Error(`${c.label}: validation rejected`);
          const owned = receipt.response as Record<string, unknown>;
          results.push({ label: c.label, resultHash: await hash(response.result), observationHash: await hash(response.observations), validatedResultHash: await hash(owned.result), validatedObservationHash: await hash(owned.observations) });
          if (!tamperedKinds.has(c.input.kind)) {
            for (const field of ['result', 'observations']) {
              const changed = structuredClone(response);
              const alter = (value: unknown): boolean => {
                if (!value || typeof value !== 'object') return false;
                const record = value as Record<string, unknown>;
                if (field === 'result' && Array.isArray(record.real) && Array.isArray(record.imaginary)) { record.real[0] = record.real[0] === '0' ? '1' : '0'; return true; }
                if (field === 'observations' && typeof record.coordinate === 'number') { record.coordinate += .25; return true; }
                return Object.values(record).some(alter);
              };
              if (!alter(changed[field])) throw Error('No mathematical observation to forge');
              const rejected = await rpc(workers[1], { version: 1, requestId, input: c.input, response: changed });
              if (rejected.phase !== 'rejected' || rejected.error !== 'invalid-response') throw Error('Forged production observation admitted');
            }
            tamperedKinds.add(c.input.kind);
          }
        }
        return { results, tamperedKinds: [...tamperedKinds], probeWorkersCreated: 2, probeWorkersTerminated: 2 };
      } finally { workers.forEach(worker => worker.terminate()); }
    }, { assets: [calculate, validate], inputs: qubitWorkerCases.map(c => ({ label: c.label, input: c.input })) });
    assert.equal(report.results.length, 595); assert.equal(report.tamperedKinds.length, 5);
    report.results.forEach((row, i) => {
      assert.equal(row.label, qubitWorkerCases[i].label); assert.equal(row.resultHash, expected[i].result, row.label);
      assert.equal(row.observationHash, expected[i].observations, row.label);
      assert.equal(row.validatedResultHash, row.resultHash); assert.equal(row.validatedObservationHash, row.observationHash);
    });
    assert.deepEqual(errors, []);
    writeFileSync(root + '/production-assets.json', JSON.stringify({ assets, ...report, independentResultCases: 595, resultAndCoordinateTamperKindsRejected: 5, productionNextAssetsVerified: true, qaWorkerFactoryOverride: false, errors }, null, 2) + '\n');
    console.log(JSON.stringify({ preparations: 595, exactAndCoordinateTamperKindsRejected: 5, assets: assets.map(asset => ({ bytes: asset.bytes, type: asset.type })), probeWorkersTerminated: 2 }));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
