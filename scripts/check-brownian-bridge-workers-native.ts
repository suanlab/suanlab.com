import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { brownianBridgeExamples, type BrownianBridgeInput } from '../src/lib/learning-brownian-bridge-input';
import { calculateBrownianBridge } from '../src/lib/learning-brownian-bridge';

const root = '.runtime/learning/brownian-bridge-worker';
mkdirSync(root, { recursive: true });
const reference = JSON.parse(readFileSync('scripts/fixtures/brownian-bridge-references.json', 'utf8')) as { cases: { id: string; input: BrownianBridgeInput }[] };
const cases = [...reference.cases, ...brownianBridgeExamples];
const stable = (value: unknown): string => Array.isArray(value) ? `[${value.map(stable).join(',')}]` : value && typeof value === 'object' ? `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stable((value as Record<string, unknown>)[key])}`).join(',')}}` : JSON.stringify(value);
const hash = (value: unknown) => createHash('sha256').update(stable(value)).digest('hex');

async function main() {
  const codes: Record<string, string> = {};
  for (const [kind, entry] of Object.entries({ calculate: 'src/lib/learning-brownian-bridge.worker.ts', validate: 'src/lib/learning-brownian-bridge-validation.worker.ts' })) {
    const bundle = await build({ entryPoints: [entry], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true });
    codes[kind] = bundle.outputFiles[0].text;
    if (kind === 'validate') for (const name of ['learning-brownian-bridge.ts', 'learning-brownian-bridge-evaluator.ts', 'learning-quantum-hilbert-core.ts']) assert(!Object.keys(bundle.metafile!.inputs).includes('src/lib/' + name));
    writeFileSync(`${root}/${kind}.js`, codes[kind]);
    writeFileSync(`${root}/${kind}-dependency-graph.json`, JSON.stringify(bundle.metafile, null, 2) + '\n');
  }
  const expected = cases.map(c => hash(calculateBrownianBridge(c.input)));
  const browser = await chromium.launch({ headless: true }), errors: string[] = [];
  try {
    const context = await browser.newContext();
    await context.addInitScript({ content: 'globalThis.__name=fn=>fn;' });
    const page = await context.newPage(); page.on('pageerror', e => errors.push(e.message));
    await page.route('**/__brownian_bridge_workers__/', route => route.fulfill({ contentType: 'text/html', body: '<!doctype html><html lang="en"><title>Brownian bridge Worker checks</title><body><label>Typing<input id="typing"></label></body></html>' }));
    for (const kind of ['calculate', 'validate']) await page.route(`**/__brownian_bridge_workers__/${kind}.js`, route => route.fulfill({ contentType: 'text/javascript', body: codes[kind] }));
    await page.goto('http://127.0.0.1:4175/__brownian_bridge_workers__/');
    const work = page.evaluate(async inputs => {
      const workers = [new Worker('calculate.js', { type: 'module' }), new Worker('validate.js', { type: 'module' })];
      let started = 0; const lifecycle = { created: workers.length, terminated: 0 };
      const stable = (value: unknown): string => Array.isArray(value) ? `[${value.map(stable).join(',')}]` : value && typeof value === 'object' ? `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stable((value as Record<string, unknown>)[key])}`).join(',')}}` : JSON.stringify(value);
      const digest = async (value: unknown) => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(stable(value))))).map(x => x.toString(16).padStart(2, '0')).join('');
      const rpc = (port: Worker, message: unknown) => new Promise<Record<string, unknown>>((resolve, reject) => {
        const timer = setTimeout(() => reject(Error('Native Worker deadline')), 30000);
        port.onerror = e => { clearTimeout(timer); e.preventDefault(); reject(Error(e.message)); };
        port.onmessage = e => { if (e.data.phase === 'started') { started++; return; } clearTimeout(timer); resolve(e.data); };
        port.postMessage(message);
      });
      const rows = []; let rejectedMutations = 0, rejectedAttributions = 0;
      const state = window as unknown as { __brownianBridgeBatch: { started: boolean; finished: boolean } };
      state.__brownianBridgeBatch = { started: true, finished: false };
      try {
        for (const [i, c] of inputs.entries()) {
          const requestId = i + 1, response = await rpc(workers[0], { version: 1, requestId, input: c.input });
          if (!response.ok) throw Error(c.id + ': calculation rejected');
          const receipt = await rpc(workers[1], { version: 1, requestId, input: c.input, response });
          if (receipt.phase !== 'validated') throw Error(c.id + ': validation rejected');
          const accepted = receipt.response as Record<string, unknown>;
          rows.push({ id: c.id, hash: await digest(response.result), validatedHash: await digest(accepted.result) });
          const changed = structuredClone(response), result = changed.result as { integrationDimension: number };
          result.integrationDimension += 1;
          if ((await rpc(workers[1], { version: 1, requestId, input: c.input, response: changed })).phase !== 'rejected') throw Error('Wrong Brownian bridge integration dimension accepted');
          rejectedMutations++;
          if ((await rpc(workers[1], { version: 1, requestId: requestId + 1000, input: c.input, response })).phase !== 'rejected') throw Error('Wrong request ID accepted');
          rejectedAttributions++;
        }
        return { rows, started, rejectedMutations, rejectedAttributions, lifecycle };
      } finally { state.__brownianBridgeBatch.finished = true; workers.forEach(worker => { worker.terminate(); lifecycle.terminated++; }); }
    }, cases.map(c => ({ id: c.id, input: c.input })));
    await page.waitForFunction(() => (window as unknown as { __brownianBridgeBatch: { started: boolean } }).__brownianBridgeBatch?.started);
    await page.locator('#typing').fill('Brownian bridge calculations leave typing responsive');
    assert.equal(await page.locator('#typing').inputValue(), 'Brownian bridge calculations leave typing responsive');
    const typingBeforeBatchFinished = await page.evaluate(() => !(window as unknown as { __brownianBridgeBatch: { finished: boolean } }).__brownianBridgeBatch.finished);
    assert(typingBeforeBatchFinished);
    const result = await work;
    result.rows.forEach((row, i) => { assert.equal(row.hash, expected[i], row.id); assert.equal(row.validatedHash, row.hash); });
    assert.equal(result.lifecycle.created, 2); assert.equal(result.lifecycle.terminated, 2); assert.equal(result.rows.length, cases.length); assert.equal(result.started, cases.length); assert.deepEqual(errors, []);
    const report = { ...result, independentPythonFixtureCases: reference.cases.length, explicitExamples: brownianBridgeExamples.length,  requestsIncludeRepeatedExamples: true, actualModuleWorkers: true, typingBeforeBatchFinished, productionNextAssetsVerified: false, readerRegistered: false, errors };
    writeFileSync(`${root}/native-workers.json`, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ nativeRequests: result.rows.length, rejectedMutations: result.rejectedMutations, rejectedAttributions: result.rejectedAttributions, typingBeforeBatchFinished }));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
