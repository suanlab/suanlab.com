/** Real Chromium module Workers in an intercepted localhost probe, not Next production assets. */
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { thermalFieldDefault, thermalFieldExamples, type ThermalFieldInput } from '../src/lib/learning-thermal-field-input';
import { evaluateThermalFieldRequest } from '../src/lib/learning-thermal-field-evaluator';
import type { ThermalFieldCalculationController, ThermalFieldCalculationState } from '../src/lib/learning-thermal-field-controller';

const root = '.runtime/learning/thermal-field-workers';
mkdirSync(root, { recursive: true });
const python = spawnSync('python3', ['scripts/learning/verify-thermal-field-foundation.py'], { encoding: 'utf8', timeout: 120000, maxBuffer: 8000000 });
assert.equal(python.status, 0, python.stderr);
const references = JSON.parse(python.stdout) as { cases: { input: ThermalFieldInput; expected: Record<string, unknown> }[] };
const cases = [...references.cases.map((x, i) => ({ id: `foundation-${i + 1}`, input: x.input })), ...thermalFieldExamples.map(x => ({ id: `example-${x.id}`, input: x.input }))];
const stable = (value: unknown): string => Array.isArray(value) ? `[${value.map(stable).join(',')}]` : value && typeof value === 'object'
  ? `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stable((value as Record<string, unknown>)[key])}`).join(',')}}` : JSON.stringify(value);
const hash = (value: unknown) => createHash('sha256').update(stable(value)).digest('hex');
const nodeOutputs = cases.map(c => {
  const output = evaluateThermalFieldRequest({ version: 1, requestId: 1, input: c.input });
  if (!output.ok) assert.fail(`${c.id}: ${output.error}`);
  return output;
});
references.cases.forEach((c, i) => {
  const exact = nodeOutputs[i].result.exact;
  for (const [key, value] of Object.entries(c.expected)) assert.deepEqual(exact[key as keyof typeof exact], value, `${i}:${key}`);
});

async function main() {
  const codes: Record<string, string> = {};
  for (const [kind, entry] of Object.entries({ calculate: 'src/lib/learning-thermal-field.worker.ts', validate: 'src/lib/learning-thermal-field-validation.worker.ts' })) {
    const bundle = await build({ entryPoints: [entry], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true });
    codes[kind] = bundle.outputFiles[0].text;
    if (kind === 'validate') for (const forbidden of ['learning-thermal-field.ts', 'learning-thermal-field-evaluator.ts', 'learning-thermal-field-observations.ts', 'learning-quantum-hilbert-core.ts', 'learning-real-time-interval.ts']) assert(!Object.keys(bundle.metafile!.inputs).includes(`src/lib/${forbidden}`));
    writeFileSync(`${root}/${kind}.js`, codes[kind]);
    writeFileSync(`${root}/${kind}-dependencies.json`, JSON.stringify(bundle.metafile, null, 2) + '\n');
  }
  const controller = await build({ stdin: { contents: `export {ThermalFieldCalculationController as Controller} from ${JSON.stringify(path.resolve('src/lib/learning-thermal-field-controller.ts'))};`, resolveDir: process.cwd(), loader: 'ts' }, bundle: true, write: false, platform: 'browser', format: 'iife', globalName: 'ThermalWorkerProbe', metafile: true });
  for (const forbidden of ['learning-thermal-field.ts', 'learning-thermal-field-validation.ts', 'learning-thermal-field-observations.ts']) assert(!Object.keys(controller.metafile!.inputs).includes(`src/lib/${forbidden}`));
  writeFileSync(`${root}/controller-dependencies.json`, JSON.stringify(controller.metafile, null, 2) + '\n');
  const forged = structuredClone(nodeOutputs[0]);
  if (forged.result.exact.kind !== 'normalizable') assert.fail('wrong forge preparation');
  forged.result.exact.thermalEnergy = '999';
  codes.mutated = `self.onmessage=e=>{const value=${JSON.stringify(forged)};value.requestId=e.data.requestId;value.input=e.data.input;value.inputKey=JSON.stringify(e.data.input);self.postMessage(value);};`;
  codes.silent = 'self.onmessage=()=>{};';
  const browser = await chromium.launch({ headless: true });
  const errors: string[] = [];
  try {
    const context = await browser.newContext();
    await context.addInitScript({ content: 'globalThis.__name=fn=>fn;' });
    const page = await context.newPage(); page.on('pageerror', e => errors.push(e.message));
    const base = 'http://127.0.0.1:4175/__thermal_field_worker_probe__/';
    await page.route(base, route => route.fulfill({ contentType: 'text/html', body: '<!doctype html><html lang="en"><title>Thermal field Worker probe</title><label>Typing<input id="typing"></label></html>' }));
    for (const [kind, body] of Object.entries(codes)) await page.route(`${base}${kind}.js`, route => route.fulfill({ contentType: 'text/javascript', body }));
    await page.goto(base);
    await page.addScriptTag({ content: controller.outputFiles[0].text });
    const work = page.evaluate(async ({ cases, base }) => {
      const win = window as unknown as { ThermalWorkerProbe: { Controller: typeof ThermalFieldCalculationController }; batchStarted: boolean; batchFinished: boolean };
      const Controller = win.ThermalWorkerProbe.Controller;
      const ports: { worker: Worker; terminated: number; listeners: Set<EventListenerOrEventListenerObject> }[] = [];
      let active = 0, peak = 0, frames = 0, ticking = true;
      const tick = () => { if (ticking) { frames++; requestAnimationFrame(tick); } }; requestAnimationFrame(tick);
      const factory = (kind: string) => () => {
        const worker = new Worker(`${base}${kind}.js`, { type: 'module' }), port = { worker, terminated: 0, listeners: new Set<EventListenerOrEventListenerObject>() };
        ports.push(port); active++; peak = Math.max(peak, active);
        const terminate = worker.terminate.bind(worker);
        worker.terminate = () => { port.terminated++; active--; terminate(); };
        const add = worker.addEventListener.bind(worker), remove = worker.removeEventListener.bind(worker);
        worker.addEventListener = ((type: string, listener: EventListenerOrEventListenerObject, options?: boolean | AddEventListenerOptions) => {
          if (type === 'messageerror') port.listeners.add(listener); add(type, listener, options);
        }) as Worker['addEventListener'];
        worker.removeEventListener = ((type: string, listener: EventListenerOrEventListenerObject, options?: boolean | EventListenerOptions) => {
          if (type === 'messageerror') port.listeners.delete(listener); remove(type, listener, options);
        }) as Worker['removeEventListener'];
        return worker;
      };
      const stable = (value: unknown): string => Array.isArray(value) ? `[${value.map(stable).join(',')}]` : value && typeof value === 'object'
        ? `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stable((value as Record<string, unknown>)[key])}`).join(',')}}` : JSON.stringify(value);
      const hash = async (value: unknown) => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(stable(value))))).map(x => x.toString(16).padStart(2, '0')).join('');
      const rows = [];
      win.batchStarted = true; win.batchFinished = false;
      for (const c of cases) {
        let finish!: (state: ThermalFieldCalculationState) => void;
        const done = new Promise<ThermalFieldCalculationState>(resolve => { finish = resolve; });
        const phases: string[] = [];
        const instance = new Controller({ calculate: factory('calculate'), validate: factory('validate') }, state => {
          phases.push(state.status === 'pending' ? state.phase : state.status);
          if (state.status === 'ready' || state.status === 'error') finish(state);
        });
        instance.run(c.input);
        const state = await done;
        instance.dispose();
        if (state.status !== 'ready') throw Error(`${c.id}: ${JSON.stringify(state)}`);
        rows.push({ id: c.id, hash: await hash(state.response.result), phases });
        if (active !== 0) throw Error(`${c.id}: leaked ${active} Workers`);
      }
      ticking = false; win.batchFinished = true;
      return { rows, frames, created: ports.length, active, peak, allTerminatedOnce: ports.every(p => p.terminated === 1), handlersCleared: ports.every(p => !p.worker.onmessage && !p.worker.onerror && !p.worker.onmessageerror && p.listeners.size === 0) };
    }, { cases, base });
    await page.waitForFunction(() => (window as unknown as { batchStarted: boolean }).batchStarted);
    await page.locator('#typing').fill('worker calculations leave typing available');
    const typingBeforeBatchFinished = await page.evaluate(() => !(window as unknown as { batchFinished: boolean }).batchFinished);
    const batch = await work;
    batch.rows.forEach((row, i) => { assert.equal(row.hash, hash(nodeOutputs[i].result), row.id); assert.deepEqual(row.phases, ['starting', 'calculating', 'validating', 'ready']); });
    assert(batch.allTerminatedOnce && batch.handlersCleared); assert.equal(batch.active, 0); assert(typingBeforeBatchFinished && batch.frames > 1);
    const lifecycle = await page.evaluate(async ({ base, input, replacement }) => {
      const Controller = (window as unknown as { ThermalWorkerProbe: { Controller: typeof ThermalFieldCalculationController } }).ThermalWorkerProbe.Controller;
      const ports: { worker: Worker; terminated: number; listeners: Set<EventListenerOrEventListenerObject> }[] = [];
      let active = 0;
      const factory = (kind: string, decodeError = false) => () => {
        const worker = new Worker(`${base}${kind}.js`, { type: 'module' }), port = { worker, terminated: 0, listeners: new Set<EventListenerOrEventListenerObject>() };
        ports.push(port); active++;
        const terminate = worker.terminate.bind(worker); worker.terminate = () => { port.terminated++; active--; terminate(); };
        const add = worker.addEventListener.bind(worker), remove = worker.removeEventListener.bind(worker);
        worker.addEventListener = ((type: string, listener: EventListenerOrEventListenerObject, options?: boolean | AddEventListenerOptions) => {
          if (type === 'messageerror') port.listeners.add(listener); add(type, listener, options);
        }) as Worker['addEventListener'];
        worker.removeEventListener = ((type: string, listener: EventListenerOrEventListenerObject, options?: boolean | EventListenerOptions) => {
          if (type === 'messageerror') port.listeners.delete(listener); remove(type, listener, options);
        }) as Worker['removeEventListener'];
        if (decodeError) queueMicrotask(() => worker.dispatchEvent(new MessageEvent('messageerror')));
        return worker;
      };
      const outcomes = [];
      for (const scenario of ['replacement', 'cancel', 'wrong-math', 'silent-validator', 'calculation-decode-error', 'validation-decode-error']) {
        let finish!: (state: ThermalFieldCalculationState) => void;
        const done = new Promise<ThermalFieldCalculationState>(resolve => { finish = resolve; });
        const phases: string[] = [];
        const instance = new Controller({ calculate: factory(scenario === 'wrong-math' ? 'mutated' : 'calculate', scenario === 'calculation-decode-error'),
          validate: factory(scenario === 'silent-validator' ? 'silent' : 'validate', scenario === 'validation-decode-error') }, state => {
          phases.push(state.status === 'pending' ? state.phase : state.status);
          if (state.status !== 'pending') finish(state);
        }, {}, scenario === 'silent-validator' ? 1500 : 30000);
        instance.run(input);
        if (scenario === 'replacement') instance.run(replacement);
        if (scenario === 'cancel') instance.cancel();
        const state = await done; instance.dispose();
        outcomes.push({ scenario, status: state.status, requestId: state.requestId, error: state.status === 'error' ? state.error : null, phases });
        if (active !== 0) throw Error(`${scenario}: ${active} live Workers`);
      }
      return { outcomes, created: ports.length, active, allTerminatedOnce: ports.every(p => p.terminated === 1), handlersCleared: ports.every(p => !p.worker.onmessage && !p.worker.onerror && !p.worker.onmessageerror && p.listeners.size === 0) };
    }, { base, input: thermalFieldDefault, replacement: thermalFieldExamples[2].input });
    assert.deepEqual(lifecycle.outcomes.map(x => [x.scenario, x.status, x.error]), [
      ['replacement', 'ready', null], ['cancel', 'idle', null], ['wrong-math', 'error', 'invalid-response'], ['silent-validator', 'error', 'timeout'],
      ['calculation-decode-error', 'error', 'invalid-response'], ['validation-decode-error', 'error', 'invalid-response'],
    ]);
    assert.equal(lifecycle.outcomes[0].requestId, 2); assert(lifecycle.outcomes[3].phases.includes('validating'));
    assert(lifecycle.allTerminatedOnce && lifecycle.handlersCleared); assert.equal(lifecycle.active, 0); assert.deepEqual(errors, []);
    writeFileSync(`${root}/native-verification.json`, JSON.stringify({ status: 'native-esbuild-workers-verified', requests: cases.length,
      exactPythonReferenceModels: references.cases.length, examples: thermalFieldExamples.length, typingBeforeBatchFinished, batch, lifecycle,
      productionNextAssetsVerified: false, readerRegistered: false, errors }, null, 2) + '\n');
    console.log(JSON.stringify({ nativeRequests: cases.length, activeAfter: batch.active + lifecycle.active, frames: batch.frames, typingBeforeBatchFinished, lifecycle: lifecycle.outcomes }));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
