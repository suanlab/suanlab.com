/** Actual Chromium module Workers on an intercepted localhost page.
 * This verifies native lifecycle and arithmetic, not Next production assets.
 */
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { blockOrderExamples, type BlockOrderInput } from '../src/lib/learning-block-order-input';
import { evaluateBlockOrderRequest } from '../src/lib/learning-block-order-evaluator';
import type { BlockOrderCalculationController, BlockOrderCalculationState } from '../src/lib/learning-block-order-controller';

const root = '.runtime/learning/block-order-workers';
mkdirSync(root,{ recursive: true });
const oracle = spawnSync('python3',['scripts/learning/verify-block-order-foundation.py'],{ encoding: 'utf8', timeout: 120000, maxBuffer: 16000000 });
assert.equal(oracle.status,0,oracle.stderr);
const refs = JSON.parse(oracle.stdout) as { cases: { input: BlockOrderInput; expected: Record<string,unknown> }[] };
const maximum: BlockOrderInput = { shape: [4,4], blockShape: [2,2], boundary: 'periodic', disagreementActivity: '999983/999979',
  blockUpActivities: ['999961/999959','999953/999931','999917/999907','999883/999863'], selectedUpCounts: [1,2,3,1], witnessPair: [0,3] };
const cases = [...refs.cases.map((r,i) => ({ id: `independent-${i+1}`, input: r.input })), ...blockOrderExamples, { id: 'maximum-rational-activities', input: maximum }];
const stable = (value: unknown): string => Array.isArray(value) ? `[${value.map(stable).join(',')}]` : value && typeof value === 'object'
  ? `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${stable((value as Record<string,unknown>)[k])}`).join(',')}}` : JSON.stringify(value);
const hash = (value: unknown) => createHash('sha256').update(stable(value)).digest('hex');
const nodeResponses = cases.map(({ input }) => {
  const r = evaluateBlockOrderRequest({ version: 1, requestId: 1, input });
  assert(r.ok,JSON.stringify(r)); return r;
});
refs.cases.forEach(({ expected },i) => {
  const { profileChunks, ...rest } = nodeResponses[i].result.exact, raw = { ...rest, profiles: profileChunks.flat() };
  for (const [key,value] of Object.entries(expected)) {
    if (key === 'selected') for (const [field,v] of Object.entries(value as Record<string,unknown>)) assert.deepEqual(raw.selected[field as keyof typeof raw.selected],v);
    else assert.deepEqual(raw[key as keyof typeof raw],value);
  }
});

async function main() {
  const code: Record<string,string> = {};
  for (const [kind,entry] of Object.entries({ calculate: 'src/lib/learning-block-order.worker.ts', validate: 'src/lib/learning-block-order-validation.worker.ts' })) {
    const bundle = await build({ entryPoints: [entry], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true });
    code[kind] = bundle.outputFiles[0].text;
    writeFileSync(`${root}/${kind}.js`,code[kind]);
    writeFileSync(`${root}/${kind}-dependencies.json`,JSON.stringify(bundle.metafile,null,2)+'\n');
    if (kind === 'validate') for (const f of ['learning-block-order.ts','learning-block-order-observations.ts','learning-finite-markov-exact.ts','learning-interval-transcendentals.ts'])
      assert(!Object.keys(bundle.metafile!.inputs).includes(`src/lib/${f}`));
  }
  const controller = await build({ entryPoints: ['src/lib/learning-block-order-controller.ts'], bundle: true, write: false, platform: 'browser', format: 'iife', globalName: 'BlockOrderWorkerProbe', metafile: true });
  for (const f of ['learning-block-order.ts','learning-block-order-validation.ts','learning-block-order-observations.ts']) assert(!Object.keys(controller.metafile!.inputs).includes(`src/lib/${f}`));
  writeFileSync(`${root}/controller-dependencies.json`,JSON.stringify(controller.metafile,null,2)+'\n');
  for (const kind of ['mutated-exact','mutated-observation','wrong-attribution','oversized']) {
    const copy = structuredClone(nodeResponses[0]);
    if (kind === 'mutated-exact') copy.result.exact.selected.conditionalSpinMeans[0] = '999';
    if (kind === 'mutated-observation') copy.result.observations.logPartition = { lower: '0', upper: '0' };
    if (kind === 'wrong-attribution') copy.inputKey = 'wrong';
    if (kind === 'oversized') Object.assign(copy.result,{ excess: 'x'.repeat(65537) });
    code[kind] = `self.onmessage=e=>{const r=${JSON.stringify(copy)};r.requestId=e.data.requestId;self.postMessage(r);};`;
  }
  code.silent = 'self.onmessage=()=>{};';
  code.throwing = 'self.onmessage=()=>{throw Error("deliberate worker failure");};';
  const browser = await chromium.launch({ headless: true });
  const errors: string[] = [];
  try {
    const context = await browser.newContext();
    await context.addInitScript({ content: 'globalThis.__name=fn=>fn;' });
    const page = await context.newPage();
    page.on('pageerror',error => errors.push(error.message));
    const base = 'http://127.0.0.1:4175/__block_order_native_probe__/';
    await page.route(base,route => route.fulfill({ contentType: 'text/html', body: '<!doctype html><html lang="en"><title>Block order Worker probe</title><label>Typing<input id="typing"></label></html>' }));
    for (const [kind,body] of Object.entries(code)) await page.route(`${base}${kind}.js`,route => route.fulfill({ contentType: 'text/javascript', body }));
    await page.goto(base);
    await page.addScriptTag({ content: controller.outputFiles[0].text });
    const work = page.evaluate(async ({ cases,base }) => {
      const win = window as unknown as { BlockOrderWorkerProbe: { BlockOrderCalculationController: typeof BlockOrderCalculationController }; batchStarted: boolean; batchFinished: boolean };
      const Controller = win.BlockOrderWorkerProbe.BlockOrderCalculationController;
      const ports: { worker: Worker; terminated: number; listeners: Set<EventListenerOrEventListenerObject> }[] = [];
      let active = 0, peak = 0, frames = 0, ticking = true;
      const tick = () => { if (ticking) { frames++; requestAnimationFrame(tick); } }; requestAnimationFrame(tick);
      const factory = (kind: string, afterCreate?: (worker: Worker) => void) => () => {
        const worker = new Worker(`${base}${kind}.js`,{ type: 'module' }), p = { worker, terminated: 0, listeners: new Set<EventListenerOrEventListenerObject>() };
        ports.push(p); active++; peak = Math.max(peak,active);
        const terminate = worker.terminate.bind(worker), add = worker.addEventListener.bind(worker), remove = worker.removeEventListener.bind(worker);
        worker.terminate = () => { p.terminated++; active--; terminate(); };
        worker.addEventListener = ((type: string, listener: EventListenerOrEventListenerObject, options?: AddEventListenerOptions | boolean) => {
          if (type === 'messageerror') p.listeners.add(listener); add(type,listener,options);
        }) as Worker['addEventListener'];
        worker.removeEventListener = ((type: string, listener: EventListenerOrEventListenerObject, options?: EventListenerOptions | boolean) => {
          if (type === 'messageerror') p.listeners.delete(listener); remove(type,listener,options);
        }) as Worker['removeEventListener'];
        afterCreate?.(worker);
        return worker;
      };
      const digest = async (value: unknown) => {
        const stable = (v: unknown): string => Array.isArray(v) ? `[${v.map(stable).join(',')}]` : v && typeof v === 'object'
          ? `{${Object.keys(v).sort().map(k => `${JSON.stringify(k)}:${stable((v as Record<string,unknown>)[k])}`).join(',')}}` : JSON.stringify(v);
        return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(stable(value))))).map(v => v.toString(16).padStart(2,'0')).join('');
      };
      const run = async (input: BlockOrderInput, calc = 'calculate', gate = 'validate', deadline = 30000) => {
        let instance: BlockOrderCalculationController | null = null;
        try {
          return await new Promise<BlockOrderCalculationState>(resolve => {
            instance = new Controller({ calculate: factory(calc), validate: factory(gate) },state => {
              if (state.status === 'ready' || state.status === 'error') resolve(state);
            },{},deadline);
            instance.run(input);
          });
        } finally { (instance as BlockOrderCalculationController | null)?.dispose(); }
      };
      const rows = [];
      win.batchStarted = true;
      for (const c of cases) {
        const state = await run(c.input);
        if (state.status !== 'ready') throw Error(`${c.id}: ${JSON.stringify(state)}`);
        rows.push({ id: c.id, hash: await digest(state.response), profiles: state.response.result.exact.profileChunks.flat().length });
      }
      const failures = [];
      for (const [calc,gate,deadline] of [['mutated-exact','validate',30000],['mutated-observation','validate',30000],['wrong-attribution','validate',30000],
        ['oversized','validate',30000],['throwing','validate',30000],['silent','validate',50],['calculate','silent',1200]] as const) {
        const state = await run(cases[0].input,calc,gate,deadline);
        if (state.status !== 'error') throw Error(`forgery or failure accepted: ${calc}/${gate}`);
        failures.push({ calc,gate,error: state.error });
      }
      const cancellations = [];
      for (const phase of ['calculate','validate'] as const) {
        let instance: BlockOrderCalculationController | null = null;
        await new Promise<void>(resolve => {
          const cancel = () => queueMicrotask(() => { instance!.cancel(); resolve(); });
          instance = new Controller({ calculate: factory('calculate',phase === 'calculate' ? cancel : undefined), validate: factory('validate',phase === 'validate' ? cancel : undefined) },() => {});
          instance.run(cases[0].input);
        });
        const controller = instance as BlockOrderCalculationController | null;
        if (controller!.getSnapshot().status !== 'idle') throw Error('cancel did not become idle');
        controller!.dispose(); cancellations.push(phase);
      }
      const decodeFailures = [];
      for (const phase of ['calculate','validate'] as const) {
        let instance: BlockOrderCalculationController | null = null;
        const state = await new Promise<BlockOrderCalculationState>(resolve => {
          const decode = (worker: Worker) => queueMicrotask(() => worker.dispatchEvent(new MessageEvent('messageerror')));
          instance = new Controller({ calculate: factory('calculate',phase === 'calculate' ? decode : undefined), validate: factory('validate',phase === 'validate' ? decode : undefined) },s => { if (s.status === 'error') resolve(s); });
          instance.run(cases[0].input);
        });
        if (state.status !== 'error' || state.error !== 'invalid-response') throw Error('native decode error not handled');
        (instance as BlockOrderCalculationController | null)!.dispose(); decodeFailures.push(phase);
      }
      // Retry and replacement use the same controller instance, exercising
      // sequence attribution across actual native port lifetimes.
      const retryFactories = { calculate: factory('silent'), validate: factory('validate') };
      let receiveRetry: (state: BlockOrderCalculationState) => void = () => {};
      const retry = new Controller(retryFactories,state => { if (state.status === 'error' || state.status === 'ready') receiveRetry(state); },{},1200);
      let recovered: BlockOrderCalculationState;
      try {
        const failed = await new Promise<BlockOrderCalculationState>(resolve => { receiveRetry = resolve; retry.run(cases[0].input); });
        if (failed.status !== 'error' || failed.error !== 'timeout') throw Error('silent initial retry request did not time out');
        retryFactories.calculate = factory('calculate');
        recovered = await new Promise<BlockOrderCalculationState>(resolve => { receiveRetry = resolve; retry.run(cases[0].input); });
        if (recovered.status !== 'ready' || recovered.requestId !== 2) throw Error('same-controller native retry failed');
      } finally { retry.dispose(); }
      const retryResultHash = await digest(recovered.response.result);
      let replacement: BlockOrderCalculationController | null = null, replaced = false;
      let replacementState: BlockOrderCalculationState;
      try {
        replacementState = await new Promise<BlockOrderCalculationState>(resolve => {
          replacement = new Controller({ calculate: factory('calculate'), validate: factory('validate') },state => {
            if (!replaced && state.status === 'pending' && state.phase === 'calculating') {
              replaced = true; replacement!.run(cases[1].input);
            } else if (state.status === 'ready' || state.status === 'error') resolve(state);
          });
          replacement.run(cases[0].input);
        });
        if (replacementState.status !== 'ready' || replacementState.requestId !== 2 || replacementState.inputKey !== JSON.stringify(cases[1].input)) throw Error('native replacement admitted wrong input');
      } finally { (replacement as BlockOrderCalculationController | null)?.dispose(); }
      ticking = false; win.batchFinished = true;
      return { rows,failures,cancellations,decodeFailures,retryResultHash,sameControllerRetry: true, replacementRequestId: replacementState.requestId,
        frames,peak,active,ports: ports.map(p => ({ terminated: p.terminated, listeners: p.listeners.size,
        messageHandler: p.worker.onmessage !== null, errorHandler: p.worker.onerror !== null, decodeHandler: p.worker.onmessageerror !== null })) };
    },{ cases,base });
    await page.waitForFunction(() => (window as unknown as { batchStarted: boolean }).batchStarted);
    await page.locator('#typing').fill('typing while independent Workers calculate');
    const result = await work;
    assert.equal(await page.locator('#typing').inputValue(),'typing while independent Workers calculate');
    result.rows.forEach((row,i) => assert.equal(row.hash,hash(nodeResponses[i]),row.id));
    assert.equal(result.retryResultHash,hash(nodeResponses[0].result));
    assert.equal(result.replacementRequestId,2);
    assert.deepEqual(result.failures.map(r => r.error),['invalid-response','invalid-response','invalid-response','resource-limit','worker-failed','timeout','timeout']);
    assert.equal(result.active,0); assert(result.frames > 2);
    assert(result.ports.every(p => p.terminated === 1 && p.listeners === 0 && !p.messageHandler && !p.errorHandler && !p.decodeHandler));
    assert.deepEqual(errors,[]);
    writeFileSync(`${root}/verification.json`,JSON.stringify({ status: 'native-module-workers-verified', productionNextAssets: false,
      models: result.rows.length, independentReferenceModels: refs.cases.length, workerPorts: result.ports.length, typingResponsive: true, browserPageErrors: errors, ...result },null,2)+'\n');
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
