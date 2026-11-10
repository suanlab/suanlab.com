/** Isolated real-browser Worker/React lifecycle probe. This does not claim
 * verification of Next reader integration or its emitted worker chunk URLs.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdirSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { finiteSizeKinds as kinds, type FiniteSizeInput } from '../src/lib/learning-finite-size-protocol';
const run = '.runtime/learning/finite-size-worker';
const inputs: FiniteSizeInput[] = [
 { kind: kinds[0], parameters: { dimension: 2, lengthOne: 2, lengthTwo: 2, width: 3, growthMode: 'uniform-dilation', scale: 4, spinOne: 1, spinTwo: -1, couplingNumerator: 1, couplingDenominator: 1 } },
 { kind: kinds[1], parameters: { siteCount: 4, occupiedWeight: 1, emptyWeight: 1, commonNumerator: 1, commonDenominator: 2, commonScaling: 'fixed', epsilonNumerator: 1, epsilonDenominator: 2 } },
 { kind: kinds[2], parameters: { siteCount: 4, activityNumerator: 2, activityDenominator: 1, sizePath: 'fixed', approachSide: 1, offsetNumerator: 1, offsetDenominator: 2 } },
];
const heavy: FiniteSizeInput[] = [
 { kind: kinds[0], parameters: { ...(inputs[0].parameters as Extract<FiniteSizeInput, { kind: typeof kinds[0] }>['parameters']), dimension: 3, lengthOne: 1000000, lengthTwo: 1000000, width: 1000000, scale: 1000000, couplingDenominator: 1000000 } },
 { kind: kinds[1], parameters: { ...(inputs[1].parameters as Extract<FiniteSizeInput, { kind: typeof kinds[1] }>['parameters']), siteCount: 512, occupiedWeight: 1, emptyWeight: 999999, commonNumerator: 1, commonDenominator: 1000000, commonScaling: 'inverse-size-squared', epsilonNumerator: 999999, epsilonDenominator: 1000000 } },
 { kind: kinds[2], parameters: { ...(inputs[2].parameters as Extract<FiniteSizeInput, { kind: typeof kinds[2] }>['parameters']), siteCount: 512, activityNumerator: 1, activityDenominator: 1000000 } },
];
async function main() {
 mkdirSync(run, { recursive: true });
 const probe = `${run}/browser-probe.tsx`;
 writeFileSync(probe, `import React,{useState,StrictMode} from 'react';
import {createRoot} from 'react-dom/client';import {flushSync} from 'react-dom';
import {useFiniteSizeCalculation} from '@/components/learning/use-finite-size-calculation';
import {finiteSizeInputKey,isFiniteSizeSuccessForInput} from '@/lib/learning-finite-size-protocol';
const w=window as any;w.probe={renders:[],input:null,status:'idle'};const root=createRoot(document.getElementById('root')!);
function Probe(){const [input,setInput]=useState<any>(null),[enabled,setEnabled]=useState(true);const state=useFiniteSizeCalculation(input,enabled);
const renderedKey=input?finiteSizeInputKey(input):null;const responseKey=state.response?.inputKey??null;
w.probe.renders.push({renderedKey,responseKey,status:state.status});Object.assign(w.probe,{input,status:state.status,phase:state.phase,response:state.response,error:state.error,valid:state.response?isFiniteSizeSuccessForInput(state.response,input,state.requestId!):null,
change:(p:any)=>flushSync(()=>setInput(p)),enable:(v:boolean)=>flushSync(()=>setEnabled(v)),cancel:state.cancel,retry:state.retry,unmount:()=>flushSync(()=>root.unmount())});
return <output>{state.status}</output>;}root.render(<StrictMode><Probe/></StrictMode>);`);
 const [mainBundle, workerBundle] = await Promise.all([
  build({ entryPoints: [probe], bundle: true, write: false, format: 'esm', platform: 'browser', target: 'es2022', define: { 'process.env.NODE_ENV': '"development"' }, logLevel: 'silent' }),
  build({ entryPoints: ['src/lib/learning-finite-size.worker.ts'], bundle: true, write: false, format: 'esm', platform: 'browser', target: 'es2022', logLevel: 'silent' }),
 ]);
 const server = createServer((req, res) => { const url = new URL(req.url ?? '/', 'http://localhost'); if (url.pathname === '/probe.js' || url.pathname === '/learning-finite-size.worker.ts') { res.setHeader('Content-Type', 'text/javascript'); res.end(url.pathname === '/probe.js' ? mainBundle.outputFiles[0].text : workerBundle.outputFiles[0].text); } else { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><html><body><div id="root"></div><script type="module" src="/probe.js"></script></body></html>'); } });
 await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve)); const address = server.address(); assert.ok(address && typeof address === 'object');
 const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
 const report = { scope: 'Isolated browser Worker and React StrictMode hook; Next reader chunks pending integration', normalCases: 0, maximumPayloadCases: 0, startedAckCancellations: 0, recoveredCases: 0, replacements: 0, disableCases: 0, unavailableAndRetryCases: 0, inFlightUnmounts: 0, workersCreated: 0, workersTerminated: 0, liveWorkers: -1, staleReadyRenders: -1, errors: [] as string[] };
 try {
  const context = await browser.newContext(); await context.addInitScript(() => {
   const w = window as any, Native = Worker, stats = w.workerStats = { created: 0, terminated: 0, live: new Set<Worker>(), cancelled: 0, replaced: 0, disabled: 0, unmounted: 0 };
   w.Worker = class extends Native { ended = false; constructor(...args: ConstructorParameters<typeof Worker>) { if (w.failWorker) throw Error('worker disabled for probe'); super(...args); stats.created++; stats.live.add(this); this.addEventListener('message', event => { if (event.data?.phase !== 'started') return; if (w.cancelOnStart) { w.cancelOnStart = false; w.probe.cancel(); stats.cancelled++; } else if (w.replaceOnStart) { const next = w.replaceOnStart; w.replaceOnStart = null; w.probe.change(next); stats.replaced++; } else if (w.disableOnStart) { w.disableOnStart = false; w.probe.enable(false); stats.disabled++; } else if (w.unmountOnStart) { w.unmountOnStart = false; w.probe.unmount(); stats.unmounted++; } }); } terminate() { if (!this.ended) { this.ended = true; stats.terminated++; stats.live.delete(this); } super.terminate(); } };
  });
  const page = await context.newPage(); page.on('pageerror', e => report.errors.push(e.message)); await page.goto(`http://127.0.0.1:${address.port}/`); await page.waitForFunction(() => typeof (window as any).probe?.change === 'function');
  const ready = async () => { await page.waitForFunction(() => (window as any).probe.status === 'ready', undefined, { timeout: 30000 }); assert.equal(await page.evaluate(() => (window as any).probe.valid), true); };
  for (const input of inputs) { await page.evaluate(input => (window as any).probe.change(input), input); await ready(); report.normalCases++; }
  for (let i = 0; i < 3; i++) {
   await page.evaluate(input => { (window as any).cancelOnStart = true; (window as any).probe.change(input); }, heavy[i]); await page.waitForFunction(n => (window as any).workerStats.cancelled === n && (window as any).probe.status === 'idle', i + 1); report.startedAckCancellations++;
   await page.evaluate(input => (window as any).probe.change(input), inputs[i]); await ready(); report.recoveredCases++;
   await page.evaluate(input => (window as any).probe.change(input), heavy[i]); await ready();
   const payload = await page.evaluate(() => { const r = (window as any).probe.response.result; return { siteCount: r.current.geometry?.siteCount, tailRatio: r.current.eventTailRatioToIndependent?.value, tailResolution: r.current.eventTailRatioToIndependent?.resolution, correctionValue: r.current.pressureCorrection?.value, correctionSign: r.current.pressureCorrection?.sign }; });
   if (i === 0) assert.equal(payload.siteCount, '2000000000000000000000000000000000000'); if (i === 1) { assert.equal(payload.tailRatio, null); assert.equal(payload.tailResolution, 'exact-rational'); } if (i === 2) { assert.equal(payload.correctionValue, null); assert.equal(payload.correctionSign, 1); } report.maximumPayloadCases++;
  }
  await page.evaluate(({ input, next }) => { (window as any).replaceOnStart = next; (window as any).probe.change(input); }, { input: heavy[1], next: inputs[2] }); await page.waitForFunction(() => (window as any).workerStats.replaced === 1 && (window as any).probe.status === 'ready' && (window as any).probe.response.kind === 'two-phase-partition-zeros-rounding'); report.replacements++;
  await page.evaluate(input => { (window as any).disableOnStart = true; (window as any).probe.change(input); }, heavy[1]); await page.waitForFunction(() => (window as any).workerStats.disabled === 1 && (window as any).probe.status === 'idle' && (window as any).probe.response === null && (window as any).workerStats.live.size === 0); report.disableCases++;
  await page.evaluate(() => { (window as any).failWorker = true; (window as any).probe.enable(true); }); await page.waitForFunction(() => (window as any).probe.error === 'unavailable'); await page.evaluate(() => { (window as any).failWorker = false; (window as any).probe.retry(); }); await ready(); report.unavailableAndRetryCases++;
  await page.evaluate(input => { (window as any).unmountOnStart = true; (window as any).probe.change(input); }, heavy[2]); await page.waitForFunction(() => (window as any).workerStats.unmounted === 1 && (window as any).workerStats.live.size === 0); report.inFlightUnmounts++;
  const stats = await page.evaluate(() => { const w = window as any; return { created: w.workerStats.created, terminated: w.workerStats.terminated, live: w.workerStats.live.size, stale: w.probe.renders.filter((r: any) => r.responseKey !== null && r.responseKey !== r.renderedKey).length }; }); Object.assign(report, { workersCreated: stats.created, workersTerminated: stats.terminated, liveWorkers: stats.live, staleReadyRenders: stats.stale });
  assert.equal(report.workersCreated, report.workersTerminated); assert.equal(report.liveWorkers, 0); assert.equal(report.staleReadyRenders, 0); assert.deepEqual(report.errors, []); await context.close(); writeFileSync(`${run}/browser-verification.json`, JSON.stringify(report, null, 2) + '\n'); console.log(JSON.stringify(report));
 } finally { await browser.close(); await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve())); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
