/** Native-browser fixed-comparison sequence probe, before reader integration. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdirSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { finiteSizeKinds as kinds, finiteSizePresetInput as preset } from '../src/lib/learning-finite-size';
import { finiteSizeConceptSteps as steps, finiteSizeConceptInputs as inputs } from '../src/lib/learning-finite-size-concepts';
async function main() {
 const run = '.runtime/learning/finite-size-teaching'; mkdirSync(run, { recursive: true }); const probe = `${run}/series-probe.tsx`;
 writeFileSync(probe, `import React,{useState,StrictMode} from 'react';import {createRoot} from 'react-dom/client';import {flushSync} from 'react-dom';
import {useFiniteSizeSeries} from '@/components/learning/use-finite-size-series';import {finiteSizeInputKey,isFiniteSizeInput,isFiniteSizeSuccessForInput} from '@/lib/learning-finite-size-protocol';
const w=window as any;w.probe={renders:[],status:'pending'};const root=createRoot(document.getElementById('root')!);
function Probe(){const [inputs,setInputs]=useState<any[]>([]),series=useFiniteSizeSeries(inputs);const expected=inputs.map(input=>isFiniteSizeInput(input)?finiteSizeInputKey(input):'invalid-input');const actual=series.responses.map(r=>r.inputKey);
w.probe.renders.push({expected,actual,status:series.status});Object.assign(w.probe,{status:series.status,responses:series.responses,valid:series.responses.every(r=>isFiniteSizeSuccessForInput(r,r.input,r.requestId)),change:(v:any[])=>flushSync(()=>setInputs(v)),retry:series.retry,unmount:()=>flushSync(()=>root.unmount())});return <output>{series.status}</output>;}root.render(<StrictMode><Probe/></StrictMode>);`);
 const [mainBundle, workerBundle] = await Promise.all([
  build({ entryPoints: [probe], bundle: true, write: false, format: 'esm', platform: 'browser', target: 'es2022', define: { 'process.env.NODE_ENV': '"development"' }, logLevel: 'silent' }),
  build({ entryPoints: ['src/lib/learning-finite-size.worker.ts'], bundle: true, write: false, format: 'esm', platform: 'browser', target: 'es2022', logLevel: 'silent' }),
 ]);
 const server = createServer((req, res) => { const path = new URL(req.url ?? '/', 'http://localhost').pathname; if (path === '/probe.js' || path === '/learning-finite-size.worker.ts') { res.setHeader('Content-Type', 'text/javascript'); res.end(path === '/probe.js' ? mainBundle.outputFiles[0].text : workerBundle.outputFiles[0].text); } else { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><html><body><div id="root"></div><script type="module" src="/probe.js"></script></body></html>'); } });
 await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve)); const address = server.address(); assert.ok(address && typeof address === 'object'); const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
 const report = { scope: 'Isolated native browser/React StrictMode sequence hook, not actual Next reader or teaching views', conceptSteps: 0, successfulFixedCalculations: 0, replacementCases: 0, emptyCases: 0, invalidCases: 0, unavailableAndRetryCases: 0, inFlightUnmounts: 0, workersCreated: 0, workersTerminated: 0, liveWorkers: -1, staleReadyRenders: -1, partialPendingRenders: -1, errors: [] as string[] };
 try {
  const context = await browser.newContext(); await context.addInitScript(() => {
   const w = window as any, Native = Worker, stats = w.workerStats = { created: 0, terminated: 0, live: new Set<Worker>(), replacements: 0, unmounts: 0 };
   w.Worker = class extends Native { ended = false; constructor(...args: ConstructorParameters<typeof Worker>) { if (w.failWorker) throw Error('probe unavailable'); super(...args); stats.created++; stats.live.add(this); this.addEventListener('message', event => { if (event.data?.phase !== 'started') return; if (w.replaceOnStart) { const next = w.replaceOnStart; w.replaceOnStart = null; w.probe.change(next); stats.replacements++; } else if (w.unmountOnStart) { w.unmountOnStart = false; w.probe.unmount(); stats.unmounts++; } }); } terminate() { if (!this.ended) { this.ended = true; stats.terminated++; stats.live.delete(this); } super.terminate(); } };
  });
  const page = await context.newPage(); page.on('pageerror', error => report.errors.push(error.message)); await page.goto(`http://127.0.0.1:${address.port}/`);
  const ready = async (count: number) => { await page.waitForFunction(count => (window as any).probe.status === 'ready' && (window as any).probe.responses.length === count, count); assert.equal(await page.evaluate(() => (window as any).probe.valid), true); };
  await ready(0); assert.equal(await page.evaluate(() => (window as any).workerStats.created), 0); report.emptyCases++;
  for (const kind of kinds) for (const index of steps[kind].keys()) { const samples = inputs(kind, index); await page.evaluate(samples => (window as any).probe.change(samples), samples); await ready(samples.length); report.conceptSteps++; report.successfulFixedCalculations += samples.length; }
  const next = [preset(kinds[0], 0), preset(kinds[0], 1)]; await page.evaluate(({ heavy, next }) => { (window as any).replaceOnStart = next; (window as any).probe.change([heavy]); }, { heavy: preset(kinds[1], 14), next }); await page.waitForFunction(() => (window as any).workerStats.replacements === 1 && (window as any).probe.status === 'ready' && (window as any).probe.responses.length === 2); report.replacementCases++;
  const beforeInvalid = await page.evaluate(() => (window as any).workerStats.created); await page.evaluate(() => (window as any).probe.change([null])); await page.waitForFunction(() => (window as any).probe.status === 'error' && (window as any).probe.responses.length === 0); assert.equal(await page.evaluate(() => (window as any).workerStats.created), beforeInvalid); report.invalidCases++;
  await page.evaluate(input => { (window as any).failWorker = true; (window as any).probe.change([input]); }, preset(kinds[2], 0)); await page.waitForFunction(() => (window as any).probe.status === 'error'); await page.evaluate(() => { (window as any).failWorker = false; (window as any).probe.retry(); }); await ready(1); report.unavailableAndRetryCases++;
  await page.evaluate(() => (window as any).probe.change([])); await ready(0); report.emptyCases++;
  await page.evaluate(input => { (window as any).unmountOnStart = true; (window as any).probe.change([input]); }, preset(kinds[1], 14)); await page.waitForFunction(() => (window as any).workerStats.unmounts === 1 && (window as any).workerStats.live.size === 0); report.inFlightUnmounts++;
  const stats = await page.evaluate(() => { const w = window as any; return { created: w.workerStats.created, terminated: w.workerStats.terminated, live: w.workerStats.live.size, stale: w.probe.renders.filter((r: any) => r.status === 'ready' && JSON.stringify(r.expected) !== JSON.stringify(r.actual)).length, partial: w.probe.renders.filter((r: any) => r.status !== 'ready' && r.actual.length > 0).length }; }); Object.assign(report, { workersCreated: stats.created, workersTerminated: stats.terminated, liveWorkers: stats.live, staleReadyRenders: stats.stale, partialPendingRenders: stats.partial });
  assert.equal(report.conceptSteps, 18); assert.equal(report.workersCreated, report.workersTerminated); assert.equal(report.liveWorkers, 0); assert.equal(report.staleReadyRenders, 0); assert.equal(report.partialPendingRenders, 0); assert.deepEqual(report.errors, []); await context.close(); writeFileSync(`${run}/series-browser-verification.json`, JSON.stringify(report, null, 2) + '\n'); console.log(JSON.stringify(report));
 } finally { await browser.close(); await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve())); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
