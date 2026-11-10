/** Standalone native-worker teaching views. No reader registration or export mutation. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { detailedBalanceKinds as kinds, detailedBalancePresetInput as preset, detailedBalancePresets as presets } from '../src/lib/learning-detailed-balance';
import { detailedBalanceConceptSteps as steps, detailedBalanceConceptInputs as samples } from '../src/lib/learning-detailed-balance-concepts';
import { detailedBalanceInputKey as key } from '../src/lib/learning-detailed-balance-protocol';
const destination = '.runtime/learning/detailed-balance-views';
const entry = `import React,{useState} from 'react';import{createRoot}from'react-dom/client';
import Concept from './src/components/learning/detailed-balance-concept';
import {DetailedBalanceResultView}from'./src/components/learning/detailed-balance-view';
import {useDetailedBalanceCalculation}from'./src/components/learning/use-detailed-balance-calculation';
import {detailedBalanceKinds as kinds,detailedBalancePresets as presets,detailedBalancePresetInput as preset,detailedBalanceTitles as titles}from'./src/lib/learning-detailed-balance';
function Result({kind,index,language}){const input=React.useMemo(()=>preset(kind,index),[kind,index]),s=useDetailedBalanceCalculation(input);return <section id="experiment" aria-label={language==='ko'?'결과 미리보기':'Result preview'} data-result-state={s.status}>{s.status==='error'?<button onClick={s.retry}>Retry</button>:s.response?<DetailedBalanceResultView response={s.response} language={language}/>:<p role="status">{language==='ko'?'계산 중':'Calculating'}</p>}</section>;}
function App(){const query=new URLSearchParams(location.search),kind=kinds[Number(query.get('kind'))||0],language=query.get('lang')==='en'?'en':'ko',[index,setIndex]=useState(0),[mounted,setMounted]=useState(true);window.viewProbe={setMounted};return <main className="ai-book"><div className="container ai-book-wrap"><h1>{titles[kind][language]}</h1>{mounted&&<><Concept kind={kind} language={language}/><label>{language==='ko'?'검증용 예제':'Verification preparation'}<select className="bg-background text-foreground border border-border max-w-full" aria-label={language==='ko'?'검증용 예제':'Verification preparation'} value={index} onChange={e=>setIndex(Number(e.target.value))}>{presets[kind].map((p,i)=><option key={p.id} value={i}>{p.label[language]}</option>)}</select></label><Result kind={kind} index={index} language={language}/></>}</div></main>;}createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);`;
async function main() {
 mkdirSync(destination, { recursive: true });
 const [app, worker] = await Promise.all([
  build({ stdin: { contents: entry, loader: 'jsx', resolveDir: resolve('.'), sourcefile: 'detailed-balance-view-preview.jsx' }, bundle: true, write: false, platform: 'browser', format: 'esm', jsx: 'automatic', metafile: true, logLevel: 'silent' }),
  build({ entryPoints: ['src/lib/learning-detailed-balance.worker.ts'], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true, logLevel: 'silent' }),
 ]);
 for (const file of Object.keys(app.metafile!.inputs)) assert.doesNotMatch(file, /learning-(stationary-cycle|hastings-proposal|two-state-mixing|finite-markov-exact|binary-products|rational-gcd|detailed-balance-evaluator)\./);
 writeFileSync(`${destination}/main-dependencies.json`, JSON.stringify(app.metafile, null, 2));
 const css = readdirSync('out/_next/static/css').filter(s => s.endsWith('.css')), styles = css.map(s => readFileSync(`out/_next/static/css/${s}`, 'utf8')).join('\n');
 let failWorker = false;
 const server = createServer((request, response) => {
  const path = new URL(request.url!, 'http://localhost').pathname;
  if (path.endsWith('learning-detailed-balance.worker.ts')) { if (failWorker) { response.writeHead(404).end(); return; } response.setHeader('Content-Type', 'application/javascript'); response.end(worker.outputFiles[0].text); }
  else if (path === '/main.js') { response.setHeader('Content-Type', 'application/javascript'); response.end(app.outputFiles[0].text); }
  else { response.setHeader('Content-Type', 'text/html'); response.end(`<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Detailed balance preview</title><style>${styles}</style></head><body><div id="root"></div><script type="module" src="/main.js"></script></body></html>`); }
 });
 await new Promise<void>(done => server.listen(0, '127.0.0.1', done));
 const address = server.address(); assert.ok(address && typeof address !== 'string'); const base = `http://127.0.0.1:${address.port}`;
 const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
 const report = { displayCases: 0, conceptSteps: 0, predictionChoices: 0, fixedPreparationChoices: 0, resultPreparations: 0, axeStates: 0, tablePages: 0, exactInspections: 0, matrixChoices: 0, logs: 0, failureRecovery: false, unmountRemount: false, rapidStepReplacement: false, workerCreated: 0, workerTerminated: 0, workerStarted: 0, staleFrames: [] as string[], errors: [] as string[] };
 try {
  for (const language of ['ko', 'en'] as const) for (const dark of [false, true]) for (const width of [320, 1440]) for (let family = 0; family < 3; family++) {
   const kind = kinds[family], context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
   const expectedKeys = Object.fromEntries(kinds.map(k => [k, steps[k].map((_, i) => samples(k, i).map(key))]));
   await context.addInitScript(({ language, dark, expectedKeys }) => {
    document.addEventListener('DOMContentLoaded', () => { document.documentElement.lang = language; document.documentElement.classList.toggle('dark', dark); });
    const state = (window as any).viewStats = { created: 0, terminated: 0, started: 0, live: new Set(), stale: [] as string[] };
    const Native = window.Worker;
    window.Worker = class extends Native { done = false; constructor(url: string | URL, options?: WorkerOptions) { super(url, options); state.created++; state.live.add(this); this.addEventListener('message', e => { if (e.data?.phase === 'started' || e.data?.type === 'started') state.started++; }); } terminate() { if (!this.done) { this.done = true; state.terminated++; state.live.delete(this); } super.terminate(); } };
    new MutationObserver(() => {
     const concept = document.querySelector('[data-detailed-balance-concept]'), selected = concept?.querySelector('[data-detailed-balance-result-key]');
     if (selected) { const kind = concept!.getAttribute('data-detailed-balance-concept')!, step = Number(concept!.getAttribute('data-detailed-balance-step')), key = selected.getAttribute('data-detailed-balance-result-key')!; if (!(expectedKeys as any)[kind][step].includes(key)) state.stale.push(`${kind}:${step}`); }
    }).observe(document, { subtree: true, childList: true, attributes: true });
   }, { language, dark, expectedKeys });
   const page = await context.newPage(); page.on('pageerror', e => report.errors.push(e.message));
   const ready = async (index: number) => { await page.waitForFunction(key => document.querySelector('#experiment [data-detailed-balance-result-key]')?.getAttribute('data-detailed-balance-result-key') === key, key(preset(kind, index)), { timeout: 90000 }); };
   const check = async () => {
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${kind}/${language}/${width} overflow`);
    if (language === 'en') assert.doesNotMatch(await page.locator('body').innerText(), /[가-힣]/);
   };
   const axe = async () => { const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze(); assert.deepEqual(result.violations.map(v => ({ id: v.id, nodes: v.nodes.map(n => n.target) })), []); report.axeStates++; };
   await page.goto(`${base}/?kind=${family}&lang=${language}`); await ready(0); await page.locator('[data-detailed-balance-concept-state="ready"]').waitFor({ timeout: 90000 }); await check(); await axe(); report.displayCases++;
   if (dark && width === 320) {
    for (let step = 0; step < 6; step++) {
     await page.locator('.ai-concept-steps button').nth(step).click(); const concept = page.locator(`[data-detailed-balance-concept][data-detailed-balance-step="${step}"][data-detailed-balance-concept-state="ready"]`); await concept.waitFor({ timeout: 90000 });
     const fixed = concept.getByRole('group', { name: language === 'ko' ? '고정 준비 선택' : 'Choose a fixed preparation' }).getByRole('button');
     for (let i = 0; i < samples(kind, step).length; i++) { await fixed.nth(i).click(); assert.equal(await concept.locator('[data-detailed-balance-result-key]').getAttribute('data-detailed-balance-result-key'), key(samples(kind, step)[i])); report.fixedPreparationChoices++; }
     for (let answer = 0; answer < 3; answer++) { await concept.locator('input[type=radio]').nth(answer).check(); await concept.getByRole('button', { name: language === 'ko' ? '예측 확인' : 'Check prediction', exact: true }).click(); assert.ok((await concept.locator('.ai-answer strong').innerText()).includes(answer === steps[kind][step].check.answer ? language === 'ko' ? '맞았습니다.' : 'Correct.' : language === 'ko' ? '다시 생각해보세요.' : 'Try again.')); report.predictionChoices++; }
     await check(); report.conceptSteps++;
    }
    for (let index = 0; index < presets[kind].length; index++) {
     await page.getByRole('combobox').selectOption(String(index)); await ready(index); const result = page.locator('#experiment [data-detailed-balance-result]'); report.resultPreparations++;
     const log = result.getByRole('button', { name: language === 'ko' ? '거리 로그 표시' : 'Log distance display', exact: true });
     if (await log.count()) { await log.click(); report.logs++; }
     if (family === 0) await result.getByRole('button', { name: language === 'ko' ? '정상 참조 흐름 보기' : 'Show stationary-reference flow', exact: true }).click();
     if (family === 1) { for (const corrected of [true, false]) { await result.getByRole('button', { name: corrected ? language === 'ko' ? 'Hastings 보정' : 'Hastings corrected' : language === 'ko' ? '목표 비율만·비교 커널' : 'Target-ratio-only comparison', exact: true }).click(); const matrices = result.getByRole('group', { name: language === 'ko' ? '행렬 선택' : 'Choose a matrix' }).getByRole('button'); for (let j = 0; j < 10; j++) { await matrices.nth(j).click(); report.matrixChoices++; } } }
     const details = result.locator('details'); for (let j = 0; j < await details.count(); j++) { await details.nth(j).locator('summary').click(); assert.doesNotMatch(await details.nth(j).innerText(), /\[object Object\]/); report.exactInspections++; }
     for (const table of await result.locator('[data-detailed-balance-paged]').all()) { const next = table.getByRole('button', { name: language === 'ko' ? '다음 기록' : 'Next records', exact: true }); while (await next.isEnabled()) { await next.click(); report.tablePages++; } }
     await check();
     if ([5, 9, 11, 12].includes(index)) await axe();
    }
    if (family === 0 && language === 'en') { await page.locator('#experiment').evaluate(element => element.scrollIntoView({block:'start'})); await page.screenshot({ path: `${destination}/cycle-en-dark-mobile.png`, fullPage: false }); }
    if (family === 1 && language === 'ko') { await page.locator('#experiment').evaluate(element => element.scrollIntoView({block:'start'})); await page.screenshot({ path: `${destination}/hastings-ko-dark-mobile.png`, fullPage: false }); }
    if (family === 2 && language === 'en') { await page.locator('#experiment').evaluate(element => element.scrollIntoView({block:'start'})); await page.screenshot({ path: `${destination}/mixing-en-dark-mobile.png`, fullPage: false }); }
    // Replacing a real series during loading must cancel obsolete native work.
    await page.locator('.ai-concept-steps button').nth(5).click(); await page.locator('.ai-concept-steps button').nth(1).click(); await page.locator('[data-detailed-balance-step="1"][data-detailed-balance-concept-state="ready"]').waitFor({ timeout: 90000 }); report.rapidStepReplacement = true;
    failWorker = true; await page.locator('.ai-concept-steps button').nth(0).click(); await page.locator('[data-detailed-balance-concept-state="error"]').waitFor({ timeout: 90000 }); failWorker = false; await page.getByRole('button', { name: language === 'ko' ? '개념 계산 다시 시도' : 'Retry concept calculation' }).click(); await page.locator('[data-detailed-balance-concept-state="ready"]').waitFor({ timeout: 90000 }); report.failureRecovery = true;
    await page.evaluate(() => (window as any).viewProbe.setMounted(false)); await page.waitForFunction(() => (window as any).viewStats.live.size === 0); await page.evaluate(() => (window as any).viewProbe.setMounted(true)); await ready(presets[kind].length - 1); await page.locator('[data-detailed-balance-concept-state="ready"]').waitFor({ timeout: 90000 }); report.unmountRemount = true;
   }
   await page.evaluate(() => (window as any).viewProbe.setMounted(false)); await page.waitForFunction(() => (window as any).viewStats.live.size === 0);
   const stats = await page.evaluate(() => { const s = (window as any).viewStats; return { created: s.created, terminated: s.terminated, started: s.started, stale: s.stale }; });
   assert.equal(stats.created, stats.terminated); report.workerCreated += stats.created; report.workerTerminated += stats.terminated; report.workerStarted += stats.started; report.staleFrames.push(...stats.stale); await context.close();
   console.log(JSON.stringify({ kind, language, dark, width, created: stats.created, terminated: stats.terminated }));
  }
  assert.deepEqual(report.errors, []); assert.deepEqual(report.staleFrames, []); assert.equal(report.displayCases, 24); assert.equal(report.conceptSteps, 36); assert.equal(report.resultPreparations, 72);
  writeFileSync(`${destination}/browser-verification.json`, JSON.stringify(report, null, 2)); console.log(JSON.stringify(report));
 } finally { await browser.close(); await new Promise<void>((done, reject) => server.close(error => error ? reject(error) : done())); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
