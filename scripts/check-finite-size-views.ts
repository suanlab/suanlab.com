/** Standalone native-worker teaching views; preserves the registered export. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { finiteSizeKinds as kinds, finiteSizePresetInput as preset, finiteSizePresets as presets } from '../src/lib/learning-finite-size';
import { finiteSizeConceptSteps as steps, finiteSizeConceptInputs as samples } from '../src/lib/learning-finite-size-concepts';
import { finiteSizeInputKey as key } from '../src/lib/learning-finite-size-protocol';
const destination = '.runtime/learning/finite-size-views';
const entry = `import React,{useState} from 'react';import{createRoot}from'react-dom/client';
import Concept from './src/components/learning/finite-size-concept';import{FiniteSizeResultView}from'./src/components/learning/finite-size-view';
import{useFiniteSizeCalculation}from'./src/components/learning/use-finite-size-calculation';
import{finiteSizeKinds as kinds,finiteSizePresets as presets,finiteSizePresetInput as preset,finiteSizeTitles as titles}from'./src/lib/learning-finite-size';
function Result({kind,index,language}){const input=React.useMemo(()=>preset(kind,index),[kind,index]),s=useFiniteSizeCalculation(input);return <section id="experiment" aria-label={language==='ko'?'결과 미리보기':'Result preview'} data-result-state={s.status}>{s.status==='error'?<button onClick={s.retry}>{language==='ko'?'결과 다시 시도':'Retry result'}</button>:s.response?<FiniteSizeResultView response={s.response} language={language}/>:<p role="status">{language==='ko'?'계산 중':'Calculating'}</p>}</section>;}
function App(){const query=new URLSearchParams(location.search),kind=kinds[Number(query.get('kind'))||0],language=query.get('lang')==='en'?'en':'ko',[index,setIndex]=useState(0),[mounted,setMounted]=useState(true);window.viewProbe={setMounted};return <main className="ai-book"><div className="container ai-book-wrap"><h1>{titles[kind][language]}</h1>{mounted&&<><Concept kind={kind} language={language}/><label>{language==='ko'?'검증용 예제':'Verification preparation'}<select className="max-w-full border border-border bg-background text-foreground" aria-label={language==='ko'?'검증용 예제':'Verification preparation'} value={index} onChange={e=>setIndex(Number(e.target.value))}>{presets[kind].map((p,i)=><option key={p.id} value={i}>{p.label[language]}</option>)}</select></label><Result kind={kind} index={index} language={language}/></>}</div></main>;}createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);`;
async function main() {
 mkdirSync(destination, { recursive: true });
 const [app, worker] = await Promise.all([
  build({ stdin: { contents: entry, loader: 'jsx', resolveDir: resolve('.'), sourcefile: 'finite-size-view-preview.jsx' }, bundle: true, write: false, platform: 'browser', format: 'esm', jsx: 'automatic', metafile: true, logLevel: 'silent' }),
  build({ entryPoints: ['src/lib/learning-finite-size.worker.ts'], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true, logLevel: 'silent' }),
 ]);
 for (const file of Object.keys(app.metafile!.inputs)) assert.doesNotMatch(file, /learning-(boundary-extensivity|spatial-self-averaging|two-phase-rounding|finite-markov-exact|binary-products|rational-gcd|interval-transcendentals|finite-size-evaluator)\./);
 writeFileSync(`${destination}/main-dependencies.json`, JSON.stringify(app.metafile, null, 2));
 const styles = readdirSync('out/_next/static/css').filter(s => s.endsWith('.css')).map(s => readFileSync(`out/_next/static/css/${s}`, 'utf8')).join('\n'); let failWorker = false;
 const server = createServer((request, response) => { const path = new URL(request.url!, 'http://localhost').pathname;
  if (path.endsWith('learning-finite-size.worker.ts')) { if (failWorker) { response.writeHead(404).end(); return; } response.setHeader('Content-Type', 'application/javascript'); response.end(worker.outputFiles[0].text); }
  else if (path === '/main.js') { response.setHeader('Content-Type', 'application/javascript'); response.end(app.outputFiles[0].text); }
  else { response.setHeader('Content-Type', 'text/html'); response.end(`<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Finite size preview</title><style>${styles}</style></head><body><div id="root"></div><script type="module" src="/main.js"></script></body></html>`); }
 });
 await new Promise<void>(done => server.listen(0, '127.0.0.1', done)); const address = server.address(); assert.ok(address && typeof address !== 'string'); const base = `http://127.0.0.1:${address.port}`;
 const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
 const report = { scope: 'Standalone native-worker concept/result views; actual Next chunks/readers pending', displayCases: 0, conceptSteps: 0, predictionChoices: 0, fixedPreparationChoices: 0, resultPreparations: 0, axeStates: 0, tablePages: 0, certificates: 0, graphChoices: 0, growthRecordChoices: 0, countSelections: 0, rootSelections: 0, logChoices: 0, failureRecoveryCases: 0, unmountRemountCases: 0, rapidStepReplacementCases: 0, workersCreated: 0, workersTerminated: 0, staleFrames: [] as string[], errors: [] as string[] };
 try {
  for (const language of ['ko', 'en'] as const) for (const dark of [false, true]) for (const width of [320, 1440]) for (let family = 0; family < 3; family++) {
   const kind = kinds[family], context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
   const expectedKeys = Object.fromEntries(kinds.map(k => [k, steps[k].map((_, i) => samples(k, i).map(key))]));
   await context.addInitScript(({ language, dark, expectedKeys }) => {
    document.addEventListener('DOMContentLoaded', () => { document.documentElement.lang = language; document.documentElement.classList.toggle('dark', dark); });
    const state = (window as any).viewStats = { created: 0, terminated: 0, live: new Set(), stale: [] as string[] }; const Native = window.Worker;
    window.Worker = class extends Native { done = false; constructor(url: string | URL, options?: WorkerOptions) { super(url, options); state.created++; state.live.add(this); } terminate() { if (!this.done) { this.done = true; state.terminated++; state.live.delete(this); } super.terminate(); } };
    new MutationObserver(() => { const concept = document.querySelector('[data-finite-size-concept]'), selected = concept?.querySelector('[data-finite-size-result-key]'); if (selected) { const kind = concept!.getAttribute('data-finite-size-concept')!, step = Number(concept!.getAttribute('data-finite-size-step')), key = selected.getAttribute('data-finite-size-result-key')!; if (!(expectedKeys as any)[kind][step].includes(key)) state.stale.push(`${kind}:${step}`); } }).observe(document, { subtree: true, childList: true, attributes: true });
   }, { language, dark, expectedKeys });
   const page = await context.newPage(); page.on('pageerror', e => report.errors.push(e.message));
   const ready = async (index: number) => { await page.waitForFunction(key => document.querySelector('#experiment [data-finite-size-result-key]')?.getAttribute('data-finite-size-result-key') === key, key(preset(kind, index)), { timeout: 90000 }); };
   const check = async () => { assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${kind}/${language}/${width} overflow`); if (language === 'en') assert.doesNotMatch(await page.locator('body').innerText(), /[가-힣]/); assert.doesNotMatch(await page.locator('body').innerText(), /NaN|\[object Object\]/); };
   const axe = async () => { const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze(); assert.deepEqual(result.violations.map(v => ({ id: v.id, nodes: v.nodes.map(n => n.target) })), []); report.axeStates++; };
   await page.goto(`${base}/?kind=${family}&lang=${language}`); await ready(0); await page.locator('[data-finite-size-concept-state="ready"]').waitFor({ timeout: 90000 }); await check(); await axe(); report.displayCases++;
   if (dark && width === 320) {
    for (let step = 0; step < 6; step++) {
     await page.locator('.ai-concept-steps button').nth(step).click(); const concept = page.locator(`[data-finite-size-concept][data-finite-size-step="${step}"][data-finite-size-concept-state="ready"]`); await concept.waitFor({ timeout: 90000 });
     const fixed = concept.getByRole('group', { name: language === 'ko' ? '고정 준비 선택' : 'Choose a fixed preparation' }).getByRole('button');
     for (let i = 0; i < samples(kind, step).length; i++) { await fixed.nth(i).click(); assert.equal(await concept.locator('[data-finite-size-result-key]').getAttribute('data-finite-size-result-key'), key(samples(kind, step)[i])); report.fixedPreparationChoices++; }
     for (let answer = 0; answer < 3; answer++) { await concept.locator('input[type=radio]').nth(answer).check(); await concept.getByRole('button', { name: language === 'ko' ? '예측 확인' : 'Check prediction', exact: true }).click(); assert.ok((await concept.locator('.ai-answer strong').innerText()).includes(answer === steps[kind][step].check.answer ? language === 'ko' ? '맞았습니다.' : 'Correct.' : language === 'ko' ? '다시 생각해보세요.' : 'Try again.')); report.predictionChoices++; }
     await check(); report.conceptSteps++;
    }
    for (let index = 0; index < presets[kind].length; index++) {
     await page.getByRole('combobox', { name: language === 'ko' ? '검증용 예제' : 'Verification preparation', exact: true }).selectOption(String(index)); await ready(index); const result = page.locator('#experiment [data-finite-size-result]'); report.resultPreparations++;
     const log = result.getByRole('button', { name: family === 0 ? language === 'ko' ? '에너지 절댓값 로그 표시' : 'Log energy magnitude display' : family === 1 ? language === 'ko' ? '확률 로그 표시' : 'Log probability display' : language === 'ko' ? '압력 보정 로그 표시' : 'Log pressure-correction display', exact: true }); await log.click(); report.logChoices++;
     if (family === 0) { const graphs = result.getByRole('group', { name: language === 'ko' ? '결합 그래프' : 'Interaction graph' }).getByRole('button'); for (let j = 0; j < 3; j++) { await graphs.nth(j).click(); assert.equal(await graphs.nth(j).getAttribute('aria-pressed'), 'true'); report.graphChoices++; } const select = result.getByRole('combobox', { name: language === 'ko' ? '공간 확대 기록 선택' : 'Choose spatial growth record', exact: true }); await select.selectOption('0'); assert.equal(await result.locator('[data-finite-size-record-size]').getAttribute('data-finite-size-record-size'), '1'); report.growthRecordChoices++; }
     if (family === 1) { await result.getByRole('slider', { name: language === 'ko' ? '선택 점유 수' : 'Selected count', exact: true }).press('End'); const p = preset(kind, index).parameters; assert.ok('siteCount' in p); assert.equal(await result.locator('[data-finite-size-event-count]').getAttribute('data-finite-size-event-count'), String(p.siteCount)); report.countSelections++; await result.getByRole('button', { name: language === 'ko' ? 'N배 분산 표시' : 'Show N-scaled variance', exact: true }).click(); }
     if (family === 2) { await result.getByRole('slider', { name: language === 'ko' ? '선택 복소 근' : 'Selected complex root', exact: true }).press('End'); const p = preset(kind, index).parameters; assert.ok('siteCount' in p); const region = result.getByRole('region', { name: language === 'ko' ? '선택 근의 정확 위상과 근사 좌표' : 'Selected root: exact phase and approximate coordinates', exact: true }); assert.equal(await region.locator('tbody tr').first().locator('td').innerText(), String(p.siteCount - 1)); report.rootSelections++; }
     const details = result.locator('details'); for (let j = 0; j < await details.count(); j++) { await details.nth(j).locator('summary').click(); assert.doesNotMatch(await details.nth(j).innerText(), /\[object Object\]/); report.certificates++; }
     for (const table of await result.locator('[data-finite-size-paged]').all()) { const next = table.getByRole('button', { name: language === 'ko' ? '다음 기록' : 'Next records', exact: true }); while (await next.isEnabled()) { await next.click(); report.tablePages++; } }
     await check(); if ([9, 10, 12, 14].includes(index)) await axe();
    }
    await page.locator('#experiment').evaluate(element => element.scrollIntoView({ block: 'start' })); await page.screenshot({ path: `${destination}/${family}-${language}-dark-mobile.png`, fullPage: false });
    await page.locator('.ai-concept-steps button').nth(5).click(); await page.locator('.ai-concept-steps button').nth(1).click(); await page.locator('[data-finite-size-step="1"][data-finite-size-concept-state="ready"]').waitFor({ timeout: 90000 }); report.rapidStepReplacementCases++;
    failWorker = true; await page.locator('.ai-concept-steps button').nth(0).click(); await page.locator('[data-finite-size-concept-state="error"]').waitFor({ timeout: 90000 }); failWorker = false; await page.getByRole('button', { name: language === 'ko' ? '개념 계산 다시 시도' : 'Retry concept calculation', exact: true }).click(); await page.locator('[data-finite-size-concept-state="ready"]').waitFor({ timeout: 90000 }); report.failureRecoveryCases++;
    await page.evaluate(() => (window as any).viewProbe.setMounted(false)); await page.waitForFunction(() => (window as any).viewStats.live.size === 0); await page.evaluate(() => (window as any).viewProbe.setMounted(true)); await ready(presets[kind].length - 1); await page.locator('[data-finite-size-concept-state="ready"]').waitFor({ timeout: 90000 }); report.unmountRemountCases++;
   }
   await page.evaluate(() => (window as any).viewProbe.setMounted(false)); await page.waitForFunction(() => (window as any).viewStats.live.size === 0); const stats = await page.evaluate(() => { const s = (window as any).viewStats; return { created: s.created, terminated: s.terminated, stale: s.stale }; }); assert.equal(stats.created, stats.terminated); report.workersCreated += stats.created; report.workersTerminated += stats.terminated; report.staleFrames.push(...stats.stale); await context.close(); console.log(JSON.stringify({ kind, language, dark, width, workersCreated: stats.created, workersTerminated: stats.terminated }));
  }
  assert.deepEqual(report.errors, []); assert.deepEqual(report.staleFrames, []); assert.equal(report.displayCases, 24); assert.equal(report.conceptSteps, 36); assert.equal(report.resultPreparations, 88); writeFileSync(`${destination}/browser-verification.json`, JSON.stringify(report, null, 2)); console.log(JSON.stringify(report));
 } finally { await browser.close(); await new Promise<void>((done, reject) => server.close(error => error ? reject(error) : done())); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
