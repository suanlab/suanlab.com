/** Isolated React preview with real module Workers and retained site CSS.
 * This stage does not register the unit or verify Next production Worker URLs.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { build } from 'esbuild';
import { chromium, type Page } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { blockOrderExamples } from '../src/lib/learning-block-order-input';
import { blockOrderInputKey } from '../src/lib/learning-block-order-protocol';
import { blockOrderSteps, blockOrderExperiments, blockOrderQuestions } from '../src/data/block-order-teaching';

const root = '.runtime/learning/block-order-visuals', folder = path.resolve(root,'preview');
mkdirSync(folder,{ recursive: true });
async function main() {
  for (const [name,entry] of Object.entries({ calculate: 'src/lib/learning-block-order.worker.ts',validate: 'src/lib/learning-block-order-validation.worker.ts' }))
    await build({ entryPoints: [entry],bundle: true,platform: 'browser',format: 'esm',outfile: `${folder}/${name}.js`,logLevel: 'silent' });
  await build({ stdin: { contents: `
import React,{useState,useEffect}from'react';import{createRoot}from'react-dom/client';import Lab from './src/components/learning/block-order-lab';
function Preview(){const[language,setLanguage]=useState(new URLSearchParams(location.search).get('lang')==='ko'?'ko':'en'),[mounted,setMounted]=useState(true);
useEffect(()=>{document.documentElement.lang=language},[language]);return <main className="ai-book" style={{maxWidth:1200,margin:'auto',padding:16}}><header><h1>Local order parameter preview</h1><button id="language" onClick={()=>setLanguage(language==='en'?'ko':'en')}>Language</button><button id="theme" onClick={()=>document.documentElement.classList.toggle('dark')}>Theme</button><button id="mount" onClick={()=>setMounted(p=>!p)}>Mount or unmount</button></header>{mounted&&<Lab language={language}/>}</main>}
createRoot(document.getElementById('root')).render(<Preview/>);`,loader: 'tsx',resolveDir: process.cwd() },bundle: true,platform: 'browser',format: 'esm',splitting: true,outdir: folder,logLevel: 'silent',
    plugins: [{ name: 'preview-worker-urls',setup(builder) { builder.onLoad({ filter: /learning-block-order-worker-factory\.ts$/ },() => ({ contents: "export const createBlockOrderWorker=()=>new Worker('/calculate.js',{type:'module'});export const createBlockOrderValidationWorker=()=>new Worker('/validate.js',{type:'module'});",loader: 'ts' })); } }] });
  const styles = readdirSync('out/_next/static/css').filter(s => s.endsWith('.css')).map(s => `<link rel="stylesheet" href="/_next/static/css/${s}">`).join('');
  writeFileSync(`${folder}/index.html`,`<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Local order parameter QA</title>${styles}<link rel="stylesheet" href="/stdin.css"><div id="root"></div><script type="module" src="/stdin.js"></script></html>`);
  const server = createServer((request,response) => {
    try {
      const relative = decodeURIComponent(new URL(request.url!,'http://localhost').pathname).replace(/^\//,'') || 'index.html';
      if (relative.includes('..')) throw Error('Invalid path');
      const file = relative.startsWith('_next/') ? path.resolve('out',relative) : path.join(folder,relative);
      response.setHeader('Content-Type',file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : file.endsWith('.woff2') ? 'font/woff2' : 'text/html'); response.end(readFileSync(file));
    } catch { response.statusCode = 404; response.end('Not found'); }
  });
  await new Promise<void>(resolve => server.listen(0,'127.0.0.1',resolve));
  const address = server.address(); assert(address && typeof address !== 'string'); const base = `http://127.0.0.1:${address.port}`;
  writeFileSync(`${root}/preview-server.json`,JSON.stringify({ pid: process.pid,base,ownership: 'this probe only; close in finally' },null,2)+'\n');
  const browser = await chromium.launch({ headless: true,args: ['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
  const errors: string[] = [], rows = [];
  const ready = async (page: Page,key?: string) => page.waitForFunction(key => { const lab = document.querySelector('[data-block-status]'); return lab?.getAttribute('data-block-status') === 'ready' && (!key || lab.getAttribute('data-block-input-key') === key); },key,{ timeout: 60000 });
  const noOverflow = async (page: Page) => assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1),'document overflow');
  try {
    for (const language of ['en','ko'] as const) for (const dark of [false,true]) for (const width of [320,1440]) {
      const context = await browser.newContext({ viewport: { width,height: 900 } });
      await context.addInitScript(() => {
        (globalThis as unknown as { __name: unknown }).__name = (fn: unknown) => fn;
        const ports = { created: 0,active: 0,terminated: 0,double: 0 }, Original = Worker;
        (window as unknown as { blockPorts: typeof ports }).blockPorts = ports;
        window.Worker = class extends Original {
          private stopped = false;
          constructor(url: string | URL,options?: WorkerOptions) { super(url,options); ports.created++; ports.active++; }
          terminate() { if (this.stopped) { ports.double++; return; } this.stopped = true; ports.active--; ports.terminated++; super.terminate(); }
        };
      });
      const page = await context.newPage(); page.on('pageerror',error => errors.push(error.message));
      await page.goto(`${base}/?lang=${language}`); if (dark) await page.locator('#theme').click(); await ready(page);
      const t = (ko: string,en: string) => language === 'ko' ? ko : en, lab = page.locator('[data-block-status]');
      for (let i = 0; i < blockOrderSteps.length; i++) {
        await lab.getByRole('button',{ name: `${i+1}. ${blockOrderSteps[i].title[language]}`,exact: true }).click();
        await lab.locator(`[data-block-concept="${i}"]`).waitFor();
        assert.equal(await lab.locator('.block-order-concept-svg').count(),1);
      }
      for (let i = 0; i < blockOrderExperiments.length; i++) {
        await lab.getByLabel(t('안내 실험','Guided experiment'),{ exact: true }).selectOption(String(i));
        await lab.getByRole('button',{ name: t('해설 보기','Show explanation'),exact: true }).click();
        assert.equal((await lab.locator('[data-block-guide-explanation]').textContent())?.trim(),blockOrderExperiments[i].explanation[language]);
      }
      for (const e of blockOrderExamples) {
        await lab.getByLabel(t('전체 예제','All examples'),{ exact: true }).selectOption(e.id); await ready(page,blockOrderInputKey(e.input));
        assert.equal(await lab.locator('[data-block-result="verified"]').count(),1);
      }
      await lab.getByRole('button',{ name: t('첫 예제로 초기화','Reset to first example'),exact: true }).click(); await ready(page,blockOrderInputKey(blockOrderExamples[0].input));
      const created = await page.evaluate(() => (window as unknown as { blockPorts: { created: number } }).blockPorts.created);
      await lab.getByLabel(t('공간에 표시할 값','Quantity shown in space'),{ exact: true }).selectOption('profile');
      const before = await lab.getByRole('table',{ name: t('그림에 표시한 위치별 정확한 값','Exact site values shown in the drawing'),exact: true }).innerText();
      await lab.getByRole('button',{ name: t('다음 예제 배치','Next example arrangement'),exact: true }).click();
      assert.equal(await lab.getByRole('table',{ name: t('그림에 표시한 위치별 정확한 값','Exact site values shown in the drawing'),exact: true }).innerText(),before);
      await lab.getByLabel(t('공간에 표시할 값','Quantity shown in space'),{ exact: true }).selectOption('configuration');
      await lab.getByRole('button',{ name: t('예제 배치 순서대로 보기','Play example arrangements'),exact: true }).click();
      await page.waitForFunction(() => document.querySelector('[data-block-example]')?.textContent?.includes('mask=9'));
      await lab.getByRole('button',{ name: t('배치 재생 멈춤','Pause arrangements'),exact: true }).click();
      assert.equal(await page.evaluate(() => (window as unknown as { blockPorts: { created: number } }).blockPorts.created),created);
      await lab.getByRole('button',{ name: t('3D 격자 보기','Show 3D lattice'),exact: true }).click(); await lab.locator('[data-block-spatial-plot="3d"]').waitFor();
      await lab.locator('canvas').focus(); await page.keyboard.press('ArrowRight'); await page.keyboard.press('Home');
      await lab.locator('[data-block-spatial-plot]').screenshot({ path: `${root}/${language}-${dark ? 'dark' : 'light'}-${width}-spatial.png` });
      await lab.locator('.block-order-concept').screenshot({ path: `${root}/${language}-${dark ? 'dark' : 'light'}-${width}-concept.png` });
      await lab.getByRole('button',{ name: t('3D 격자 닫기','Close 3D lattice'),exact: true }).click();
      const applied = await lab.locator('[data-block-applied]').innerText();
      await lab.getByLabel(t('불일치 활동도 x','Disagreement activity x'),{ exact: true }).fill('1e-3');
      await lab.getByRole('button',{ name: t('입력 적용','Apply input'),exact: true }).click(); assert(await lab.getByRole('alert').count());
      assert.equal(await lab.locator('[data-block-applied]').innerText(),applied);
      assert.equal(await page.evaluate(() => (window as unknown as { blockPorts: { created: number } }).blockPorts.created),created);
      await lab.getByLabel(t('불일치 활동도 x','Disagreement activity x'),{ exact: true }).fill('1');
      await lab.getByRole('button',{ name: t('입력 적용','Apply input'),exact: true }).click(); await ready(page,blockOrderInputKey(blockOrderExamples.find(e => e.id === 'uncoupled-counting')!.input));
      const selected = lab.getByRole('table',{ name: t('선택 프로필의 정확한 값','Exact values for the selected profile'),exact: true });
      assert((await selected.getByRole('row').filter({ hasText: /^P\(k\)/ }).innerText()).includes('1/4'));
      await lab.getByLabel(t('전체 예제','All examples'),{ exact: true }).selectOption('boundary-versus-mean'); await ready(page,blockOrderInputKey(blockOrderExamples[2].input));
      await lab.getByLabel(t('공간에 표시할 값','Quantity shown in space'),{ exact: true }).selectOption('conditional');
      const sites = lab.getByRole('table',{ name: t('그림에 표시한 위치별 정확한 값','Exact site values shown in the drawing'),exact: true });
      assert.deepEqual(await sites.locator('tbody td').allTextContents(),['1/3','2/3','2/3','1/3']);
      const questions = lab.locator('fieldset.information-question');
      for (let i = 0; i < blockOrderQuestions.length; i++) {
        const q = questions.nth(i), correct = blockOrderQuestions[i].correct;
        await q.getByRole('radio').nth((correct+1)%3).check(); await q.locator('[data-block-answer="incorrect"]').waitFor();
        await q.getByRole('radio').nth(correct).check(); await q.locator('[data-block-answer="correct"]').waitFor();
      }
      await lab.getByLabel(t('전체 예제','All examples'),{ exact: true }).selectOption('two-dimensional'); await ready(page,blockOrderInputKey(blockOrderExamples.find(e => e.id === 'two-dimensional')!.input));
      const summary = lab.locator('summary').filter({ hasText: t('전체 프로필 분포와 미시 상태 수','Full profile distribution and microscopic counts') }); await summary.click();
      assert((await lab.locator('[data-block-profile-page]').innerText()).includes('625'));
      if (language === 'en' && !dark && width === 1440 || language === 'ko' && dark && width === 320) {
        for (let i = 0; i < 24; i++) await lab.getByRole('button',{ name: t('다음 프로필 페이지','Next profile page'),exact: true }).click();
        assert((await lab.locator('[data-block-profile-page]').innerText()).includes('25/25'));
        await lab.locator('[data-block-profile-index="624"]').click();
        await ready(page,blockOrderInputKey({ ...blockOrderExamples.find(e => e.id === 'two-dimensional')!.input,selectedUpCounts: [4,4,4,4] }));
        assert((await lab.locator('[data-block-applied]').innerText()).includes('k=(4, 4, 4, 4)'));
      }
      if (language === 'en' && dark && width === 320) {
        const stress = { ...blockOrderExamples.find(e => e.id === 'two-dimensional')!.input,disagreementActivity: '999983/999979',
          blockUpActivities: ['999961/999959','999953/999931','999917/999907','999883/999863'],selectedUpCounts: [1,2,3,1] };
        await lab.getByLabel('Disagreement activity x',{ exact: true }).fill(stress.disagreementActivity);
        await lab.getByLabel('Blockwise up activities y',{ exact: true }).fill(stress.blockUpActivities.join(', '));
        await lab.getByLabel('Selected profile up-counts k',{ exact: true }).fill('1, 2, 3, 1');
        await lab.getByRole('button',{ name: 'Apply input',exact: true }).click(); await ready(page,blockOrderInputKey(stress));
        await lab.locator('summary').filter({ hasText: 'Exact operands and logarithm enclosures' }).click(); await noOverflow(page);
        await lab.locator('summary').filter({ hasText: 'Full profile distribution and microscopic counts' }).click(); await noOverflow(page);
        assert((await lab.locator('[data-block-profile-page]').innerText()).includes('625'));
      }
      await noOverflow(page);
      assert.equal(await page.evaluate(() => document.querySelector('html')!.lang),language);
      if (language === 'en') assert(!/[가-힣]/.test(await lab.innerText()));
      const axe = await new AxeBuilder({ page }).include('main').analyze();
      writeFileSync(`${root}/axe-${language}-${dark ? 'dark' : 'light'}-${width}.json`,JSON.stringify(axe.violations,null,2)+'\n');
      assert.deepEqual(axe.violations,[]);
      await page.locator('#mount').click(); assert.equal(await page.locator('canvas').count(),0);
      const ports = await page.evaluate(() => (window as unknown as { blockPorts: { created: number;active: number;terminated: number;double: number } }).blockPorts);
      assert.equal(ports.active,0); assert.equal(ports.created,ports.terminated); assert.equal(ports.double,0);
      rows.push({ language,dark,width,conceptSteps: 6,guideExplanations: 6,examples: 9,explainedQuestions: 8,axeViolations: 0,ports });
      await context.close();
    }
    // Storage and reduced-motion checks use a separate context and explicit
    // restoration before the first calculation; their ports are not in rows.
    const context = await browser.newContext({ viewport: { width: 320,height: 900 },reducedMotion: 'reduce' });
    await context.addInitScript(() => { (globalThis as unknown as { __name: unknown }).__name = (fn: unknown) => fn; });
    const page = await context.newPage(); page.on('pageerror',e => errors.push(e.message)); await page.goto(base); await ready(page);
    const lab = page.locator('[data-block-status]');
    assert(await lab.getByRole('button',{ name: 'Play example arrangements',exact: true }).isDisabled());
    await lab.getByRole('button',{ name: 'Next example arrangement',exact: true }).click();
    await lab.getByRole('button',{ name: '4. The boundary differs from the average',exact: true }).click();
    assert.equal(await lab.locator('.information-step').evaluate(e => getComputedStyle(e).animationName),'none');
    await lab.getByLabel('Disagreement activity x',{ exact: true }).fill('1/');
    await page.reload(); await ready(page); assert.equal(await lab.getByLabel('Disagreement activity x',{ exact: true }).inputValue(),'1/');
    assert((await lab.locator('[data-block-applied]').innerText()).includes('x=1/2'));
    await lab.locator('[data-block-concept="3"]').waitFor();
    await page.locator('#language').click(); assert(/[가-힣]/.test(await lab.innerText()));
    await page.locator('#language').click(); assert(!/[가-힣]/.test(await lab.innerText()));
    await lab.getByLabel('All examples',{ exact: true }).selectOption('two-dimensional');
    await lab.getByRole('button',{ name: 'Cancel calculation',exact: true }).click();
    await page.waitForFunction(() => document.querySelector('[data-block-status]')?.getAttribute('data-block-status') === 'idle');
    assert.equal(await lab.locator('[data-block-result]').count(),0);
    await lab.getByRole('button',{ name: 'Recalculate',exact: true }).click(); await ready(page,blockOrderInputKey(blockOrderExamples.find(e => e.id === 'two-dimensional')!.input));
    await noOverflow(page);
    await page.locator('#mount').click(); await context.close();
    const fallback = await browser.newContext();
    await fallback.addInitScript(() => {
      (globalThis as unknown as { __name: unknown }).__name = (fn: unknown) => fn;
      const original = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function(this: HTMLCanvasElement,type: string,...args: unknown[]) { return type.includes('webgl') ? null : original.apply(this,[type,...args] as never); } as typeof original;
      Object.defineProperty(window,'localStorage',{ get() { throw Error('storage denied'); } });
    });
    const fallbackPage = await fallback.newPage(); fallbackPage.on('pageerror',e => errors.push(e.message)); await fallbackPage.goto(base); await ready(fallbackPage);
    const fallbackLab = fallbackPage.locator('[data-block-status]');
    await fallbackLab.getByRole('button',{ name: 'Show 3D lattice',exact: true }).click();
    await fallbackLab.getByText('3D is unavailable. Continue comparing with the2D map and exact tables.',{ exact: true }).waitFor();
    assert.equal(await fallbackLab.locator('[data-block-storage="unavailable"]').count(),1);
    assert.equal(await fallbackLab.locator('.block-order-grid-svg').count(),1); await fallbackPage.locator('#mount').click(); await fallback.close();
    const corrupt = await browser.newContext();
    await corrupt.addInitScript(() => { (globalThis as unknown as { __name: unknown }).__name = (fn: unknown) => fn; localStorage.setItem('suanlab-block-order-v1','{'); });
    const corruptPage = await corrupt.newPage(); corruptPage.on('pageerror',e => errors.push(e.message)); await corruptPage.goto(base); await ready(corruptPage);
    assert.equal(await corruptPage.locator('[data-block-storage="corrupt"]').count(),1); await corruptPage.locator('#mount').click(); await corrupt.close();
    assert.deepEqual(errors,[]);
    writeFileSync(`${root}/verification.json`,JSON.stringify({ status: 'isolated-react-visuals-verified',registered: false,actualReaderVerified: false,productionNextAssets: false,
      states: rows,conceptInteractions: 48,guideExplanations: 48,quizCorrectAndIncorrectPairs: 64,reducedMotion: true,webglFallback: true,deniedStorage: true,
      unfinishedDraftRestored: true,localeSwitchPreservesState: true,cancelAndRetry: true,corruptStorage: true,maximumRationalResponseMobileOverflowChecked: true,pageErrors: errors },null,2)+'\n');
  } finally { await browser.close(); await new Promise<void>(resolve => server.close(() => resolve())); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
