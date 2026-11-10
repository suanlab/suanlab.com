import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFileSync, mkdirSync, writeFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { thermalFieldQuestions, thermalFieldSteps, thermalFieldExperiments } from '../src/data/thermal-field-teaching';
import { thermalFieldExamples } from '../src/lib/learning-thermal-field-input';

const root = '.runtime/learning/thermal-field-visuals', folder = path.resolve(root, 'preview');
mkdirSync(folder, { recursive: true });
async function main() {
  for (const [name, entry] of Object.entries({ calculate: 'src/lib/learning-thermal-field.worker.ts', validate: 'src/lib/learning-thermal-field-validation.worker.ts' })) {
    await build({ entryPoints: [entry], bundle: true, platform: 'browser', format: 'esm', outfile: `${folder}/${name}.js` });
  }
  await build({ stdin: { contents: `
import React,{useState,useEffect} from 'react';import{createRoot}from'react-dom/client';
import Lab from './src/components/learning/thermal-field-lab';
function Preview(){const[language,setLanguage]=useState(new URLSearchParams(location.search).get('lang')==='ko'?'ko':'en');
useEffect(()=>{document.documentElement.lang=language},[language]);
return <main className="ai-book" style={{maxWidth:1200,margin:'auto',padding:16}}><header><h1>Thermal field preview</h1><button onClick={()=>setLanguage(language==='en'?'ko':'en')}>Language</button><button onClick={()=>document.documentElement.classList.toggle('dark')}>Theme</button></header><Lab language={language}/></main>}
createRoot(document.getElementById('root')).render(<Preview/>);`, loader: 'tsx', resolveDir: process.cwd() }, bundle: true, platform: 'browser', format: 'esm', splitting: true, outdir: folder,
    plugins: [{ name: 'local-worker-urls', setup(builder) { builder.onLoad({ filter: /learning-thermal-field-worker-factory\.ts$/ }, () => ({ contents: "export const createThermalFieldWorker=()=>new Worker('/calculate.js',{type:'module'});export const createThermalFieldValidationWorker=()=>new Worker('/validate.js',{type:'module'});", loader: 'ts' })); } }] });
  const styles = readdirSync('out/_next/static/css').filter(s => s.endsWith('.css')).map(s => `<link rel="stylesheet" href="/_next/static/css/${s}">`).join('');
  writeFileSync(`${folder}/index.html`, `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Thermal field QA</title>${styles}<link rel="stylesheet" href="/stdin.css"><div id="root"></div><script type="module" src="/stdin.js"></script></html>`);
  const server = createServer((request, response) => {
    try {
      const url = new URL(request.url!, 'http://localhost'), relative = decodeURIComponent(url.pathname).replace(/^\//, '') || 'index.html';
      if (relative.includes('..')) throw Error('Invalid path');
      const file = relative.startsWith('_next/') ? path.resolve('out', relative) : path.join(folder, relative);
      response.setHeader('Content-Type', file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : file.endsWith('.woff2') ? 'font/woff2' : 'text/html');
      response.end(readFileSync(file));
    } catch { response.statusCode = 404; response.end('Not found'); }
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address(); assert(address && typeof address !== 'string');
  const base = `http://127.0.0.1:${address.port}`;
  const browser = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const errors: string[] = [], states = [];
  try {
    for (const language of ['en', 'ko'] as const) for (const dark of [false, true]) for (const width of [320, 1440]) {
      const context = await browser.newContext({ viewport: { width, height: 900 } });
      await context.addInitScript(() => {
        (globalThis as unknown as { __name: unknown }).__name = (fn: unknown) => fn;
        const counters = { created: 0, active: 0, terminated: 0, double: 0 }, Original = Worker;
        (window as unknown as { thermalPorts: typeof counters }).thermalPorts = counters;
        window.Worker = class extends Original {
          private stopped = false;
          constructor(url: string | URL, options?: WorkerOptions) { super(url, options); counters.created++; counters.active++; }
          terminate() { if (this.stopped) { counters.double++; return; } this.stopped = true; counters.active--; counters.terminated++; super.terminate(); }
        };
      });
      const page = await context.newPage(); page.on('pageerror', e => errors.push(e.message));
      await page.goto(`${base}/?lang=${language}`); if (dark) await page.locator('header button').filter({ hasText: 'Theme' }).click();
      const lab = page.locator('[data-thermal-status]');
      const ready = async () => { await page.waitForFunction(() => document.querySelector('[data-thermal-status]')?.getAttribute('data-thermal-status') === 'ready'); };
      await ready();
      const t = (ko: string, en: string) => language === 'ko' ? ko : en;
      for (let i = 0; i < thermalFieldSteps.length; i++) {
        await lab.getByRole('button', { name: `${i + 1}. ${thermalFieldSteps[i].title[language]}`, exact: true }).click();
        await page.locator(`[data-thermal-concept="${i}"]`).waitFor();
        assert.equal(await lab.locator('.thermal-concept-svg').count(), 1);
      }
      for (let i = 0; i < thermalFieldExperiments.length; i++) {
        await lab.getByLabel(t('안내 실험', 'Guided experiment'), { exact: true }).selectOption(String(i));
        await lab.getByRole('button', { name: t('해설 보기', 'Show explanation'), exact: true }).click();
        assert.equal((await lab.locator('[data-thermal-guide-explanation]').textContent())?.trim(), thermalFieldExperiments[i].explanation[language]);
      }
      for (const example of thermalFieldExamples) {
        await lab.getByLabel(t('전체 예제', 'All examples'), { exact: true }).selectOption(example.id); await ready();
        const nonnormal = ['massless-divergence', 'zero-pivot-indefinite'].includes(example.id);
        assert.equal(await lab.locator(`[data-thermal-result="${nonnormal ? 'not-normalizable' : 'normalizable'}"]`).count(), 1);
      }
      await lab.getByRole('button', { name: t('첫 예제로 초기화', 'Reset to first example'), exact: true }).click(); await ready();
      const before = await page.evaluate(() => (window as unknown as { thermalPorts: { created: number } }).thermalPorts.created);
      await lab.locator('[data-thermal-cell="0-0"]').focus(); await page.keyboard.press('ArrowRight');
      await lab.locator('[data-thermal-selected]').filter({ hasText: '[0,1]' }).waitFor();
      await lab.getByLabel(t('관찰할 행렬', 'Matrix to inspect'), { exact: true }).selectOption('precision');
      await lab.getByLabel(t('그림의 척도', 'Plot scale'), { exact: true }).selectOption('signed-log');
      await lab.getByRole('button', { name: t('성분 순서대로 보기', 'Play entries'), exact: true }).click();
      await lab.locator('[data-thermal-selected]').filter({ hasText: '[1,0]' }).waitFor();
      await lab.getByRole('button', { name: t('재생 일시정지', 'Pause entries'), exact: true }).click();
      assert.equal(await page.evaluate(() => (window as unknown as { thermalPorts: { created: number } }).thermalPorts.created), before);
      await lab.getByRole('button', { name: t('3D 행렬 보기', 'Show 3D matrix'), exact: true }).click();
      await lab.locator('[data-gaussian-matrix-plot="3d"]').waitFor();
      await lab.locator('canvas').focus(); await page.keyboard.press('ArrowRight'); await page.keyboard.press('Home');
      await page.screenshot({ path: `${root}/${language}-${dark ? 'dark' : 'light'}-${width}.png`, fullPage: true });
      await page.screenshot({ path: `${root}/${language}-${dark ? 'dark' : 'light'}-${width}-3d.png` });
      await lab.locator('.thermal-concept').screenshot({ path: `${root}/${language}-${dark ? 'dark' : 'light'}-${width}-concept.png` });
      await lab.getByRole('button', { name: t('3D 행렬 닫기', 'Close 3D matrix'), exact: true }).click();
      await lab.getByLabel(t('온도 T', 'Temperature T'), { exact: true }).fill('1e-100');
      await lab.getByRole('button', { name: t('입력 적용', 'Apply input'), exact: true }).click();
      assert(await lab.getByRole('alert').count());
      assert.equal(await page.evaluate(() => (window as unknown as { thermalPorts: { created: number } }).thermalPorts.created), before);
      await lab.getByLabel(t('온도 T', 'Temperature T'), { exact: true }).fill('3');
      await lab.getByRole('button', { name: t('입력 적용', 'Apply input'), exact: true }).click(); await ready();
      assert((await lab.locator('[data-thermal-result]').innerText()).includes('T=3'));
      const meanTable = lab.getByRole('table', { name: t('선택한 위치의 정확한 값', 'Exact values at the selected site'), exact: true });
      const oldMean = await meanTable.getByRole('row').filter({ hasText: 'μᵢ' }).textContent();
      const energyTable = lab.getByRole('table', { name: t('정확한 유리수 관측량', 'Exact rational observables'), exact: true });
      const oldU = await energyTable.getByRole('row').filter({ hasText: /^U/ }).textContent();
      await lab.getByLabel(`r: ${t('1차 온도 미분', 'First temperature derivative')}`, { exact: true }).fill('1');
      await lab.getByRole('button', { name: t('입력 적용', 'Apply input'), exact: true }).click(); await ready();
      assert.equal(await meanTable.getByRole('row').filter({ hasText: 'μᵢ' }).textContent(), oldMean);
      assert.notEqual(await energyTable.getByRole('row').filter({ hasText: /^U/ }).textContent(), oldU);
      const questions = lab.locator('fieldset.information-question');
      for (let i = 0; i < thermalFieldQuestions.length; i++) {
        const q = questions.nth(i), correct = thermalFieldQuestions[i].correct;
        await q.getByRole('radio').nth((correct + 1) % 3).check(); await q.locator('[data-thermal-answer="incorrect"]').waitFor();
        await q.getByRole('radio').nth(correct).check(); await q.locator('[data-thermal-answer="correct"]').waitFor();
      }
      if (language === 'en') assert(!/[가-힣]/.test(await lab.innerText()));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'document overflow');
      const axe = await new AxeBuilder({ page }).include('[data-thermal-status]').analyze(); assert.deepEqual(axe.violations.map(x => ({ id: x.id, nodes: x.nodes.map(n => n.target) })), []);
      const ports = await page.evaluate(() => (window as unknown as { thermalPorts: { created: number; active: number; terminated: number; double: number } }).thermalPorts);
      assert.equal(ports.active, 0); assert.equal(ports.double, 0); assert.equal(ports.created, ports.terminated);
      states.push({ language, dark, width, concepts: 6, guides: 6, examples: 14, questions: 8, threeD: true, playback: true, customJetPreservesMean: true, axeViolations: axe.violations.length, ports });
      await context.close();
    }
    const reduced = await browser.newContext({ reducedMotion: 'reduce' });
    await reduced.addInitScript(() => {
      const original = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function(this: HTMLCanvasElement, ...args: unknown[]) {
        return args[0] === 'webgl' || args[0] === 'webgl2' ? null : Reflect.apply(original, this, args);
      } as typeof original;
    });
    const rp = await reduced.newPage(); rp.on('pageerror', e => errors.push(e.message)); await rp.goto(base);
    await rp.waitForFunction(() => document.querySelector('[data-thermal-status]')?.getAttribute('data-thermal-status') === 'ready');
    assert(await rp.getByRole('button', { name: 'Play entries', exact: true }).isDisabled());
    assert.equal(await rp.locator('.information-step').evaluate(el => getComputedStyle(el).animationName), 'none');
    await rp.getByRole('button', { name: 'Show 3D matrix', exact: true }).click();
    await rp.getByText('3D is unavailable. Continue comparing with the heat map and exact tables.', { exact: true }).waitFor();
    assert.equal(await rp.locator('[data-thermal-cell]').count(), 4); await reduced.close();
    assert.deepEqual(errors, []);
    writeFileSync(`${root}/verification.json`, JSON.stringify({ status: 'isolated-react-preview-verified', states, reducedMotion: true, webglFallback: true, errors, productionNextAssetsVerified: false, readerRegistered: false }, null, 2) + '\n');
    console.log(JSON.stringify({ states: states.length, errors }));
  } finally { await browser.close(); await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve())); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
