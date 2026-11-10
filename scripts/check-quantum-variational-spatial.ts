/** Native WebGL and lifecycle QA of unregistered Ritz teaching components. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { quantumVariationalPresets as presets, quantumVariationalKinds as kinds } from '../src/lib/learning-quantum-variational';
const root = process.env.SUANLAB_QV_SPATIAL_OUTPUT ?? '.runtime/learning/quantum-variational-spatial';
const kind = kinds[1];
async function main() {
  const source = `import React,{useState}from'react';import{createRoot}from'react-dom/client';import{LanguageProvider}from'./src/components/language-provider';import{SiteMotionProvider}from'./src/components/motion/motion-provider';import Lab from'./src/components/learning/quantum-variational-lab';import Cover from'./src/components/learning/quantum-variational-cover';function App(){const[mounted,setMounted]=useState(true);return <div className="ai-book"><header style={{padding:16}}><h1>Ritz space verification</h1><button onClick={()=>setMounted(v=>!v)}>Mount / unmount</button><button onClick={()=>document.documentElement.classList.toggle('dark')}>Theme</button></header><main id="main-content" style={{maxWidth:1100,margin:'auto',padding:16}}><div style={{height:160}}><Cover kind="nonorthogonal-ritz-subspace-certificates"/></div>{mounted&&<Lab kind="nonorthogonal-ritz-subspace-certificates"/>}</main></div>}createRoot(document.getElementById('root')).render(<LanguageProvider><SiteMotionProvider><App/></SiteMotionProvider></LanguageProvider>);`;
  const client = await build({ stdin: { contents: source, resolveDir: process.cwd(), loader: 'tsx' }, bundle: true, write: false, metafile: true, jsx: 'automatic', format: 'esm', platform: 'browser', target: 'es2022', define: { 'process.env.NODE_ENV': '"production"' } });
  const worker = await build({ entryPoints: ['src/lib/learning-quantum-variational.worker.ts'], bundle: true, write: false, format: 'esm', platform: 'browser', target: 'es2022' });
  const graph = Object.keys(client.metafile!.inputs);
  assert(!graph.some(p => /learning-(rayleigh|ritz|tensor-roots|quantum-variational-evaluator)\.ts$/.test(p)));
  writeFileSync(`${root}/browser-dependency-graph.json`, JSON.stringify(graph, null, 2));
  writeFileSync(`${root}/spatial-client.js`, client.outputFiles[0].text);
  const exported = readFileSync('out/book/online/physics-math-for-ai/index.html', 'utf8');
  const css = [...new Set([...exported.matchAll(/href="([^"]+\.css)"/g)].map(m => m[1]))].map(href => `<link rel="stylesheet" href="${href}">`).join('');
  assert(css);
  const html = `<!doctype html><html lang="en"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ritz space QA</title>${css}</head><body><div id="root"></div><script type="module" src="/__qv-spatial.js"></script></body></html>`;
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox', '--enable-unsafe-swiftshader'] });
  const report = { cases: 0, real3D: 0, views: 0, drags: 0, touchDrags: 0, residualChanges: 0, flatProjections: 0, actualContextLosses: 0, contextRecoveries: 0, initializationFallbacks: 0, initializationRecoveries: 0, subspaceDimensions: 0, complexFallbacks: 0, themeChanges: 0, idleDrawChecks: 0, workerCreated: 0, workerTerminated: 0, webglCreated: 0, axeCases: 0, errors: [] as string[], readerIntegrated: false };
  try {
    for (const language of ['ko', 'en'] as const) for (const width of [320, 1440]) {
      const context = await browser.newContext({ viewport: { width, height: 960 }, deviceScaleFactor: width === 320 ? 2 : 1, reducedMotion: 'reduce', hasTouch: width === 320 });
      await context.addInitScript(({ language, width }) => {
        localStorage.setItem('suanlab-language', language);
        document.addEventListener('DOMContentLoaded', () => document.documentElement.classList.toggle('dark', width === 1440));
        const stats = { created: 0, terminated: 0, live: new Set<Worker>(), glCreated: 0, draws: 0, observers: new Set<unknown>(), observerStacks: new Map<unknown, string>() };
        (window as any).qvSpatialStats = stats;
        const Original = window.Worker;
        window.Worker = class extends Original {
          constructor(url: string | URL, options?: WorkerOptions) { super(url, options); stats.created++; stats.live.add(this); }
          terminate() { if (stats.live.delete(this)) stats.terminated++; super.terminate(); }
        };
        const getContext = HTMLCanvasElement.prototype.getContext;
        const seen = new WeakSet<object>();
        HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, ...args: any[]) {
          const gl = (getContext as any).apply(this, args);
          if (gl && (args[0] === 'webgl2' || args[0] === 'webgl') && !seen.has(gl)) {
            seen.add(gl); stats.glCreated++;
            for (const method of ['drawArrays', 'drawElements', 'drawArraysInstanced', 'drawElementsInstanced']) {
              const original = gl[method];
              if (original) gl[method] = function (...values: any[]) { stats.draws++; return original.apply(this, values); };
            }
          }
          return gl;
        } as typeof getContext;
        for (const name of ['ResizeObserver', 'IntersectionObserver', 'MutationObserver'] as const) {
          const Base = window[name];
          (window as any)[name] = class extends (Base as any) {
            observe(...args: any[]) { stats.observers.add(this); stats.observerStacks.set(this, `${name}: ${new Error().stack}`); return super.observe(...args); }
            disconnect() { stats.observers.delete(this); return super.disconnect(); }
          };
        }
      }, { language, width });
      const page = await context.newPage(); page.on('pageerror', e => report.errors.push(e.message));
      await page.route('**/__qv-spatial/', r => r.fulfill({ contentType: 'text/html', body: html }));
      await page.route('**/__qv-spatial.js', r => r.fulfill({ contentType: 'text/javascript', body: client.outputFiles[0].text }));
      await page.route('**/learning-quantum-variational.worker.ts', r => r.fulfill({ contentType: 'text/javascript', body: worker.outputFiles[0].text }));
      await page.goto('http://localhost:4175/__qv-spatial/');
      const lab = page.locator('[data-qv-lab]');
      const ready = () => page.locator('[data-qv-lab][data-qv-state="ready"]').waitFor({ timeout: 120000 });
      const examples = lab.getByRole('group', { name: language === 'ko' ? '실험 예제' : 'Experiment examples', exact: true });
      await ready();
      const figure = lab.locator('figure[data-qv-space]');
      await figure.scrollIntoViewIfNeeded(); await figure.locator('canvas').waitFor({ timeout: 30000 });
      assert.equal(await figure.getAttribute('data-qv-space'), '3d'); report.real3D++;
      const key = await lab.getAttribute('data-qv-input');
      const hash = async () => createHash('sha256').update(await figure.locator('canvas').screenshot()).digest('hex');
      const perspective = await hash();
      await figure.getByRole('button', { name: language === 'ko' ? '정면' : 'Front', exact: true }).focus(); await page.keyboard.press('Enter');
      const front = await hash(); assert.notEqual(front, perspective); report.views++;
      await figure.getByRole('button', { name: language === 'ko' ? '위에서' : 'Top', exact: true }).click();
      assert.notEqual(await hash(), front); report.views++;
      await figure.getByRole('button', { name: language === 'ko' ? '시점 초기화' : 'Reset view', exact: true }).click();
      const beforeDrag = await hash(), box = (await figure.locator('canvas').boundingBox())!;
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2); await page.mouse.down();
      await page.mouse.move(box.x + box.width / 2 + 40, box.y + box.height / 2 - 20, { steps: 8 }); await page.mouse.up();
      assert.notEqual(await hash(), beforeDrag); report.drags++;
      if (width === 320) {
        const beforeTouch = await hash(), cdp = await context.newCDPSession(page);
        const start = { x: box.x + box.width * 0.75, y: box.y + box.height / 2 };
        await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [start] });
        for (let i = 1; i <= 6; i++) { await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: start.x - i * 10, y: start.y }] }); await page.waitForTimeout(20); }
        await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] }); await cdp.detach();
        assert.notEqual(await hash(), beforeTouch); report.touchDrags++;
      }
      const residuals = lab.getByRole('group', { name: language === 'ko' ? '그릴 잔차 선택' : 'Choose the residual to draw', exact: true });
      for (const index of [1, 2, 0]) { await residuals.getByRole('button').nth(index).click(); assert.equal(await figure.getAttribute('data-qv-space'), '3d'); report.residualChanges++; }
      assert.equal(await lab.getAttribute('data-qv-input'), key);
      await page.waitForTimeout(100);
      const draws = await page.evaluate(() => (window as any).qvSpatialStats.draws);
      await page.waitForTimeout(350);
      assert.equal(await page.evaluate(() => (window as any).qvSpatialStats.draws), draws); report.idleDrawChecks++;
      await figure.locator('canvas').evaluate((c: HTMLCanvasElement) => {
        const gl = c.getContext('webgl2')!, extension = gl.getExtension('WEBGL_lose_context');
        if (!extension) throw new Error('WEBGL_lose_context unavailable'); extension.loseContext();
      });
      await page.waitForFunction(() => document.querySelector('figure[data-qv-space]')?.getAttribute('data-qv-space-status') === '2d');
      assert.equal(await figure.locator('canvas').count(), 0); report.actualContextLosses++;
      await figure.getByRole('button', { name: language === 'ko' ? '3D 다시 시도' : 'Retry 3D', exact: true }).click();
      await figure.locator('canvas').waitFor(); report.contextRecoveries++;
      await figure.getByRole('button', { name: language === 'ko' ? '2D 보기' : 'Show 2D', exact: true }).click();
      assert.equal(await figure.locator('canvas').count(), 0);
      await figure.getByRole('button', { name: 'e₀ / e₂', exact: true }).focus(); await page.keyboard.press('Enter');
      assert.equal(await figure.getAttribute('data-qv-space'), 'xz'); report.flatProjections++;
      await figure.getByRole('button', { name: 'e₀ / e₁', exact: true }).click();
      assert.equal(await figure.getAttribute('data-qv-space'), 'xy'); report.flatProjections++;
      await figure.getByRole('button', { name: language === 'ko' ? '3D 보기' : 'Show 3D', exact: true }).click(); await figure.locator('canvas').waitFor();
      await page.getByRole('button', { name: 'Theme', exact: true }).click();
      await figure.scrollIntoViewIfNeeded(); await page.waitForTimeout(100);
      assert.equal(await lab.getAttribute('data-qv-input'), key); report.themeChanges++;
      for (const index of [13, 14, 0]) {
        await examples.getByRole('button', { name: presets[kind][index].label[language], exact: true }).click(); await ready();
        await figure.scrollIntoViewIfNeeded(); await figure.locator('canvas').waitFor(); report.subspaceDimensions++;
      }
      await page.screenshot({ path: `${root}/spatial-${language}-${width}.png` });
      // Playwright owns a page-lifetime listener-removal observer. Its captured
      // stack identifies it explicitly; application observers must all close.
      await page.evaluate(() => {
        const s = (window as any).qvSpatialStats;
        (window as any).qvApplicationObservers = [...s.observers].filter(o => !String(s.observerStacks.get(o)).includes('InjectedScript._setupGlobalListenersRemovalDetection'));
      });
      const axe = await new AxeBuilder({ page }).include('#root').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
      writeFileSync(`${root}/axe-${language}-${width}.json`, JSON.stringify(axe.violations, null, 2)); assert.deepEqual(axe.violations, []); report.axeCases++;
      await examples.getByRole('button', { name: presets[kind][6].label[language], exact: true }).click(); await ready();
      assert.equal(await lab.locator('[data-qv-space="unavailable"]').count(), 1); assert.equal(await lab.locator('canvas').count(), 0); report.complexFallbacks++;
      if (language === 'en') assert(!/[가-힣]/.test(await page.locator('body').innerText()));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      await page.getByRole('button', { name: 'Mount / unmount', exact: true }).click();
      assert.equal(await lab.count(), 0);
      await page.waitForFunction(() => (window as any).qvSpatialStats.live.size === 0 && (window as any).qvApplicationObservers.every((observer: unknown) => !(window as any).qvSpatialStats.observers.has(observer))).catch(async error => {
        const details = await page.evaluate(() => { const s = (window as any).qvSpatialStats; return { liveWorkers: s.live.size, created: s.created, terminated: s.terminated, observers: (window as any).qvApplicationObservers.filter((o: unknown) => s.observers.has(o)).map((o: unknown) => s.observerStacks.get(o)) }; });
        writeFileSync(`${root}/cleanup-failure.json`, JSON.stringify(details, null, 2)); throw error;
      });
      assert.equal(await page.locator('canvas').count(), 0);
      const stats = await page.evaluate(() => { const s = (window as any).qvSpatialStats; return { created: s.created, terminated: s.terminated, glCreated: s.glCreated }; });
      assert.equal(stats.created, stats.terminated); report.workerCreated += stats.created; report.workerTerminated += stats.terminated; report.webglCreated += stats.glCreated;
      report.cases++; console.log(`PASS ${language}/${width}`); await context.close();
    }
    for (const failure of ['webgl-unavailable', 'resize-setup'] as const) {
      const context = await browser.newContext({ viewport: { width: 800, height: 900 }, reducedMotion: 'reduce' });
      await context.addInitScript(failure => {
        localStorage.setItem('suanlab-language', 'en');
        (window as any).qvInitializationFailure = failure;
        const getContext = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, ...args: any[]) {
          if ((window as any).qvInitializationFailure === 'webgl-unavailable' && String(args[0]).startsWith('webgl')) return null;
          return (getContext as any).apply(this, args);
        } as typeof getContext;
        const observe = ResizeObserver.prototype.observe;
        ResizeObserver.prototype.observe = function (target, options) {
          if ((window as any).qvInitializationFailure === 'resize-setup' && target.querySelector('canvas')) throw new Error('Expected resize setup failure');
          return observe.call(this, target, options);
        };
      }, failure);
      const page = await context.newPage(); page.on('pageerror', e => report.errors.push(e.message));
      await page.route('**/__qv-spatial/', r => r.fulfill({ contentType: 'text/html', body: html }));
      await page.route('**/__qv-spatial.js', r => r.fulfill({ contentType: 'text/javascript', body: client.outputFiles[0].text }));
      await page.route('**/learning-quantum-variational.worker.ts', r => r.fulfill({ contentType: 'text/javascript', body: worker.outputFiles[0].text }));
      await page.goto('http://localhost:4175/__qv-spatial/');
      await page.locator('[data-qv-lab][data-qv-state="ready"]').waitFor({ timeout: 120000 });
      const figure = page.locator('figure[data-qv-space]'); await figure.scrollIntoViewIfNeeded();
      await page.waitForFunction(() => document.querySelector('figure[data-qv-space]')?.getAttribute('data-qv-space-status') === '2d');
      assert.equal(await figure.locator('canvas').count(), 0); assert.equal(await figure.locator('svg').count(), 1); report.initializationFallbacks++;
      await page.evaluate(() => { (window as any).qvInitializationFailure = ''; });
      await figure.getByRole('button', { name: 'Retry 3D', exact: true }).click(); await figure.locator('canvas').waitFor(); report.initializationRecoveries++;
      await page.getByRole('button', { name: 'Mount / unmount', exact: true }).click(); assert.equal(await page.locator('canvas').count(), 0);
      console.log(`PASS ${failure}`); await context.close();
    }
    assert.deepEqual(report.errors, []);
    writeFileSync(`${root}/browser-verification.json`, JSON.stringify(report, null, 2)); console.log(JSON.stringify(report));
  } finally { await browser.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
