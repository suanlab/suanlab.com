/** Exercise teaching controls with real browser storage, input and Workers. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { quantumVariationalKinds as kinds, quantumVariationalPresets as presets, quantumVariationalInputKey as key } from '../src/lib/learning-quantum-variational';
const root = '.runtime/learning/quantum-variational-spatial';
async function main() {
  const source = `import React,{useState}from'react';import{createRoot}from'react-dom/client';import{LanguageProvider}from'./src/components/language-provider';import{SiteMotionProvider,MotionControls}from'./src/components/motion/motion-provider';import Lab from'./src/components/learning/quantum-variational-lab';function App(){const[mounted,setMounted]=useState(true);return <div className="ai-book"><header style={{position:'sticky',top:0,zIndex:10,background:'hsl(var(--background))',padding:8}}><h1>Variational UI verification</h1><button onClick={()=>setMounted(v=>!v)}>Mount / unmount</button><MotionControls/></header><main style={{maxWidth:1100,margin:'auto',padding:16}}>{mounted&&<Lab kind={window.qvKind}/>}</main></div>}createRoot(document.getElementById('root')).render(<LanguageProvider><SiteMotionProvider><App/></SiteMotionProvider></LanguageProvider>);`;
  const client = await build({ stdin: { contents: source, resolveDir: process.cwd(), loader: 'tsx' }, bundle: true, write: false, jsx: 'automatic', format: 'esm', platform: 'browser', target: 'es2022', define: { 'process.env.NODE_ENV': '"production"' } });
  const worker = await build({ entryPoints: ['src/lib/learning-quantum-variational.worker.ts'], bundle: true, write: false, format: 'esm', platform: 'browser', target: 'es2022' });
  const exported = readFileSync('out/book/online/physics-math-for-ai/index.html', 'utf8');
  const css = [...new Set([...exported.matchAll(/href="([^"]+\.css)"/g)].map(m => m[1]))].map(href => `<link rel="stylesheet" href="${href}">`).join('');
  const html = `<!doctype html><html lang="en"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Variational UI QA</title>${css}</head><body><div id="root"></div><script type="module" src="/__qv-ui.js"></script></body></html>`;
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox', '--enable-unsafe-swiftshader'] });
  const report = { cases: 0, validRestores: 0, corruptFallbacks: 0, wrongKindFallbacks: 0, deniedStorage: 0, unavailableRetries: 0, executionRetries: 0, invalidDraftResets: 0, startedCancels: 0, cancelledRetries: 0, startedReplacements: 0, startedUnmounts: 0, remounts: 0, keyboardScrolls: 0, touchScrolls: 0, reducedMotionChecks: 0, playbackChecks: 0, globalPauseChecks: 0, offscreenPauseChecks: 0, workerCreated: 0, workerTerminated: 0, expectedWorkerErrors: 0, errors: [] as string[] };
  try {
    for (const kind of kinds) for (const cache of ['corrupt', 'wrong-kind', 'denied', 'valid'] as const) {
      const context = await browser.newContext({ viewport: { width: 320, height: 960 }, deviceScaleFactor: 2, hasTouch: true, isMobile: true, reducedMotion: 'reduce' });
      await context.addInitScript(({ kind, cache, value, wrong }) => {
        (window as any).qvKind = kind;
        localStorage.setItem('suanlab-language', 'en');
        const storageKey = `suanlab-quantum-variational-${kind}`;
        localStorage.setItem(storageKey, cache === 'corrupt' ? '{broken' : JSON.stringify({ version: 1, value: cache === 'wrong-kind' ? wrong : value }));
        const get = Storage.prototype.getItem, set = Storage.prototype.setItem;
        const stats = { created: 0, terminated: 0, live: new Set<Worker>(), writes: [] as string[], failure: '' };
        (window as any).qvUIStats = stats;
        Storage.prototype.getItem = function (key: string) { if (cache === 'denied' && key === storageKey) throw new DOMException('Storage denied', 'SecurityError'); return get.call(this, key); };
        Storage.prototype.setItem = function (key: string, value: string) { if (key === storageKey) { if (cache === 'denied') throw new DOMException('Storage denied', 'SecurityError'); stats.writes.push(value); } return set.call(this, key, value); };
        const Original = window.Worker;
        window.Worker = class extends Original {
          private blob: string | null;
          constructor(url: string | URL, options?: WorkerOptions) {
            const failure = stats.failure; stats.failure = '';
            if (failure === 'unavailable') throw new Error('Unavailable for QA');
            const blob = failure === 'execution' ? URL.createObjectURL(new Blob(['throw new Error("qv-expected-native-worker-failure")'], { type: 'text/javascript' })) : null;
            super(blob ?? url, options); this.blob = blob; stats.created++; stats.live.add(this);
          }
          terminate() { if (stats.live.delete(this)) stats.terminated++; if (this.blob) { URL.revokeObjectURL(this.blob); this.blob = null; } super.terminate(); }
        };
      }, { kind, cache, value: presets[kind][1].input, wrong: presets[kind === kinds[0] ? kinds[1] : kinds[0]][0].input });
      const page = await context.newPage();
      page.on('pageerror', e => { if (e.message.includes('qv-expected-native-worker-failure')) report.expectedWorkerErrors++; else report.errors.push(e.message); });
      await page.route('**/__qv-ui/', r => r.fulfill({ contentType: 'text/html', body: html }));
      await page.route('**/__qv-ui.js', r => r.fulfill({ contentType: 'text/javascript', body: client.outputFiles[0].text }));
      await page.route('**/learning-quantum-variational.worker.ts', r => r.fulfill({ contentType: 'text/javascript', body: worker.outputFiles[0].text }));
      await page.goto('http://localhost:4175/__qv-ui/');
      const lab = page.locator('[data-qv-lab]'), ready = () => page.locator('[data-qv-lab][data-qv-state="ready"]').waitFor({ timeout: 120000 });
      await ready();
      const restored = cache === 'valid' ? 1 : 0;
      assert.equal(await lab.getAttribute('data-qv-input'), key(presets[kind][restored].input));
      if (cache === 'denied') { assert(await lab.getByText('Browser storage is unavailable. You can continue experimenting in this session.', { exact: true }).isVisible()); report.deniedStorage++; }
      else {
        const writes = await page.evaluate(() => (window as any).qvUIStats.writes);
        assert(writes.length > 0); assert.equal(key(JSON.parse(writes[0]).value), key(presets[kind][restored].input));
        if (cache === 'valid') report.validRestores++; else if (cache === 'corrupt') report.corruptFallbacks++; else report.wrongKindFallbacks++;
      }
      const examples = lab.getByRole('group', { name: 'Experiment examples', exact: true });
      if (cache === 'valid') {
        for (const failure of ['unavailable', 'execution']) {
          await page.evaluate(f => { (window as any).qvUIStats.failure = f; }, failure);
          const index = failure === 'unavailable' ? 0 : 1;
          await examples.getByRole('button', { name: presets[kind][index].label.en, exact: true }).click();
          await page.locator('[data-qv-lab][data-qv-state="error"]').waitFor();
          assert.equal(await lab.locator('[data-qv-result]').count(), 0);
          await lab.getByRole('button', { name: 'Retry calculation', exact: true }).click(); await ready();
          assert.equal(await lab.locator('[data-qv-result]').getAttribute('data-qv-result-key'), key(presets[kind][index].input));
          if (failure === 'unavailable') report.unavailableRetries++; else report.executionRetries++;
        }
        await examples.getByRole('button', { name: presets[kind][0].label.en, exact: true }).click(); await ready();
        const editor = lab.locator('details').filter({ has: page.locator('[data-qv-field]') });
        await editor.locator('summary').click();
        const denominator = lab.locator('[data-qv-field="hDenominator"]');
        await denominator.fill('0'); await denominator.press('Enter'); assert.equal(await denominator.getAttribute('aria-invalid'), 'true');
        await lab.getByRole('button', { name: 'Reset to first example', exact: true }).click(); await ready();
        assert.equal(await denominator.inputValue(), '1'); assert.equal(await denominator.getAttribute('aria-invalid'), 'false'); report.invalidDraftResets++;
        await editor.locator('summary').click();
        const reading = lab.locator('[data-qv-reading]');
        await reading.scrollIntoViewIfNeeded();
        assert(await reading.getByRole('button', { name: 'Follow the components', exact: true }).isDisabled());
        const original = await reading.getAttribute('data-qv-reading');
        await reading.getByRole('button', { name: 'Next component', exact: true }).click();
        assert.notEqual(await reading.getAttribute('data-qv-reading'), original); report.reducedMotionChecks++;
        const diagram = lab.locator('[data-qv-diagram]').first(); await diagram.focus(); await page.keyboard.press('ArrowRight');
        await page.waitForFunction(() => document.querySelector('[data-qv-lab] [data-qv-diagram]')!.scrollLeft > 0); report.keyboardScrolls++;
        await diagram.evaluate(e => { e.scrollLeft = 0; }); await diagram.scrollIntoViewIfNeeded();
        const box = (await diagram.boundingBox())!, cdp = await context.newCDPSession(page);
        const start = { x: box.x + box.width * 0.85, y: box.y + Math.min(80, box.height / 2) };
        await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [start] });
        for (let i = 1; i <= 6; i++) { await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: start.x - i * 18, y: start.y }] }); await page.waitForTimeout(20); }
        await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
        await page.waitForFunction(() => document.querySelector('[data-qv-lab] [data-qv-diagram]')!.scrollLeft > 20); report.touchScrolls++; await cdp.detach();
        if (kind === kinds[0]) {
          await page.emulateMedia({ reducedMotion: 'no-preference' }); await reading.scrollIntoViewIfNeeded();
          const follow = reading.getByRole('button', { name: 'Follow the components', exact: true }); await follow.waitFor();
          await page.waitForFunction(() => ![...document.querySelectorAll<HTMLButtonElement>('[data-qv-reading] button')].find(b => b.textContent === 'Follow the components')!.disabled);
          await follow.click(); const before = await reading.getAttribute('data-qv-reading');
          await page.waitForFunction(v => document.querySelector('[data-qv-reading]')!.getAttribute('data-qv-reading') !== v, before); report.playbackChecks++;
          await page.getByRole('button', { name: 'Pause site motion', exact: true }).click();
          await page.waitForFunction(() => document.documentElement.dataset.siteMotion === 'paused');
          const paused = await reading.getAttribute('data-qv-reading'); await page.waitForTimeout(1100);
          assert.equal(await reading.getAttribute('data-qv-reading'), paused); report.globalPauseChecks++;
          await reading.getByRole('button', { name: 'Stop reading playback', exact: true }).click();
          assert.equal(await reading.getByRole('button', { name: 'Follow the components', exact: true }).getAttribute('aria-pressed'), 'false');
          await page.getByRole('button', { name: 'Play site motion', exact: true }).click(); await reading.scrollIntoViewIfNeeded();
          await reading.getByRole('button', { name: 'Follow the components', exact: true }).click();
          await page.evaluate(() => window.scrollTo(0, 0)); await page.waitForTimeout(100);
          const offscreen = await reading.getAttribute('data-qv-reading'); await page.waitForTimeout(1100);
          assert.equal(await reading.getAttribute('data-qv-reading'), offscreen); report.offscreenPauseChecks++;
          await page.emulateMedia({ reducedMotion: 'reduce' });
        } else {
          const long = () => examples.getByRole('button', { name: presets[kind][9].label.en, exact: true }).click();
          const started = () => page.locator('[data-qv-lab][data-qv-phase="calculating"]').waitFor();
          await long(); await started(); await lab.getByRole('button', { name: 'Cancel calculation', exact: true }).click();
          assert.equal(await lab.locator('[data-qv-result]').count(), 0); assert.equal(await page.evaluate(() => (window as any).qvUIStats.live.size), 0); report.startedCancels++;
          await lab.getByRole('button', { name: 'Retry calculation', exact: true }).click(); await ready(); report.cancelledRetries++;
          await examples.getByRole('button', { name: presets[kind][0].label.en, exact: true }).click(); await ready();
          await long(); await started(); await examples.getByRole('button', { name: presets[kind][0].label.en, exact: true }).click(); await ready();
          assert.equal(await lab.locator('[data-qv-result]').getAttribute('data-qv-result-key'), key(presets[kind][0].input)); report.startedReplacements++;
          await long(); await started(); await page.getByRole('button', { name: 'Mount / unmount', exact: true }).click();
          assert.equal(await page.evaluate(() => (window as any).qvUIStats.live.size), 0); report.startedUnmounts++;
          await page.getByRole('button', { name: 'Mount / unmount', exact: true }).click();
          await examples.getByRole('button', { name: presets[kind][0].label.en, exact: true }).click(); await ready(); report.remounts++;
        }
      }
      assert(!/[가-힣]/.test(await page.locator('body').innerText()));
      if (await lab.count()) await page.getByRole('button', { name: 'Mount / unmount', exact: true }).click();
      await page.waitForFunction(() => (window as any).qvUIStats.live.size === 0);
      const stats = await page.evaluate(() => ({ created: (window as any).qvUIStats.created, terminated: (window as any).qvUIStats.terminated }));
      assert.equal(stats.created, stats.terminated); report.workerCreated += stats.created; report.workerTerminated += stats.terminated;
      report.cases++; console.log(`PASS ${kind}/${cache}`); await context.close();
    }
    assert.deepEqual(report.errors, []);
    writeFileSync(`${root}/ui-verification.json`, JSON.stringify(report, null, 2)); console.log(JSON.stringify(report));
  } finally { await browser.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
