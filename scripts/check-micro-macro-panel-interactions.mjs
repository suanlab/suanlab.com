/* Isolated actual panels; Next reader/export verification remains separate. */
import webpackPackage from 'next/dist/compiled/webpack/webpack.js';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
webpackPackage.init(); const webpack = webpackPackage.webpack;
const root = path.resolve('.runtime/learning/micro-macro-panels/interactions');
(async () => {
  await fs.mkdir(root, { recursive: true });
  const loader = path.join(root, 'transform.cjs');
  await fs.writeFile(loader, `module.exports=function(source){return require(${JSON.stringify(fileURLToPath(import.meta.resolve('esbuild')))}).transformSync(source,{loader:this.resourcePath.endsWith('.tsx')?'tsx':'ts',format:'esm',target:'es2022',jsx:'automatic'}).code};`);
  const stats = await new Promise((resolve, reject) => {
    const compiler = webpack({ mode: 'development', devtool: false, context: process.cwd(), target: 'web',
      entry: path.resolve('scripts/learning/micro-macro-panels-interactions-harness.tsx'),
      plugins: [new webpack.ProvidePlugin({ process: fileURLToPath(import.meta.resolve('next/dist/build/polyfills/process.js')) })],
      output: { path: root, filename: 'app.js', chunkFilename: 'worker-[name]-[contenthash].js', publicPath: 'auto' },
      resolve: { extensions: ['.tsx', '.ts', '.js'], alias: { '@': path.resolve('src') } },
      module: { rules: [{ test: /\.tsx?$/, exclude: /node_modules/, use: loader }] },
      optimization: { minimize: false },
    });
    compiler.run((error, result) => compiler.close(closeError => {
      if (error || closeError) reject(error || closeError);
      else if (result.hasErrors()) reject(new Error(result.toString({ all: false, errors: true })));
      else resolve({ assets: result.compilation.getAssets().map(asset => ({ name: asset.name })) });
    }));
  });
  const assets = new Set(stats.assets.map(a => a.name).filter(n => n.endsWith('.js')));
  const workerAssets = [...assets].filter(n => n.startsWith('worker-')); assert.equal(workerAssets.length, 1);
  const mainSource = await fs.readFile(path.join(root, 'app.js'), 'utf8');
  assert.doesNotMatch(mainSource, /function (?:binaryMacrostateModel|macroPartitionModel|noisyMacroObservationModel)\(/);
  const cssRoot = path.resolve('out/_next/static/css'), cssFiles = (await fs.readdir(cssRoot)).filter(n => n.endsWith('.css'));
  const css = (await Promise.all(cssFiles.map(n => fs.readFile(path.join(cssRoot, n), 'utf8')))).join('\n');
  const server = http.createServer(async (request, response) => {
    const name = (request.url || '').split('?')[0].slice(1);
    if (assets.has(name)) { response.setHeader('Content-Type', 'application/javascript; charset=utf-8'); response.end(await fs.readFile(path.join(root, name))); return; }
    if (name === 'site.css') { response.setHeader('Content-Type', 'text/css; charset=utf-8'); response.end(css); return; }
    if (name.startsWith('_next/static/media/') || name.startsWith('assets/')) {
      const base = path.resolve(name.startsWith('assets/') ? 'public' : 'out'), file = path.resolve(base, name);
      if (file.startsWith(base + path.sep)) { try { response.end(await fs.readFile(file)); return; } catch {} }
      response.writeHead(404); response.end(); return;
    }
    response.setHeader('Content-Type', 'text/html; charset=utf-8'); response.end('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Panel verification</title><link rel="stylesheet" href="/site.css"></head><body><div id="root"></div><script type="module" src="/app.js"></script></body></html>');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser; const errors = [];
  try {
    browser = await chromium.launch({ headless: true, args: ['--disable-dev-shm-usage'] });
    const context = await browser.newContext(); const page = await context.newPage(); page.on('pageerror', error => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const kinds = await page.evaluate(() => Object.keys(window.microMacroPanelPresets));
    const ready = (kind, key) => page.waitForFunction(({ kind, key }) => {
      const lab = document.querySelector('[data-micro-macro-lab]'); return lab?.getAttribute('data-micro-macro-lab') === kind && lab.getAttribute('data-micro-macro-current') === key;
    }, { kind, key }, { timeout: 60000 });
    const notes = [], presetsFor = kind => page.evaluate(kind => window.microMacroPanelPresets[kind], kind);
    await page.getByTestId('language-en').click();
    await page.setViewportSize({ width: 1440, height: 900 });
    for (const [index, kind] of kinds.entries()) {
      await page.getByTestId(`kind-${index}`).click();
      const presets = await presetsFor(kind), lab = page.locator('[data-micro-macro-lab]');
      await page.waitForFunction(() => document.querySelector('[data-micro-macro-lab]')?.getAttribute('data-micro-macro-ready') === 'true');
      await lab.getByRole('button', { name: presets[1].label.en, exact: true }).click(); await ready(kind, presets[1].key);
      const saved = await page.evaluate(kind => JSON.parse(localStorage.getItem('suanlab-micro-macro-' + kind)), kind); assert.equal(saved.version, 1);
      await page.reload(); await page.getByTestId(`kind-${index}`).click(); await ready(kind, presets[1].key);
      const guides = lab.locator('.ai-guided-choices button');
      const guidePresets = await page.evaluate(kind => window.microMacroPanelGuides[kind], kind);
      for (let g = 0; g < 3; g++) {
        await guides.nth(g).click(); assert.equal(await guides.nth(g).getAttribute('aria-pressed'), 'true');
        await lab.getByRole('button', { name: 'Load this example', exact: true }).click();
        await ready(kind, presets[guidePresets[g]].key);
        const reveal = lab.locator('summary').filter({ hasText: 'Reveal the explanation' }); await reveal.click();
      }
      await lab.locator('summary').filter({ hasText: 'Extended settings:' }).click();
      await lab.getByRole('textbox', { name: 'Extended parameter JSON' }).fill('{invalid');
      await lab.getByRole('button', { name: 'Apply extended settings', exact: true }).click();
      await lab.getByRole('alert').waitFor(); assert.match(await lab.getByRole('alert').textContent(), /not applied/);
      const previousValidKey = await lab.getAttribute('data-micro-macro-current'), invalidParameters = JSON.parse(presets[0].key).parameters;
      if (index === 0) invalidParameters.sites = 4097;
      if (index === 1) invalidParameters.trialWeights = invalidParameters.trialWeights.map(() => 0);
      if (index === 2) invalidParameters.channelWeights[0] = invalidParameters.channelWeights[0].map(() => 0);
      await lab.getByRole('textbox', { name: 'Extended parameter JSON' }).fill(JSON.stringify(invalidParameters));
      await lab.getByRole('button', { name: 'Apply extended settings', exact: true }).click();
      assert.equal(await lab.getAttribute('data-micro-macro-current'), previousValidKey);
      await lab.getByRole('button', { name: 'Reset all experiment settings', exact: true }).click(); await ready(kind, presets[0].key);
      await page.evaluate(kind => localStorage.setItem('suanlab-micro-macro-' + kind, '{bad'), kind);
      await page.reload(); await page.getByTestId(`kind-${index}`).click(); await ready(kind, presets[0].key);
      notes.push({ kind, storageReload: true, malformedSavedFallback: true, invalidDraftRetainedValidSettings: true, invalidNumericDraftRetainedValidSettings: true, guides: 3 });
      console.log(JSON.stringify(notes.at(-1)));
    }
    await page.getByTestId('kind-0').click(); const binaryPresets = await presetsFor(kinds[0]);
    const binaryLab = page.locator('[data-micro-macro-lab]');
    await binaryLab.getByRole('button', { name: binaryPresets[5].label.en, exact: true }).click(); await ready(kinds[0], binaryPresets[5].key);
    await binaryLab.getByRole('button', { name: 'Next counts', exact: true }).click();
    assert.equal(await binaryLab.getByRole('region', { name: 'Exact shell counts · 32 rows per page', exact: true }).locator('tbody tr').first().locator('th').textContent(), '32');
    await binaryLab.getByRole('button', { name: 'Next block counts', exact: true }).click();
    assert.equal(await binaryLab.getByRole('region', { name: 'Exact fixed-K block counts · 32 rows per page', exact: true }).locator('tbody tr').first().locator('th').textContent(), '32');
    await binaryLab.getByRole('button', { name: binaryPresets[0].label.en, exact: true }).click(); await ready(kinds[0], binaryPresets[0].key);
    const lab = page.locator('[data-micro-macro-lab]');
    await lab.getByRole('button', { name: 'Scan comparison examples', exact: true }).click();
    await page.waitForFunction(() => Number(document.querySelector('[data-micro-macro-lab]')?.getAttribute('data-micro-macro-frame')) > 0);
    await page.getByRole('button', { name: 'Pause site motion', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('[data-micro-macro-lab]')?.getAttribute('data-micro-macro-playing') === 'false');
    await lab.getByRole('button', { name: 'Step comparison example', exact: true }).click();
    await page.emulateMedia({ reducedMotion: 'reduce' });
    assert.equal(await lab.getByRole('button', { name: 'Scan comparison examples', exact: true }).isDisabled(), true);
    await page.emulateMedia({ reducedMotion: 'no-preference' }); await page.getByRole('button', { name: 'Play site motion', exact: true }).click();
    await lab.getByRole('button', { name: 'Scan comparison examples', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('[data-micro-macro-lab]')?.getAttribute('data-micro-macro-playing') === 'true');
    await page.evaluate(() => { const spacer = document.createElement('div'); spacer.id = 'outside-panel'; spacer.style.height = '2000px'; document.body.append(spacer); window.scrollTo(0, document.documentElement.scrollHeight); });
    await page.waitForFunction(() => document.querySelector('[data-micro-macro-lab]')?.getAttribute('data-micro-macro-playing') === 'false');
    await page.evaluate(() => { document.getElementById('outside-panel').remove(); window.scrollTo(0, 0); });
    await page.getByTestId('kind-2').click(); const noisyPresets = await presetsFor(kinds[2]); await ready(kinds[2], noisyPresets[0].key);
    const parameters = { preparationWeights: Array.from({ length: 16 }, (_, i) => i % 2 ? Number.MAX_VALUE / (i + 1) : Number.MIN_VALUE * (i + 1)),
      channelWeights: Array.from({ length: 16 }, (_, i) => Array.from({ length: 16 }, (_, j) => j === i ? Number.MAX_VALUE / (i + 1) : Number.MIN_VALUE * (j + 1))) };
    await lab.locator('summary').filter({ hasText: 'Extended settings:' }).click();
    await lab.getByRole('textbox', { name: 'Extended parameter JSON' }).fill(JSON.stringify(parameters));
    const before = await page.evaluate(() => ({ ...window.microMacroInteractionProbe }));
    const stoppedDuringDebounce = await lab.evaluate(el => new Promise(resolve => {
      [...el.querySelectorAll('button')].find(b => b.textContent === 'Apply extended settings').click();
      requestAnimationFrame(() => { const stop = [...el.querySelectorAll('button')].find(b => b.textContent === 'Stop calculation'); if (stop) stop.click(); resolve(Boolean(stop)); });
    }));
    assert.equal(stoppedDuringDebounce, true);
    await page.waitForTimeout(350); assert.equal(await lab.getAttribute('data-micro-macro-state'), 'idle');
    assert.equal(await page.evaluate(() => window.microMacroInteractionProbe.started), before.started);
    await lab.getByRole('button', { name: 'Retry calculation', exact: true }).click();
    await page.waitForFunction(previous => window.microMacroInteractionProbe.acknowledged > previous && window.microMacroInteractionProbe.live === 1, before.acknowledged);
    const started = await page.evaluate(() => ({ ...window.microMacroInteractionProbe }));
    assert.equal(await lab.locator('[data-micro-macro-result]').count(), 0);
    await page.getByTestId('language-ko').click(); await page.getByTestId('theme').click();
    await page.waitForFunction(previous => window.microMacroInteractionProbe.ticks > previous + 4, started.ticks);
    await page.getByTestId('kind-1').click();
    await page.waitForFunction(() => window.microMacroInteractionProbe.live === 0 && document.querySelector('[data-micro-macro-result]')?.getAttribute('data-micro-macro-result') === 'finite-macro-partition-conditional-entropy');
    await page.getByTestId('language-en').click(); await page.getByTestId('kind-2').click();
    const heavyKey = JSON.stringify({ kind: kinds[2], parameters }); await ready(kinds[2], heavyKey);
    const heavyAxe = await new AxeBuilder({ page }).include('[data-micro-macro-lab]').analyze(); assert.deepEqual(heavyAxe.violations, []);
    await lab.getByRole('button', { name: 'Erasure probability 1/4', exact: true }).click(); await ready(kinds[2], noisyPresets[2].key);
    await lab.getByRole('combobox').selectOption('2');
    await lab.locator('[data-micro-macro-result]').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(root, 'noisy-desktop-en-dark.png') });
    await lab.getByRole('button', { name: noisyPresets[6].label.en, exact: true }).click(); await ready(kinds[2], noisyPresets[6].key);
    await lab.getByRole('combobox').selectOption('2');
    assert.match(await lab.textContent(), /true zero probability, so its posterior is undefined/);
    await page.getByTestId('kind-1').click(); const weightedPresets = await presetsFor(kinds[1]);
    await lab.getByRole('button', { name: weightedPresets[5].label.en, exact: true }).click(); await ready(kinds[1], weightedPresets[5].key);
    await lab.locator('[data-micro-macro-result]').getByRole('combobox').selectOption('2'); assert.match(await lab.textContent(), /contains no states/);
    await lab.locator('[data-micro-macro-result]').getByRole('combobox').selectOption('1'); assert.match(await lab.textContent(), /Undefined conditional/);
    await page.setViewportSize({ width: 320, height: 900 }); await page.getByTestId('language-ko').click();
    await page.evaluate(() => document.documentElement.classList.remove('dark')); await page.getByTestId('kind-0').click();
    await lab.getByRole('button', { name: binaryPresets[0].label.ko, exact: true }).click(); await ready(kinds[0], binaryPresets[0].key);
    await lab.locator('[data-micro-macro-result]').scrollIntoViewIfNeeded(); await page.screenshot({ path: path.join(root, 'binary-mobile-ko-light.png') });
    const blockedContext = await browser.newContext();
    await blockedContext.addInitScript(() => { Storage.prototype.getItem = () => { throw new Error('blocked'); }; Storage.prototype.setItem = () => { throw new Error('blocked'); }; });
    const blocked = await blockedContext.newPage(); await blocked.goto(page.url());
    await blocked.waitForFunction(() => document.querySelector('[data-micro-macro-result]'));
    assert.match(await blocked.locator('[data-micro-macro-lab]').textContent(), /브라우저 저장을 사용할 수 없습니다/); await blockedContext.close();
    assert.deepEqual(errors, []);
    const probe = await page.evaluate(() => ({ ...window.microMacroInteractionProbe })); assert.equal(probe.live, 0); assert.equal(probe.started, probe.terminated);
    await fs.writeFile(path.join(root, 'result.json'), JSON.stringify({ scope: 'Isolated actual panel controls/worker/storage/motion; actual Next readers remain pending', notes, zeroMassAndEmptySelection: true, exactLargeCountPagination: true, motion: { globalPause: true, reducedMotion: true, manualStep: true, actualOffscreenPause: true }, stoppedDuringDebounce, heavyStopRetryReplacementAndCompletion: true, blockedStorage: true, workerProbe: probe, errors }, null, 2));
    console.log(JSON.stringify({ families: notes.length, guides: 9, liveWorkers: probe.live, errors: errors.length }));
  } finally { if (browser) await browser.close(); await new Promise(resolve => server.close(resolve)); }
})().catch(error => { console.error(error); process.exitCode = 1; });
