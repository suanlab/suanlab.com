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
const root = path.resolve('.runtime/learning/ensemble-panels/interactions');
(async () => {
  await fs.mkdir(root, { recursive: true });
  const loader = path.join(root, 'transform.cjs');
  await fs.writeFile(loader, `module.exports=function(source){return require(${JSON.stringify(fileURLToPath(import.meta.resolve('esbuild')))}).transformSync(source,{loader:this.resourcePath.endsWith('.tsx')?'tsx':'ts',format:'esm',target:'es2022',jsx:'automatic'}).code};`);
  const stats = await new Promise((resolve, reject) => {
    const compiler = webpack({ mode: 'development', devtool: false, context: process.cwd(), target: 'web',
      entry: path.resolve('scripts/learning/ensemble-panels-interactions-harness.tsx'),
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
  assert.doesNotMatch(mainSource, /function (?:finiteReservoirModel|canonicalBinaryModel|grandCanonicalOccupancyModel)\(/);
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
    const kinds = await page.evaluate(() => Object.keys(window.ensemblePanelPresets));
    const ready = (kind, key) => page.waitForFunction(({ kind, key }) => {
      const lab = document.querySelector('[data-ensemble-lab]'); return lab?.getAttribute('data-ensemble-lab') === kind && lab.getAttribute('data-ensemble-current') === key;
    }, { kind, key }, { timeout: 60000 });
    const notes = [], presetsFor = kind => page.evaluate(kind => window.ensemblePanelPresets[kind], kind);
    await page.getByTestId('language-en').click();
    await page.setViewportSize({ width: 1440, height: 900 });
    for (const [index, kind] of kinds.entries()) {
      await page.getByTestId(`kind-${index}`).click();
      const presets = await presetsFor(kind), lab = page.locator('[data-ensemble-lab]');
      await page.waitForFunction(() => document.querySelector('[data-ensemble-lab]')?.getAttribute('data-ensemble-ready') === 'true');
      await lab.getByRole('button', { name: presets[1].label.en, exact: true }).click(); await ready(kind, presets[1].key);
      const saved = await page.evaluate(kind => JSON.parse(localStorage.getItem('suanlab-ensemble-' + kind)), kind); assert.equal(saved.version, 1);
      await page.reload(); await page.getByTestId(`kind-${index}`).click(); await ready(kind, presets[1].key);
      const guides = lab.locator('.ai-guided-choices button');
      const guidePresets = await page.evaluate(kind => window.ensemblePanelGuides[kind], kind);
      for (let g = 0; g < 3; g++) {
        await guides.nth(g).click(); assert.equal(await guides.nth(g).getAttribute('aria-pressed'), 'true');
        await lab.getByRole('button', { name: 'Load this example', exact: true }).click();
        await ready(kind, presets[guidePresets[g]].key);
        const reveal = lab.locator('summary').filter({ hasText: 'Reveal the explanation' }); await reveal.click();
      }
      await lab.locator('summary').filter({ hasText: 'Extended parameters' }).click();
      await lab.getByRole('textbox', { name: 'Extended parameter JSON' }).fill('{invalid');
      await lab.getByRole('button', { name: 'Apply extended settings', exact: true }).click();
      await lab.getByRole('alert').waitFor(); assert.match(await lab.getByRole('alert').textContent(), /not applied/);
      const previousValidKey = await lab.getAttribute('data-ensemble-current'), invalidParameters = JSON.parse(presets[0].key).parameters;
      if (index === 0) invalidParameters.subsystemSites = 4097;
      if (index === 1) invalidParameters.energyGap = -1;
      if (index === 2) invalidParameters.chemicalPotential = null;
      await lab.getByRole('textbox', { name: 'Extended parameter JSON' }).fill(JSON.stringify(invalidParameters));
      await lab.getByRole('button', { name: 'Apply extended settings', exact: true }).click();
      assert.equal(await lab.getAttribute('data-ensemble-current'), previousValidKey);
      await lab.getByRole('button', { name: 'Reset all experiment settings', exact: true }).click(); await ready(kind, presets[0].key);
      await page.evaluate(kind => localStorage.setItem('suanlab-ensemble-' + kind, '{bad'), kind);
      await page.reload(); await page.getByTestId(`kind-${index}`).click(); await ready(kind, presets[0].key);
      notes.push({ kind, storageReload: true, malformedSavedFallback: true, invalidDraftRetainedValidSettings: true, invalidNumericDraftRetainedValidSettings: true, guides: 3 });
      console.log(JSON.stringify(notes.at(-1)));
    }
    const lab = page.locator('[data-ensemble-lab]');
    await page.getByTestId('kind-0').click(); const presets = await presetsFor(kinds[0]);
    await lab.getByRole('button', { name: presets[5].label.en, exact: true }).click(); await ready(kinds[0], presets[5].key);
    await lab.getByRole('button', { name: 'Next distribution rows', exact: true }).click();
    assert.equal(await lab.getByRole('region', { name: 'Distribution and exact counts · 32 rows per page', exact: true }).locator('tbody tr').first().locator('th').textContent(), '32');
    await lab.getByRole('button', { name: 'Previous distribution rows', exact: true }).click();
    assert.equal(await lab.getByRole('region', { name: 'Distribution and exact counts · 32 rows per page', exact: true }).locator('tbody tr').first().locator('th').textContent(), '0');
    await lab.getByRole('button', { name: 'Scan comparison examples', exact: true }).click();
    await page.waitForFunction(() => Number(document.querySelector('[data-ensemble-lab]')?.getAttribute('data-ensemble-frame')) > 0);
    await page.getByRole('button', { name: 'Pause site motion', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('[data-ensemble-lab]')?.getAttribute('data-ensemble-playing') === 'false');
    await lab.getByRole('button', { name: 'Step comparison example', exact: true }).click();
    await page.emulateMedia({ reducedMotion: 'reduce' });
    assert.equal(await lab.getByRole('button', { name: 'Scan comparison examples', exact: true }).isDisabled(), true);
    await page.emulateMedia({ reducedMotion: 'no-preference' }); await page.getByRole('button', { name: 'Play site motion', exact: true }).click();
    await lab.getByRole('button', { name: 'Scan comparison examples', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('[data-ensemble-lab]')?.getAttribute('data-ensemble-playing') === 'true');
    await page.evaluate(() => { const spacer = document.createElement('div'); spacer.id = 'outside-panel'; spacer.style.height = '2000px'; document.body.append(spacer); window.scrollTo(0, document.documentElement.scrollHeight); });
    await page.waitForFunction(() => document.querySelector('[data-ensemble-lab]')?.getAttribute('data-ensemble-playing') === 'false');
    await page.evaluate(() => { document.getElementById('outside-panel').remove(); window.scrollTo(0, 0); });
    await lab.getByRole('button', { name: 'Reset all experiment settings', exact: true }).click(); await ready(kinds[0], presets[0].key);
    await lab.locator('summary').filter({ hasText: 'Extended parameters' }).click();
    const parameters = { subsystemSites: 2048, reservoirSites: 2048, totalExcitations: 1 };
    await lab.getByRole('textbox', { name: 'Extended parameter JSON' }).fill(JSON.stringify(parameters));
    const before = await page.evaluate(() => ({ ...window.ensembleInteractionProbe }));
    const stoppedDuringDebounce = await lab.evaluate(el => new Promise(resolve => {
      [...el.querySelectorAll('button')].find(b => b.textContent === 'Apply extended settings').click();
      requestAnimationFrame(() => { const stop = [...el.querySelectorAll('button')].find(b => b.textContent === 'Stop calculation'); if (stop) stop.click(); resolve(Boolean(stop)); });
    }));
    assert.equal(stoppedDuringDebounce, true);
    await page.waitForTimeout(350); assert.equal(await lab.getAttribute('data-ensemble-state'), 'idle');
    assert.equal(await page.evaluate(() => window.ensembleInteractionProbe.started), before.started);
    await lab.getByRole('button', { name: 'Retry calculation', exact: true }).click();
    await page.waitForFunction(previous => window.ensembleInteractionProbe.acknowledged > previous && window.ensembleInteractionProbe.live === 1, before.acknowledged);
    const started = await page.evaluate(() => ({ ...window.ensembleInteractionProbe }));
    assert.equal(await lab.locator('[data-ensemble-result]').count(), 0);
    await page.getByTestId('language-ko').click(); await page.getByTestId('theme').click();
    await page.waitForFunction(previous => window.ensembleInteractionProbe.ticks > previous + 4, started.ticks);
    await page.getByTestId('kind-1').click();
    await page.waitForFunction(() => window.ensembleInteractionProbe.live === 0 && document.querySelector('[data-ensemble-result]')?.getAttribute('data-ensemble-result') === 'canonical-binary-energy-fluctuations');
    await page.getByTestId('language-en').click(); await page.getByTestId('kind-0').click();
    await ready(kinds[0], JSON.stringify({ kind: kinds[0], parameters }));
    assert.ok(await lab.getByRole('button', { name: 'Next distribution rows', exact: true }).isEnabled());
    const heavyAxe = await new AxeBuilder({ page }).include('[data-ensemble-lab]').analyze(); assert.deepEqual(heavyAxe.violations, []);
    for (const [index, kind] of kinds.entries()) {
      await page.getByTestId('kind-' + index).click(); const examples = await presetsFor(kind);
      const chosen = index === 0 ? 0 : index === 1 ? 6 : 4;
      await lab.getByRole('button', { name: examples[chosen].label.en, exact: true }).click(); await ready(kind, examples[chosen].key);
      await lab.locator('[data-ensemble-result]').scrollIntoViewIfNeeded();
      await page.screenshot({ path: path.join(root, 'family-' + index + '-desktop-en.png'), fullPage: true });
    }
    // Condition on a true-zero count, then distinguish the threshold and tail.
    const grandPresets = await presetsFor(kinds[2]);
    await lab.getByRole('button', { name: grandPresets[3].label.en, exact: true }).click(); await ready(kinds[2], grandPresets[3].key);
    await lab.getByRole('spinbutton', { name: 'Particle count N to inspect', exact: true }).fill('0');
    await lab.getByRole('region', { name: 'Inspect patterns at a fixed particle count', exact: true }).getByText('The chosen count has true zero probability, so its conditional pattern probability is undefined.', { exact: true }).waitFor();
    await lab.getByRole('button', { name: grandPresets[4].label.en, exact: true }).click(); await ready(kinds[2], grandPresets[4].key);
    await lab.getByRole('spinbutton', { name: 'Particle count N to inspect', exact: true }).fill('1');
    const conditionalText = await lab.getByRole('region', { name: 'Inspect patterns at a fixed particle count', exact: true }).textContent();
    assert.match(conditionalText, /01 · 10/); assert.match(conditionalText, /P\(N\)=0.5/);
    // A blocked real worker asset must expose recovery without a main-thread fallback.
    await page.getByTestId('kind-1').click(); const canonicalPresets = await presetsFor(kinds[1]);
    await ready(kinds[1], canonicalPresets[6].key);
    await context.route('**/worker-*', route => route.abort());
    await lab.getByRole('button', { name: canonicalPresets[3].label.en, exact: true }).click();
    await page.waitForFunction(() => document.querySelector('[data-ensemble-lab]')?.getAttribute('data-ensemble-state') === 'error');
    assert.equal(await lab.locator('[data-ensemble-result]').count(), 0);
    await context.unroute('**/worker-*');
    await lab.getByRole('button', { name: 'Retry calculation', exact: true }).click(); await ready(kinds[1], canonicalPresets[3].key);
    const final = await page.evaluate(() => ({ ...window.ensembleInteractionProbe }));
    assert.equal(final.live, 0); assert.equal(final.started, final.terminated); assert.deepEqual(errors, []);
    await fs.writeFile(path.join(root, 'result.json'), JSON.stringify({ scope: 'Isolated actual ensemble panels; no Next reader registration', notes, pagination: true, conditionalInspection: true, blockedWorkerRecovery: true, motionPauseReducedManualOffscreen: true, stopDuringDebounce: true, heavyStartedResponsiveReplacementRemount: true, workerAssets, final, errors }, null, 2));
    console.log(JSON.stringify({ verified: true, final, errors }));
  } finally { if (browser) await browser.close(); await new Promise(resolve => server.close(resolve)); }
})().catch(error => { console.error(error); process.exitCode = 1; });
