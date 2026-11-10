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
const root = path.resolve('.runtime/learning/micro-macro-panels/browser');
(async () => {
  await fs.mkdir(root, { recursive: true });
  const loader = path.join(root, 'transform.cjs');
  await fs.writeFile(loader, `module.exports=function(source){return require(${JSON.stringify(fileURLToPath(import.meta.resolve('esbuild')))}).transformSync(source,{loader:this.resourcePath.endsWith('.tsx')?'tsx':'ts',format:'esm',target:'es2022',jsx:'automatic'}).code};`);
  const stats = await new Promise((resolve, reject) => {
    const compiler = webpack({ mode: 'development', devtool: false, context: process.cwd(), target: 'web',
      entry: path.resolve('scripts/learning/micro-macro-panels-browser-harness.tsx'),
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
  let browser; const errors = [], cases = [];
  try {
    browser = await chromium.launch({ headless: true, args: ['--disable-dev-shm-usage'] });
    const context = await browser.newContext(); const page = await context.newPage(); page.on('pageerror', error => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const kinds = await page.evaluate(() => Object.keys(window.microMacroPanelPresets));
    const ready = (kind, key) => page.waitForFunction(({ kind, key }) => {
      const lab = document.querySelector('[data-micro-macro-lab]'); return lab?.getAttribute('data-micro-macro-lab') === kind && lab.getAttribute('data-micro-macro-current') === key;
    }, { kind, key }, { timeout: 60000 });
    for (const size of [320, 1440]) for (const dark of [false, true]) for (const language of ['ko', 'en']) {
      await page.setViewportSize({ width: size, height: 900 });
      await page.evaluate(dark => document.documentElement.classList.toggle('dark', dark), dark);
      await page.getByTestId(`language-${language}`).click();
      for (const [index, kind] of kinds.entries()) {
        await page.getByTestId(`kind-${index}`).click();
        const lab = page.locator('[data-micro-macro-lab]');
        await page.waitForFunction(() => document.querySelector('[data-micro-macro-lab]')?.getAttribute('data-micro-macro-ready') === 'true');
        const presets = await page.evaluate(kind => window.microMacroPanelPresets[kind], kind);
        for (const [preset, p] of presets.entries()) {
          await lab.getByRole('button', { name: p.label[language], exact: true }).click(); await ready(kind, p.key);
          const metrics = await lab.evaluate(el => ({ hangul: /[가-힣]/.test(el.textContent), overflow: document.documentElement.scrollWidth > innerWidth + 1,
            invalidSvg: [...el.querySelectorAll('svg *')].some(node => [...node.attributes].some(a => /NaN|Infinity/.test(a.value))) }));
          assert.equal(metrics.overflow, false, JSON.stringify({ kind, preset, size, dark, language, metrics }));
          assert.equal(metrics.invalidSvg, false); if (language === 'en') assert.equal(metrics.hangul, false);
          const axe = await new AxeBuilder({ page }).include('[data-micro-macro-lab]').analyze();
          assert.deepEqual(axe.violations.map(v => ({ id: v.id, nodes: v.nodes.map(n => n.target) })), [], JSON.stringify({ kind, preset, size, dark, language }));
          cases.push({ kind, preset, size, dark, language });
          await fs.writeFile(path.join(root, 'progress.json'), JSON.stringify({ cases, errors }, null, 2));
        }
        console.log(JSON.stringify({ verified: cases.length, size, dark, language, kind }));
      }
    }
    await fs.writeFile(path.join(root, 'matrix.json'), JSON.stringify({ scope: 'Isolated real panels with existing exported CSS and webpack worker assets; actual Next readers not yet registered', cases, errors, workerAssets, cssFiles }, null, 2));
    assert.equal(cases.length, 160); assert.deepEqual(errors, []);
  } finally { if (browser) await browser.close(); await new Promise(resolve => server.close(resolve)); }
})().catch(error => { console.error(error); process.exitCode = 1; });
