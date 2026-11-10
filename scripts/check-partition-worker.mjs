/* Isolated browser/hook/worker validation. This is not the Next reader export. */
import webpackPackage from 'next/dist/compiled/webpack/webpack.js';
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
webpackPackage.init();
const webpack = webpackPackage.webpack;
const root = path.resolve('.runtime/learning/partition-worker/browser');
(async () => {
  await fs.mkdir(root, { recursive: true });
  const loader = path.join(root, 'transform.cjs');
  await fs.writeFile(loader, `module.exports=function(source){return require(${JSON.stringify(fileURLToPath(import.meta.resolve('esbuild')))}).transformSync(source,{loader:this.resourcePath.endsWith('.tsx')?'tsx':'ts',format:'esm',target:'es2022',jsx:'automatic'}).code};`);
  const stats = await new Promise((resolve, reject) => {
    const compiler = webpack({ mode: 'development', devtool: false, context: process.cwd(), target: 'web',
      entry: path.resolve('scripts/learning/partition-worker-browser-harness.tsx'),
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
  const assets = new Set(stats.assets.map(asset => asset.name).filter(name => name.endsWith('.js')));
  const workerAssets = [...assets].filter(name => name.startsWith('worker-'));
  assert.equal(workerAssets.length, 1);
  const workerSource = await fs.readFile(path.join(root, workerAssets[0]), 'utf8');
  for (const name of ['partitionCumulantModel', 'coupledPartitionModel', 'energyLadderModel']) assert.ok(workerSource.includes('function ' + name + '('));
  const clientSource = await fs.readFile(path.join(root, 'app.js'), 'utf8');
  assert.doesNotMatch(clientSource, /function (?:partitionCumulantModel|coupledPartitionModel|energyLadderModel)\(/);
  const server = http.createServer(async (request, response) => {
    const candidate = (request.url || '').split('?')[0].slice(1);
    const file = assets.has(candidate) ? candidate : null;
    response.setHeader('Content-Type', file ? 'application/javascript; charset=utf-8' : 'text/html; charset=utf-8');
    response.end(file ? await fs.readFile(path.join(root, file)) : '<!doctype html><html lang="en"><title>Worker verification</title><body><div id="root"></div><script type="module" src="/app.js"></script></body></html>');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  const errors = [], workerRequests = [];
  try {
    browser = await chromium.launch({ headless: true, args: ['--disable-dev-shm-usage'] });
    const page = await browser.newPage();
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => { if (workerAssets.some(asset => request.url().endsWith('/' + asset))) workerRequests.push(request.url()); });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const latest = () => page.evaluate(() => structuredClone(window.partitionProbe));
    const ready = (kind, size = 0) => page.waitForFunction(({ kind, size }) => {
      const p = window.partitionProbe; return p.latest.status === 'ready' && p.latest.kind === kind && (!size || p.latest.size === size);
    }, { kind, size }, { timeout: 60000 });
    await ready('finite-spectrum-partition-cumulant-generators', 3);
    await page.getByTestId('canonical').click(); await ready('coupled-binary-sites-partition-factorization');
    await page.getByTestId('grand').click(); await ready('infinite-energy-ladder-normalization-cutoff', 8);
    await page.getByTestId('heavy').click();
    await page.waitForFunction(() => window.partitionProbe.latest.status === 'pending' && window.partitionProbe.latest.phase === 'calculating' && window.partitionProbe.live === 1);
    const heavyStart = await latest();
    await page.getByTestId('click').click();
    await page.waitForFunction(previous => window.partitionProbe.clicks === 1 && window.partitionProbe.heartbeats > previous + 4, heavyStart.heartbeats);
    const during = await latest(); assert.equal(during.latest.status, 'pending'); assert.equal(during.started, heavyStart.started);
    await page.getByTestId('reservoir').click(); await ready('finite-spectrum-partition-cumulant-generators', 3);
    const replaced = await latest(); assert.equal(replaced.live, 0); assert.equal(replaced.latest.size, 3); assert.equal(replaced.staleResults, 0);
    await page.getByTestId('heavy').click();
    await page.waitForFunction(() => window.partitionProbe.latest.status === 'pending' && window.partitionProbe.latest.phase === 'calculating' && window.partitionProbe.live === 1);
    await page.getByTestId('enabled').click(); await page.waitForFunction(() => window.partitionProbe.latest.status === 'idle' && window.partitionProbe.live === 0);
    await page.getByTestId('enabled').click(); await page.waitForFunction(() => window.partitionProbe.latest.phase === 'calculating' && window.partitionProbe.live === 1);
    await page.getByTestId('cancel').click(); await page.waitForFunction(() => window.partitionProbe.latest.status === 'idle' && window.partitionProbe.live === 0);
    await page.getByTestId('retry').click(); await page.waitForFunction(() => window.partitionProbe.latest.status === 'pending' && window.partitionProbe.latest.phase === 'calculating' && window.partitionProbe.live === 1);
    await page.getByTestId('toggle').click(); await page.waitForFunction(() => window.partitionProbe.live === 0);
    await page.getByTestId('grand').click(); await page.getByTestId('toggle').click(); await ready('infinite-energy-ladder-normalization-cutoff', 8);
    const beforeCompletion = await latest();
    await page.getByTestId('heavy').click(); await ready('infinite-energy-ladder-normalization-cutoff', 4096);
    const completed = await latest(); assert.equal(completed.live, 0); assert.ok(completed.latest.value > .95 && completed.latest.value < .97);
    assert.equal(completed.staleResults, 0); assert.ok(completed.heartbeats - beforeCompletion.heartbeats > 4);
    await page.context().route('**/worker-*.js', route => route.abort());
    await page.getByTestId('reservoir').click();
    await page.waitForFunction(() => window.partitionProbe.latest.status === 'error' && window.partitionProbe.live === 0);
    const loadFailure = await latest();
    await page.context().unroute('**/worker-*.js');
    await page.getByTestId('retry').click(); await ready('finite-spectrum-partition-cumulant-generators', 3);
    const recovered = await latest();
    await page.getByTestId('toggle').click(); await page.waitForFunction(() => window.partitionProbe.live === 0);
    assert.deepEqual(errors, []); assert.ok(workerRequests.length >= 3);
    const afterUnmount = await latest(); assert.equal(afterUnmount.started, afterUnmount.terminated);
    const result = { afterUnmount, scope: 'Isolated webpack-built actual-browser React StrictMode + separate worker asset; Next static export and reader integration still pending', webpackWorkerAssets: workerAssets, mainBundleContainsCalculatorFunctions: false,
      heavyStarted: heavyStart.started, duringHeavy: during, afterReplacement: replaced, afterHeavyCompletion: completed, afterBlockedWorkerLoad: loadFailure, afterRecovery: recovered,
      workerAssetRequests: workerRequests.length, runtimeErrors: errors };
    await fs.writeFile(path.join(root, 'result.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify({ liveWorkers: completed.live, staleResults: completed.staleResults, heartbeats: completed.heartbeats,
      maxHeartbeatGapMs: completed.maxGap, requests: workerRequests.length, errors: errors.length }));
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
