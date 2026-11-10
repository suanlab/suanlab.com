/* Real static-export readers on the existing preview server; no app mocks. */
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
const root = '.runtime/learning/ensemble-integration';
const base = 'http://127.0.0.1:4175';
const { presets } = JSON.parse(await fs.readFile(`${root}/specs.json`, 'utf8'));
const kinds = Object.keys(presets);
const browser = await chromium.launch({ headless: true });
const result = { errors: [], checks: [] };
try {
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 }, reducedMotion: 'no-preference' });
  await context.addInitScript(() => {
    localStorage.setItem('suanlab-language', 'en');
    localStorage.setItem('theme', 'light');
    const probe = { started: 0, acknowledged: 0, terminated: 0, live: 0, ticks: 0 };
    window.readerWorkerProbe = probe;
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      constructor(url, options) {
        super(url, options); probe.started++; probe.live++; this.stopped = false;
        this.addEventListener('message', event => { if (event.data?.phase === 'started') probe.acknowledged++; });
      }
      terminate() { if (!this.stopped) { this.stopped = true; probe.terminated++; probe.live--; } super.terminate(); }
    };
    setInterval(() => probe.ticks++, 16);
  });
  const page = await context.newPage();
  page.on('pageerror', error => result.errors.push(error.message));
  const lab = page.locator('[data-ensemble-lab]');
  const ready = (kind, key) => page.waitForFunction(({ kind, key }) => {
    const el = document.querySelector('[data-ensemble-lab]');
    return el?.getAttribute('data-ensemble-lab') === kind && el.getAttribute('data-ensemble-current') === key;
  }, { kind, key }, { timeout: 90000 });
  await page.goto(`${base}/book/online/physics-math-for-ai/${kinds[0]}/`);
  await ready(kinds[0], presets[kinds[0]][0].key);
  const workerUrls = new Set(); page.on('worker', worker => workerUrls.add(worker.url()));
  await page.locator('.ai-concept-steps button').nth(3).click();
  await page.waitForFunction(() => document.querySelector('[data-ensemble-concept]')?.getAttribute('data-ensemble-concept-state') === 'ready');
  assert.ok(workerUrls.size > 0);
  const blockedUrl = [...workerUrls][0]; await context.route(blockedUrl, route => route.abort());
  await page.locator('.ai-concept-steps button').nth(2).click();
  await page.waitForFunction(() => document.querySelector('[data-ensemble-concept]')?.getAttribute('data-ensemble-concept-state') === 'error');
  assert.equal(await page.locator('[data-ensemble-concept-result]').count(), 0);
  await context.unroute(blockedUrl);
  await page.locator('[data-ensemble-concept]').getByRole('button', { name: 'Retry calculation', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('[data-ensemble-concept]')?.getAttribute('data-ensemble-concept-state') === 'ready');
  await page.locator('.ai-concept-steps button').nth(3).click();
  await page.locator('.ai-concept-steps button').nth(0).click();
  await page.waitForFunction(() => document.querySelector('[data-ensemble-concept]')?.getAttribute('data-ensemble-step') === '0' && document.querySelector('[data-ensemble-concept]')?.getAttribute('data-ensemble-concept-state') === 'ready');
  await page.waitForFunction(() => window.readerWorkerProbe.live === 0);
  result.checks.push('fixed series worker-load error, recovery and rapid step replacement without stale results');
  await lab.getByRole('button', { name: 'Scan comparison examples', exact: true }).scrollIntoViewIfNeeded();
  await lab.getByRole('button', { name: 'Scan comparison examples', exact: true }).click();
  await page.waitForFunction(() => Number(document.querySelector('[data-ensemble-lab]')?.getAttribute('data-ensemble-frame')) > 0);
  await page.locator('footer').getByRole('button', { name: 'Pause site motion', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('[data-ensemble-lab]')?.getAttribute('data-ensemble-playing') === 'false');
  const frame = Number(await lab.getAttribute('data-ensemble-frame'));
  await lab.getByRole('button', { name: 'Step comparison example', exact: true }).click();
  await ready(kinds[0], presets[kinds[0]][frame + 1].key);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  assert.equal(await lab.getByRole('button', { name: 'Scan comparison examples', exact: true }).isDisabled(), true);
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.locator('footer').getByRole('button', { name: 'Play site motion', exact: true }).click();
  await lab.getByRole('button', { name: 'Reset all experiment settings', exact: true }).click();
  await ready(kinds[0], presets[kinds[0]][0].key);
  await lab.getByRole('button', { name: 'Scan comparison examples', exact: true }).scrollIntoViewIfNeeded();
  await lab.getByRole('button', { name: 'Scan comparison examples', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('[data-ensemble-lab]')?.getAttribute('data-ensemble-playing') === 'true');
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForFunction(() => document.querySelector('[data-ensemble-lab]')?.getAttribute('data-ensemble-playing') === 'false');
  result.checks.push('automatic comparison, global pause, manual step, reduced motion, actual offscreen pause');
  // Use client-side navigation to prove disposal rather than closing a document.
  await page.locator('main a[href="/book/interactive/"]').first().click();
  await page.waitForURL('**/book/interactive/');
  await page.locator(`main a[href="/book/online/physics-math-for-ai/"]`).click();
  await page.waitForURL('**/book/online/physics-math-for-ai/');
  await page.locator(`main a.ai-unit-card[href="/book/online/physics-math-for-ai/${kinds[0]}/"]`).first().click();
  await ready(kinds[0], presets[kinds[0]][0].key);
  await page.waitForFunction(() => window.readerWorkerProbe.live === 0);
  const parameters = { subsystemSites: 2048, reservoirSites: 2048, totalExcitations: 1 };
  await lab.locator('summary').filter({ hasText: 'Extended parameters' }).click();
  await lab.getByRole('textbox', { name: 'Extended parameter JSON' }).fill(JSON.stringify(parameters));
  const before = await page.evaluate(() => ({ ...window.readerWorkerProbe }));
  const stopped = await lab.evaluate(el => new Promise(resolve => {
    [...el.querySelectorAll('button')].find(b => b.textContent === 'Apply extended settings').click();
    requestAnimationFrame(() => {
      const stop = [...el.querySelectorAll('button')].find(b => b.textContent === 'Stop calculation');
      stop?.click(); resolve(Boolean(stop));
    });
  }));
  assert.equal(stopped, true);
  await page.waitForTimeout(350);
  assert.equal(await lab.getAttribute('data-ensemble-state'), 'idle');
  assert.equal(await page.evaluate(() => window.readerWorkerProbe.started), before.started);
  await lab.getByRole('button', { name: 'Retry calculation', exact: true }).click();
  await page.waitForFunction(previous => window.readerWorkerProbe.acknowledged > previous && window.readerWorkerProbe.live === 1, before.acknowledged);
  assert.equal(await lab.locator('[data-ensemble-result]').count(), 0);
  const ticks = await page.evaluate(() => window.readerWorkerProbe.ticks);
  await page.locator('header').getByRole('button', { name: 'KO', exact: true }).click();
  await page.locator('header').getByRole('button', { name: '테마 전환', exact: true }).click();
  await page.waitForFunction(previous => window.readerWorkerProbe.ticks > previous + 4 && document.documentElement.lang === 'ko' && document.documentElement.classList.contains('dark'), ticks);
  await page.locator('main a[href="/book/interactive/"]').first().click();
  await page.waitForURL('**/book/interactive/');
  await page.waitForFunction(() => window.readerWorkerProbe.live === 0 && !document.querySelector('[data-ensemble-lab]'));
  result.checks.push('debounce stop without worker, started heavy job, responsive language/theme, client navigation disposes worker');
  await page.locator('header').getByRole('button', { name: 'EN', exact: true }).click();
  await page.locator('main a[href="/book/online/physics-math-for-ai/"]').click();
  await page.waitForURL('**/book/online/physics-math-for-ai/');
  await page.locator(`main a.ai-unit-card[href="/book/online/physics-math-for-ai/${kinds[0]}/"]`).first().click();
  await ready(kinds[0], JSON.stringify({ kind: kinds[0], parameters }));
  await lab.locator('.ai-presets button').nth(0).click();
  await ready(kinds[0], presets[kinds[0]][0].key);
  await page.waitForTimeout(500);
  assert.equal(await lab.getAttribute('data-ensemble-current'), presets[kinds[0]][0].key);
  await page.waitForFunction(() => window.readerWorkerProbe.live === 0);
  result.checks.push('saved heavy calculation completes, replacement remains current, all workers terminated');
  result.workerProbe = await page.evaluate(() => ({ ...window.readerWorkerProbe }));
  assert.equal(result.workerProbe.started, result.workerProbe.terminated);
  assert.deepEqual(result.errors, []);
  await fs.writeFile(`${root}/reader-interactions-result.json`, JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result));
  await context.close();
} finally { await browser.close(); }
