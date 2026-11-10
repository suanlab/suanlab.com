/* Actual emitted Next worker, heavy saved preparation and client navigation. */
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
const root = '.runtime/learning/boltzmann-gibbs-integration', base = 'http://127.0.0.1:4175';
const { presets } = JSON.parse(await fs.readFile(`${root}/specs.json`, 'utf8'));
const kind = 'finite-gibbs-conditionals-scan-support', book = '/book/online/physics-math-for-ai/';
const heavy = { ...presets[kind][0].input, parameters: { ...presets[kind][0].input.parameters,
  couplingEnergy: Number.MIN_VALUE, firstFieldEnergy: 800 * 1.380649e-23 * Number.MAX_VALUE,
  secondFieldEnergy: 800 * 1.380649e-23 * Number.MAX_VALUE, temperature: Number.MAX_VALUE, observationSteps: [64] } };
const browser = await chromium.launch({ headless: true });
const result = { scope: 'actual exported Next reader and native worker assets', checks: [], errors: [], workerUrls: [] };
try {
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 }, reducedMotion: 'reduce' });
  await context.route('**/*', route => new URL(route.request().url()).origin === base ? route.continue() : route.abort());
  await context.addInitScript(({ kind, heavy }) => {
    localStorage.setItem('suanlab-language', 'en'); localStorage.setItem('theme', 'light');
    localStorage.setItem(`suanlab-boltzmann-${kind}`, JSON.stringify({ version: 1, value: heavy }));
    const p = { started: 0, terminated: 0, live: 0, heavyAcknowledged: 0, heavyLive: 0, heavyCompleted: 0, heavyCancelled: 0, ticks: 0, staleFrames: 0 };
    window.nativeBoltzmannLifecycle = p;
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      constructor(url, options) {
        super(url, options); this.stopped = false; this.heavy = false; p.started++; p.live++;
        this.addEventListener('message', event => {
          if (this.heavy && event.data?.phase === 'started') p.heavyAcknowledged++;
          if (this.heavy && event.data?.ok === true) { this.completed = true; p.heavyCompleted++; }
        });
      }
      postMessage(message, ...rest) {
        this.heavy = message?.input?.parameters?.temperature === Number.MAX_VALUE;
        if (this.heavy) p.heavyLive++;
        return super.postMessage(message, ...rest);
      }
      terminate() { if (!this.stopped) { this.stopped = true; p.terminated++; p.live--; if (this.heavy) { p.heavyLive--; if (!this.completed) p.heavyCancelled++; } } super.terminate(); }
    };
    setInterval(() => p.ticks++, 16);
    new MutationObserver(() => {
      const lab = document.querySelector('[data-boltzmann-lab]'), concept = document.querySelector('[data-boltzmann-concept]');
      if (lab && lab.dataset.boltzmannState !== 'ready' && lab.querySelector('[data-boltzmann-result]')) p.staleFrames++;
      if (concept && concept.dataset.boltzmannConceptState !== 'ready' && concept.querySelector('[data-boltzmann-concept-result]')) p.staleFrames++;
    }).observe(document, { subtree: true, childList: true, attributes: true });
  }, { kind, heavy });
  const page = await context.newPage(); page.on('pageerror', e => result.errors.push(e.message));
  page.on('worker', worker => { if (!result.workerUrls.includes(worker.url())) result.workerUrls.push(worker.url()); });
  const lab = page.locator('[data-boltzmann-lab]');
  const ready = () => page.locator('[data-boltzmann-state="ready"]').waitFor({ timeout: 120000 });
  await page.goto(`${base}${book}${kind}/`);
  await page.waitForFunction(() => window.nativeBoltzmannLifecycle.heavyAcknowledged > 0 && window.nativeBoltzmannLifecycle.heavyLive > 0);
  const ticks = await page.evaluate(() => window.nativeBoltzmannLifecycle.ticks);
  await page.locator('header').getByRole('button', { name: 'KO', exact: true }).click();
  await page.locator('header').getByRole('button', { name: '테마 전환', exact: true }).click();
  await page.waitForFunction(ticks => window.nativeBoltzmannLifecycle.ticks > ticks + 4 && document.documentElement.lang === 'ko' && document.documentElement.classList.contains('dark'), ticks);
  await page.locator('main a[href="/book/interactive/"]').first().click(); await page.waitForURL('**/book/interactive/');
  await page.waitForFunction(() => window.nativeBoltzmannLifecycle.live === 0 && !document.querySelector('[data-boltzmann-lab]'));
  result.checks.push('heavy saved request acknowledges in the actual worker; language/theme and event loop respond; client navigation disposes workers');
  await page.locator('header').getByRole('button', { name: 'EN', exact: true }).click();
  await page.locator(`main a[href="${book}"]`).click(); await page.waitForURL('**' + book);
  await page.locator(`main a.ai-unit-card[href="${book}${kind}/"]`).click();
  await ready(); assert.equal(JSON.parse(await lab.getAttribute('data-boltzmann-current')).parameters.temperature, Number.MAX_VALUE);
  await lab.locator('.ai-presets button').nth(0).click(); await ready();
  await page.waitForFunction(key => document.querySelector('[data-boltzmann-lab]').dataset.boltzmannCurrent === key, presets[kind][0].key);
  result.checks.push('saved heavy preparation completes and replacement exposes only the matching ordinary preparation');

  const workerUrl = result.workerUrls[0]; assert.ok(workerUrl.startsWith(base + '/_next/static/'));
  await context.route(workerUrl, route => route.abort());
  await lab.locator('.ai-presets button').nth(1).click(); await page.locator('[data-boltzmann-state="error"]').waitFor();
  assert.equal(await lab.locator('[data-boltzmann-result]').count(), 0);
  await context.unroute(workerUrl);
  await lab.getByRole('button', { name: 'Retry calculation', exact: true }).click(); await ready();
  result.checks.push('actual worker asset load failure hides previous output and native retry recovers');

  await context.route(workerUrl, route => route.abort());
  await page.locator('.ai-concept-steps button').nth(2).click(); await page.locator('[data-boltzmann-concept-state="error"]').waitFor();
  assert.equal(await page.locator('[data-boltzmann-concept-result]').count(), 0);
  await context.unroute(workerUrl);
  await page.locator('[data-boltzmann-concept]').getByRole('button', { name: 'Retry calculation', exact: true }).click();
  await page.locator('[data-boltzmann-concept-state="ready"]').waitFor();
  await page.locator('.ai-concept-steps button').nth(3).click(); await page.locator('.ai-concept-steps button').nth(0).click();
  await page.locator('[data-boltzmann-step="0"][data-boltzmann-concept-state="ready"]').waitFor();
  await page.waitForFunction(() => window.nativeBoltzmannLifecycle.live === 0);
  result.checks.push('fixed-series asset failure, explicit recovery and rapid step replacement never expose stale or partial preparations');
  result.metrics = await page.evaluate(() => ({ ...window.nativeBoltzmannLifecycle }));
  assert.equal(result.metrics.started, result.metrics.terminated); assert.equal(result.metrics.heavyLive, 0); assert.equal(result.metrics.staleFrames, 0);
  assert.ok(result.metrics.heavyCancelled >= 1, 'Client navigation must dispose an acknowledged heavy request before its result');
  assert.ok(result.metrics.heavyCompleted >= 1, 'The restored heavy request must also be allowed to complete');
  assert.deepEqual(result.errors, []);
  await fs.writeFile(`${root}/reader-lifecycle-result.json`, JSON.stringify(result, null, 2) + '\n'); console.log(JSON.stringify(result));
  await context.close();
} finally { await browser.close(); }
