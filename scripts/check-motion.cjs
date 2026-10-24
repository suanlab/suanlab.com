const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('playwright');
const { settleMotion } = require('./browser-motion.cjs');
const base = process.env.SUANLAB_TEST_BASE || 'http://127.0.0.1:4175';

(async () => {
  fs.mkdirSync('.runtime', { recursive: true });
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox'] });
  const errors = [], scenarios = [];
  async function context(options = {}, initialize = true) {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, ...options });
    if (initialize) await ctx.addInitScript(() => { try { localStorage.setItem('suanlab-language', 'en'); } catch {} });
    await ctx.route('**/*', route => new URL(route.request().url()).origin === new URL(base).origin ? route.continue() : route.abort());
    ctx.on('page', page => {
      page.on('pageerror', error => errors.push({ route: page.url(), message: error.message }));
      page.on('console', message => {
        if (message.type() === 'error' && /Hydration|did not match|matching|Encountered two children/.test(message.text())) errors.push({ route: page.url(), message: message.text() });
      });
    });
    return ctx;
  }
  async function go(page, route) {
    const response = await page.goto(base + route);
    assert.equal(response.status(), 200, route);
    await page.waitForFunction(() => document.documentElement.lang === 'en');
    await page.evaluate(() => document.fonts.ready);
  }
  async function settle(page) {
    await settleMotion(page);
  }
  try {
    const live = await context({ reducedMotion: 'no-preference' });
    const page = await live.newPage();
    await go(page, '/');
    await page.waitForFunction(() => document.documentElement.dataset.siteMotion === 'running');
    await page.locator('.identity-hero').getByRole('button', { name: 'Pause site motion', exact: true }).click();
    await page.waitForFunction(() => document.documentElement.dataset.siteMotion === 'paused' && document.querySelector('.identity-hero')?.dataset.motion === 'paused');
    assert.equal(await page.evaluate(() => localStorage.getItem('suanlab-motion-paused')), 'true');
    assert.ok(await page.locator('.identity-hero').innerText());
    await go(page, '/research/');
    await page.waitForFunction(() => document.documentElement.dataset.siteMotion === 'paused');
    await page.getByRole('button', { name: 'Play site motion', exact: true }).first().click();
    await page.waitForFunction(() => document.documentElement.dataset.siteMotion === 'running');

    for (const index of [0, 1, 2, 3, 1]) {
      const step = page.locator(`[data-story-step="${index}"]`);
      await step.evaluate(node => node.scrollIntoView({ block: 'center' }));
      await page.waitForFunction(index => document.querySelector(`[data-story-step="${index}"]`)?.dataset.active === 'true', index);
    }
    await settle(page);
    await page.screenshot({ path: '.runtime/motion-research-desktop.png' });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), 1440);
    scenarios.push('Global pause persists across routes; scroll stages follow both scroll directions.');

    await go(page, '/publication/');
    const bars = page.locator('[data-publication-timeline] button[data-count]');
    const total = await bars.evaluateAll(nodes => nodes.reduce((sum, node) => sum + Number(node.dataset.count), 0));
    assert.equal(total, 293);
    const year = page.locator('[data-publication-timeline] button[data-year="2026"]');
    const count = Number(await year.getAttribute('data-count'));
    await year.click();
    await page.waitForFunction(() => document.querySelectorAll('[id^="publication-"].motion-card').length === Number(document.querySelector('button[data-year="2026"]').dataset.count));
    assert.equal(await page.locator('[id^="publication-"].motion-card').count(), count);
    assert.equal(await page.locator('label:has(select)').last().locator('select').inputValue(), '2026');
    await page.getByRole('button', { name: 'All years', exact: true }).click();
    await page.waitForFunction(() => document.querySelectorAll('[id^="publication-"].motion-card').length === 293);
    const input = page.getByRole('textbox', { name: '검색 / Search', exact: true });
    await input.fill('UrbanReplayBench');
    await page.locator('#publication-658').waitFor();
    assert.equal(await input.evaluate(node => node === document.activeElement), true);
    assert.equal(await page.locator('[id^="publication-"].motion-card').count(), 1);
    await settle(page);
    await page.screenshot({ path: '.runtime/motion-publication-desktop.png' });
    scenarios.push('Timeline totals match all 293 publications; year filtering and search retain focus.');

    await go(page, '/suan/');
    await page.getByRole('img', { name: 'Visited countries world map', exact: true }).waitFor();
    const select = page.getByRole('combobox', { name: 'Trip / conference route', exact: true });
    assert.equal(await select.inputValue(), '39');
    assert.equal(await page.locator('.map-route').count(), 1);
    await select.selectOption('35');
    assert.equal(await page.locator('.map-route').count(), 0);
    await select.selectOption('2');
    assert.equal(await page.locator('.map-route').count(), 2);
    await page.locator('.visited-map').scrollIntoViewIfNeeded();
    await settle(page);
    await page.screenshot({ path: '.runtime/motion-map-desktop.png' });
    scenarios.push('Map connects countries from selected trip records; single-country trips draw no route.');

    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.waitForFunction(() => document.documentElement.dataset.siteMotion === 'paused');
    assert.equal(await page.getByRole('button', { name: 'System setting: reduced motion', exact: true }).isDisabled(), true);
    await go(page, '/');
    await page.waitForFunction(() => document.querySelector('.identity-hero')?.dataset.motion === 'paused');
    await page.locator('[data-motion-count]').first().scrollIntoViewIfNeeded();
    assert.equal(await page.locator('[data-motion-count]').first().innerText(), '293');
    assert.equal(await page.evaluate(() => document.getAnimations().filter(animation => animation.playState === 'running' && animation.effect?.getTiming().iterations === Infinity).length), 0);
    scenarios.push('Changing the OS motion preference stops animations and leaves final readable counts.');
    await live.close();

    const blocked = await context();
    await blocked.addInitScript(() => {
      const read = Storage.prototype.getItem, write = Storage.prototype.setItem;
      Storage.prototype.getItem = function(key) { if (key === 'suanlab-motion-paused') throw new DOMException('Denied', 'SecurityError'); return read.call(this, key); };
      Storage.prototype.setItem = function(key, value) { if (key === 'suanlab-motion-paused') throw new DOMException('Denied', 'SecurityError'); return write.call(this, key, value); };
    });
    const blockedPage = await blocked.newPage();
    await go(blockedPage, '/');
    await blockedPage.locator('.identity-hero').getByRole('button', { name: 'Pause site motion', exact: true }).click();
    await blockedPage.waitForFunction(() => document.documentElement.dataset.siteMotion === 'paused');
    await blocked.close();
    scenarios.push('Denied preference storage does not break the motion control.');

    const staticContext = await context({ javaScriptEnabled: false }, false);
    const staticPage = await staticContext.newPage();
    for (const route of ['/', '/research/', '/publication/']) {
      assert.equal((await staticPage.goto(base + route)).status(), 200);
      assert.ok(await staticPage.locator('#main-content h1').isVisible());
      assert.ok(await staticPage.locator('#main-content').innerText());
      if (route === '/research/') assert.equal(await staticPage.locator('[data-story-step]').count(), 4);
      if (route === '/publication/') assert.equal(await staticPage.locator('[id^="publication-"].motion-card').count(), 293);
    }
    await staticContext.close();
    scenarios.push('Static HTML retains content, research stages and all publication records without JavaScript.');

    fs.writeFileSync('.runtime/motion-check.json', JSON.stringify({ scenarios, errors }, null, 2));
    assert.deepEqual(errors, [], 'See .runtime/motion-check.json for runtime errors.');
    console.log(`Motion acceptance passed: ${scenarios.length} scenarios, zero runtime or hydration errors.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
