/* Verify the actual exported discovery paths without changing the preview server. */
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
const root = '.runtime/learning/boltzmann-gibbs-integration', base = 'http://127.0.0.1:4175';
const { teaching } = JSON.parse(await fs.readFile(`${root}/specs.json`, 'utf8'));
const browser = await chromium.launch({ headless: true });
const result = { menuCases: 0, breadcrumbs: 0, searches: 0, library: false, onlineBookExcluded: false, sitemap: false, errors: [] };
try {
  for (const language of ['ko', 'en']) {
    const context = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
    await context.addInitScript(language => localStorage.setItem('suanlab-language', language), language);
    const page = await context.newPage(); page.on('pageerror', e => result.errors.push(e.message));
    for (const kind of Object.keys(teaching)) {
      await page.goto(`${base}/book/online/physics-math-for-ai/${kind}/`);
      await page.waitForFunction(language => document.documentElement.lang === language, language);
      const crumb = page.locator('main a[href="/book/interactive/"]').first(); await crumb.waitFor(); result.breadcrumbs++;
      const parent = page.locator('header nav a[href="/book/"]'); await parent.focus(); await parent.press('ArrowDown');
      assert.equal(await page.getByRole('menuitem', { name: 'Interactive Book', exact: true }).getAttribute('aria-current'), 'page');
      assert.equal(await page.getByRole('menuitem', { name: 'Online Book', exact: true }).getAttribute('aria-current'), null);
      await page.keyboard.press('Escape'); result.menuCases++;
      await page.goto(`${base}/search/?q=${encodeURIComponent(teaching[kind].unit.title[language])}`);
      await page.locator(`main a[href="/book/online/physics-math-for-ai/${kind}/"]`).first().waitFor(); result.searches++;
    }
    await page.goto(`${base}/book/interactive/`); await page.locator('main a[href="/book/online/physics-math-for-ai/"]').waitFor(); result.library = true;
    await page.goto(`${base}/book/online/`); assert.equal(await page.locator('main a[href*="physics-math-for-ai"]').count(), 0); assert.ok(await page.locator('main a[href^="/book/online/"]').count()); result.onlineBookExcluded = true;
    await page.goto(`${base}/book/online/physics-math-for-ai/`);
    for (const kind of Object.keys(teaching)) assert.ok(await page.locator(`main a.ai-unit-card[href="/book/online/physics-math-for-ai/${kind}/"]`).count());
    await page.setViewportSize({ width: 320, height: 900 });
    await page.getByRole('button', { name: language === 'en' ? 'Open menu' : '메뉴 열기', exact: true }).click();
    await page.getByRole('dialog').getByRole('link', { name: 'Interactive Book', exact: true }).click(); await page.waitForURL('**/book/interactive/'); result.menuCases++;
    await context.close();
  }
  const xml = await (await fetch(base + '/sitemap.xml')).text();
  assert.ok(xml.includes('/book/interactive/')); for (const kind of Object.keys(teaching)) assert.ok(xml.includes(`/book/online/physics-math-for-ai/${kind}/`)); result.sitemap = true;
  assert.deepEqual(result.errors, []);
  await fs.writeFile(`${root}/navigation-result.json`, JSON.stringify(result, null, 2) + '\n'); console.log(JSON.stringify(result));
} finally { await browser.close(); }
