// Run against a freshly built export served by npm start.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const { settleMotion } = require('./browser-motion.cjs');
const base = process.env.SUANLAB_TEST_BASE || 'http://127.0.0.1:4175';
const routes = ['/', '/suan/', '/research/', '/project/', '/publication/', '/lecture/',
  '/lecture/ai/present/', '/course/', '/youtube/', '/book/', '/blog/', '/search/',
  '/qt/', '/qt/2015-02-01/', '/prompts/', '/deadlines/'];

(async () => {
  fs.mkdirSync('.runtime', { recursive: true });
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox'] });
  const errors = [], layouts = [];
  async function context(language = 'en', options = {}) {
    const ctx = await browser.newContext({ viewport: { width: 1600, height: 900 }, ...options });
    await ctx.addInitScript(language => { try { localStorage.setItem('suanlab-language', language); } catch { /* Cross-origin frames can deny storage. */ } }, language);
    await ctx.route('**/*', route => new URL(route.request().url()).origin === new URL(base).origin ? route.continue() : route.abort());
    ctx.on('page', page => page.on('pageerror', error => errors.push({ route: page.url(), message: error.message })));
    return ctx;
  }
  async function go(page, route, language = 'en') {
    const response = await page.goto(base + route);
    assert.equal(response.status(), 200, route);
    await page.waitForFunction(language => document.documentElement.lang === language, language);
    await page.waitForFunction(() => !document.querySelector('header button[disabled]'));
    await page.evaluate(() => document.fonts.ready);
  }
  try {
    if (!process.env.SUANLAB_RESPONSIVE_ONLY) {
    const system = await context('en', { colorScheme: 'dark' });
    const page = await system.newPage();
    await go(page, '/');
    await page.waitForFunction(() => document.documentElement.classList.contains('dark'));
    await page.getByRole('button', { name: 'Toggle theme', exact: true }).click();
    await page.waitForFunction(() => !document.documentElement.classList.contains('dark'));
    assert.equal(await page.evaluate(() => localStorage.getItem('theme')), 'light');
    assert.ok(!/[가-힣]/.test(await page.locator('.identity-slide h2').allTextContents().then(text => text.join(''))));
    assert.ok(!/[가-힣]/.test(await page.locator('footer').innerText()));
    assert.ok(await page.locator('a[href*="publication-658"]').count(), 'UrbanReplayBench homepage card');

    // Parent links retain Enter navigation; submenu keys never access pooled events.
    const research = page.locator('header nav').getByRole('link', { name: 'RESEARCH', exact: true });
    await research.focus(); await research.press('ArrowDown');
    await page.waitForFunction(() => document.activeElement?.getAttribute('role') === 'menuitem');
    const menu = page.getByRole('menu', { name: 'RESEARCH', exact: true });
    const items = menu.getByRole('menuitem');
    assert.equal(await items.first().evaluate(node => node === document.activeElement), true);
    await page.keyboard.press('End');
    assert.equal(await items.last().evaluate(node => node === document.activeElement), true);
    await page.keyboard.press('ArrowDown');
    assert.equal(await items.first().evaluate(node => node === document.activeElement), true);
    await page.keyboard.press('ArrowUp');
    assert.equal(await items.last().evaluate(node => node === document.activeElement), true);
    await page.keyboard.press('Home');
    assert.equal(await items.first().evaluate(node => node === document.activeElement), true);
    await page.keyboard.press('Escape');
    assert.equal(await page.getByRole('button', { name: 'RESEARCH submenu', exact: true }).evaluate(node => node === document.activeElement), true);
    assert.equal(await menu.count(), 0);
    await research.focus(); await research.press('Enter');
    await page.waitForURL('**/research/');

    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole('button', { name: 'Open menu', exact: true }).click();
    await page.getByRole('dialog').getByRole('link', { name: 'BLOG', exact: true }).click();
    await page.waitForURL('**/blog/');
    await page.waitForFunction(() => !document.querySelector('[role="dialog"]') && getComputedStyle(document.body).pointerEvents !== 'none');
    await page.getByRole('button', { name: 'Open menu', exact: true }).click();
    await page.getByRole('dialog').getByRole('link', { name: 'BLOG', exact: true }).click();
    await page.getByRole('dialog').waitFor({ state: 'detached' });
    await system.close();
    console.log('Navigation passed: system theme, desktop keyboard, mobile route and same-route menu closure.');

    const lists = await context(); const listPage = await lists.newPage();
    await go(listPage, '/qt/');
    const john = listPage.getByRole('heading', { name: '요한복음', exact: true }).locator('xpath=ancestor::div[contains(@class,"rounded-xl")]');
    await john.getByRole('link', { name: /View \d+ more/ }).click();
    await listPage.waitForFunction(() => new URLSearchParams(location.search).get('book') === '요한복음');
    assert.equal(await listPage.locator('[role="tabpanel"] h3').count(), 1);
    assert.equal(await listPage.locator('[role="tabpanel"] a[href^="/qt/"]').count(), 120);
    await listPage.reload();
    await listPage.waitForFunction(() => document.querySelectorAll('[role="tabpanel"] a[href^="/qt/"]').length === 120);
    await listPage.goBack();
    await listPage.waitForFunction(() => document.querySelectorAll('[role="tabpanel"] h3').length > 1);
    await listPage.getByRole('textbox', { name: 'Search reflections', exact: true }).fill(' 하나님의 어린양 ');
    await listPage.locator('[role="tabpanel"] a[href^="/qt/2017-01-04"]').waitFor();
    assert.equal(await listPage.locator('[role="tabpanel"] a[href^="/qt/"]').count(), 1);
    for (const route of ['/qt/', '/qt/2015-02-01/']) {
      await go(listPage, route);
      await settleMotion(listPage);
      const audit = await new AxeBuilder({ page: listPage }).analyze();
      fs.writeFileSync(`.runtime/qt-${route === '/qt/' ? 'list' : 'detail'}-accessibility.json`, JSON.stringify(audit.violations, null, 2));
      assert.equal(audit.violations.length, 0, `QT accessibility: ${route}`);
      assert.equal(await listPage.locator('main').count(), 1);
    }

    // Follow actual detail links, then verify direct URLs, reload and history.
    await go(listPage, '/blog/20261005-paper-2609-37989-tabfm-auto-self-evolving-pipel/');
    const tagLink = listPage.locator('a[href^="/blog"][href*="tag="]').first();
    const tagHref = await tagLink.getAttribute('href');
    const tag = new URL(tagHref, base).searchParams.get('tag');
    const categoryHref = await listPage.locator('a[href^="/blog"][href*="category="]').first().getAttribute('href');
    const category = new URL(categoryHref, base).searchParams.get('category');
    await tagLink.click();
    await listPage.locator('aside button[aria-pressed="true"]').filter({ hasText: tag }).waitFor();
    assert.ok((await listPage.locator('p[role="status"]').innerText()).includes(tag));
    await go(listPage, categoryHref);
    await listPage.locator('aside button[aria-pressed="true"]').filter({ hasText: category }).waitFor();
    await go(listPage, '/blog/?page=2');
    await listPage.waitForFunction(() => document.querySelector('button[aria-current="page"]')?.textContent === '2');
    const second = await listPage.locator('a:has(h2)').first().getAttribute('href');
    await listPage.getByRole('button', { name: 'Next', exact: true }).click();
    await listPage.waitForFunction(() => new URLSearchParams(location.search).get('page') === '3');
    await listPage.goBack();
    await listPage.waitForFunction(() => document.querySelector('button[aria-current="page"]')?.textContent === '2');
    assert.equal(await listPage.locator('a:has(h2)').first().getAttribute('href'), second);
    await listPage.reload();
    await listPage.waitForFunction(() => document.querySelector('button[aria-current="page"]')?.textContent === '2');
    await listPage.getByRole('textbox', { name: 'Search blog', exact: true }).fill('zz-no-results-98462');
    await listPage.getByText('No results found.', { exact: true }).waitFor();
    await listPage.getByRole('button', { name: 'Clear filters', exact: true }).first().click();
    await listPage.waitForFunction(() => document.querySelector('button[aria-current="page"]')?.textContent === '1');
    console.log('List filters passed: QT full-book/search, blog detail links, pagination, reload, back and empty states.');

    await go(listPage, '/search/?q=RAG&type=prompt');
    await listPage.locator('a[href="/prompts/?builder=rag-system"]').waitFor();
    const snippetLink = listPage.locator('a[href^="/prompts/?snippet="]').first();
    const snippetHref = await snippetLink.getAttribute('href');
    const snippetId = new URL(snippetHref, base).searchParams.get('snippet');
    await listPage.locator('a[href="/prompts/?builder=rag-system"]').click();
    await listPage.getByRole('heading', { name: 'RAG Architecture & Retrieval Diagnosis', exact: true }).waitFor();
    await go(listPage, snippetHref);
    await listPage.waitForFunction(id => document.activeElement?.id === `prompt-${id}`, snippetId);
    assert.equal(await listPage.locator(`#prompt-${snippetId} button[aria-expanded]`).getAttribute('aria-expanded'), 'true');
    await go(listPage, '/prompts/?builder=unknown-id');
    await listPage.getByRole('tab', { name: /Builder/ }).waitFor();
    await go(listPage, '/search/?q=UrbanReplayBench');
    await listPage.locator('a[href*="publication-658"]').waitFor();
    assert.equal(await listPage.locator('#main-content h1').innerText(), 'Search');
    for (const theme of ['light', 'dark']) {
      await listPage.evaluate(theme => localStorage.setItem('theme', theme), theme);
      await go(listPage, '/blog/20261005-paper-2609-37989-tabfm-auto-self-evolving-pipel/');
      await listPage.getByRole('heading', { name: 'Comments', exact: true }).waitFor();
      assert.equal(await listPage.evaluate(() => document.documentElement.classList.contains('dark')), theme === 'dark');
    }
    await lists.close();
    console.log('Deep links and hydration passed: prompt builders/snippets, English search, UrbanReplayBench and both stored comment themes.');
    }

    for (const language of process.env.SUANLAB_NAVIGATION_ONLY ? [] : ['ko', 'en']) {
      const ctx = await context(language); const layoutPage = await ctx.newPage();
      for (const width of [320, 768, 1440]) {
        await layoutPage.setViewportSize({ width, height: 900 });
        for (const route of routes) {
          await go(layoutPage, route, language);
          const result = await layoutPage.evaluate(() => ({ viewport: innerWidth, scroll: document.documentElement.scrollWidth,
            overflow: [...document.querySelectorAll('main *, header, footer')].filter(node => node.getBoundingClientRect().right > innerWidth + 1 && getComputedStyle(node).visibility !== 'hidden').slice(0, 8).map(node => ({ tag: node.tagName, class: node.className, text: node.textContent.slice(0, 60) })) }));
          layouts.push({ route, language, width, ...result });
          if (width === 320 && route === '/blog/') await layoutPage.screenshot({ path: `.runtime/blog-320-${language}.png` });
        }
      }
      await ctx.close();
    }
    if (layouts.length) fs.writeFileSync('.runtime/responsive-check.json', JSON.stringify(layouts, null, 2));
    fs.writeFileSync('.runtime/navigation-errors.json', JSON.stringify(errors, null, 2));
    assert.ok(layouts.every(result => result.scroll <= result.viewport), 'See .runtime/responsive-check.json for overflow.');
    assert.deepEqual(errors, [], 'Uncaught browser errors');
    console.log(`Navigation acceptance passed: ${layouts.length} additional layout cases; zero uncaught browser errors.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
