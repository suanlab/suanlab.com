// Audit every UI template; original-language prose and formal citations are marked explicitly.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('playwright');
const base = process.env.SUANLAB_TEST_BASE || 'http://127.0.0.1:4175';
const paths = [...fs.readFileSync('out/sitemap.xml', 'utf8').matchAll(/<loc>([^<]+)<\/loc>/g)].map(match => new URL(match[1]).pathname);
const routes = [...new Set([...paths.filter(path => !/^\/(blog|qt)\/[^/]+/.test(path)), ...['blog', 'qt'].flatMap(type => { const entries = paths.filter(path => new RegExp(`^/${type}/[^/]+`).test(path)); return [entries[0], entries[Math.floor(entries.length / 2)], entries[entries.length - 1]]; }), '/qt/2017-01-11/', '/qt/2016-02-18/', '/qt/2015-10-03/', '/qt/2015-09-28/', '/qt/2014-10-06/', '/404.html'])].filter(Boolean);

async function untranslated(page) {
  return page.evaluate(() => {
    const out = [], source = 'script,style,noscript,pre,code,[lang="ko"],[data-original-language],[aria-hidden="true"]';
    const visible = element => {
      const node = element.tagName === 'OPTION' ? element.closest('select') : element;
      return node && node.getClientRects().length && getComputedStyle(node).visibility !== 'hidden' && getComputedStyle(node).display !== 'none';
    };
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      const text = node.textContent.trim().replace(/\s+/g, ' '), element = node.parentElement;
      if (/[가-힣]/.test(text) && !element.closest(source) && visible(element)) out.push(text);
    }
    for (const element of document.querySelectorAll('[title],[alt],[placeholder],[aria-label]')) {
      if (element.closest(source) || !visible(element)) continue;
      for (const attribute of ['title', 'alt', 'placeholder', 'aria-label']) {
        const text = element.getAttribute(attribute);
        if (text && /[가-힣]/.test(text)) out.push(`${attribute}: ${text}`);
      }
    }
    return [...new Set(out)];
  });
}

(async () => {
  fs.mkdirSync('.runtime', { recursive: true });
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox'] });
  const rows = [], errors = [];
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    await context.addInitScript(() => { try { localStorage.setItem('suanlab-language', 'en'); localStorage.setItem('suanlab-motion-paused', 'true'); } catch {} });
    await context.route('**/*', route => new URL(route.request().url()).origin === new URL(base).origin ? route.continue() : route.abort());
    const page = await context.newPage();
    page.on('pageerror', error => errors.push({ url: page.url(), message: error.message }));
    page.on('console', message => { if (message.type() === 'error' && /Hydration|did not match|matching|Encountered two children/.test(message.text())) errors.push({ url: page.url(), message: message.text() }); });
    async function audit(route, state = 'default') {
      const texts = await untranslated(page);
      rows.push({ route, state, title: await page.title(), texts });
      fs.writeFileSync('.runtime/language-check.json', JSON.stringify({ rows, errors }, null, 2));
      console.log(`${route} ${state}: ${texts.length} untranslated strings`);
    }
    for (const route of routes) {
      await page.goto(base + route);
      await page.waitForFunction(() => document.documentElement.lang === 'en');
      await page.evaluate(() => document.fonts.ready);
      if (route === '/suan/') await page.getByRole('img', { name: 'Visited countries world map', exact: true }).waitFor();
      await audit(route);
      if (!/^\/(blog|qt)\/[^/]+/.test(route)) {
        // Titles for original articles deliberately keep their source names.
        const title = await page.title();
        if (/[가-힣]/.test(title)) errors.push({ route, message: `Untranslated document title: ${title}` });
      }
      if (route === '/youtube/') for (const name of ['Popular videos', 'Recent videos']) {
        await page.getByRole('button', { name, exact: true }).click();
        await audit(route, name);
      }
      if (/\/present\/$/.test(route)) {
        await page.getByRole('button', { name: 'Presenter notes', exact: true }).click();
        const select = page.getByRole('combobox', { name: 'Select slide', exact: true });
        const count = await select.locator('option').count();
        for (let index = 0; index < count; index++) {
          await select.selectOption(String(index));
          await audit(route, `slide ${index + 1}`);
        }
      }
    }
    // Locale switches must preserve the same input node and the user's active filter.
    await page.goto(base + '/publication/');
    await page.waitForFunction(() => document.documentElement.lang === 'en');
    const input = page.getByRole('textbox', { name: 'Search', exact: true });
    await input.fill('UrbanReplayBench');
    await input.evaluate(node => { window.__localeInput = node; });
    await page.getByRole('button', { name: 'KO', exact: true }).first().click();
    await page.waitForFunction(() => document.documentElement.lang === 'ko');
    assert.equal(await page.evaluate(() => window.__localeInput === document.querySelector('main input')), true);
    assert.equal(await page.locator('main input').first().inputValue(), 'UrbanReplayBench');
    await page.getByRole('button', { name: 'EN', exact: true }).first().click();
    await page.waitForFunction(() => document.documentElement.lang === 'en');
    assert.equal(await input.inputValue(), 'UrbanReplayBench');
    await page.goto(base + '/project/');
    await page.waitForFunction(() => document.documentElement.lang === 'en');
    assert.ok(!/[가-힣]/.test(await page.title()));
    await page.getByRole('button', { name: 'KO', exact: true }).first().click();
    await page.waitForFunction(() => document.documentElement.lang === 'ko');
    assert.ok(/[가-힣]/.test(await page.locator('main').innerText()));
    await page.getByRole('button', { name: 'EN', exact: true }).first().click();
    await page.waitForFunction(() => document.documentElement.lang === 'en');
    await page.reload();
    await page.waitForFunction(() => document.documentElement.lang === 'en');
    await audit('/project/', 'locale persistence');
    await page.goto(base + '/contact/');
    await page.waitForFunction(() => document.documentElement.lang === 'en');
    await page.locator('footer a[href="/privacy"], footer a[href="/privacy/"]').first().click();
    await page.waitForURL('**/privacy/');
    await page.waitForFunction(() => document.title.includes('Privacy Policy'));
    await page.getByRole('button', { name: 'KO', exact: true }).first().click();
    await page.waitForFunction(() => document.documentElement.lang === 'ko');
    await page.waitForFunction(() => document.title.includes('개인정보처리방침'));
    await page.getByRole('button', { name: 'EN', exact: true }).first().click();
    await page.waitForFunction(() => document.documentElement.lang === 'en');
    await page.waitForFunction(() => document.title.includes('Privacy Policy'));
    await audit('/privacy/', 'client navigation and locale title');
    fs.writeFileSync('.runtime/language-check.json', JSON.stringify({ rows, errors }, null, 2));
    assert.deepEqual(errors, [], 'Runtime or document-title errors; see .runtime/language-check.json');
    assert.ok(rows.every(row => row.texts.length === 0), 'Untranslated UI; see .runtime/language-check.json');
    console.log(`English localization passed: ${routes.length} routes, ${rows.length} states.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
