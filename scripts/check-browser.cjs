// Start a static server for out/ before running this acceptance check.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const { settleMotion } = require('./browser-motion.cjs');
const routes = [
  '/', '/suan/', '/research/', '/research/nlp/', '/project/', '/publication/',
  '/lecture/', '/lecture/ai/', '/lecture/ai/notes/', '/lecture/ai/present/',
  '/course/', '/youtube/', '/youtube/nlp/', '/book/', '/book/online/computer-vision/',
  '/blog/', '/blog/20261005-paper-2609-37989-tabfm-auto-self-evolving-pipel/',
  '/search/', '/qt/', '/qt/2015-02-01/', '/book/online/', '/book/published/',
  '/deadlines/', '/prompts/', '/contact/', '/privacy/', '/terms/',
  '/research/ds/', '/research/dl/', '/research/cv/', '/research/graphs/',
  '/research/st/', '/research/asp/', '/lecture/ml/', '/lecture/ml/present/',
];
const base = process.env.SUANLAB_TEST_BASE || 'http://127.0.0.1:4175';
const origin = new URL(fs.readFileSync('out/index.html', 'utf8').match(/<link rel="canonical" href="([^"]+)"/)[1]).origin;
// Serve the local export under its public origin so cross-origin widgets can load
// its styles without the browser's public-to-loopback network restriction.
async function routeExport(context) {
  await context.route(origin + '/**', async route => {
    const url = new URL(route.request().url());
    const response = await route.fetch({ url: new URL(url.pathname + url.search, base).href });
    await route.fulfill({ response });
  });
}

(async () => {
  fs.mkdirSync('.runtime', { recursive: true });
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox'] });
  const results = [];
  try {
    for (const theme of ['light', 'dark']) {
      const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
      await routeExport(context);
      await context.addInitScript(theme => {
        localStorage.setItem('theme', theme);
        localStorage.setItem('suanlab-language', theme === 'dark' ? 'en' : 'ko');
      }, theme);
      const page = await context.newPage();
      await page.route('https://www.youtube.com/**', route => route.abort());
      await page.route('https://www.googletagmanager.com/**', route => route.abort());
      for (const route of routes) {
        const response = await page.goto(origin + route);
        assert.equal(response.status(), 200, `Unexpected page status: ${route}`);
        await page.locator('#main-content h1').first().waitFor();
        await settleMotion(page);
        const audit = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
        results.push({ route, theme, overflow, violations: audit.violations });
        console.log(`${theme} ${route}: ${audit.violations.length} violations; overflow=${overflow}`);
        if (route === '/') await page.screenshot({ path: `.runtime/home-${theme}.png` });
      }
      await context.close();
    }
    fs.writeFileSync('.runtime/browser-check.json', JSON.stringify({ base, origin, checkedAt: new Date().toISOString(), results }, null, 2));
    assert.ok(results.every(result => !result.overflow && !result.violations.length), 'See .runtime/browser-check.json for failures.');
    // Regression: denied language storage must not replace the whole application.
    const blocked = await browser.newContext({ viewport: { width: 390, height: 844 } });
    await routeExport(blocked);
    await blocked.addInitScript(() => {
      const read = Storage.prototype.getItem, write = Storage.prototype.setItem;
      Storage.prototype.getItem = function(key) {
        if (key === 'suanlab-language') throw new DOMException('Storage denied', 'SecurityError');
        return read.call(this, key);
      };
      Storage.prototype.setItem = function(key, value) {
        if (key === 'suanlab-language') throw new DOMException('Storage denied', 'SecurityError');
        return write.call(this, key, value);
      };
    });
    const blockedPage = await blocked.newPage();
    await blockedPage.route('https://www.googletagmanager.com/**', route => route.abort());
    await blockedPage.goto(origin + '/');
    await blockedPage.locator('#main-content h1').waitFor();
    await blockedPage.getByRole('button', { name: 'EN', exact: true }).click();
    await blockedPage.waitForFunction(() => document.documentElement.lang === 'en');
    assert.ok(await blockedPage.locator('#main-content h1').isVisible());
    await blockedPage.getByRole('button', { name: 'KO', exact: true }).click();
    await blockedPage.waitForFunction(() => document.documentElement.lang === 'ko');
    await blocked.close();

    // Regression: code-split builders retain edited drafts and sanitize shared inputs.
    const tools = await browser.newContext();
    await routeExport(tools);
    await tools.addInitScript(() => localStorage.setItem('suanlab-language', 'en'));
    const toolPage = await tools.newPage();
    await toolPage.route('https://www.googletagmanager.com/**', route => route.abort());
    await toolPage.goto(origin + '/prompts/');
    await toolPage.getByRole('button', { name: 'Open builder', exact: true }).first().click();
    const editor = toolPage.getByRole('textbox', { name: 'Edit generated prompt', exact: true });
    await editor.fill('An edited draft retained through navigation.');
    const storedDraft = await toolPage.waitForFunction(() => {
      const key = Object.keys(localStorage).find(key => key.startsWith('suanlab-prompt-draft-'));
      return key && JSON.parse(localStorage.getItem(key)).output === 'An edited draft retained through navigation.' ? key : false;
    });
    const draftKey = await storedDraft.jsonValue();
    await toolPage.reload();
    await toolPage.getByRole('button', { name: 'Open builder', exact: true }).first().click();
    assert.equal(await editor.inputValue(), 'An edited draft retained through navigation.');
    const builderId = draftKey.replace('suanlab-prompt-draft-', '');
    const share = Buffer.from(encodeURIComponent(JSON.stringify({ b: builderId, v: { context: ['invalid type'], unknown: 'ignored' } }))).toString('base64');
    await toolPage.goto(origin + '/prompts/#' + share);
    await editor.waitFor();
    assert.notEqual(await editor.inputValue(), 'An edited draft retained through navigation.');
    await toolPage.getByRole('button', { name: 'Back', exact: true }).click();
    await toolPage.getByRole('tab', { name: /^Library/ }).click();
    await toolPage.getByRole('button', { name: /Add my prompt/ }).click();
    const dialog = toolPage.getByRole('dialog');
    await dialog.getByRole('textbox', { name: /Title/ }).fill('Regression personal prompt');
    await dialog.getByRole('textbox', { name: /Content/ }).fill('Explain {{topic}}');
    await dialog.getByRole('button', { name: 'Save', exact: true }).click();
    await toolPage.waitForFunction(() => (localStorage.getItem('suanlab-custom-prompts') || '').includes('Regression personal prompt'));
    await toolPage.getByRole('tab', { name: /^Workflows/ }).click();
    await toolPage.getByRole('button', { name: 'Start / Resume', exact: true }).first().click();
    await toolPage.getByRole('textbox', { name: 'Result from your AI tool', exact: true }).fill('Actual first-step model response.');
    const requiredFields = toolPage.locator('[aria-required="true"]');
    for (let i = 0; i < await requiredFields.count(); i++) {
      const field = requiredFields.nth(i);
      if (await field.evaluate(node => node.matches('input, textarea'))) await field.fill('Regression research context');
    }
    await toolPage.getByRole('button', { name: 'Save result & complete step', exact: true }).click();
    await toolPage.waitForFunction(() => (localStorage.getItem('suanlab-prompt-workflow-state') || '').includes('Actual first-step model response.'));
    await toolPage.getByRole('button', { name: 'Start / Resume', exact: true }).first().click();
    assert.ok((await editor.inputValue()).includes('Actual first-step model response.'));
    await tools.close();
    console.log('Interaction regressions passed: denied storage, language switching, draft persistence, shared inputs, personal prompts and workflow handoff.');

    const widget = await browser.newContext();
    await routeExport(widget);
    await widget.addInitScript(() => { localStorage.setItem('theme', 'dark'); localStorage.setItem('suanlab-language', 'en'); });
    const widgetPage = await widget.newPage();
    await widgetPage.route('https://www.googletagmanager.com/**', route => route.abort());
    await widgetPage.route('https://giscus.app/**', route => {
      if (route.request().url().endsWith('/client.js')) return route.fulfill({ contentType: 'application/javascript', body: `
        const owner = document.currentScript;
        window.giscusInitializations = [...(window.giscusInitializations || []), { theme: owner.dataset.theme, lang: owner.dataset.lang }];
        const frame = document.createElement('iframe');
        frame.src = 'https://giscus.app/regression-fixture';
        frame.title = 'Comments fixture';
        owner.parentElement.appendChild(frame);
      ` });
      return route.fulfill({ contentType: 'text/html', body: `<html><body><textarea aria-label="Comment draft"></textarea><script>
        window.addEventListener('message', event => { if (event.data?.giscus?.setConfig) window.widgetConfig = event.data.giscus.setConfig; });
      </script></body></html>` });
    });
    await widgetPage.goto(origin + '/blog/20261005-paper-2609-37989-tabfm-auto-self-evolving-pipel/');
    const commentFrame = widgetPage.frameLocator('iframe[title="Comments fixture"]');
    const commentDraft = commentFrame.getByRole('textbox', { name: 'Comment draft', exact: true });
    await commentDraft.fill('Keep this comment draft when the theme or language changes.');
    const initialWidget = await widgetPage.evaluate(() => window.giscusInitializations);
    assert.deepEqual(initialWidget, [{ theme: new URL('/assets/styles/giscus-dark.css', origin).href, lang: 'en' }]);
    await widgetPage.getByRole('button', { name: 'Toggle theme', exact: true }).click();
    await widgetPage.getByRole('button', { name: 'KO', exact: true }).click();
    await widgetPage.waitForFunction(() => document.documentElement.lang === 'ko' && !document.documentElement.classList.contains('dark'));
    const frame = widgetPage.frames().find(frame => frame.url().includes('/regression-fixture'));
    await frame.waitForFunction(() => window.widgetConfig?.theme === 'light' && window.widgetConfig?.lang === 'ko');
    assert.equal(await commentDraft.inputValue(), 'Keep this comment draft when the theme or language changes.');
    assert.equal(await widgetPage.evaluate(() => window.giscusInitializations.length), 1);
    await widget.close();
    console.log('Comment regression passed: one initialization and theme/language updates preserve the draft.');

    const mapContext = await browser.newContext();
    await routeExport(mapContext);
    await mapContext.addInitScript(() => localStorage.setItem('suanlab-language', 'en'));
    const mapPage = await mapContext.newPage();
    const mapDataUrl = origin + '/geo/countries-110m.json';
    await mapPage.route(mapDataUrl, route => route.abort());
    await mapPage.goto(origin + '/suan/');
    await mapPage.getByText('Could not load the map.', { exact: true }).waitFor();
    await mapPage.unroute(mapDataUrl);
    await mapPage.getByRole('button', { name: 'Retry map', exact: true }).click();
    await mapPage.getByRole('img', { name: 'Visited countries world map', exact: true }).waitFor();
    await mapContext.close();
    console.log('Map regression passed: failed download shows recovery and retry renders the map.');

    // Check the real, visible comment frame after its public-origin theme loads.
    const liveComments = await browser.newContext({ viewport: { width: 390, height: 844 } });
    await routeExport(liveComments);
    await liveComments.addInitScript(() => { localStorage.setItem('theme', 'dark'); localStorage.setItem('suanlab-language', 'en'); });
    const livePage = await liveComments.newPage();
    await livePage.route('https://www.googletagmanager.com/**', route => route.abort());
    await livePage.goto(origin + '/blog/20261005-paper-2609-37989-tabfm-auto-self-evolving-pipel/');
    const liveIframe = livePage.locator('iframe.giscus-frame');
    await liveIframe.waitFor();
    await liveIframe.scrollIntoViewIfNeeded();
    const liveFrame = await (await liveIframe.elementHandle()).contentFrame();
    await liveFrame.waitForURL('https://giscus.app/**');
    await liveFrame.waitForFunction(() => getComputedStyle(document.documentElement).backgroundColor === 'rgb(13, 17, 23)', {}, { timeout: 15_000 });
    await liveFrame.locator('.gsc-comments-count').waitFor({ timeout: 15_000 });
    const liveAudit = await new AxeBuilder({ page: livePage }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
    fs.writeFileSync('.runtime/comments-live-dark.json', JSON.stringify(liveAudit.violations, null, 2));
    assert.equal(liveAudit.violations.length, 0, 'See .runtime/comments-live-dark.json.');
    await liveComments.close();
    console.log('Visible live comments passed: dark canvas loaded and zero accessibility violations.');

    console.log(`Browser acceptance passed: ${results.length} page/theme combinations.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
