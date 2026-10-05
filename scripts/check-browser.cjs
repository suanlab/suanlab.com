// Start a static server for out/ before running this acceptance check.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const AxeBuilder = require('@axe-core/playwright').default;
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
(async () => {
  fs.mkdirSync('.runtime', { recursive: true });
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox'] });
  const results = [];
  try {
    for (const theme of ['light', 'dark']) {
      const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
      await context.addInitScript(theme => {
        localStorage.setItem('theme', theme);
        localStorage.setItem('suanlab-language', theme === 'dark' ? 'en' : 'ko');
      }, theme);
      const page = await context.newPage();
      await page.route('https://www.youtube.com/**', route => route.abort());
      await page.route('https://www.googletagmanager.com/**', route => route.abort());
      for (const route of routes) {
        await page.goto(base + route);
        const audit = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
        results.push({ route, theme, overflow, violations: audit.violations });
        console.log(`${theme} ${route}: ${audit.violations.length} violations; overflow=${overflow}`);
        if (route === '/') await page.screenshot({ path: `.runtime/home-${theme}.png` });
      }
      await context.close();
    }
    fs.writeFileSync('.runtime/browser-check.json', JSON.stringify({ base, checkedAt: new Date().toISOString(), results }, null, 2));
    assert.ok(results.every(result => !result.overflow && !result.violations.length), 'See .runtime/browser-check.json for failures.');
    console.log(`Browser acceptance passed: ${results.length} page/theme combinations.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
