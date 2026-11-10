import fs from 'node:fs/promises';
import { chromium } from 'playwright';
const root = '.runtime/learning/partition-integration';
const { teaching } = JSON.parse(await fs.readFile(`${root}/specs.json`, 'utf8'));
const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
  await context.addInitScript(() => { localStorage.setItem('suanlab-language', 'en'); localStorage.setItem('theme', 'light'); });
  const page = await context.newPage();
  for (const [i, kind] of Object.keys(teaching).entries()) {
    await page.goto(`http://127.0.0.1:4175/book/online/physics-math-for-ai/${kind}/`);
    await page.waitForFunction(() => document.documentElement.lang === 'en' && document.querySelector('[data-partition-concept]')?.getAttribute('data-partition-concept-state') === 'ready');
    await page.evaluate(() => document.fonts.ready);
    // Element capture only: keep the global sticky header from occluding the diagram.
    await page.addStyleTag({ content: 'header { visibility: hidden !important; }' });
    await page.locator('.ai-concept-content').screenshot({ path: `${root}/reader-concept-${i}.png` });
  }
  await context.close();
} finally { await browser.close(); }
