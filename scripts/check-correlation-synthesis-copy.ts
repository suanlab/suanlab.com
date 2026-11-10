import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
const root = '.runtime/learning/correlation-function-topic-audit';
const units = ['finite-gaussian-field-sources-connected-correlations', 'interacting-nongaussian-connected-correlations', 'spatial-spectra-decay-and-correlation-length'];
async function main() {
  const language = process.env.CORRELATION_AUDIT_LANGUAGE, theme = process.env.CORRELATION_AUDIT_THEME;
  assert(language === 'ko' || language === 'en'); assert(theme === 'light' || theme === 'dark');
  const base = process.env.CORRELATION_PREVIEW_BASE ?? 'http://127.0.0.1:4191';
  const browser = await chromium.launch({ headless: true }); const rows = [];
  try {
    for (const width of [320, 1440]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await context.addInitScript(({ language, theme }) => { localStorage.setItem('suanlab-language', language); localStorage.setItem('theme', theme); }, { language, theme });
      const page = await context.newPage(); const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
      for (const [index, unit] of units.entries()) {
        await page.goto(`${base}/book/online/physics-math-for-ai/${unit}/`);
        const prose = page.locator('.ai-prose'); await prose.waitFor();
        const text = await prose.innerText();
        if (language === 'en') assert(!/[가-힣]/.test(text));
        for (const target of index === 0 ? units.slice(1) : index === 1 ? units.slice(2) : units.slice(0, 2)) {
          const link = prose.locator(`a[href="/book/online/physics-math-for-ai/${target}/"]`).last();
          await link.scrollIntoViewIfNeeded(); assert(await link.isVisible());
          const response = await page.request.get(`${base}/book/online/physics-math-for-ai/${target}/`); assert.equal(response.status(), 200);
        }
        if (index === 2) {
          const table = prose.locator('table').last(); assert.equal(await table.locator('tbody tr').count(), 3);
          assert(text.includes('1/180') && text.includes('3/5') && text.includes('a C/T'));
          await table.scrollIntoViewIfNeeded();
          assert.equal(await table.getAttribute('tabindex'), '0'); assert(await table.getAttribute('aria-label'));
          await table.focus(); assert(await table.evaluate(element => element === document.activeElement));
          const axe = await new AxeBuilder({ page }).include('.ai-theory').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze(); assert.deepEqual(axe.violations.map(v => v.id), []);
          await page.screenshot({ path: `${root}/synthesis-${language}-${theme}-${width}.png` });
        }
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)); assert.deepEqual(errors, []);
        rows.push({ unit, language, theme, width, crosslinksVerified: true, revisedCopyVerified: true, noPageOverflow: true });
      }
      await context.close();
    }
    writeFileSync(`${root}/synthesis-copy-${language}-${theme}.json`, JSON.stringify({ rows }, null, 2) + '\n');
  } finally { await browser.close(); }
}
mkdirSync(root, { recursive: true }); main().catch(error => { console.error(error); process.exitCode = 1; });
