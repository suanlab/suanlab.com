import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
const root = '.runtime/learning/quartic-asymptotic-teaching', { url } = JSON.parse(readFileSync(`${root}/review-server.json`, 'utf8'));
async function main() {
  const browser = await chromium.launch({ headless: true }), states = [];
  try {
    for (const language of ['ko', 'en'] as const) {
      const context = await browser.newContext({ viewport: { width: language === 'ko' ? 320 : 1440, height: 900 }, reducedMotion: 'reduce' });
      await context.addInitScript({content:'globalThis.__name=fn=>fn;'});
      await context.addInitScript(language => { localStorage.setItem('suanlab-language', language); if (language === 'ko') localStorage.setItem('theme', 'dark'); const Native = Worker; (window as unknown as { __ports: { created: number; closed: number } }).__ports = { created: 0, closed: 0 }; window.Worker = class extends Native { ended: boolean; constructor(url: string | URL, options?: WorkerOptions) { super(url, options); this.ended = false; (window as unknown as { __ports: { created: number } }).__ports.created++; } terminate() { if (!this.ended) { this.ended = true; (window as unknown as { __ports: { closed: number } }).__ports.closed++; super.terminate(); } } }; }, language);
      const page = await context.newPage(), errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
      await page.goto(url); const t = (ko: string, en: string) => language === 'ko' ? ko : en;
      const ready = async () => { await page.locator('[data-quartic-phase="ready"]').waitFor({ timeout: 30000 }); await page.waitForFunction(() => { const p = (window as unknown as { __ports: { created: number; closed: number } }).__ports; return p.created === p.closed; }); };
      await ready(); await page.getByLabel(t('전체 예제', 'All examples'), { exact: true }).selectOption('gaussian-exact');
      await page.waitForFunction(() => document.querySelector('[data-quartic-result]')?.getAttribute('data-quartic-result-key') === JSON.stringify({ coupling: '0', order: 12 })); await ready();
      assert.equal(await page.locator('[data-quartic-metric=partial]').innerText(), '1'); assert.equal(await page.locator('[data-quartic-metric=error-upper]').innerText(), '0');
      const figure = page.locator('[data-quartic-integrand]'); assert.equal(await figure.locator('[data-quartic-zero-coupling]').count(), 1);
      await figure.scrollIntoViewIfNeeded(); await figure.getByRole('button', { name: t('3D 보기', 'Show 3D'), exact: true }).click(); await figure.locator('canvas').waitFor();
      await figure.screenshot({ path: `${root}/zero-axis-${language}.png` }); const before = await figure.locator('canvas').screenshot(); await figure.locator('canvas').press('ArrowRight'); assert(!before.equals(await figure.locator('canvas').screenshot()));
      const axe = await new AxeBuilder({ page }).include('main').analyze(); assert.deepEqual(axe.violations.map(v => v.id), []);
      await page.getByRole('button', { name: t('첫 예제로 초기화', 'Reset to first example'), exact: true }).click();await page.waitForFunction(() => document.querySelector('[data-quartic-result]')?.getAttribute('data-quartic-result-key') === JSON.stringify({ coupling: '1/10', order: 12 }));await ready();
      assert.equal(await page.locator('[data-quartic-zero-coupling]').count(), 0); assert.equal(await page.locator('[data-quartic-metric=partial]').innerText(), '613/640');
      await figure.scrollIntoViewIfNeeded(); await figure.getByRole('button', { name: t('3D 보기', 'Show 3D'), exact: true }).click(); await figure.locator('canvas').waitFor();
      if (language === 'en') assert(!/[가-힣]/.test(await page.locator('main').innerText())); assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      const ports = await page.evaluate(() => (window as unknown as { __ports: { created: number; closed: number } }).__ports); assert.equal(ports.created, ports.closed); assert.deepEqual(errors, []);
      states.push({ language, zeroAxisCoincident: true, positiveMappingRetained: true, keyboardCamera: true, accessibility: true, ports, errors }); await context.close();
    }
    writeFileSync(`${root}/zero-axis.json`, JSON.stringify({ states, accessibility: 2, positiveMappingUnitInvariant: '4*fraction-2', readerRegistered: false }, null, 2) + '\n');
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
