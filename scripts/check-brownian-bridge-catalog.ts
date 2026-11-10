import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import { brownianBridgeReader } from '../src/data/brownian-bridge-reader';
import { curriculumDomains } from '../src/data/learning-curriculum';
const root = '.runtime/learning/brownian-bridge-reader-integration', base = process.env.BROWNIAN_BRIDGE_PREVIEW_BASE ?? 'http://127.0.0.1:4201';
const path = '/book/online/physics-math-for-ai/', target = path + brownianBridgeReader.unit.id + '/';
async function main() {
  const browser = await chromium.launch({ headless: true }), rows = [];
  try {
    for (const language of ['ko', 'en'] as const) {
      const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
      await context.addInitScript(language => localStorage.setItem('suanlab-language', language), language);
      const page = await context.newPage(), errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
      await page.goto(base + '/book/interactive/'); await page.locator('.ai-book-entry').click(); await page.locator('.ai-book-overview').waitFor();
      assert.equal(await page.locator('.ai-unit-card').count(), 598);
      const card = page.locator('.ai-unit-card[href="' + target + '"]'); assert.equal(await card.count(), 1); assert((await card.innerText()).includes(brownianBridgeReader.unit.title[language]));
      const image = card.locator('[data-brownian-reference]'); assert.equal(await image.getAttribute('data-brownian-reference'), '0'); assert.equal(await image.evaluate(el => getComputedStyle(el).minWidth), '0px');
      await page.locator('.ai-curriculum >summary').click(); await page.getByLabel(language === 'ko' ? '영역 또는 주제 찾기' : 'Find a domain or topic', { exact: true }).fill(curriculumDomains.find(d => d.id === 'field-multiscale')!.title[language]);
      await page.locator('#curriculum-field-multiscale >summary').click(); await page.locator('#curriculum-field-multiscale a[href="' + target + '"]').click(); await page.locator('[data-brownian-phase="ready"]').waitFor({ timeout: 60000 });
      assert.equal(await page.locator('h1').innerText(), brownianBridgeReader.unit.title[language]);
      await page.goto(base + '/search/'); await page.getByRole('searchbox').fill(language === 'ko' ? 'Brownian bridge' : 'Brownian Bridges'); await page.locator('main a[href="' + target + '"]').waitFor();
      assert((await page.locator('main a[href="' + target + '"]').innerText()).includes(brownianBridgeReader.unit.title[language]));
      await page.goto(base + '/book/online/'); assert.equal(await page.locator('a[href="' + target + '"]').count(), 0);
      for (const [id, attribute] of [['finite-gaussian-field-sources-connected-correlations', 'gaussian'], ['interacting-nongaussian-connected-correlations', 'nongaussian'], ['spatial-spectra-decay-and-correlation-length', 'spatial']] as const) {
        await page.goto(base + path + id + '/'); await page.locator(`[data-${attribute}-phase="ready"]`).waitFor({ timeout: 60000 }); assert.equal(await page.locator('h1').count(), 1);
      }
      assert.deepEqual(errors, []); rows.push({ language, catalogCards: 598, libraryEntry: true, catalogCover: true, domainSearch: true, globalSearch: true, onlineBookListSeparate: true, priorReaders: 3, errors }); await context.close();
    }
    writeFileSync(root + '/catalog-and-regression.json', JSON.stringify({ rows, productionNextAssetsVerified: true }, null, 2) + '\n'); console.log(JSON.stringify(rows));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
