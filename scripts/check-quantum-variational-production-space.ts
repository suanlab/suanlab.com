/** Inspect assets actually requested by the exported reader's lazy 3D view. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';
const root = '.runtime/learning/quantum-variational-integration';
const width = process.env.SUANLAB_QV_WIDTH === '320' ? 320 : 1600;
async function main() {
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox', '--enable-unsafe-swiftshader'] });
  try {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, deviceScaleFactor: width === 320 ? 2 : 1, hasTouch: width === 320, reducedMotion: 'reduce' });
    await page.addInitScript(() => localStorage.setItem('suanlab-language', 'en'));
    const errors: string[] = [], bodies: Promise<{ url: string; status: number; body: Buffer }>[] = [];
    let capture = false;
    page.on('pageerror', e => errors.push(e.message));
    page.on('response', response => {
      if (capture && response.request().resourceType() === 'script' && new URL(response.url()).pathname.startsWith('/_next/static/')) {
        bodies.push(response.body().then(body => ({ url: response.url(), status: response.status(), body })));
      }
    });
    await page.goto('http://localhost:4175/book/online/physics-math-for-ai/nonorthogonal-ritz-subspace-certificates/');
    await page.locator('[data-qv-lab][data-qv-state="ready"]').waitFor({ timeout: 120000 });
    await page.locator('[data-qv-concept-state="ready"]').waitFor({ timeout: 120000 });
    const lab = page.locator('[data-qv-lab]'), inputKey = await lab.getAttribute('data-qv-input');
    assert.equal(await page.locator('canvas').count(), 0, '3D must wait until its view approaches the viewport');
    capture = true;
    const figure = lab.locator('figure[data-qv-space]');
    await figure.scrollIntoViewIfNeeded(); await figure.locator('canvas').waitFor({ timeout: 30000 });
    await page.waitForFunction(() => document.querySelector('[data-qv-lab] figure[data-qv-space]')?.getAttribute('data-qv-space') === '3d');
    await page.waitForTimeout(100); capture = false;
    const results = await Promise.allSettled(bodies), assets: { url: string; path: string; bytes: number; sha256: string }[] = [];
    for (const result of results) {
      if (result.status === 'rejected') throw result.reason;
      const { url, status, body } = result.value, path = `out${new URL(url).pathname}`;
      assert.equal(status, 200); assert.deepEqual(body, readFileSync(path));
      assets.push({ url, path, bytes: body.length, sha256: createHash('sha256').update(body).digest('hex') });
    }
    assert(assets.length > 0, 'The lazy spatial renderer must request production JavaScript');
    await figure.getByRole('button', { name: 'Top', exact: true }).focus(); await page.keyboard.press('Enter');
    await figure.locator('canvas').evaluate((canvas: HTMLCanvasElement) => {
      const extension = canvas.getContext('webgl2')!.getExtension('WEBGL_lose_context');
      if (!extension) throw new Error('Missing context loss extension'); extension.loseContext();
    });
    await page.waitForFunction(() => document.querySelector('[data-qv-lab] figure[data-qv-space]')?.getAttribute('data-qv-space-status') === '2d');
    assert.equal(await figure.locator('canvas').count(), 0);
    await figure.getByRole('button', { name: 'Retry 3D', exact: true }).click(); await figure.locator('canvas').waitFor();
    assert.equal(await lab.getAttribute('data-qv-input'), inputKey);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    const box = (await figure.locator('canvas').boundingBox())!; assert(box.width <= width);
    await page.screenshot({ path: `${root}/production-space-en-light-${width}.png` });
    await page.getByRole('navigation', { name: 'Book navigation', exact: true }).getByRole('link').click();
    await page.waitForURL('**/book/online/physics-math-for-ai/'); assert.equal(await page.locator('canvas').count(), 0);
    assert.deepEqual(errors, []);
    const report = { viewportWidth: width, deviceScaleFactor: width === 320 ? 2 : 1, lazyProductionAssets: assets, actual3D: true, actualContextLossAndRecovery: true, cameraOnlyInputPreserved: true, clientNavigationCleanup: true, errors };
    writeFileSync(`${root}/production-space${width === 320 ? '-mobile' : ''}.json`, JSON.stringify(report, null, 2)); console.log(JSON.stringify(report));
  } finally { await browser.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
