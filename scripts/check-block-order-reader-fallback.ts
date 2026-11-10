/** Next production reader with WebGL disabled and reduced motion enabled. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
const root = '.runtime/learning/block-order-reader-integration'; mkdirSync(root,{ recursive: true });
const candidate = process.env.BLOCK_ORDER_READER_CANDIDATE ?? '/tmp/suanlab-block-order-reader-candidate-20261111';
async function main() {
  const server = createServer((request,response) => {
    try {
      const relative = decodeURIComponent(new URL(request.url!,'http://localhost').pathname); if (relative.includes('..')) throw Error('Path');
      const file = path.join(candidate,'out',relative.endsWith('/') ? relative+'index.html' : relative);
      response.setHeader('Content-Type',file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : file.endsWith('.woff2') ? 'font/woff2' : 'text/html'); response.end(readFileSync(file));
    } catch { response.statusCode = 404; response.end('Not found'); }
  });
  await new Promise<void>(resolve => server.listen(0,'127.0.0.1',resolve)); const address = server.address(); assert(address && typeof address !== 'string');
  const browser = await chromium.launch({ headless: true }), errors: string[] = [];
  try {
    const context = await browser.newContext({ viewport: { width: 320,height: 950 },reducedMotion: 'reduce' });
    await context.addInitScript(() => {
      (globalThis as unknown as { __name: unknown }).__name = (fn: unknown) => fn;
      localStorage.setItem('suanlab-language','en');
      const original = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function(this: HTMLCanvasElement,type: string,...args: unknown[]) { return type.includes('webgl') ? null : original.apply(this,[type,...args] as never); } as typeof original;
    });
    const page = await context.newPage(); page.on('pageerror',error => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${address.port}/book/online/physics-math-for-ai/microscopic-blocks-and-local-order-parameters/`);
    const lab = page.locator('[data-block-status="ready"]'); await lab.waitFor({ timeout: 120000 });
    assert(await lab.getByRole('button',{ name: 'Play example arrangements',exact: true }).isDisabled());
    assert.equal(await page.locator('[data-block-concept] .information-step').evaluate(element => getComputedStyle(element).animationName),'none');
    await lab.getByRole('button',{ name: 'Show 3D lattice',exact: true }).click();
    await lab.getByText('3D is unavailable. Continue comparing with the2D map and exact tables.',{ exact: true }).waitFor();
    assert.equal(await lab.locator('.block-order-grid-svg').count(),1);
    assert.equal(await lab.getByRole('table',{ name: 'Exact site values shown in the drawing',exact: true }).locator('tbody tr').count(),4);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1));
    const axe = await new AxeBuilder({ page }).include('.ai-reader').analyze(); assert.deepEqual(axe.violations,[]); assert.deepEqual(errors,[]);
    writeFileSync(`${root}/reader-fallback.json`,JSON.stringify({ status: 'actual-next-reader-fallback-verified',webglUnavailable: true,reducedMotion: true,
      staticMapRetained: true,exactValuesRetained: true,axeViolations: 0,documentOverflow: false,errors },null,2)+'\n');
  } finally { await browser.close(); await new Promise<void>(resolve => server.close(() => resolve())); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
