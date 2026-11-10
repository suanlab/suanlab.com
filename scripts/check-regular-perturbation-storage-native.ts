import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { regularPerturbationDefault } from '../src/lib/learning-regular-perturbation-input';
import { calculateRegularPerturbation } from '../src/lib/learning-regular-perturbation';
const root = '.runtime/learning/regular-perturbation-teaching', key = 'suanlab-regular-perturbation-v1';
const { url } = JSON.parse(readFileSync(`${root}/review-server.json`, 'utf8'));
const maximum = JSON.parse(readFileSync('scripts/fixtures/regular-perturbation-budget.json', 'utf8'))[0].input;
type Harness = Window & { __control: { mode: string; created: number; terminated: number; active: Set<Worker> } };
async function main() {
  const browser = await chromium.launch({ headless: true }), rows = [];
  try {
    for (const scenario of ['corrupt', 'bad-version', 'oversized', 'read-denied', 'write-denied', 'maximum'] as const) {
      const context = await browser.newContext({ viewport: { width: 320, height: 900 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, reducedMotion: scenario === 'maximum' ? 'no-preference' : 'reduce' });
      await context.addInitScript(({ scenario, key, maximum }) => {
        localStorage.setItem('suanlab-language', 'en');
        if (scenario === 'corrupt') localStorage.setItem(key, '{');
        if (scenario === 'bad-version') localStorage.setItem(key, JSON.stringify({ version: 2, value: maximum }));
        if (scenario === 'oversized') localStorage.setItem(key, ' '.repeat(4097));
        if (scenario === 'maximum') localStorage.setItem(key, JSON.stringify({ version: 1, value: maximum }));
        const get = Storage.prototype.getItem, set = Storage.prototype.setItem;
        Storage.prototype.getItem = function(name) { if (name === key && scenario === 'read-denied') throw Error('storage denied'); return get.call(this, name); };
        Storage.prototype.setItem = function(name, value) { if (name === key && scenario === 'write-denied') throw Error('storage denied'); return set.call(this, name, value); };
        const Native = Worker, control = { mode: 'normal', created: 0, terminated: 0, active: new Set<Worker>() }; (window as unknown as Harness).__control = control;
        window.Worker = class extends Native {
          channel: string;
          constructor(url: string | URL, options?: WorkerOptions) { super(url, options); this.channel = options?.name ?? ''; control.created++; control.active.add(this); }
          postMessage(message: unknown, options?: StructuredSerializeOptions | Transferable[]) { if (this.channel === 'regular-perturbation-calculation' && control.mode === 'hold') return; if (this.channel === 'regular-perturbation-calculation' && control.mode === 'throw') throw Error('postMessage failed'); if (Array.isArray(options)) super.postMessage(message, options); else super.postMessage(message, options); }
          terminate() { if (control.active.delete(this)) { control.terminated++; super.terminate(); } }
        };
      }, { scenario, key, maximum });
      const page = await context.newPage(), errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
      await page.goto(url);
      const ready = async () => { await page.waitForFunction(() => document.querySelector('[data-perturbation-phase]')?.getAttribute('data-perturbation-phase') === 'ready', null, { timeout: 30000 }); await page.waitForFunction(() => (window as unknown as Harness).__control.active.size === 0); };
      await ready();
      const expectedInput = scenario === 'maximum' ? maximum : regularPerturbationDefault;
      assert.equal(await page.locator('[data-perturbation-result]').getAttribute('data-perturbation-result-key'), JSON.stringify(expectedInput));
      if (scenario !== 'maximum') assert.equal(await page.locator('[data-perturbation-storage]').getAttribute('data-perturbation-storage'), scenario === 'read-denied' || scenario === 'write-denied' ? 'unavailable' : 'corrupt');
      const result = calculateRegularPerturbation(expectedInput);
      assert.equal(await page.locator('[data-perturbation-metric=bound]').innerText(), result.approximations.at(-1)!.errorBound);
      const report: Record<string, unknown> = { scenario, savedInputRestored: scenario === 'maximum', continuedExperiment: true };
      if (scenario === 'maximum') {
        assert(await page.locator('[data-perturbation-off-window]').count());
        const previous = await page.evaluate(() => (window as unknown as Harness).__control.created);
        const slider = page.getByLabel('Selected order', { exact: true }); await slider.fill('1');
        await page.getByRole('button', { name: 'Play order changes', exact: true }).click();
        await page.waitForFunction(() => (document.querySelector('input[type=range]') as HTMLInputElement)?.value !== '1');
        await page.getByRole('button', { name: 'Pause order playback', exact: true }).click();
        assert.equal(await page.evaluate(() => (window as unknown as Harness).__control.created), previous);
        await page.getByLabel('Displacement plotting window', { exact: true }).selectOption('linear');
        const energy = page.locator('[data-perturbation-energy]'); await energy.scrollIntoViewIfNeeded(); await energy.getByRole('button', { name: 'Show 3D', exact: true }).click(); const canvas = energy.locator('canvas'); await canvas.waitFor();
        const before = await canvas.screenshot(), box = await canvas.boundingBox(); assert(box);
        const cdp = await context.newCDPSession(page), x = box.x + box.width / 2, y = box.y + box.height / 2;
        await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] });
        await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: x + 45, y }] });
        await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
        assert(!before.equals(await canvas.screenshot()), 'Touch changes the energy camera');
        await energy.screenshot({ path: `${root}/maximum-mobile-3d.png` });
        const height = await page.evaluate(() => document.documentElement.scrollHeight); assert(height < 25000, `Bounded page height ${height}`);
        report.pageHeight = height; report.touchCamera = true; report.playbackWithoutCalculation = true;
        await page.evaluate(() => { (window as unknown as Harness).__control.mode = 'hold'; });
        await page.getByLabel('Coupling λ', { exact: true }).fill('1/2'); await page.getByRole('button', { name: 'Apply input', exact: true }).click();
        await page.waitForFunction(() => document.querySelector('[data-perturbation-phase]')?.getAttribute('data-perturbation-phase') === 'pending'); assert.equal(await page.locator('[data-perturbation-result]').count(), 0);
        await page.getByRole('button', { name: 'Cancel calculation', exact: true }).click(); await page.waitForFunction(() => (window as unknown as Harness).__control.active.size === 0);
        await page.evaluate(() => { (window as unknown as Harness).__control.mode = 'throw'; });
        await page.getByRole('button', { name: 'Recalculate', exact: true }).click(); await page.waitForFunction(() => document.querySelector('[data-perturbation-phase]')?.getAttribute('data-perturbation-phase') === 'error');
        await page.evaluate(() => { (window as unknown as Harness).__control.mode = 'normal'; });
        await page.getByRole('button', { name: 'Recalculate', exact: true }).click(); await ready(); report.cancelErrorRetry = true;
      }
      const axe = await new AxeBuilder({ page }).include('main').analyze(); assert.deepEqual(axe.violations.map(v => v.id), []);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)); assert(!/[가-힣]/.test(await page.locator('main').innerText()));
      const counts = await page.evaluate(() => { const c = (window as unknown as Harness).__control; return { created: c.created, terminated: c.terminated, active: c.active.size }; });
      assert.equal(counts.created, counts.terminated); assert.equal(counts.active, 0); assert.deepEqual(errors, []);
      rows.push({ ...report, ...counts, accessibility: true, errors }); console.log(JSON.stringify(rows.at(-1))); await context.close();
    }
    writeFileSync(`${root}/storage-maximum.json`, JSON.stringify({ rows, accessibility: 6, readerRegistered: false, productionNextAssetsVerified: false }, null, 2) + '\n');
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
