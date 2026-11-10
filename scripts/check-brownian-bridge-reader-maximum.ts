import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { brownianBridgeDefault, brownianBridgeExamples } from '../src/lib/learning-brownian-bridge-input';
import { brownianBridgeInputKey } from '../src/lib/learning-brownian-bridge-protocol';
import { brownianBridgeStorageKey } from '../src/components/learning/use-saved-brownian-bridge';
const root = '.runtime/learning/brownian-bridge-reader-integration';
type Control = { mode: string; created: number; closed: number; active: number };
async function main() {
  const browser = await chromium.launch({ headless: true }), url=(process.env.BROWNIAN_BRIDGE_PREVIEW_BASE??'http://127.0.0.1:4201')+'/book/online/physics-math-for-ai/brownian-bridge-time-slicing-measure/', rows = [];
  const maximum = brownianBridgeExamples.find(e => e.id === 'maximum-coprime')!.input;
  try {
    const context = await browser.newContext({ viewport: { width: 320, height: 1000 }, hasTouch: true, isMobile: true, reducedMotion: 'no-preference' });
    await context.addInitScript({ content: 'globalThis.__name=fn=>fn;' });
    await context.addInitScript(() => {
      localStorage.setItem('suanlab-language', 'en'); localStorage.setItem('theme', 'dark');
      const Native = Worker, active = new Set<Worker>(), control = { mode: 'normal', created: 0, closed: 0, active: 0 }; (window as unknown as { __brownianControl: Control }).__brownianControl = control;
      window.Worker = class extends Native {
        channel: string;
        constructor(url: string | URL, options?: WorkerOptions) { super(url, options); this.channel = options?.name ?? ''; active.add(this); control.created++; control.active = active.size; }
        postMessage(value: unknown, options?: StructuredSerializeOptions | Transferable[]) { if (this.channel === 'brownian-bridge-calculation' && control.mode === 'hold') return; if (this.channel === 'brownian-bridge-calculation' && control.mode === 'throw') throw Error('Injected postMessage failure'); if (Array.isArray(options)) super.postMessage(value, options); else super.postMessage(value, options); }
        terminate() { if (active.delete(this)) { control.closed++; control.active = active.size; } super.terminate(); }
      };
    });
    const page = await context.newPage(), errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
    const ready = async (input: object) => { await page.waitForFunction(key => document.querySelector('[data-brownian-result]')?.getAttribute('data-brownian-result-key') === key, brownianBridgeInputKey(input as typeof maximum), { timeout: 60000 }); await page.waitForFunction(() => (window as unknown as { __brownianControl: Control }).__brownianControl.active === 0); };
    const counts = () => page.evaluate(() => { const c = (window as unknown as { __brownianControl: Control }).__brownianControl; return { created: c.created, closed: c.closed, active: c.active }; });
    await page.goto(url); await ready(brownianBridgeDefault);
    const originalKey = await page.locator('[data-brownian-result]').getAttribute('data-brownian-result-key'), beforeInvalid = await counts();
    await page.getByLabel('Interval count N', { exact: true }).fill('4.0000000000000001'); await page.getByRole('button', { name: 'Apply input', exact: true }).click();
    assert((await page.locator('[data-brownian-phase] [role=alert]').innerText()).includes('previously applied')); assert.equal(await page.locator('[data-brownian-result]').getAttribute('data-brownian-result-key'), originalKey); assert.deepEqual(await counts(), beforeInvalid);
    const signedFraction='-999999999983/999999999989';
    await page.getByLabel('Interval count N', {exact:true}).fill('4');
    await page.getByLabel('Start value a', {exact:true}).fill(signedFraction); await page.getByLabel('Auxiliary point source j', {exact:true}).fill(signedFraction);
    await page.getByRole('button',{name:'Apply input',exact:true}).click(); await ready({...brownianBridgeDefault,start:signedFraction,source:signedFraction});
    const one=brownianBridgeExamples.find(e=>e.id==='one-interior-coordinate')!.input;
    await page.getByLabel('All examples',{exact:true}).selectOption('one-interior-coordinate');await ready(one);
    await page.getByLabel('Displayed matrix',{exact:true}).selectOption('precision');
    assert.equal(await page.getByRole('group',{name:'Interior precision Q',exact:true}).getByRole('button').count(),1);
    await page.getByRole('button',{name:'Show 3D matrix',exact:true}).click();await page.locator('[data-brownian-matrix-plot="3d"]').waitFor();
    await page.getByRole('button',{name:'Close 3D matrix',exact:true}).click();
    await page.getByLabel('All examples', { exact: true }).selectOption('maximum-coprime'); await ready(maximum);
    await page.getByLabel('Displayed matrix',{exact:true}).selectOption('position');
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
    assert.equal(await page.getByRole('group', { name: 'Position covariance C', exact: true }).getByRole('button').count(), 33 * 33);
    await page.getByRole('button', { name: 'Show 3D matrix', exact: true }).click(); await page.locator('[data-brownian-matrix-plot="3d"]').waitFor({ timeout: 60000 });
    const canvas = page.locator('[data-brownian-matrix-plot] canvas'); await canvas.scrollIntoViewIfNeeded(); const beforeTouch = await canvas.screenshot(), box = await canvas.boundingBox(); assert(box);
    const cdp = await context.newCDPSession(page);
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: box.x + box.width * .65, y: box.y + box.height * .5 }] });
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: box.x + box.width * .35, y: box.y + box.height * .5 }] });
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] }); await cdp.detach();
    assert(!beforeTouch.equals(await canvas.screenshot({ path: `${root}/maximum-mobile-3d.png` })));
    const beforePlay = await counts(); await page.getByRole('button', { name: 'Play time highlighting', exact: true }).click();
    const oldTime = await page.getByLabel('Highlighted time index', { exact: true }).inputValue();
    await page.waitForFunction(old => (document.querySelector('input[type=range]') as HTMLInputElement).value !== old, oldTime);
    await page.getByRole('button', { name: 'Pause time highlighting', exact: true }).click(); assert.deepEqual(await counts(), beforePlay);
    await page.evaluate(() => { (window as unknown as { __brownianControl: Control }).__brownianControl.mode = 'hold'; });
    const tiny = brownianBridgeExamples.find(e => e.id === 'minimum-positive-factors')!.input;
    await page.getByLabel('All examples', { exact: true }).selectOption('minimum-positive-factors'); await page.locator('[data-brownian-phase="pending"]').waitFor();
    assert.equal(await page.locator('[data-brownian-result]').count(), 0); await page.getByRole('button', { name: 'Cancel calculation', exact: true }).click(); await page.locator('[data-brownian-phase="idle"]').waitFor(); assert.equal((await counts()).active, 0);
    await page.evaluate(() => { (window as unknown as { __brownianControl: Control }).__brownianControl.mode = 'normal'; }); await page.getByRole('button', { name: 'Recalculate', exact: true }).click(); await ready(tiny);
    await page.evaluate(() => { (window as unknown as { __brownianControl: Control }).__brownianControl.mode = 'throw'; }); await page.getByLabel('All examples', { exact: true }).selectOption('maximum-coprime'); await page.locator('[data-brownian-phase="error"]').waitFor(); assert.equal((await counts()).active, 0);
    await page.evaluate(() => { (window as unknown as { __brownianControl: Control }).__brownianControl.mode = 'normal'; }); await page.getByRole('button', { name: 'Recalculate', exact: true }).click(); await ready(maximum);
    await page.getByLabel('Displayed matrix', { exact: true }).selectOption('precision'); await page.getByLabel('Matrix display scale', { exact: true }).selectOption('signed-log');
    const timeSlider = page.getByLabel('Highlighted time index', { exact: true }); await timeSlider.focus(); await timeSlider.press('Home'); for (let i = 0; i < 3; i++) await timeSlider.press('ArrowRight');
    await page.getByRole('button',{name:'4. A point source tilts the mean',exact:true}).click();
    await page.getByLabel('Guided experiment',{exact:true}).selectOption('3');await page.getByRole('button',{name:'Show explanation',exact:true}).click();
    await page.locator('.ai-question').nth(7).locator('input').nth(1).check();await page.locator('.ai-question').nth(7).getByRole('button').click();await page.locator('.ai-completion input').check();
    const stored = await page.evaluate(key => JSON.parse(localStorage.getItem(key)!), brownianBridgeStorageKey); assert.deepEqual(stored.input, maximum); assert.equal(stored.display.selectedTime, 3); assert.equal(stored.display.kind, 'precision');assert.equal(stored.learning.experiment,3);assert.equal(stored.learning.explanationOpen,true);
    const axe = await new AxeBuilder({ page }).include('main').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze(); assert.deepEqual(axe.violations.map(v => v.id), []);
    const beforeReload = await counts(); assert.equal(beforeReload.created, beforeReload.closed);
    await page.reload(); await ready(maximum); assert.equal(await page.getByLabel('Displayed matrix', { exact: true }).inputValue(), 'precision'); assert.equal(await page.getByLabel('Matrix display scale', { exact: true }).inputValue(), 'signed-log'); assert.equal(await page.getByLabel('Highlighted time index', { exact: true }).inputValue(), '3');
    assert.equal(await page.locator('[data-brownian-concept]').getAttribute('data-brownian-concept'),'3');assert.equal(await page.getByLabel('Guided experiment',{exact:true}).inputValue(),'3');assert(await page.locator('[data-brownian-guide-explanation]').isVisible());assert(await page.locator('.ai-question').nth(7).locator('input').nth(1).isChecked());assert(await page.locator('.ai-completion input').isChecked());
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)); assert.deepEqual(errors, []);
    const afterReload = await counts(); assert.equal(afterReload.created, afterReload.closed); await page.locator('[data-brownian-result]').scrollIntoViewIfNeeded(); await page.screenshot({ path: `${root}/maximum-mobile.png` });
    rows.push({ case: 'maximum-mobile-lifecycle-and-reload', language: 'en', maximumIntervals: 32, fullPositionMatrixButtons: 1089, oneInteriorPrecisionThreeD: true, touchRotation: true, playbackDoesNotCalculate: true, invalidDraftRejected: true,fullSignedFractionLengthAccepted: true, cancelRetry: true, errorRetry: true, savedInputAndDisplayRestored: true, learningProgressRestored: true, beforeReload, afterReload, accessibility: true, errors }); await context.close();
    for (const language of ['ko', 'en']) for (const mode of ['corrupt-input', 'corrupt-display', 'unavailable']) {
      const c = await browser.newContext({ viewport: { width: 320, height: 1000 } });
      await c.addInitScript(({ key, language, mode, input }) => {
        localStorage.setItem('suanlab-language', language);
        if (mode === 'unavailable') Storage.prototype.setItem = () => { throw Error('Storage unavailable'); };
        else localStorage.setItem(key, JSON.stringify({ version: mode === 'corrupt-input' ? 2 : 1, input, learning:{step:0,experiment:0,explanationOpen:false,answers:Array(8).fill(-1)}, display: { kind: 'precision', scale: 'linear', selectedTime: 0, selectedCell: 9999 } }));
      }, { key: brownianBridgeStorageKey, language, mode, input: maximum });
      const p = await c.newPage(); await p.goto(url); await p.waitForFunction(key => document.querySelector('[data-brownian-result]')?.getAttribute('data-brownian-result-key') === key, brownianBridgeInputKey(brownianBridgeDefault));
      await p.locator(`[data-brownian-storage="${mode === 'unavailable' ? 'unavailable' : 'corrupt'}"]`).waitFor();
      const a = await new AxeBuilder({ page: p }).include('main').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze(); assert.deepEqual(a.violations.map(v => v.id), []);
      rows.push({ case: mode, language, defaultRestored: true, accessibility: true }); await c.close();
    }
    writeFileSync(`${root}/reader-maximum.json`, JSON.stringify({ rows }, null, 2) + '\n');
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
