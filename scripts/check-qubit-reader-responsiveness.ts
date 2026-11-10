import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import { qubitTeachingIds as ids } from '../src/lib/learning-qubit-teaching-input';
import { qubitLabBrowserInstrumentation } from './learning/qubit-lab-browser-instrumentation';
const root = '.runtime/learning/quantum-qubit-reader-integration', base = process.env.QUBIT_PREVIEW_BASE ?? 'http://localhost:4176';
type RuntimeWindow = Window & { __qubitMode: string; __qubitLifecycle: { created: number; terminated: number }; __qubitPulse?: { last: number; gap: number; samples: number; stale: number; timer: number } };
async function main() {
  const browser = await chromium.launch({ headless: true }), errors: string[] = [], measurements = [], storageFlows = [];
  try {
    const context = await browser.newContext(); await context.addInitScript({ content: qubitLabBrowserInstrumentation });
    await context.addInitScript(() => localStorage.setItem('suanlab-language', 'en'));
    const page = await context.newPage(); page.on('pageerror', error => errors.push(error.message));
    for (const [kind, preset] of [[ids[0], 8], [ids[0], 7], [ids[1], 3], [ids[2], 5], [ids[3], 6], [ids[3], 7]] as const) {
      await page.goto(`${base}/book/online/physics-math-for-ai/${kind}/`, { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => document.querySelector('[data-qubit-lab]')?.getAttribute('data-qubit-phase') === 'ready');
      await page.getByText('Custom input · edit exact states and operations', { exact: true }).click();
      await page.evaluate(() => {
        const w = window as unknown as RuntimeWindow;
        const pulse = { last: performance.now(), gap: 0, samples: 0, stale: 0, timer: 0 };
        pulse.timer = window.setInterval(() => {
          const now = performance.now();
          if (document.querySelector('[data-qubit-lab]')?.getAttribute('data-qubit-phase') === 'pending') {
            pulse.gap = Math.max(pulse.gap, now - pulse.last); pulse.samples++;
            if (document.querySelector('[data-qubit-result]')) pulse.stale++;
          }
          pulse.last = now;
        }, 10); w.__qubitPulse = pulse;
      });
      await page.getByLabel('All input examples', { exact: true }).selectOption(String(preset));
      await page.waitForFunction(() => document.querySelector('[data-qubit-lab]')?.getAttribute('data-qubit-phase') === 'pending');
      const start = performance.now(); await page.getByLabel('Exact input JSON', { exact: true }).fill('{ typing while the current input is checked'); const typingMilliseconds = performance.now() - start;
      await page.waitForFunction(() => document.querySelector('[data-qubit-lab]')?.getAttribute('data-qubit-phase') === 'ready');
      const pulse = await page.evaluate(() => { const pulse = (window as unknown as RuntimeWindow).__qubitPulse!; clearInterval(pulse.timer); return pulse; });
      const lifecycle = await page.evaluate(() => (window as unknown as RuntimeWindow).__qubitLifecycle);
      assert.equal(lifecycle.created, lifecycle.terminated); assert.equal(pulse.stale, 0); assert(pulse.samples > 0);
      assert(pulse.gap < 500, `${kind}/${preset} heartbeat`); assert(typingMilliseconds < 1000);
      measurements.push({ kind, preset, typingMilliseconds, maximumPendingHeartbeatGapMilliseconds: pulse.gap, pendingSamples: pulse.samples, staleReady: pulse.stale, lifecycle });
    }
    await page.evaluate(() => { (window as unknown as RuntimeWindow).__qubitMode = 'validation-hang'; });
    await page.getByLabel('All input examples', { exact: true }).selectOption('6');
    await page.waitForFunction(() => document.querySelector('[data-qubit-calculation-status]')?.textContent?.includes('Independently checking'));
    const start = performance.now(); await page.getByRole('button', { name: 'Cancel calculation', exact: true }).click();
    const cancelMilliseconds = performance.now() - start;
    assert.equal(await page.locator('[data-qubit-result]').count(), 0);
    const stopped = await page.evaluate(() => (window as unknown as RuntimeWindow).__qubitLifecycle); assert.equal(stopped.created, stopped.terminated);
    await page.evaluate(() => { (window as unknown as RuntimeWindow).__qubitMode = 'normal'; }); await page.getByRole('button', { name: 'Recalculate', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('[data-qubit-lab]')?.getAttribute('data-qubit-phase') === 'ready');
    assert(cancelMilliseconds < 1000); await context.close();
    for (const kind of ids) for (const language of ['ko', 'en']) for (const denied of [false, true]) {
      const context = await browser.newContext();
      await context.addInitScript({ content: qubitLabBrowserInstrumentation });
      await context.addInitScript(({ kind, language, denied }) => {
        localStorage.setItem('suanlab-language', language); const key = 'suanlab-qubit-' + kind;
        if (!denied) localStorage.setItem(key, 'invalid saved JSON');
        else {
          const get = Storage.prototype.getItem, set = Storage.prototype.setItem;
          Storage.prototype.getItem = function (name) { if (name === key) throw Error('denied'); return get.call(this, name); };
          Storage.prototype.setItem = function (name, value) { if (name === key) throw Error('denied'); set.call(this, name, value); };
        }
      }, { kind, language, denied });
      const page = await context.newPage(); page.on('pageerror', error => errors.push(error.message));
      await page.goto(`${base}/book/online/physics-math-for-ai/${kind}/`, { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => document.querySelector('[data-qubit-lab]')?.getAttribute('data-qubit-phase') === 'ready');
      await page.waitForFunction(language => document.querySelector('.ai-reader')?.getAttribute('lang') === language, language);
      const text = await page.locator('[data-qubit-lab]').innerText();
      assert(text.includes(denied ? language === 'ko' ? '저장을 사용할 수 없습니다' : 'storage is unavailable' : language === 'ko' ? '첫 예제로 복원' : 'first example was restored'));
      if (language === 'en') assert(!/[가-힣]/.test(text));
      const lifecycle = await page.evaluate(() => (window as unknown as RuntimeWindow).__qubitLifecycle); assert.equal(lifecycle.created, lifecycle.terminated);
      storageFlows.push({ kind, language, denied, restoredAndCalculated: true, workersReleased: true }); await context.close();
    }
    assert.deepEqual(errors, []);
    writeFileSync(root + '/reader-responsiveness.json', JSON.stringify({ measurements, validationCancellationMilliseconds: cancelMilliseconds, validationCancellationAndRetry: true, storageFlows, actualReaderVerified: true, qaWorkerFactoryOverride: false, errors }, null, 2) + '\n');
    console.log(JSON.stringify({ preparations: measurements.length, storageFlows: storageFlows.length, cancelMilliseconds, typingMilliseconds: measurements.map(m => m.typingMilliseconds), heartbeatMilliseconds: measurements.map(m => m.maximumPendingHeartbeatGapMilliseconds) }));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
