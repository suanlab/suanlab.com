import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { chromium, type Page } from 'playwright';
import { qubitTeachingIds, qubitTeachingInput } from '../src/lib/learning-qubit-teaching-input';
import { qubitLabBrowserInstrumentation } from './learning/qubit-lab-browser-instrumentation';

const runtime = '.runtime/learning/quantum-qubit-labs';
async function main() {
  const graph = JSON.parse(readFileSync(runtime + '/dependency-graphs.json', 'utf8')) as { files: Record<string, string>; css: { path: string }[] };
  const browser = await chromium.launch({ headless: true }), errors: string[] = [], flows: unknown[] = [];
  const phase = async (page: Page, desired: string) => page.waitForFunction(value => document.querySelector('[data-qubit-lab]')?.getAttribute('data-qubit-phase') === value, desired);
  const mode = async (page: Page, value: string) => page.evaluate(value => { (window as unknown as { __qubitMode: string }).__qubitMode = value; }, value);
  const released = async (page: Page) => { const life = await page.evaluate(() => (window as unknown as { __qubitLifecycle: { created: number; terminated: number } }).__qubitLifecycle); assert.equal(life.created, life.terminated); };
  const mount = async (page: Page, kind: string, language: string) => {
    await page.goto(`http://localhost:4175/__qubit_lab__/?unit=${kind}&lang=${language}`); await phase(page, 'ready');
    if (await page.locator('main').getAttribute('data-language') !== language) await page.locator('[data-preview-language]').click();
    await page.waitForFunction(value => document.querySelector('main')?.getAttribute('data-language') === value, language);
  };
  try {
    for (const language of ['ko', 'en']) for (const kind of qubitTeachingIds) {
      for (const storageMode of ['corrupt', 'denied', 'write-denied']) {
        const context = await browser.newContext({ viewport: { width: 320, height: 1000 }, reducedMotion: 'reduce' });
        await context.addInitScript({ content: qubitLabBrowserInstrumentation });
        await context.addInitScript(({ key, storageMode, saved }) => {
          if (storageMode === 'corrupt') localStorage.setItem(key, '{"version":1,"value":{"kind":"wrong-family"}}');
          else if (storageMode === 'denied') { Storage.prototype.getItem = () => { throw new Error('denied'); }; Storage.prototype.setItem = () => { throw new Error('denied'); }; }
          else { localStorage.setItem(key, saved); Storage.prototype.setItem = () => { throw new Error('write denied'); }; }
        }, { key: 'suanlab-qubit-' + kind, storageMode, saved: JSON.stringify({ version: 1, value: qubitTeachingInput(kind, 1) }) });
        const page = await context.newPage(); await routes(page); await mount(page, kind, language);
        const text = await page.locator('main').innerText();
        assert(text.includes(storageMode === 'corrupt' ? language === 'ko' ? '첫 예제로 복원' : 'first example was restored' : language === 'ko' ? '저장을 사용할 수 없습니다' : 'storage is unavailable'));
        if (language === 'en') assert(!/[가-힣]/.test(text));
        if (storageMode === 'write-denied') {
          const all = page.getByLabel(language === 'ko' ? '전체 입력 예제' : 'All input examples', { exact: true });
          assert.equal(await all.inputValue(), '1'); await all.selectOption('0'); await phase(page, 'ready');
          await page.reload(); await phase(page, 'ready');
          if (await page.locator('main').getAttribute('data-language') !== language) await page.locator('[data-preview-language]').click();
          assert.equal(await all.inputValue(), '1');
        }
        await released(page); flows.push({ kind, language, storageMode, recovered: true }); await context.close();
      }
      const context = await browser.newContext({ viewport: { width: 320, height: 1000 }, reducedMotion: 'reduce' });
      await context.addInitScript({ content: qubitLabBrowserInstrumentation });
      const page = await context.newPage(); await routes(page); await mount(page, kind, language);
      const all = page.getByLabel(language === 'ko' ? '전체 입력 예제' : 'All input examples', { exact: true });
      const cancel = page.getByRole('button', { name: language === 'ko' ? '계산 취소' : 'Cancel calculation', exact: true });
      const retry = page.getByRole('button', { name: language === 'ko' ? '다시 계산' : 'Recalculate', exact: true });
      for (const failure of ['hang', 'validation-hang', 'unavailable', 'broken', 'tamper']) {
        await mode(page, failure); await all.selectOption(await all.inputValue() === '0' ? '1' : '0');
        if (failure === 'hang' || failure === 'validation-hang') {
          await phase(page, 'pending');
          assert.equal(await page.locator('[data-qubit-result]').count(), 0);
          if (failure === 'validation-hang') await page.waitForFunction(() => document.querySelector('[data-qubit-calculation-status]')?.textContent?.includes('Independently') || document.querySelector('[data-qubit-calculation-status]')?.textContent?.includes('독립적으로'));
          await cancel.click(); await phase(page, 'idle');
        } else {
          await phase(page, 'error'); assert.equal(await page.locator('[data-qubit-result]').count(), 0);
        }
        await released(page); await mode(page, 'normal'); await retry.click(); await phase(page, 'ready'); await released(page);
        flows.push({ kind, language, failure, failureHiddenResult: true, workersReleased: true, retry: true });
      }
      // Keep structurally valid but nonphysical input distinct from JSON errors.
      await page.getByText(language === 'ko' ? '직접 입력 · 정확한 상태와 연산 수정' : 'Custom input · edit exact states and operations', { exact: true }).click();
      const editor = page.getByLabel(language === 'ko' ? '정확 입력 JSON' : 'Exact input JSON', { exact: true });
      const invalid = JSON.parse(await editor.inputValue());
      const zero = { real: ['0', '0'], imaginary: ['0', '0'] };
      if (invalid.kind === 'circuit') invalid.input = invalid.input.map(() => zero); else invalid.state = [[zero, zero], [zero, zero]];
      await editor.fill(JSON.stringify(invalid)); await page.getByRole('button', { name: language === 'ko' ? '입력 적용' : 'Apply input', exact: true }).click();
      await phase(page, 'error'); assert.equal(await page.locator('[data-qubit-result]').count(), 0); await released(page);
      await page.getByRole('button', { name: language === 'ko' ? '첫 예제로 초기화' : 'Reset to first example', exact: true }).click(); await phase(page, 'ready');
      const custom = JSON.parse(await editor.inputValue());
      const one = { real: ['1', '0'], imaginary: ['0', '0'] };
      if (custom.kind === 'circuit') custom.input = [{ real: ['2', '0'], imaginary: ['0', '0'] }, zero];
      else if (custom.kind === 'channel') custom.state = [[one, zero], [zero, zero]];
      else if (custom.kind === 'environment') custom.parameter = '1/3';
      else custom.state = [[{ real: ['2/3', '0'], imaginary: ['0', '0'] }, zero], [zero, { real: ['1/3', '0'], imaginary: ['0', '0'] }]];
      await editor.fill(JSON.stringify(custom)); await page.getByRole('button', { name: language === 'ko' ? '입력 적용' : 'Apply input', exact: true }).click(); await phase(page, 'ready');
      assert.equal(await all.inputValue(), 'custom'); await page.reload(); await phase(page, 'ready'); assert.equal(await all.inputValue(), 'custom');
      assert.deepEqual(JSON.parse(await editor.inputValue()), custom); flows.push({ kind, language, customInputAndReload: true });
      // Fake main-page clock exercises the real configured 30s UI boundary;
      // the independent native Worker suite separately uses wall-clock timers.
      await page.clock.install({ time: new Date('2030-01-01T00:00:00Z') });
      await page.clock.pauseAt(new Date('2030-01-01T00:00:01Z'));
      await mode(page, 'hang'); await all.selectOption('1'); await phase(page, 'pending');
      await page.clock.fastForward(29999); assert.equal(await page.locator('[data-qubit-lab]').getAttribute('data-qubit-phase'), 'pending');
      await page.clock.fastForward(1); await phase(page, 'error'); await released(page); await page.clock.resume();
      await mode(page, 'normal'); await retry.click(); await phase(page, 'ready');
      if (kind === qubitTeachingIds[0]) {
        const before = await page.evaluate(() => (window as unknown as { __qubitLifecycle: { created: number } }).__qubitLifecycle.created);
        const slider = page.getByLabel(language === 'ko' ? '회로 단계' : 'Circuit step', { exact: true });
        await slider.press('Home'); assert.equal(await page.locator('[data-qubit-prefix]').getAttribute('data-qubit-prefix'), '0');
        await slider.press('End');
        const after = await page.evaluate(() => (window as unknown as { __qubitLifecycle: { created: number } }).__qubitLifecycle.created);
        assert.equal(after, before); flows.push({ kind, language, prefixKeyboardUsesPrecomputedResponse: true });
        const play = page.getByRole('button', { name: language === 'ko' ? '단계 재생' : 'Play steps', exact: true });
        assert(await play.isDisabled());
        await page.emulateMedia({ reducedMotion: 'no-preference' }); await play.scrollIntoViewIfNeeded();
        await play.click(); await page.waitForFunction(() => document.querySelector('[data-qubit-prefix]')?.getAttribute('data-qubit-prefix') === '1');
        await page.emulateMedia({ reducedMotion: 'reduce' });
        await page.waitForFunction(() => document.querySelector('[data-qubit-playback]')?.getAttribute('data-qubit-playback') === 'stopped');
        const stoppedAt = await page.locator('[data-qubit-prefix]').getAttribute('data-qubit-prefix');
        await page.waitForTimeout(900); assert.equal(await page.locator('[data-qubit-prefix]').getAttribute('data-qubit-prefix'), stoppedAt);
        assert(await play.isDisabled());
        await page.emulateMedia({ reducedMotion: 'no-preference' }); await play.click();
        await page.evaluate(() => { const spacer = document.createElement('div'); spacer.id = 'qa-spacer'; spacer.style.height = '2000px'; document.querySelector('main')!.prepend(spacer); window.scrollTo(0, 0); });
        await page.waitForFunction(() => document.querySelector('[data-qubit-playback]')?.getAttribute('data-qubit-playback') === 'stopped');
        await page.evaluate(() => document.querySelector('#qa-spacer')?.remove()); await page.emulateMedia({ reducedMotion: 'reduce' });
        assert.equal(await page.evaluate(() => (window as unknown as { __qubitLifecycle: { created: number } }).__qubitLifecycle.created), before);
        flows.push({ kind, language, playbackUsesComputedPrefixes: true, reducedMotionStopsPlayback: true, offscreenStopsPlayback: true });
      }
      if (kind === qubitTeachingIds[1]) {
        await all.selectOption('3'); await phase(page, 'ready'); await page.getByLabel(language === 'ko' ? '자세히 볼 상태' : 'State to inspect', { exact: true }).selectOption('bellOutput');
        assert.equal(await page.locator('[data-qubit-bloch]').count(), 0);
        flows.push({ kind, language, invalidBellOutputHasNoBloch: true });
      }
      if (kind === qubitTeachingIds[3]) {
        await all.selectOption('3'); await phase(page, 'ready'); await page.getByLabel(language === 'ko' ? '자세히 볼 상태' : 'State to inspect', { exact: true }).selectOption('conditionalAfterFeedback1');
        assert.equal(await page.locator('[data-qubit-bloch]').count(), 0);
        assert((await page.locator('[data-qubit-result]').innerText()).includes(language === 'ko' ? '정의되지 않습니다' : 'undefined for a zero-probability'));
        await all.selectOption('7'); await phase(page, 'ready'); const inspect = page.getByLabel(language === 'ko' ? '자세히 볼 상태' : 'State to inspect', { exact: true });
        const blochTable = page.locator('table').filter({ has: page.locator('caption', { hasText: language === 'ko' ? 'Bloch 성분과 정확 값' : 'Bloch components and exact values' }) });
        assert.equal(await blochTable.locator('tbody tr').nth(1).locator('code').innerText(), '1');
        await inspect.selectOption('dephasedRecord'); assert.equal(await blochTable.locator('tbody tr').nth(1).locator('code').innerText(), '0');
        flows.push({ kind, language, zeroBranchHasNoBloch: true, deferredRecordYOneToZero: true });
      }
      // Unmount while calculation is pending; neither result nor Worker leaks.
      await mode(page, 'hang'); await all.selectOption('0'); await phase(page, 'pending');
      const next = qubitTeachingIds[(qubitTeachingIds.indexOf(kind) + 1) % qubitTeachingIds.length];
      await page.locator('#unit-select').selectOption(next); await phase(page, 'pending');
      assert.equal(await page.locator('[data-qubit-result]').count(), 0);
      await cancel.click(); await phase(page, 'idle'); await released(page);
      flows.push({ kind, language, physicalInputErrorAndReset: true, thirtySecondClockBoundary: true, unmountCancelsWork: true });
      console.log(JSON.stringify({ kind, language, recoveryFlows: flows.length })); await context.close();
    }
    assert.deepEqual(errors, []);
    writeFileSync(runtime + '/native-recovery.json', JSON.stringify({ flows, errors, actualNativeWorkers: true, uiDeadlineClockSimulated: true, actualReaderVerified: false }, null, 2) + '\n');
    console.log(JSON.stringify({ recoveryFlows: flows.length }));
  } finally { await browser.close(); }

  async function routes(page: Page) {
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/__qubit_lab__/**', route => {
      const name = new URL(route.request().url()).pathname.replace('/__qubit_lab__/', '');
      if (name === 'broken.js') return route.fulfill({ contentType: 'text/javascript', body: 'throw Error("intentional Worker startup failure")' });
      if (graph.files[name]) return route.fulfill({ contentType: name.endsWith('.css') ? 'text/css' : 'text/javascript', body: readFileSync(runtime + '/' + name) });
      return route.fulfill({ contentType: 'text/html', body: `<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Recovery check</title>${graph.css.map(css => `<link rel="stylesheet" href="${css.path.slice(3)}">`).join('')}<link rel="stylesheet" href="/__qubit_lab__/assets/preview.css"></head><body><div id="root"></div><script type="module" src="/__qubit_lab__/assets/preview.js"></script></body></html>` });
    });
    await page.route('**/_next/static/css/*.css', route => route.fulfill({ contentType: 'text/css', body: readFileSync('out' + new URL(route.request().url()).pathname) }));
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
