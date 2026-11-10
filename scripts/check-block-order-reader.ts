/** Actual statically exported reader; assets are served unchanged from candidate/out. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { blockOrderReader } from '../src/data/block-order-reader';
import { blockOrderExamples } from '../src/lib/learning-block-order-input';
import { blockOrderInputKey } from '../src/lib/learning-block-order-protocol';
import { blockOrderTeachingId as id } from '../src/lib/learning-block-order-teaching';
import { emptyBlockOrderState, blockOrderStorageKey } from '../src/lib/learning-block-order-state';
const root = '.runtime/learning/block-order-reader-integration'; mkdirSync(root,{ recursive: true });
const candidate = process.env.BLOCK_ORDER_READER_CANDIDATE ?? '/tmp/suanlab-block-order-reader-candidate-20261111';
async function main() {
  const server = createServer((request,response) => {
    try {
      const relative = decodeURIComponent(new URL(request.url!,'http://localhost').pathname); if (relative.includes('..')) throw Error('Path');
      const full = path.join(candidate,'out',relative.endsWith('/') ? relative+'index.html' : relative);
      const mime: Record<string,string> = { '.js': 'text/javascript','.css': 'text/css','.woff2': 'font/woff2','.svg': 'image/svg+xml','.png': 'image/png','.jpg': 'image/jpeg','.jpeg': 'image/jpeg','.webp': 'image/webp','.ico': 'image/x-icon','.txt': 'text/plain' };
      response.setHeader('Content-Type',mime[path.extname(full)] ?? 'text/html'); response.end(readFileSync(full));
    } catch { response.statusCode = 404; response.end('Not found'); }
  });
  await new Promise<void>(resolve => server.listen(0,'127.0.0.1',resolve)); const address = server.address(); assert(address && typeof address !== 'string');
  const base = `http://127.0.0.1:${address.port}`, url = `${base}/book/online/physics-math-for-ai/${id}/`;
  const browser = await chromium.launch({ headless: true,args: ['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] }), errors: string[] = [], rows = [];
  try {
    for (const language of ['ko','en'] as const) for (const theme of ['light','dark']) for (const width of [320,1440]) {
      const context = await browser.newContext({ viewport: { width,height: 950 },reducedMotion: 'reduce' });
      await context.addInitScript({ content: `globalThis.__name=fn=>fn;localStorage.setItem('suanlab-language',${JSON.stringify(language)});localStorage.setItem('theme',${JSON.stringify(theme)});const Native=Worker,live=new Set();globalThis.__ports={created:0,closed:0,active:0,urls:[]};globalThis.Worker=class extends Native{constructor(url,options){super(url,options);live.add(this);__ports.created++;__ports.active=live.size;__ports.urls.push(String(url));}terminate(){if(live.delete(this)){__ports.closed++;__ports.active=live.size;}super.terminate();}};` });
      const page = await context.newPage(); page.on('pageerror',e => errors.push(e.message));
      const ready = async (key?: string) => { await page.waitForFunction(key => { const lab = document.querySelector('[data-block-status]'); return lab?.getAttribute('data-block-status') === 'ready' && (!key || lab.getAttribute('data-block-input-key') === key); },key,{ timeout: 120000 }); };
      const t = (ko: string,en: string) => language === 'ko' ? ko : en;
      assert.equal((await page.goto(url))!.status(),200); await ready(); await page.waitForFunction(l => document.querySelector('.ai-reader')?.getAttribute('lang') === l,language);
      assert.equal(await page.locator('h1').innerText(),blockOrderReader.unit.title[language]); assert.equal(await page.locator('.ai-prose h2').count(),20);
      assert.equal(await page.locator('.ai-question').count(),8); assert.equal(await page.locator('fieldset.information-question').count(),0);
      for (const [i,step] of blockOrderReader.guide.steps.entries()) {
        await page.locator('[data-block-concept] .ai-concept-steps button').nth(i).click();
        assert.equal(await page.locator('[data-block-concept] .information-step h3').innerText(),step.title[language]);
      }
      for (const [i,q] of blockOrderReader.unit.questions.entries()) {
        const field = page.locator('.ai-question').nth(i), radios = field.locator('input[type=radio]');
        await radios.nth((q.answer+1)%3).check(); await field.getByRole('button').click();
        assert((await field.locator('.ai-answer strong').innerText()).startsWith(t('다시 생각해보세요','Try again.')));
        await radios.nth(q.answer).check(); await field.getByRole('button').click(); assert((await field.locator('.ai-answer').innerText()).includes(q.explanation[language]));
      }
      for (const [i,g] of blockOrderReader.guide.experiments.entries()) {
        await page.getByLabel(t('안내 실험','Guided experiment'),{ exact: true }).selectOption(String(i));
        await page.getByRole('button',{ name: t('해설 보기','Show explanation'),exact: true }).click(); assert.equal(await page.locator('[data-block-guide-explanation]').innerText(),g.explain[language]);
      }
      const lab = page.locator('[data-block-status]'), example = blockOrderExamples.find(e => e.id === 'two-dimensional')!;
      await lab.getByLabel(t('전체 예제','All examples'),{ exact: true }).selectOption(example.id); await ready(blockOrderInputKey(example.input));
      await lab.getByLabel(t('공간에 표시할 값','Quantity shown in space'),{ exact: true }).selectOption('conditional');
      await lab.getByLabel(t('살펴볼 미시 예제','Microscopic example to inspect'),{ exact: true }).selectOption('1');
      await lab.locator('summary').filter({ hasText: t('전체 프로필 분포와 미시 상태 수','Full profile distribution and microscopic counts') }).click();
      await lab.getByRole('button',{ name: t('다음 프로필 페이지','Next profile page'),exact: true }).click();
      await lab.getByLabel(t('불일치 활동도 x','Disagreement activity x'),{ exact: true }).fill('1e-');
      await page.locator('.ai-completion input').check();
      await page.waitForFunction(k => { const raw = localStorage.getItem(k); return raw && JSON.parse(raw).model.draft.disagreementActivity === '1e-'; },blockOrderStorageKey);
      await page.reload(); await ready(blockOrderInputKey(example.input)); await page.waitForFunction(l => document.querySelector('.ai-reader')?.getAttribute('lang') === l,language);
      assert.equal(await lab.getByLabel(t('불일치 활동도 x','Disagreement activity x'),{ exact: true }).inputValue(),'1e-');
      assert.equal(await lab.getByLabel(t('공간에 표시할 값','Quantity shown in space'),{ exact: true }).inputValue(),'conditional');
      assert.equal(await lab.getByLabel(t('살펴볼 미시 예제','Microscopic example to inspect'),{ exact: true }).inputValue(),'1');
      await lab.locator('summary').filter({ hasText: t('전체 프로필 분포와 미시 상태 수','Full profile distribution and microscopic counts') }).click();
      assert((await lab.locator('[data-block-profile-page]').innerText()).includes('2/25'));
      assert(await page.locator('.ai-completion input').isChecked());
      for (const [i,q] of blockOrderReader.unit.questions.entries()) assert(await page.locator('.ai-question').nth(i).locator('input[type=radio]').nth(q.answer).isChecked());
      assert.equal(await page.locator('[data-block-concept]').getAttribute('data-block-concept'),'5');
      await lab.getByRole('button',{ name: t('3D 격자 보기','Show 3D lattice'),exact: true }).click(); await lab.locator('[data-block-spatial-plot="3d"]').waitFor();
      await lab.locator('canvas').focus(); await page.keyboard.press('ArrowRight'); await page.keyboard.press('Home');
      if (language === 'en' && theme === 'light' && width === 1440) await page.locator('[data-block-concept]').screenshot({ path: `${root}/reader-concept-en.png` });
      if (width === 320) await lab.locator('[data-block-spatial-plot]').screenshot({ path: `${root}/reader-spatial-${language}-${theme}.png` });
      await lab.getByRole('button',{ name: t('3D 격자 닫기','Close 3D lattice'),exact: true }).click();
      if (language === 'en') assert(!/[가-힣]/.test(await page.locator('.ai-reader').innerText()));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1));
      const axe = await new AxeBuilder({ page }).include('.ai-reader').analyze();
      writeFileSync(`${root}/axe-${language}-${theme}-${width}.json`,JSON.stringify(axe.violations,null,2)+'\n'); assert.deepEqual(axe.violations,[]);
      const ports = await page.evaluate(() => (window as unknown as { __ports: { created: number;closed: number;active: number;urls: string[] } }).__ports);
      assert.equal(ports.active,0); assert.equal(ports.created,ports.closed); assert(ports.urls.every(u => u.startsWith(base+'/_next/static/') && u.endsWith('.js')));
      rows.push({ language,theme,width,restoredDraft: true,restoredDisplay: true,restoredStudy: true,axeViolations: 0,ports,portCounterScope: 'final document after reload' }); await context.close();
    }
    for (const variant of ['corrupt','denied']) {
      const context = await browser.newContext({ viewport: { width: 320,height: 950 },reducedMotion: 'reduce' });
      await context.addInitScript({ content: variant === 'corrupt' ? `localStorage.setItem(${JSON.stringify(blockOrderStorageKey)},'{');` : "Object.defineProperty(Storage.prototype,'getItem',{value(){throw Error('blocked')}});Object.defineProperty(Storage.prototype,'setItem',{value(){throw Error('blocked')}});" });
      const page = await context.newPage(); page.on('pageerror',e => errors.push(e.message)); await page.goto(url); await page.locator('[data-block-status="ready"]').waitFor({ timeout: 120000 });
      assert.equal(await page.locator('[data-block-storage]').getAttribute('data-block-storage'),variant === 'corrupt' ? 'corrupt' : 'unavailable'); await context.close();
    }
    const initialized = structuredClone(emptyBlockOrderState({ steps: 6,guides: 6,questions: 8,options: 3 }));
    initialized.input = { ...initialized.input,disagreementActivity: '2',selectedUpCounts: [0,2] }; initialized.draft.disagreementActivity = 'pending';
    initialized.display = { mode: 'conditional',example: 0,flat: false,page: 0 };
    const restore = await browser.newContext({ reducedMotion: 'reduce' });
    await restore.addInitScript({ content: `globalThis.__writes=[];globalThis.__requests=[];const Native=Worker;globalThis.Worker=class extends Native{postMessage(value){if(value?.input?.disagreementActivity)__requests.push(value.input.disagreementActivity);super.postMessage(value)}};const set=Storage.prototype.setItem;set.call(localStorage,'suanlab-language','en');set.call(localStorage,${JSON.stringify(blockOrderStorageKey)},${JSON.stringify(JSON.stringify({ version: 1,model: initialized }))});Storage.prototype.setItem=function(k,v){if(k===${JSON.stringify(blockOrderStorageKey)})__writes.push(JSON.parse(v).model.input.disagreementActivity);return set.call(this,k,v)};` });
    const restored = await restore.newPage(); restored.on('pageerror',e => errors.push(e.message)); await restored.goto(url); await restored.locator('[data-block-status="ready"]').waitFor({ timeout: 120000 });
    assert.equal(await restored.getByLabel('Disagreement activity x',{ exact: true }).inputValue(),'pending');
    assert((await restored.locator('[data-block-applied]').innerText()).includes('x=2'));
    assert.equal((await restored.evaluate(() => (window as unknown as { __writes: string[] }).__writes))[0],'2');
    assert.equal((await restored.evaluate(() => (window as unknown as { __requests: string[] }).__requests))[0],'2'); await restore.close();
    const life = await browser.newContext({ reducedMotion: 'reduce' });
    await life.addInitScript({ content: "globalThis.__name=fn=>fn;localStorage.setItem('suanlab-language','en');const Native=Worker,active=new Set();globalThis.__ports={created:0,closed:0,active:0};globalThis.Worker=class extends Native{constructor(url,options){super(url,options);active.add(this);__ports.created++;__ports.active=active.size;}terminate(){if(active.delete(this)){__ports.closed++;__ports.active=active.size;}super.terminate();}};" });
    await life.route(rows[0].ports.urls[0],async route => { const response = await route.fetch(); await new Promise(r => setTimeout(r,1500)); try { await route.fulfill({ response }); } catch { /* A terminated Worker may abandon this unchanged asset request. */ } });
    const lp = await life.newPage(); lp.on('pageerror',e => errors.push(e.message)); await lp.goto(url);
    await lp.locator('[data-block-status="pending"]').waitFor(); await lp.getByRole('button',{ name: 'Cancel calculation',exact: true }).click(); await lp.locator('[data-block-status="idle"]').waitFor();
    await lp.getByRole('button',{ name: 'Recalculate',exact: true }).click(); await lp.locator('[data-block-status="ready"]').waitFor({ timeout: 120000 });
    await lp.getByLabel('All examples',{ exact: true }).selectOption('two-dimensional'); await lp.locator('[data-block-status="pending"]').waitFor();
    await lp.evaluate(() => { Object.defineProperty(document,'visibilityState',{ configurable: true,get: () => 'hidden' }); document.dispatchEvent(new Event('visibilitychange')); });
    await lp.locator('[data-block-status="idle"]').waitFor(); assert.equal(await lp.locator('[data-block-result]').count(),0);
    await lp.evaluate(() => { Object.defineProperty(document,'visibilityState',{ configurable: true,get: () => 'visible' }); document.dispatchEvent(new Event('visibilitychange')); });
    await lp.locator('[data-block-status="ready"]').waitFor({ timeout: 120000 });
    const lifePorts = await lp.evaluate(() => (window as unknown as { __ports: { created: number;closed: number;active: number } }).__ports);
    assert.equal(lifePorts.active,0); assert.equal(lifePorts.created,lifePorts.closed); await life.close();
    assert.deepEqual(errors,[]);
    writeFileSync(`${root}/reader-browser.json`,JSON.stringify({ status: 'actual-static-reader-verified',rows,corruptState: true,deniedStorage: true,restoreBeforeWrite: true,
      restoreBeforeCalculation: true,cancelAndRetry: true,simulatedVisibility: true,lifePorts,productionWorkerAssets: true,errors },null,2)+'\n');
    console.log(JSON.stringify({ readerStates: rows.length,storageCases: 3,errors }));
  } finally { await browser.close(); await new Promise<void>(resolve => server.close(() => resolve())); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
