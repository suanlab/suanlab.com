import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { quarticAsymptoticSteps, quarticAsymptoticExperiments, quarticAsymptoticQuestions } from '../src/data/quartic-asymptotic-teaching';
import { quarticAsymptoticExamples, quarticAsymptoticDefault, type QuarticAsymptoticInput } from '../src/lib/learning-quartic-asymptotic-input';
import { calculateQuarticAsymptotic } from '../src/lib/learning-quartic-asymptotic';
const root = '.runtime/learning/quartic-asymptotic-teaching';
const { url } = JSON.parse(readFileSync(`${root}/review-server.json`, 'utf8'));
const examples = quarticAsymptoticExamples.map(c => ({ ...c, expected: calculateQuarticAsymptotic(c.input) }));
type Counts = { created: number; terminated: number; active: number };
type Harness = Window & { __counts: Counts };
async function main() {
  const browser = await chromium.launch({ headless: true }), states: unknown[] = []; let accessibility = 0;
  try {
    for (const language of ['ko', 'en'] as const) for (const dark of [false, true]) for (const width of [320, 1440]) {
      const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce', deviceScaleFactor: width === 320 ? 2 : 1 });
      await context.addInitScript(({ language, dark }) => {
        localStorage.setItem('suanlab-language', language); localStorage.setItem('theme', dark ? 'dark' : 'light');
        const Native = Worker, active = new Set<Worker>(), counts = { created: 0, terminated: 0, active: 0 }; (window as unknown as Harness).__counts = counts;
        window.Worker = class extends Native {
          constructor(url: string | URL, options?: WorkerOptions) { super(url, options); counts.created++; active.add(this); counts.active = active.size; }
          terminate() { if (active.delete(this)) { counts.terminated++; counts.active = active.size; super.terminate(); } }
        };
      }, { language, dark });
      const page = await context.newPage(), errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
      await page.goto(url);
      const t = (ko: string, en: string) => language === 'ko' ? ko : en;
      const ready = async (input: QuarticAsymptoticInput) => {
        await page.waitForFunction(key => document.querySelector('[data-quartic-phase]')?.getAttribute('data-quartic-phase') === 'ready' && document.querySelector('[data-quartic-result]')?.getAttribute('data-quartic-result-key') === key, JSON.stringify(input), { timeout: 30000 });
        await page.waitForFunction(() => (window as unknown as Harness).__counts.active === 0);
      };
      const axe = async () => { const result = await new AxeBuilder({ page }).include('main').analyze(); assert.deepEqual(result.violations.map(v => ({ id: v.id, nodes: v.nodes.length })), []); accessibility++; };
      await ready(quarticAsymptoticDefault);
      const select = page.getByLabel(t('전체 예제', 'All examples'), { exact: true });
      for (const c of examples) {
        await select.selectOption(c.id); await ready(c.input);
        const last = c.expected.approximations[c.expected.bestBoundOrderWithinBudget];
        for (const [key, value] of Object.entries({g:c.input.coupling,best:String(c.expected.bestBoundOrderWithinBudget),global:String(c.expected.globallyBestBoundOrder),partial:last.partial,omitted:last.firstOmittedMagnitude,'error-upper':last.errorInterval.upper})) assert.equal(await page.locator(`[data-quartic-metric="${key}"]`).innerText(), value, c.id + ':' + key);
        const bad = await page.locator('svg').evaluateAll(elements => elements.flatMap(el => [...el.querySelectorAll('*')].filter(node => ['d', 'points', 'cx', 'cy', 'x', 'y', 'width', 'height'].some(key => /NaN|Infinity/.test(node.getAttribute(key) ?? ''))).map(node => node.outerHTML)));
        assert.deepEqual(bad, []);
      }
      assert(await page.locator('[data-quartic-off-window]').count());
      await page.getByRole('button', { name: t('첫 예제로 초기화', 'Reset to first example'), exact: true }).click(); await ready(quarticAsymptoticDefault);
      const before = await page.evaluate(() => (window as unknown as Harness).__counts.created);
      for (const [i, step] of quarticAsymptoticSteps.entries()) { await page.getByRole('button', { name: `${i + 1}. ${step.title[language]}`, exact: true }).click(); assert.equal(await page.locator('.information-step > p').first().innerText(), step.body[language]); }
      assert.equal(await page.locator('.information-step').evaluate(el => getComputedStyle(el).animationName), 'none');
      for (const [i, guide] of quarticAsymptoticExperiments.entries()) { await page.getByLabel(t('안내 실험', 'Guided experiment'), { exact: true }).selectOption(String(i)); await page.getByRole('button', { name: t('해설 보기', 'Show explanation'), exact: true }).click(); assert.equal(await page.locator('[data-quartic-guide-explanation]').innerText(), guide.explanation[language]); }
      for (const [i, question] of quarticAsymptoticQuestions.entries()) { const field = page.locator('.information-question').nth(i); await field.locator('input').nth(question.correct).check(); assert((await field.locator('[role=status]').innerText()).includes(question.explanation[language])); }
      const slider = page.getByLabel(t('선택 차수', 'Selected order'), { exact: true }); await slider.fill('2');
      assert.equal(await page.locator('[data-quartic-metric=partial]').innerText(), '613/640');
      assert.equal(await page.locator('[data-quartic-metric=omitted]').innerText(), '693/25600');
      await page.getByLabel(t('적분함수 표시 범위','Kernel plotting window'),{exact:true}).selectOption('8');await page.getByLabel(t('부분합 표시 척도','Partial-sum display scale'),{exact:true}).selectOption('all');await page.getByLabel(t('적분함수의 로그 높이','Log integrand height'),{exact:true}).check();
      assert.equal(await page.evaluate(() => (window as unknown as Harness).__counts.created), before);
      assert(await page.getByRole('button', { name: t('차수 변화 재생', 'Play order changes'), exact: true }).isDisabled());
      await axe();
      const coupling=page.getByLabel(t('결합 g','Coupling g'),{exact:true}),apply=page.getByRole('button',{name:t('입력 적용','Apply input'),exact:true});
      await coupling.fill('-1');await apply.click();assert(await page.getByRole('alert').count());assert.equal(await page.evaluate(()=>(window as unknown as Harness).__counts.created),before);
      await coupling.fill('1/5');await apply.click();const custom={...quarticAsymptoticDefault,coupling:'1/5'};await ready(custom);
      await page.reload(); await ready(custom); assert.equal(await coupling.inputValue(),'1/5');
      await page.getByRole('button', { name: t('첫 예제로 초기화', 'Reset to first example'), exact: true }).click(); await ready(quarticAsymptoticDefault);
      const energy = page.locator('[data-quartic-integrand]'); await energy.scrollIntoViewIfNeeded();
      await energy.getByRole('button', { name: t('3D 보기', 'Show 3D'), exact: true }).click(); const canvas = energy.locator('canvas'); await canvas.waitFor();
      await energy.getByRole('button', { name: t('정면', 'Front'), exact: true }).click(); const initial = await canvas.screenshot(); await canvas.press('ArrowRight'); await canvas.press('ArrowUp'); assert(!initial.equals(await canvas.screenshot()));
      await canvas.evaluate(el => { const extension = (el as HTMLCanvasElement).getContext('webgl2')?.getExtension('WEBGL_lose_context'); if (!extension) throw Error('No context loss extension'); extension.loseContext(); });
      await page.locator('[data-quartic-integrand="2d"] svg').waitFor(); await axe();
      await energy.getByRole('button', { name: t('3D 다시 시도', 'Retry 3D'), exact: true }).click(); await energy.locator('canvas').waitFor();
      await energy.screenshot({ path: `${root}/${language}-${dark ? 'dark' : 'light'}-${width}.png` });
      if (language === 'en') assert(!/[가-힣]/.test(await page.locator('main').innerText()));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      const counts = await page.evaluate(() => (window as unknown as Harness).__counts); assert.equal(counts.created, counts.terminated); assert.deepEqual(errors, []);
      states.push({ language, dark, width, examples: 14, concepts: 8, guides: 6, explainedQuestions: 8, displayControlsDoNotCalculate: true, invalidInputPreservesResult: true, customInputReload: true, keyboardCamera: true, webglLossRetry: true, workerCountsWindow: 'since-final-reload', ...counts, errors });
      console.log(JSON.stringify({ cases: states.length, language, dark, width })); await context.close();
    }
    writeFileSync(`${root}/native-verification.json`, JSON.stringify({ states, accessibility, readerRegistered: false, productionNextAssetsVerified: false }, null, 2) + '\n');
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
