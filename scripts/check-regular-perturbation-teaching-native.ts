import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { regularPerturbationSteps, regularPerturbationExperiments, regularPerturbationQuestions } from '../src/data/regular-perturbation-teaching';
import { regularPerturbationExamples, regularPerturbationDefault, type RegularPerturbationInput } from '../src/lib/learning-regular-perturbation-input';
import { calculateRegularPerturbation } from '../src/lib/learning-regular-perturbation';
const root = '.runtime/learning/regular-perturbation-teaching';
const { url } = JSON.parse(readFileSync(`${root}/review-server.json`, 'utf8'));
const examples = regularPerturbationExamples.map(c => ({ ...c, expected: calculateRegularPerturbation(c.input) }));
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
      const ready = async (input: RegularPerturbationInput) => {
        await page.waitForFunction(key => document.querySelector('[data-perturbation-phase]')?.getAttribute('data-perturbation-phase') === 'ready' && document.querySelector('[data-perturbation-result]')?.getAttribute('data-perturbation-result-key') === key, JSON.stringify(input), { timeout: 30000 });
        await page.waitForFunction(() => (window as unknown as Harness).__counts.active === 0);
      };
      const axe = async () => { const result = await new AxeBuilder({ page }).include('main').analyze(); assert.deepEqual(result.violations.map(v => ({ id: v.id, nodes: v.nodes.length })), []); accessibility++; };
      await ready(regularPerturbationDefault);
      const select = page.getByLabel(t('전체 예제', 'All examples'), { exact: true });
      for (const c of examples) {
        await select.selectOption(c.id); await ready(c.input);
        const last = c.expected.approximations.at(-1)!;
        for (const [key, value] of Object.entries({ g: c.expected.dimensionlessCoupling, scale: c.expected.linearScale, partial: last.approximation, residual: last.residual, bound: last.errorBound, 'energy-bound': last.energyGapBound })) assert.equal(await page.locator(`[data-perturbation-metric="${key}"]`).innerText(), value, c.id + ':' + key);
        const bad = await page.locator('svg').evaluateAll(elements => elements.flatMap(el => [...el.querySelectorAll('*')].filter(node => ['d', 'points', 'cx', 'cy', 'x', 'y', 'width', 'height'].some(key => /NaN|Infinity/.test(node.getAttribute(key) ?? ''))).map(node => node.outerHTML)));
        assert.deepEqual(bad, []);
      }
      assert(await page.locator('[data-perturbation-off-window]').count());
      await page.getByRole('button', { name: t('첫 예제로 초기화', 'Reset to first example'), exact: true }).click(); await ready(regularPerturbationDefault);
      const before = await page.evaluate(() => (window as unknown as Harness).__counts.created);
      for (const [i, step] of regularPerturbationSteps.entries()) { await page.getByRole('button', { name: `${i + 1}. ${step.title[language]}`, exact: true }).click(); assert.equal(await page.locator('.information-step > p').first().innerText(), step.body[language]); }
      assert.equal(await page.locator('.information-step').evaluate(el => getComputedStyle(el).animationName), 'none');
      for (const [i, guide] of regularPerturbationExperiments.entries()) { await page.getByLabel(t('안내 실험', 'Guided experiment'), { exact: true }).selectOption(String(i)); await page.getByRole('button', { name: t('해설 보기', 'Show explanation'), exact: true }).click(); assert.equal(await page.locator('[data-perturbation-guide-explanation]').innerText(), guide.explanation[language]); }
      for (const [i, question] of regularPerturbationQuestions.entries()) { const field = page.locator('.information-question').nth(i); await field.locator('input').nth(question.correct).check(); assert((await field.locator('[role=status]').innerText()).includes(question.explanation[language])); }
      const slider = page.getByLabel(t('선택 차수', 'Selected order'), { exact: true }); await slider.fill('2');
      assert.equal(await page.locator('[data-perturbation-metric=partial]').innerText(), '93/100');
      assert.equal(await page.locator('[data-perturbation-metric=bound]').innerText(), '104357/10000000');
      await page.getByLabel(t('그림의 변위 구간', 'Displacement plotting window'), { exact: true }).selectOption('linear');
      assert.equal(await page.evaluate(() => (window as unknown as Harness).__counts.created), before);
      assert(await page.getByRole('button', { name: t('차수 변화 재생', 'Play order changes'), exact: true }).isDisabled());
      await axe();
      const k = page.getByLabel(t('강성 k', 'Stiffness k'), { exact: true }), h = page.getByLabel(t('원천 h', 'Source h'), { exact: true }), apply = page.getByRole('button', { name: t('입력 적용', 'Apply input'), exact: true });
      await k.fill('0'); await apply.click(); assert(await page.getByRole('alert').count()); assert.equal(await page.evaluate(() => (window as unknown as Harness).__counts.created), before);
      await k.fill('1'); await h.fill('2'); await apply.click(); const custom = { ...regularPerturbationDefault, source: '2' }; await ready(custom);
      await page.reload(); await ready(custom); assert.equal(await h.inputValue(), '2');
      await page.getByRole('button', { name: t('첫 예제로 초기화', 'Reset to first example'), exact: true }).click(); await ready(regularPerturbationDefault);
      const energy = page.locator('[data-perturbation-energy]'); await energy.scrollIntoViewIfNeeded();
      await energy.getByRole('button', { name: t('3D 보기', 'Show 3D'), exact: true }).click(); const canvas = energy.locator('canvas'); await canvas.waitFor();
      await energy.getByRole('button', { name: t('정면', 'Front'), exact: true }).click(); const initial = await canvas.screenshot(); await canvas.press('ArrowRight'); await canvas.press('ArrowUp'); assert(!initial.equals(await canvas.screenshot()));
      await canvas.evaluate(el => { const extension = (el as HTMLCanvasElement).getContext('webgl2')?.getExtension('WEBGL_lose_context'); if (!extension) throw Error('No context loss extension'); extension.loseContext(); });
      await page.locator('[data-perturbation-energy="2d"] svg').waitFor(); await axe();
      await energy.getByRole('button', { name: t('3D 다시 시도', 'Retry 3D'), exact: true }).click(); await energy.locator('canvas').waitFor();
      await energy.screenshot({ path: `${root}/${language}-${dark ? 'dark' : 'light'}-${width}.png` });
      if (language === 'en') assert(!/[가-힣]/.test(await page.locator('main').innerText()));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      const counts = await page.evaluate(() => (window as unknown as Harness).__counts); assert.equal(counts.created, counts.terminated); assert.deepEqual(errors, []);
      states.push({ language, dark, width, examples: 14, concepts: 8, guides: 6, explainedQuestions: 7, displayControlsDoNotCalculate: true, invalidInputPreservesResult: true, customInputReload: true, keyboardCamera: true, webglLossRetry: true, workerCountsWindow: 'since-final-reload', ...counts, errors });
      console.log(JSON.stringify({ cases: states.length, language, dark, width })); await context.close();
    }
    writeFileSync(`${root}/native-verification.json`, JSON.stringify({ states, accessibility, readerRegistered: false, productionNextAssetsVerified: false }, null, 2) + '\n');
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
