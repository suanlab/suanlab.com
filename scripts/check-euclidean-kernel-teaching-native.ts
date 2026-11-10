import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { euclideanKernelDefault, euclideanKernelExamples, type EuclideanKernelInput } from '../src/lib/learning-euclidean-kernel-input';
import { euclideanKernelInputKey } from '../src/lib/learning-euclidean-kernel-protocol';
import type {EuclideanKernelResult} from '../src/lib/learning-euclidean-kernel';
const reference=JSON.parse(readFileSync('scripts/fixtures/euclidean-kernel-references.json','utf8'))as{cases:{id:string;expected:EuclideanKernelResult}[]};
import { euclideanKernelSteps, euclideanKernelExperiments, euclideanKernelQuestions } from '../src/data/euclidean-kernel-teaching';
const root = '.runtime/learning/euclidean-kernel-teaching';
type Counts = { created: number; closed: number; active: number };
async function main() {
  const language = process.env.EUCLIDEAN_LANGUAGE as 'ko' | 'en', theme = process.env.EUCLIDEAN_THEME as 'light' | 'dark';
  assert(['ko', 'en'].includes(language) && ['light', 'dark'].includes(theme));
  const { url } = JSON.parse(readFileSync(`${root}/review-server.json`, 'utf8')), browser = await chromium.launch({ headless: true }), rows = [];
  try {
    for (const width of [320, 1440]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await context.addInitScript({ content: 'globalThis.__name=fn=>fn;' });
      await context.addInitScript(({ language, theme }) => {
        localStorage.setItem('suanlab-language', language); localStorage.setItem('theme', theme);
        const Native = Worker, active = new Set<Worker>(), counts = { created: 0, closed: 0, active: 0 }; (window as unknown as { __euclideanCounts: Counts }).__euclideanCounts = counts;
        window.Worker = class extends Native { constructor(url: string | URL, options?: WorkerOptions) { super(url, options); active.add(this); counts.created++; counts.active = active.size; } terminate() { if (active.delete(this)) { counts.closed++; counts.active = active.size; } super.terminate(); } };
      }, { language, theme });
      const page = await context.newPage(), errors: string[] = [], t = (ko: string, en: string) => language === 'ko' ? ko : en;
      page.on('pageerror', error => errors.push(error.message));
      const ready = async (input: EuclideanKernelInput) => { await page.waitForFunction(key => document.querySelector('[data-euclidean-result]')?.getAttribute('data-euclidean-result-key') === key, euclideanKernelInputKey(input), { timeout: 120000 }); await page.waitForFunction(() => (window as unknown as { __euclideanCounts: Counts }).__euclideanCounts.active === 0); };
      await page.goto(url); await ready(euclideanKernelDefault);
      for (const [i, s] of euclideanKernelSteps.entries()) { await page.getByRole('button', { name: `${i + 1}. ${s.title[language]}`, exact: true }).click(); assert.equal(await page.locator('.ai-concept-explanation >p').first().innerText(), s.body[language]); assert.equal(await page.locator('svg[data-euclidean-reference]').getAttribute('data-euclidean-reference'), String(i)); if (theme === 'light' && width === 1440) await page.locator('svg[data-euclidean-reference]').screenshot({ path: `${root}/reference-${language}-${i}.png` }); }
      for (const [i, guide] of euclideanKernelExperiments.entries()) { await page.getByLabel(t('안내 실험', 'Guided experiment'), { exact: true }).selectOption(String(i)); await page.getByRole('button', { name: t('해설 보기', 'Show explanation'), exact: true }).click(); assert.equal(await page.locator('[data-euclidean-guide-explanation]').innerText(), guide.explanation[language]); }
      for (const [i, question] of euclideanKernelQuestions.entries()) {
        const field = page.locator('.information-question').nth(i);
        for (const answer of [(question.correct + 1) % question.options.length, question.correct]) { await field.locator('input').nth(answer).check(); assert((await field.locator('[role=status]').innerText()).includes(question.explanation[language])); }
      }
      const examples = language === 'ko' && theme === 'light' && width === 320 ? euclideanKernelExamples : euclideanKernelExamples.filter(c => ['one-variable-hand-check', 'off-diagonal-error-comparison', 'mixed-coprime-denominators'].includes(c.id));
      for (const c of examples) {
        await page.getByLabel(t('전체 예제', 'All examples'), { exact: true }).selectOption(c.id); await ready(c.input);
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
        const expected = reference.cases.find(row=>row.id===c.id)!.expected;
        const row = page.getByRole('region', { name: t('커널과 정규화의 정확한 값', 'Exact kernel and normalization values'), exact: true }).locator('tbody tr').filter({ has: page.locator('td', { hasText: /^A_min$/ }) });
        assert.equal(await row.locator('td').nth(1).innerText(), expected.kernel.minimumAction);
      }
      await page.getByLabel(t('전체 예제', 'All examples'), { exact: true }).selectOption('one-variable-hand-check'); await ready(euclideanKernelDefault);
      const beforeDisplay = await page.evaluate(() => (window as unknown as { __euclideanCounts: Counts }).__euclideanCounts.created);
      for (const [kind, name, value] of [['position', t('위치 공분산 C', 'Position covariance C'), '0'], ['precision', t('내부 정밀도 Q', 'Interior precision Q'), '9/2']]) {
        await page.getByLabel(t('표시 행렬', 'Displayed matrix'), { exact: true }).selectOption(kind);
        const first = page.getByRole('group', { name, exact: true }).getByRole('button').first(); await first.click();
        assert((await page.locator('[data-euclidean-selected-entry]').innerText()).includes(`= ${value}.`));
        await first.press('ArrowRight'); assert.equal(await page.evaluate(() => document.activeElement?.getAttribute('aria-pressed')), 'true');
      }
      await page.getByLabel(t('행렬 표시 척도', 'Matrix display scale'), { exact: true }).selectOption('signed-log');
      await page.getByRole('button', { name: t('3D 행렬 보기', 'Show 3D matrix'), exact: true }).click();
      await page.locator('[data-euclidean-matrix-plot="3d"]').waitFor({ timeout: 120000 });
      const canvas = page.locator('[data-euclidean-matrix-plot] canvas'); await canvas.focus(); await canvas.press('ArrowLeft'); await canvas.press('Home');
      await canvas.evaluate(element => { const gl = (element as HTMLCanvasElement).getContext('webgl2'); const extension = gl?.getExtension('WEBGL_lose_context'); if (!extension) throw Error('Context-loss test unavailable'); extension.loseContext(); });
      await page.locator('[data-euclidean-matrix-plot="2d"]').waitFor(); await page.getByRole('button', { name: t('3D 다시 시도', 'Retry 3D'), exact: true }).click(); await page.locator('[data-euclidean-matrix-plot="3d"]').waitFor();
      assert.equal(await page.evaluate(() => (window as unknown as { __euclideanCounts: Counts }).__euclideanCounts.created), beforeDisplay);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
      if (language === 'en') assert(!/[가-힣]/.test(await page.locator('main').innerText()));
      const axe = await new AxeBuilder({ page }).include('main').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze(); assert.deepEqual(axe.violations.map(v => ({ id: v.id, targets: v.nodes.map(n => n.target) })), []);
      await page.locator('[data-euclidean-selected-entry]').scrollIntoViewIfNeeded(); await page.screenshot({ path: `${root}/matrix-${language}-${theme}-${width}.png` });
      const counts = await page.evaluate(() => (window as unknown as { __euclideanCounts: Counts }).__euclideanCounts); assert.equal(counts.created, counts.closed); assert.equal(counts.active, 0); assert.deepEqual(errors, []);
      rows.push({ language, theme, width, concepts: 8, guides: 6, questions: 8, exampleApplications: examples.length, threeDimensions: true, keyboard: true, contextLossRecovery: true, accessibility: true, displayControlsDoNotCalculate: true, counts, errors });
      await context.close();
    }
    writeFileSync(`${root}/native-${language}-${theme}.json`, JSON.stringify({ rows }, null, 2) + '\n');
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
