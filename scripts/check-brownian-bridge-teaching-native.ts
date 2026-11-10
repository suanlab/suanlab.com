import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { brownianBridgeDefault, brownianBridgeExamples, type BrownianBridgeInput } from '../src/lib/learning-brownian-bridge-input';
import { brownianBridgeInputKey } from '../src/lib/learning-brownian-bridge-protocol';
import { calculateBrownianBridge } from '../src/lib/learning-brownian-bridge';
import { brownianBridgeSteps, brownianBridgeExperiments, brownianBridgeQuestions } from '../src/data/brownian-bridge-teaching';
const root = '.runtime/learning/brownian-bridge-teaching';
type Counts = { created: number; closed: number; active: number };
async function main() {
  const language = process.env.BROWNIAN_LANGUAGE as 'ko' | 'en', theme = process.env.BROWNIAN_THEME as 'light' | 'dark';
  assert(['ko', 'en'].includes(language) && ['light', 'dark'].includes(theme));
  const { url } = JSON.parse(readFileSync(`${root}/review-server.json`, 'utf8')), browser = await chromium.launch({ headless: true }), rows = [];
  try {
    for (const width of [320, 1440]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await context.addInitScript({ content: 'globalThis.__name=fn=>fn;' });
      await context.addInitScript(({ language, theme }) => {
        localStorage.setItem('suanlab-language', language); localStorage.setItem('theme', theme);
        const Native = Worker, active = new Set<Worker>(), counts = { created: 0, closed: 0, active: 0 }; (window as unknown as { __brownianCounts: Counts }).__brownianCounts = counts;
        window.Worker = class extends Native { constructor(url: string | URL, options?: WorkerOptions) { super(url, options); active.add(this); counts.created++; counts.active = active.size; } terminate() { if (active.delete(this)) { counts.closed++; counts.active = active.size; } super.terminate(); } };
      }, { language, theme });
      const page = await context.newPage(), errors: string[] = [], t = (ko: string, en: string) => language === 'ko' ? ko : en;
      page.on('pageerror', error => errors.push(error.message));
      const ready = async (input: BrownianBridgeInput) => { await page.waitForFunction(key => document.querySelector('[data-brownian-result]')?.getAttribute('data-brownian-result-key') === key, brownianBridgeInputKey(input), { timeout: 60000 }); await page.waitForFunction(() => (window as unknown as { __brownianCounts: Counts }).__brownianCounts.active === 0); };
      await page.goto(url); await ready(brownianBridgeDefault);
      for (const [i, s] of brownianBridgeSteps.entries()) { await page.getByRole('button', { name: `${i + 1}. ${s.title[language]}`, exact: true }).click(); assert.equal(await page.locator('.ai-concept-explanation >p').first().innerText(), s.body[language]); assert.equal(await page.locator('svg[data-brownian-reference]').getAttribute('data-brownian-reference'), String(i)); if (theme === 'light' && width === 1440) await page.locator('svg[data-brownian-reference]').screenshot({ path: `${root}/reference-${language}-${i}.png` }); }
      for (const [i, guide] of brownianBridgeExperiments.entries()) { await page.getByLabel(t('안내 실험', 'Guided experiment'), { exact: true }).selectOption(String(i)); await page.getByRole('button', { name: t('해설 보기', 'Show explanation'), exact: true }).click(); assert.equal(await page.locator('[data-brownian-guide-explanation]').innerText(), guide.explanation[language]); }
      for (const [i, question] of brownianBridgeQuestions.entries()) {
        const field = page.locator('.information-question').nth(i);
        for (const answer of [(question.correct + 1) % question.options.length, question.correct]) { await field.locator('input').nth(answer).check(); assert((await field.locator('[role=status]').innerText()).includes(question.explanation[language])); }
      }
      const examples = language === 'ko' && theme === 'light' && width === 320 ? brownianBridgeExamples : brownianBridgeExamples.filter(c => ['one-interior-coordinate', 'middle-source', 'maximum-coprime'].includes(c.id));
      for (const c of examples) {
        await page.getByLabel(t('전체 예제', 'All examples'), { exact: true }).selectOption(c.id); await ready(c.input);
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
        const expected = calculateBrownianBridge(c.input);
        const row = page.getByRole('region', { name: t('현재 격자의 관측량', 'Current-grid observables'), exact: true }).locator('tbody tr').filter({ has: page.locator('td', { hasText: /^Var B_N$/ }) });
        assert.equal(await row.locator('td').nth(1).innerText(), expected.area.variance);
      }
      await page.getByLabel(t('전체 예제', 'All examples'), { exact: true }).selectOption('four-interval-hand-check'); await ready(brownianBridgeDefault);
      const beforeDisplay = await page.evaluate(() => (window as unknown as { __brownianCounts: Counts }).__brownianCounts.created);
      for (const [kind, name, value] of [['position', t('위치 공분산 C', 'Position covariance C'), '0'], ['increment', t('증분 공분산', 'Increment covariance'), '3/8'], ['precision', t('내부 정밀도 Q', 'Interior precision Q'), '4']]) {
        await page.getByLabel(t('표시 행렬', 'Displayed matrix'), { exact: true }).selectOption(kind);
        const first = page.getByRole('group', { name, exact: true }).getByRole('button').first(); await first.click();
        assert((await page.locator('[data-brownian-selected-entry]').innerText()).includes(`= ${value}.`));
        await first.press('ArrowRight'); assert.equal(await page.evaluate(() => document.activeElement?.getAttribute('aria-pressed')), 'true');
      }
      await page.getByLabel(t('행렬 표시 척도', 'Matrix display scale'), { exact: true }).selectOption('signed-log');
      await page.getByRole('button', { name: t('3D 행렬 보기', 'Show 3D matrix'), exact: true }).click();
      await page.locator('[data-brownian-matrix-plot="3d"]').waitFor({ timeout: 60000 });
      const canvas = page.locator('[data-brownian-matrix-plot] canvas'); await canvas.focus(); await canvas.press('ArrowLeft'); await canvas.press('Home');
      await canvas.evaluate(element => { const gl = (element as HTMLCanvasElement).getContext('webgl2'); const extension = gl?.getExtension('WEBGL_lose_context'); if (!extension) throw Error('Context-loss test unavailable'); extension.loseContext(); });
      await page.locator('[data-brownian-matrix-plot="2d"]').waitFor(); await page.getByRole('button', { name: t('3D 다시 시도', 'Retry 3D'), exact: true }).click(); await page.locator('[data-brownian-matrix-plot="3d"]').waitFor();
      assert.equal(await page.evaluate(() => (window as unknown as { __brownianCounts: Counts }).__brownianCounts.created), beforeDisplay);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
      if (language === 'en') assert(!/[가-힣]/.test(await page.locator('main').innerText()));
      const axe = await new AxeBuilder({ page }).include('main').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze(); assert.deepEqual(axe.violations.map(v => ({ id: v.id, targets: v.nodes.map(n => n.target) })), []);
      await page.locator('[data-brownian-selected-entry]').scrollIntoViewIfNeeded(); await page.screenshot({ path: `${root}/matrix-${language}-${theme}-${width}.png` });
      const counts = await page.evaluate(() => (window as unknown as { __brownianCounts: Counts }).__brownianCounts); assert.equal(counts.created, counts.closed); assert.equal(counts.active, 0); assert.deepEqual(errors, []);
      rows.push({ language, theme, width, concepts: 8, guides: 6, questions: 8, exampleApplications: examples.length, threeDimensions: true, keyboard: true, contextLossRecovery: true, accessibility: true, displayControlsDoNotCalculate: true, counts, errors });
      await context.close();
    }
    writeFileSync(`${root}/native-${language}-${theme}.json`, JSON.stringify({ rows }, null, 2) + '\n');
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
