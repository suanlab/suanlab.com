import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { spatialSpectrumDefault, spatialSpectrumExamples } from '../src/lib/learning-spatial-spectrum-input';
import { calculateSpatialSpectrum } from '../src/lib/learning-spatial-spectrum';
import { spatialSpectrumSteps, spatialSpectrumExperiments, spatialSpectrumQuestions } from '../src/data/spatial-spectrum-teaching';
const root = '.runtime/learning/spatial-spectrum-teaching', { url } = JSON.parse(readFileSync(root + '/review-server.json', 'utf8'));
type Harness = Window & { __counts: { created: number; closed: number; active: number } };
async function main() {
  const part = process.env.SPATIAL_TEACHING_LANGUAGE; if (part && !['ko', 'en'].includes(part)) throw Error('Expected ko or en');
  const expected = new Map(spatialSpectrumExamples.map(c => [c.id, calculateSpatialSpectrum(c.input).covariance[0][0]]));
  const browser = await chromium.launch({ headless: true }), states = []; let accessibility = 0;
  try {
    for (const language of ['ko', 'en'] as const) for (const dark of [false, true]) for (const width of [320, 1440]) {
      if (part && part !== language) continue;
      const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce' });
      await context.addInitScript(({ language, dark }) => {
        localStorage.setItem('suanlab-language', language); localStorage.setItem('theme', dark ? 'dark' : 'light');
        const Native = Worker, active = new Set<Worker>(), counts = { created: 0, closed: 0, active: 0 }; (window as unknown as Harness).__counts = counts;
        window.Worker = class extends Native { constructor(url: string | URL, options?: WorkerOptions) { super(url, options); active.add(this); counts.created++; counts.active = active.size; } terminate() { if (active.delete(this)) { counts.closed++; counts.active = active.size; } super.terminate(); } };
      }, { language, dark });
      const page = await context.newPage(), errors: string[] = [], t = (ko: string, en: string) => language === 'ko' ? ko : en;
      page.on('pageerror', e => errors.push(e.message));
      const ready = async (input: object) => { await page.waitForFunction(key => document.querySelector('[data-spatial-result]')?.getAttribute('data-spatial-result-key') === key, JSON.stringify(input), { timeout: 60000 }); await page.waitForFunction(() => (window as unknown as Harness).__counts.active === 0); };
      const axe = async () => { const report = await new AxeBuilder({ page }).include('main').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze(); assert.deepEqual(report.violations.map(v => ({ id: v.id, targets: v.nodes.map(n => n.target) })), []); accessibility++; };
      await page.goto(url); await ready(spatialSpectrumDefault);
      for (const [i, s] of spatialSpectrumSteps.entries()) { await page.getByRole('button', { name: `${i + 1}. ${s.title[language]}`, exact: true }).click(); assert.equal(await page.locator('.information-step >p').first().innerText(), s.body[language]); assert.equal(await page.locator('[data-spatial-reference]').getAttribute('data-spatial-reference'), String(i)); }
      for (const [i, g] of spatialSpectrumExperiments.entries()) { await page.getByLabel(t('안내 실험', 'Guided experiment'), { exact: true }).selectOption(String(i)); await page.getByRole('button', { name: t('해설 보기', 'Show explanation'), exact: true }).click(); assert.equal(await page.locator('[data-spatial-guide-explanation]').innerText(), g.explanation[language]); }
      for (const [i, q] of spatialSpectrumQuestions.entries()) { const field = page.locator('.information-question').nth(i); for (const answer of [(q.correct + 1) % 3, q.correct]) { await field.locator('input').nth(answer).check(); assert((await field.locator('[role=status]').innerText()).includes(q.explanation[language])); } }
      const ids = language === 'ko' && !dark && width === 320 ? spatialSpectrumExamples.map(c => c.id) : ['massless-ring-constrained', 'open-chain', 'maximum-coprime'];
      for (const caseId of ids) {
        const c = spatialSpectrumExamples.find(c => c.id === caseId)!;
        if (await page.locator('[data-spatial-result]').getAttribute('data-spatial-result-key') !== JSON.stringify(c.input)) { await page.getByLabel(t('전체 예제', 'All examples'), { exact: true }).selectOption(caseId); await ready(c.input); }
        const selected = page.getByRole('table', { name: t('선택 성분의 정확한 분수와 구간', 'Exact fractions and intervals of the selected entry'), exact: true });
        assert.equal(await selected.locator('tbody tr').first().locator('td').nth(3).innerText(), expected.get(caseId)); assert.equal(await page.locator('[data-spatial-cell]').count(), c.input.sites ** 2);
        const before = await page.evaluate(() => (window as unknown as Harness).__counts.created);
        await page.locator('[data-spatial-cell="0-0"]').focus(); await page.keyboard.press('ArrowRight'); await page.keyboard.press('ArrowDown'); assert.equal(await page.locator('[data-spatial-cell="1-1"]').getAttribute('aria-pressed'), 'true');
        for (const kind of ['precision', 'fourierReal', 'fourierImaginary', 'covariance']) await page.getByLabel(t('비교할 행렬', 'Matrix to inspect'), { exact: true }).selectOption(kind);
        await page.getByLabel(t('표시 척도', 'Display scale'), { exact: true }).selectOption('signed-log');
        await page.getByLabel(t('관찰 모드', 'Inspected mode'), { exact: true }).fill(String(c.input.sites - 1)); assert.equal(await page.locator('[data-spatial-inspected-mode]').getAttribute('data-spatial-inspected-mode'), String(c.input.sites - 1));
        assert(await page.getByRole('button', { name: t('모드 재생', 'Play modes'), exact: true }).isDisabled()); assert.equal(await page.evaluate(() => (window as unknown as Harness).__counts.created), before);
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), caseId + ' overflow'); if (language === 'en') assert(!/[가-힣]/.test(await page.locator('main').innerText()));
      }
      await axe(); await page.getByRole('button', { name: t('3D 행렬 보기', 'Show 3D matrix'), exact: true }).click(); await page.locator('[data-spatial-matrix-plot="3d"]').waitFor();
      const canvas = page.locator('[data-spatial-matrix-plot] canvas'); assert((await canvas.boundingBox())!.height >= 300); await canvas.focus(); const before = await canvas.screenshot(); await page.keyboard.press('ArrowRight'); assert(!before.equals(await canvas.screenshot())); await page.keyboard.press('Home');
      await canvas.evaluate(el => { const extension = (el as HTMLCanvasElement).getContext('webgl2')?.getExtension('WEBGL_lose_context'); if (!extension) throw Error('Context loss unavailable'); extension.loseContext(); });
      await page.getByRole('button', { name: t('3D 다시 시도', 'Retry 3D'), exact: true }).click(); await page.locator('[data-spatial-matrix-plot="3d"]').waitFor(); await axe(); await page.locator('[data-spatial-matrix-plot]').screenshot({ path: `${root}/matrix-${language}-${dark ? 'dark' : 'light'}-${width}.png` });
      await page.getByRole('button', { name: t('3D 행렬 닫기', 'Close 3D matrix'), exact: true }).click(); assert.equal(await page.locator('[data-spatial-matrix-plot] canvas').count(), 0);
      const key = await page.locator('[data-spatial-result]').getAttribute('data-spatial-result-key'); await page.getByLabel(t('관측 원점', 'Observation origin'), { exact: true }).fill('16'); await page.getByRole('button', { name: t('입력 적용', 'Apply input'), exact: true }).click(); assert.equal(await page.locator('form [role=alert]').count(), 1); assert.equal(await page.locator('[data-spatial-result]').getAttribute('data-spatial-result-key'), key);
      const counts = await page.evaluate(() => (window as unknown as Harness).__counts); assert.equal(counts.active, 0); assert.equal(counts.created, counts.closed); assert.deepEqual(errors, []);
      states.push({ language, dark, width, examples: ids.length, counts, errors }); console.log(JSON.stringify({ done: states.length, language, dark, width })); await context.close();
    }
    writeFileSync(root + '/native-teaching' + (part ? '-' + part : '') + '.json', JSON.stringify({ states, accessibility, readerRegistered: false, threeDimensionalViewVerified: true }, null, 2) + '\n');
  } finally { await browser.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
