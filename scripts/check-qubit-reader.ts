import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { chromium, type Page } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { qubitTeaching } from '../src/data/qubit-teaching';
import { qubitTeachingPresets } from '../src/lib/learning-qubit-teaching-input';
import { qubitLabBrowserInstrumentation } from './learning/qubit-lab-browser-instrumentation';
const root = '.runtime/learning/quantum-qubit-reader-integration', base = process.env.QUBIT_PREVIEW_BASE ?? 'http://localhost:4176';
async function main() {
  const browser = await chromium.launch({ headless: true }), errors: string[] = [], states: unknown[] = [], flows: unknown[] = [];
  const assets = new Set<string>(); let accessibilityStates = 0, conceptSteps = 0, answeredQuestions = 0;
  const ready = async (page: Page) => page.waitForFunction(() => document.querySelector('[data-qubit-lab]')?.getAttribute('data-qubit-phase') === 'ready');
  const a11y = async (page: Page, label: string) => { const result = await new AxeBuilder({ page }).include('.ai-reader').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze(); assert.deepEqual(result.violations, [], label); accessibilityStates++; };
  const life = async (page: Page) => { const result = await page.evaluate(() => (window as unknown as { __qubitLifecycle: { created: number; terminated: number } }).__qubitLifecycle); assert.equal(result.created, result.terminated); return result; };
  try {
    for (const language of ['ko', 'en'] as const) for (const dark of [false, true]) for (const width of [320, 1440]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await context.addInitScript({ content: qubitLabBrowserInstrumentation });
      await context.addInitScript(({ language, dark }) => { localStorage.setItem('suanlab-language', language); localStorage.setItem('theme', dark ? 'dark' : 'light'); }, { language, dark });
      const page = await context.newPage(); page.on('pageerror', error => errors.push(error.message));
      page.on('response', response => { const url = new URL(response.url()); if (url.origin === new URL(base).origin && url.pathname.startsWith('/_next/static/') && url.pathname.endsWith('.js')) assets.add(url.pathname); });
      for (const item of qubitTeaching) {
        const kind = item.unit.id;
        await page.goto(base + '/book/online/physics-math-for-ai/' + kind + '/', { waitUntil: 'domcontentloaded' });
        await ready(page); await page.waitForFunction(lang => document.querySelector('.ai-reader')?.getAttribute('lang') === lang, language);
        assert.equal(await page.locator('h1').innerText(), item.unit.title[language]);
        assert(!await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), kind + ' overflow');
        if (language === 'en') assert(!/[가-힣]/.test(await page.locator('.ai-reader').innerText()), kind + ' untranslated');
        assert.equal(await page.locator('[data-qubit-reader-lab]').getAttribute('data-qubit-reader-lab'), kind);
        assert.equal(await page.locator('[data-qubit-concept]').getAttribute('data-qubit-concept'), kind);
        await a11y(page, `${kind}/${language}/${dark}/${width}`);
        const beforeReload = await life(page); states.push({ kind, language, dark, width, accessibilityPassed: true, bodyOverflow: false, workersReleased: true });
        for (let step = 0; step < item.guide.steps.length; step++) {
          await page.locator('[data-qubit-concept] .ai-concept-steps button').nth(step).click();
          await page.locator('[data-qubit-concept] button[aria-controls]').press('Enter');
          assert.equal(await page.locator('[data-qubit-concept-explanation]').innerText(), item.guide.experiments[step].explain[language]); conceptSteps++;
        }
        for (const [index, question] of item.unit.questions.entries()) {
          const field = page.locator('.ai-question').nth(index), radios = field.locator('input[type=radio]');
          await radios.nth((question.answer + 1) % 3).check(); await field.getByRole('button').click();
          assert((await field.locator('.ai-answer strong').innerText()).startsWith(language === 'ko' ? '다시 생각해보세요' : 'Try again.'));
          await radios.nth(question.answer).check(); await field.getByRole('button').click();
          assert.equal(await field.locator('.ai-answer p').innerText(), question.explanation[language]); answeredQuestions++;
        }
        await page.locator('.ai-completion input').check(); await a11y(page, kind + '/quiz');
        const all = page.getByLabel(language === 'ko' ? '전체 입력 예제' : 'All input examples', { exact: true });
        if (kind === 'quantum-instruments-records-feedforward') { await all.selectOption('7'); await ready(page); assert.equal(await page.locator('[data-qubit-result]').getAttribute('data-qubit-result'), 'deferred'); await a11y(page, kind + '/deferred'); }
        const figure = page.locator('[data-qubit-bloch]').first(); await figure.scrollIntoViewIfNeeded();
        await page.waitForFunction(() => document.querySelector('[data-qubit-bloch]')?.getAttribute('data-qubit-bloch') === '3d');
        const canvas = figure.locator('canvas'); assert(await canvas.isVisible());
        await figure.getByRole('button', { name: language === 'ko' ? '정면' : 'Front', exact: true }).click();
        await figure.getByRole('button', { name: language === 'ko' ? '위에서' : 'Top', exact: true }).click();
        if (language === 'en') await page.screenshot({ path: `${root}/${kind}-${dark ? 'dark' : 'light'}-${width}.png` });
        await canvas.evaluate(canvas => { const gl = (canvas as HTMLCanvasElement).getContext('webgl2'), extension = gl?.getExtension('WEBGL_lose_context'); if (!extension) throw Error('WebGL loss not available'); extension.loseContext(); });
        await page.waitForFunction(() => document.querySelector('[data-qubit-bloch]')?.getAttribute('data-qubit-bloch') === '2d');
        await figure.getByRole('button', { name: language === 'ko' ? '3D 다시 시도' : 'Retry 3D', exact: true }).click();
        await page.waitForFunction(() => document.querySelector('[data-qubit-bloch]')?.getAttribute('data-qubit-bloch') === '3d');
        const last = qubitTeachingPresets[kind].length - 1; await all.selectOption(String(last)); await ready(page);
        const completedRun = await life(page);
        await page.reload(); await ready(page);
        assert.equal(await all.inputValue(), String(last)); assert(await page.locator('.ai-completion input').isChecked());
        for (const [index, question] of item.unit.questions.entries()) assert(await page.locator('.ai-question').nth(index).locator('input[type=radio]').nth(question.answer).isChecked());
        const afterReload = await life(page);
        await page.evaluate(() => { (window as unknown as { __qubitMode: string }).__qubitMode = 'hang'; });
        await all.selectOption('0'); await page.waitForFunction(() => document.querySelector('[data-qubit-lab]')?.getAttribute('data-qubit-phase') === 'pending');
        assert.equal(await page.locator('[data-qubit-result]').count(), 0);
        await page.locator('.ai-reader-top a').first().click(); await page.locator('.ai-book-overview').waitFor();
        assert.equal(await page.locator('[data-qubit-lab]').count(), 0); const afterUnmount = await life(page);
        flows.push({ kind, language, dark, width, beforeReload, completedRun, afterReload, afterUnmount, inputQuizCompletionReload: true, conceptStepsVerified: item.guide.steps.length, explainedQuestionsVerified: item.unit.questions.length, productionThreeDAndLossRetry: true, pendingUnmountVerified: true });
        console.log(JSON.stringify({ cases: states.length, accessibilityStates, kind, language, dark, width }));
      }
      await context.close();
    }
    writeFileSync(root + '/reader-browser-flows.json', JSON.stringify({ states, flows, accessibilityStates, conceptSteps, answeredQuestions, requestedAssetPaths: [...assets], errors, actualReaderVerified: true, qaWorkerFactoryOverride: false }, null, 2) + '\n');
    const chunks = [];
    for (const pathname of assets) {
      const response = await fetch(base + pathname); assert.equal(response.status, 200);
      const diskPath = 'out' + decodeURIComponent(pathname);
      const served = Buffer.from(await response.arrayBuffer()), disk = readFileSync(diskPath); assert.deepEqual(served, disk);
      chunks.push({ pathname, diskPath, bytes: disk.length, sha256: createHash('sha256').update(disk).digest('hex') });
    }
    assert.deepEqual(errors, []); assert.equal(states.length, 32); assert.equal(accessibilityStates, 72); assert.equal(conceptSteps, 104); assert.equal(answeredQuestions, 104);
    writeFileSync(root + '/reader.json', JSON.stringify({ states, flows, accessibilityStates, conceptSteps, answeredQuestions, productionChunks: chunks, errors, actualReaderVerified: true, qaWorkerFactoryOverride: false, productionThreeAssetsVerified: true }, null, 2) + '\n');
    console.log(JSON.stringify({ readerCases: 32, accessibilityStates, conceptSteps, answeredQuestions, productionChunks: chunks.length }));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
