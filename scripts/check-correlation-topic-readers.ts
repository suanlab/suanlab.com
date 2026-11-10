import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { gaussianFieldReader } from '../src/data/gaussian-field-reader';
import { nongaussianFieldReader } from '../src/data/nongaussian-field-reader';
import { spatialSpectrumReader } from '../src/data/spatial-spectrum-reader';
const root = '.runtime/learning/correlation-function-topic-audit';
mkdirSync(root, { recursive: true });
const families = { gaussian: gaussianFieldReader, nongaussian: nongaussianFieldReader, spatial: spatialSpectrumReader };
type Counts = { created: number; terminated: number };
async function main() {
  const family = process.env.CORRELATION_AUDIT_UNIT as keyof typeof families;
  const language = process.env.CORRELATION_AUDIT_LANGUAGE as 'ko' | 'en';
  const theme = process.env.CORRELATION_AUDIT_THEME;
  assert(Object.hasOwn(families, family) && ['ko', 'en'].includes(language) && ['light', 'dark'].includes(theme!));
  const reader = families[family], base = process.env.CORRELATION_PREVIEW_BASE ?? 'http://127.0.0.1:4175', rows = [];
  const browser = await chromium.launch({ headless: true });
  try {
    for (const width of [320, 1440]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await context.addInitScript(({ language, theme }) => {
        localStorage.setItem('suanlab-language', language); localStorage.setItem('theme', theme!);
        const Native = Worker, counts = { created: 0, terminated: 0 }; (window as unknown as { __correlationWorkers: Counts }).__correlationWorkers = counts;
        window.Worker = class extends Native { private closed = false; constructor(url: string | URL, options?: WorkerOptions) { super(url, options); counts.created++; } terminate() { if (!this.closed) { this.closed = true; counts.terminated++; } super.terminate(); } };
      }, { language, theme });
      const page = await context.newPage(), errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
      await page.goto(base + '/book/online/physics-math-for-ai/' + reader.unit.id + '/');
      await page.locator(`[data-${family}-phase="ready"]`).waitFor({ timeout: 60000 });
      assert.equal(await page.locator('h1').innerText(), reader.unit.title[language]);
      for (const [i, step] of reader.guide.steps.entries()) {
        await page.locator('.ai-intuition .ai-concept-steps button').nth(i).click();
        assert.equal(await page.locator('.ai-concept-explanation >p').first().innerText(), step.explanation[language]);
      }
      for (const [i, question] of reader.unit.questions.entries()) {
        const field = page.locator('.ai-question').nth(i), radios = field.locator('input[type=radio]');
        await radios.nth((question.answer + 1) % question.options.length).check(); await field.getByRole('button').click();
        assert((await field.locator('.ai-answer strong').innerText()).startsWith(language === 'ko' ? '다시 생각해보세요' : 'Try again.'));
        await radios.nth(question.answer).check(); await field.getByRole('button').click();
        assert.equal(await field.locator('.ai-answer p').innerText(), question.explanation[language]);
      }
      const lab = page.locator('.ai-experiment-column');
      for (const [i, guide] of reader.guide.experiments.entries()) {
        await lab.getByLabel(language === 'ko' ? '안내 실험' : 'Guided experiment', { exact: true }).selectOption(String(i));
        await lab.getByRole('button', { name: language === 'ko' ? '해설 보기' : 'Show explanation', exact: true }).click();
        assert.equal(await lab.locator(`[data-${family}-guide-explanation]`).innerText(), guide.explain[language]);
      }
      if (language === 'en') assert(!/[가-힣]/.test(await page.locator('.ai-reader').innerText()));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      const axe = await new AxeBuilder({ page }).include('.ai-reader').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze(); assert.deepEqual(axe.violations.map(v => v.id), []);
      const counts = await page.evaluate(() => (window as unknown as { __correlationWorkers: Counts }).__correlationWorkers);
      assert.equal(counts.created, counts.terminated); assert.deepEqual(errors, []);
      rows.push({ family, unitId: reader.unit.id, language, theme, width, concepts: reader.guide.steps.length, questions: reader.unit.questions.length, guides: reader.guide.experiments.length, counts, accessibility: true, errors });
      console.log(JSON.stringify(rows.at(-1))); await context.close();
    }
    writeFileSync(`${root}/readers-${family}-${language}-${theme}.json`, JSON.stringify({ rows, actualProductionReader: true }, null, 2) + '\n');
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
